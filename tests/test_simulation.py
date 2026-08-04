"""
Simulated-environment tests.

Runs the full agent/training loop against a lightweight in-process game
simulation (no BizHawk, no sockets) to verify:

  * The peek/shoot gating allows fire when and only when expected.
  * ES perturbations produce non-zero fitness variance (the update can learn).
  * A mini training run completes cleanly and produces sensible diagnostics.

Run from the project root:
    python3 -m unittest tests/test_simulation.py -v
"""

import os
import sys
import unittest

import numpy as np

# Ensure the project root is on sys.path when running this file directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    ACT_DIM, AMMO_MAX_ROUNDS, HIDDEN, OBS_DIM, PEEK_TRAVERSE_TICKS, RAM,
)
from env_timecrisis import TimeCrisisEnv
from phase_inference import PhaseInferer
from policy import PARAM_COUNT


# ---------------------------------------------------------------------------
# Simulated game
# ---------------------------------------------------------------------------

class SimulatedGame:
    """Minimal Time Crisis game simulation.

    Models:
      * a countdown timer (heuristic clear = timer jumps up)
      * an enemy health pool  (ENEMIES hits clear the screen)
      * shooting while peeking (hit probability scales with aim centering)
      * a magazine capped at AMMO_MAX_ROUNDS: once empty, further trigger
        pulls have no game-side effect (no shots_fired/shots_hit increment)
        until the character ducks back into cover, which reloads a full clip.
        This is an independent implementation from env_timecrisis.py's own
        software ammo_left tracker -- if the two ever disagree it signals a
        real bug rather than the sim just trivially agreeing with itself.
      * incoming damage while exposed
      * death (life == 0) and timeout (timer <= TIMEOUT_THRESHOLD)
    """

    TIMER_START = 5000
    ENEMIES = 6           # hits needed to clear (one full clip)
    HIT_PROB_CENTERED = 0.75   # hit chance with perfectly centred aim
    DAMAGE_PER_HIT = 20        # player HP lost per enemy shot
    DAMAGE_PROB_PER_FRAME = 0.002  # ~12% / second when exposed

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)
        self.frame_count = 0
        self._reset_state()

    def _reset_state(self):
        self.shots_fired = 0
        self.shots_hit = 0
        self.timer = self.TIMER_START
        self.life = 100
        self.enemies_left = self.ENEMIES
        self.cleared = False
        self._shoot = False
        self._peek = False
        self._aim_x = 0.5
        self._aim_y = 0.5
        self.ammo = AMMO_MAX_ROUNDS

    def reset(self):
        """Called by load_state; resets game state but not the RNG."""
        self._reset_state()

    def set_input(self, shoot: bool, peek: bool,
                  aim_x: float = 0.5, aim_y: float = 0.5):
        peek = bool(peek)
        if self._peek and not peek:
            # Ducking back into cover: reload to a full clip.
            self.ammo = AMMO_MAX_ROUNDS
        self._shoot = bool(shoot)
        self._peek  = peek
        self._aim_x = float(aim_x)
        self._aim_y = float(aim_y)

    def step_frame(self):
        """Advance one emulator frame."""
        self.frame_count += 1

        if not self.cleared:
            self.timer = max(0, self.timer - 1)

        if not self._peek or self.cleared or self.life <= 0:
            return   # in cover or episode already over

        # Shooting: a dry trigger pull (ammo == 0) has no game-side effect --
        # no shots_fired/shots_hit increment -- until a reload happens.
        if self._shoot and self.enemies_left > 0 and self.ammo > 0:
            self.ammo -= 1
            self.shots_fired = (self.shots_fired + 1) & 0xFFFF
            aim_error = abs(self._aim_x - 0.5) + abs(self._aim_y - 0.5)
            hit_prob  = max(0.05, self.HIT_PROB_CENTERED - aim_error)
            if self._rng.random() < hit_prob:
                self.shots_hit    = (self.shots_hit + 1) & 0xFFFF
                self.enemies_left -= 1
                if self.enemies_left == 0:
                    # Stage clear: jump timer upward so the Python heuristic fires.
                    self.cleared = True
                    self.timer   = (self.timer + 10_000) & 0xFFFF

        # Incoming damage while exposed
        if self._rng.random() < self.DAMAGE_PROB_PER_FRAME and self.enemies_left > 0:
            self.life = max(0, self.life - self.DAMAGE_PER_HIT)

    def read_u16(self, addr: int) -> int:
        if addr == RAM.shots_fired: return self.shots_fired
        if addr == RAM.shots_hit:   return self.shots_hit
        if addr == RAM.timer:       return self.timer
        if addr == RAM.life:        return self.life
        return 0


# ---------------------------------------------------------------------------
# Mock BridgeClient
# ---------------------------------------------------------------------------

class MockBridgeClient:
    """Drop-in replacement for BridgeClient backed by SimulatedGame."""

    def __init__(self, game: SimulatedGame):
        self._game = game

    # -- lifecycle --
    def connect(self):         pass
    def start_listening(self): pass
    def finish_connect(self):  pass
    def close(self):           pass

    # -- commands --
    def read_u16(self, addr: int) -> int:
        return self._game.read_u16(addr)

    def set_input(self, shoot: bool, peek: bool,
                  aim_x: float = 0.5, aim_y: float = 0.5):
        self._game.set_input(shoot, peek, aim_x, aim_y)

    def step_frames(self, n: int = 1):
        for _ in range(n):
            self._game.step_frame()

    def load_state(self, slot: int = 1): self._game.reset()
    def save_state(self, slot: int = 1): pass
    def frame(self) -> int:             return self._game.frame_count
    def hud(self, lines):               pass
    def hud_clear(self):                pass


# ---------------------------------------------------------------------------
# Simulated environment
# ---------------------------------------------------------------------------

class SimulatedTimeCrisisEnv(TimeCrisisEnv):
    """TimeCrisisEnv with the BridgeClient swapped for MockBridgeClient.

    Skips the real __init__ (which opens a socket); instead wires up all the
    same instance attributes by hand so every inherited method works as-is.
    """

    def __init__(self, seed: int = 0):
        game = SimulatedGame(seed=seed)
        # Manually replicate TimeCrisisEnv.__init__ without touching sockets.
        self.client             = MockBridgeClient(game)
        self.state_slot         = 1
        self.phase_infer        = PhaseInferer(vote_window=3)
        self.prev: dict         = {}
        self.start_timer        = 0
        self.ticks              = 0
        self.prev_peek          = False
        self.peek_ticks         = 0
        self.peek_lock          = 0
        self.peek_locked_value  = False
        self.ammo_left          = AMMO_MAX_ROUNDS
        self.prev_aim_x_bias    = 0.0
        self.prev_aim_y_bias    = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Output-layer bias index: w1 | b1 | w2 | b2
_B2_OFFSET = OBS_DIM * HIDDEN + HIDDEN + HIDDEN * ACT_DIM


def _theta_always(peek: bool, shoot: bool) -> np.ndarray:
    """Return a theta that unconditionally outputs the requested peek/shoot."""
    theta = np.zeros(PARAM_COUNT)
    theta[_B2_OFFSET + 0] = +50.0 if shoot else -50.0  # shoot logit
    theta[_B2_OFFSET + 1] = +50.0 if peek  else -50.0  # peek  logit
    return theta


def _theta_warm_start(seed: int = 42) -> np.ndarray:
    """Random theta with a small positive peek+shoot bias.

    Without a bias, ~50% of random seeds produce a theta whose peek output
    is negative for the fixed initial observation (timer=0.5, life=1, ammo=1,
    everything else 0), so every perturbed candidate stays in cover, all
    get fitness=-FAIL_PENALTY, and fitness std=0. The bias ensures agents
    explore shooting from generation 0 so training tests have non-trivial
    variance -- which is also the recommended real-world initialization.
    """
    rng = np.random.default_rng(seed)
    theta = rng.normal(0.0, 0.1, PARAM_COUNT)
    theta[_B2_OFFSET + 0] += 1.0  # shoot logit: P(shoot=True) >> 50 %
    theta[_B2_OFFSET + 1] += 1.0  # peek  logit: P(peek=True)  >> 50 %
    return theta


def rank_transform(fitnesses: np.ndarray) -> np.ndarray:
    ranks = np.empty_like(fitnesses, dtype=np.float64)
    ranks[np.argsort(fitnesses)] = np.arange(len(fitnesses))
    return ranks / (len(fitnesses) - 1) - 0.5


# Observation index of ammo_norm (see config.py's OBS_DIM layout comment):
# [timer, life, fired, hit, acc, last_hit, last_miss, peek_phase, ammo_norm,
#  prev_aim_x_bias, prev_aim_y_bias]
_AMMO_OBS_INDEX = 8


def _theta_reactive_ammo_duck(aim_bias: float = 0.0) -> np.ndarray:
    """Return a theta that reacts to the ammo_norm observation: peek=True
    (exposed) while ammo remains, peek=False (duck to reload) the instant
    ammo_norm hits 0. Always shoots when allowed.

    Unlike ``_theta_always``, this wires the hidden layer to the ammo_norm
    input so the *decision* genuinely depends on the agent's own ammo state,
    rather than following a fixed time schedule -- this is what "the agent
    knows it's dry-firing and reacts" looks like as a set of weights.

    ``aim_bias`` optionally saturates the aim outputs off-center (pass e.g.
    50.0) to suppress lucky one-clip clears in comparison tests; 0.0 keeps
    aim centered.
    """
    theta = np.zeros(PARAM_COUNT)
    w1, b1, w2, b2 = _unpack_theta_view(theta)
    w1[_AMMO_OBS_INDEX, :] = 20.0   # ammo_norm -> every hidden unit, saturates tanh
    w2[:, 1] = 5.0 / HIDDEN         # hidden -> peek logit
    b2[1] = -1.0                    # baseline: ammo_norm == 0 -> peek logit < 0 (duck)
    b2[0] = 50.0                    # shoot logit: always shoot when allowed
    b2[2] = aim_bias                # aim_x_bias
    b2[3] = aim_bias                # aim_y_bias
    return theta


def _unpack_theta_view(theta: np.ndarray):
    """Return (w1, b1, w2, b2) as in-place VIEWS into ``theta`` so callers can
    hand-set individual weights (mirrors policy._unpack's layout/order)."""
    i = 0
    w1 = theta[i:i + OBS_DIM * HIDDEN].reshape(OBS_DIM, HIDDEN); i += OBS_DIM * HIDDEN
    b1 = theta[i:i + HIDDEN];                                    i += HIDDEN
    w2 = theta[i:i + HIDDEN * ACT_DIM].reshape(HIDDEN, ACT_DIM); i += HIDDEN * ACT_DIM
    b2 = theta[i:i + ACT_DIM]
    return w1, b1, w2, b2


def _episode_aim_trajectory(theta: np.ndarray, seed: int = 0, max_ticks: int = 200):
    """Run an episode (capped at ``max_ticks`` ticks for speed) and return the
    per-tick (aim_x_bias, aim_y_bias) arrays the policy actually chose.

    Reads ``env.prev_aim_x_bias`` / ``env.prev_aim_y_bias`` after each step --
    env_timecrisis.py already stores the tick's raw [-1, 1] aim decision there
    (fed back as next tick's obs), so this needs no changes to production code.
    """
    env = SimulatedTimeCrisisEnv(seed=seed)
    env.reset()
    xs, ys = [], []
    for _ in range(max_ticks):
        _, done, _ = env.step(theta)
        xs.append(env.prev_aim_x_bias)
        ys.append(env.prev_aim_y_bias)
        if done:
            break
    return np.asarray(xs), np.asarray(ys)


def _print_aim_stats(label: str, xs: np.ndarray, ys: np.ndarray) -> None:
    """Print range/std/step-delta/lag-1-autocorrelation for an aim trajectory.

    Distinguishes a frozen aim (std ~ 0), jittery/random aim (low
    autocorrelation), and a smooth drift/"curve" (high autocorrelation, wide
    range) -- purely diagnostic, not a pass/fail judgment.
    """
    def axis_stats(vals: np.ndarray):
        step = float(np.abs(np.diff(vals)).mean()) if len(vals) > 1 else 0.0
        if len(vals) > 2 and vals.std() > 1e-9:
            lag1 = float(np.corrcoef(vals[:-1], vals[1:])[0, 1])
        else:
            lag1 = float("nan")
        return float(vals.min()), float(vals.max()), float(vals.std()), step, lag1

    xr = axis_stats(xs)
    yr = axis_stats(ys)
    print(f"\n[{label}] ticks={len(xs)}")
    print(f"  aim_x: range=[{xr[0]:+.3f}, {xr[1]:+.3f}]  std={xr[2]:.3f}  "
          f"mean|delta|={xr[3]:.4f}  lag1_autocorr={xr[4]:.3f}")
    print(f"  aim_y: range=[{yr[0]:+.3f}, {yr[1]:+.3f}]  std={yr[2]:.3f}  "
          f"mean|delta|={yr[3]:.4f}  lag1_autocorr={yr[4]:.3f}")


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

class SmokeSuite(unittest.TestCase):
    """Basic sanity checks: can an episode run at all?"""

    def test_episode_completes_without_error(self):
        env = SimulatedTimeCrisisEnv(seed=0)
        rng = np.random.default_rng(42)
        theta = rng.normal(0.0, 0.1, PARAM_COUNT)
        fitness, info = env.episode_fitness(theta)
        self.assertIsInstance(fitness, float)
        self.assertIn("cleared",      info)
        self.assertIn("shots_fired",  info)
        self.assertIn("shots_hit",    info)
        self.assertIn("peek_flips",   info)

    def test_episode_is_reproducible_with_same_seed(self):
        rng = np.random.default_rng(99)
        theta = rng.normal(0.0, 0.1, PARAM_COUNT)
        fit1, info1 = SimulatedTimeCrisisEnv(seed=7).episode_fitness(theta)
        fit2, info2 = SimulatedTimeCrisisEnv(seed=7).episode_fitness(theta)
        self.assertEqual(fit1, fit2)
        self.assertEqual(info1["shots_fired"], info2["shots_fired"])


# ---------------------------------------------------------------------------
# Peek / shoot gating tests
# ---------------------------------------------------------------------------

class PeekGatingSuite(unittest.TestCase):
    """Verify that the peek-to-shoot gate works exactly as intended."""

    def test_always_cover_never_fires(self):
        """peek=always-False must produce zero shots_fired."""
        env = SimulatedTimeCrisisEnv(seed=0)
        theta = _theta_always(peek=False, shoot=True)
        _, info = env.episode_fitness(theta)
        self.assertEqual(
            info["shots_fired"], 0,
            "Agent stuck in cover should fire 0 shots",
        )

    def test_always_peek_and_shoot_fires(self):
        """peek=always-True + shoot=always-True must fire shots."""
        env = SimulatedTimeCrisisEnv(seed=0)
        theta = _theta_always(peek=True, shoot=True)
        _, info = env.episode_fitness(theta)
        self.assertGreater(
            info["shots_fired"], 0,
            f"Always-peek+shoot theta fired 0 shots -- "
            f"gating logic may be permanently blocking fire.\n"
            f"info={info}",
        )

    def test_peeking_without_shooting_fires_nothing(self):
        """peek=True but shoot=False: shots_fired must stay zero."""
        env = SimulatedTimeCrisisEnv(seed=0)
        theta = _theta_always(peek=True, shoot=False)
        _, info = env.episode_fitness(theta)
        self.assertEqual(info["shots_fired"], 0)

    def test_shooting_starts_only_after_traverse(self):
        """shoot_allowed must be False for the first PEEK_TRAVERSE_TICKS ticks.

        shoot_allowed = peek AND prev_peek AND peek_ticks >= PEEK_TRAVERSE_TICKS.
        peek_ticks is updated at the END of each tick, so the gate opens at the
        START of tick PEEK_TRAVERSE_TICKS (0-indexed) -- i.e. after exactly
        PEEK_TRAVERSE_TICKS ticks have passed with peek=True.
        """
        theta = _theta_always(peek=True, shoot=True)

        env = SimulatedTimeCrisisEnv(seed=0)
        env.reset()
        shots_before_gate = 0
        for _ in range(PEEK_TRAVERSE_TICKS):   # ticks 0 .. PEEK_TRAVERSE_TICKS-1
            _, _, info = env.step(theta)
            shots_before_gate += info["shots_fired_delta"]
        self.assertEqual(
            shots_before_gate, 0,
            f"Agent fired before PEEK_TRAVERSE_TICKS={PEEK_TRAVERSE_TICKS} "
            f"ticks had elapsed -- gating window is too short",
        )
        # Tick PEEK_TRAVERSE_TICKS: peek_ticks now == PEEK_TRAVERSE_TICKS at
        # the start of this tick, so shoot_allowed must be True.
        _, _, info = env.step(theta)
        self.assertGreater(
            info["shots_fired_delta"], 0,
            f"Agent still did not fire on tick {PEEK_TRAVERSE_TICKS} -- "
            f"gate may never open",
        )


# ---------------------------------------------------------------------------
# Mini ES training tests
# ---------------------------------------------------------------------------

class MiniESTrainingSuite(unittest.TestCase):
    """Run a tiny ES loop; check learning signal is present."""

    POP  = 10
    GENS = 5
    SIGMA = 0.05
    ALPHA = 0.02

    def _run_generation(self, theta, rng, gen_offset=0):
        half = self.POP // 2
        eps_half = rng.normal(0.0, 1.0, (half, PARAM_COUNT))
        eps      = np.concatenate([eps_half, -eps_half])
        candidates = theta + self.SIGMA * eps

        fitnesses, infos = [], []
        for i, c in enumerate(candidates):
            env = SimulatedTimeCrisisEnv(seed=gen_offset + i)
            fit, info = env.episode_fitness(c)
            fitnesses.append(fit)
            infos.append(info)
        return np.asarray(fitnesses), infos, eps

    def test_fitness_variance_nonzero(self):
        """Perturbations must change fitness; otherwise ES gradient is zero."""
        rng   = np.random.default_rng(42)
        theta = _theta_warm_start()
        fitnesses, _, _ = self._run_generation(theta, rng)
        std = float(fitnesses.std())
        self.assertGreater(
            std, 0.0,
            f"fitness std={std:.4f} -- all candidates produced identical fitness; "
            "ES cannot update theta",
        )

    def test_some_candidates_fire(self):
        """At least one candidate per generation must fire shots."""
        rng   = np.random.default_rng(42)
        theta = _theta_warm_start()
        _, infos, _ = self._run_generation(theta, rng)
        total_fired = sum(info["shots_fired"] for info in infos)
        self.assertGreater(
            total_fired, 0,
            "No candidate in the population fired any shots -- "
            "peek/shoot gating may be permanently blocking fire",
        )

    def test_full_mini_run_prints_diagnostics(self):
        """Run GENS generations; print a table so the user can visually inspect."""
        rng   = np.random.default_rng(42)
        theta = _theta_warm_start()

        print(f"\n{'gen':>4}  {'mean':>8}  {'std':>7}  "
              f"{'best':>8}  {'clear%':>7}  {'shots':>6}")
        print("-" * 50)

        for gen in range(self.GENS):
            fitnesses, infos, eps = self._run_generation(
                theta, rng, gen_offset=gen * self.POP
            )
            clear_rate  = float(np.mean([1.0 if x["cleared"] else 0.0 for x in infos]))
            total_shots = int(sum(x["shots_fired"] for x in infos))

            print(f"{gen:>4}  {fitnesses.mean():>8.1f}  {fitnesses.std():>7.2f}  "
                  f"{fitnesses.max():>8.1f}  {clear_rate:>6.0%}  {total_shots:>6}")

            shaped   = rank_transform(fitnesses)
            gradient = (eps.T @ shaped) / (self.POP * self.SIGMA)
            theta    = theta + self.ALPHA * gradient

        # Weak sanity: the final mean shouldn't be wildly worse than the first.
        # (Checked after the loop so it doesn't affect diagnostic output.)
        rng2   = np.random.default_rng(42)
        theta2 = _theta_warm_start()
        f0, _, _ = self._run_generation(theta2, rng2, gen_offset=0)
        self.assertGreater(
            fitnesses.mean(), f0.mean() - 200.0,
            f"Mean fitness regressed by more than 200: "
            f"{f0.mean():.1f} -> {fitnesses.mean():.1f}",
        )


# ---------------------------------------------------------------------------
# Dry-fire / reload behavioral tests
# ---------------------------------------------------------------------------

class DryFireBehaviorSuite(unittest.TestCase):
    """Confirm the dry-fire penalty / reload bonus actually distinguish
    "never ducks after emptying the clip" from "ducks correctly on empty",
    and quantify the fitness gap between them.

    Uses a fixed seed throughout: shot/ammo depletion timing is deterministic
    (only hit/miss and incoming damage are randomized), so these assertions
    are not seed-flaky.
    """

    SEED = 0

    def test_never_duck_accumulates_dry_fire_and_no_reload(self):
        """peek=always-True, shoot=always-True (never ducks): must rack up
        dry_fire_ticks once the (now ammo-capped) clip runs out, and must
        never earn a reload bonus since it never transitions back to cover.

        Aim is deliberately saturated off-center here so a lucky one-clip
        clear (which would end the episode before dry-fire can accumulate)
        is effectively impossible -- isolating the behavior under test.
        """
        theta = _theta_always(peek=True, shoot=True)
        theta[_B2_OFFSET + 2] = 50.0  # aim_x_bias -> saturates off-center
        theta[_B2_OFFSET + 3] = 50.0  # aim_y_bias -> saturates off-center

        env = SimulatedTimeCrisisEnv(seed=self.SEED)
        fitness, info = env.episode_fitness(theta)

        self.assertGreater(
            info["dry_fire_ticks"], 0,
            f"Agent that never ducks after emptying its clip should rack up "
            f"dry_fire_ticks once ammo runs out.\ninfo={info}",
        )
        self.assertEqual(
            info["reload_correct_count"], 0,
            "An agent that never transitions back to cover should never "
            "earn a reload bonus.",
        )
        self._never_duck_fitness = fitness
        self._never_duck_info = info

    def test_reactive_duck_avoids_dry_fire_and_earns_reload(self):
        """A theta whose peek output reacts to the ammo_norm observation
        (duck the instant ammo hits 0, re-expose once reloaded) should incur
        zero dry-fire ticks and repeatedly earn the reload bonus.

        Aim is deliberately off-centered a bit (not fully saturated) so
        clearing needs several clips worth of shots rather than a lucky
        one-clip 6/6 -- otherwise the episode could end (stage clear) before
        a single duck-and-reload cycle is ever required, which isn't a
        failure of the reload mechanism, just an easy episode.
        """
        theta = _theta_reactive_ammo_duck(aim_bias=0.3)

        env = SimulatedTimeCrisisEnv(seed=self.SEED)
        fitness, info = env.episode_fitness(theta)

        self.assertEqual(
            info["dry_fire_ticks"], 0,
            f"Reactive duck-on-empty theta should never sit exposed with an "
            f"empty clip.\ninfo={info}",
        )
        self.assertGreater(
            info["reload_correct_count"], 0,
            f"Reactive duck-on-empty theta should duck (and reload) at "
            f"least once per episode.\ninfo={info}",
        )
        self._reactive_fitness = fitness
        self._reactive_info = info

    def test_reactive_duck_beats_never_duck(self):
        """Ducking correctly on empty must score materially better than
        never ducking -- confirms the reward design actually pushes ES
        toward the desired behavior (answers "do agents know they're
        dry-firing" at the incentive-design level, independent of whether
        training has converged there yet).
        """
        never_duck_theta = _theta_always(peek=True, shoot=True)
        never_duck_theta[_B2_OFFSET + 2] = 50.0
        never_duck_theta[_B2_OFFSET + 3] = 50.0
        reactive_theta = _theta_reactive_ammo_duck()

        env_a = SimulatedTimeCrisisEnv(seed=self.SEED)
        fit_never_duck, info_never_duck = env_a.episode_fitness(never_duck_theta)
        env_b = SimulatedTimeCrisisEnv(seed=self.SEED)
        fit_reactive, info_reactive = env_b.episode_fitness(reactive_theta)

        print(f"\n{'theta':>14}  {'fitness':>9}  {'dry_fire':>9}  "
              f"{'reload':>7}  {'shots_fired':>12}")
        print("-" * 58)
        print(f"{'never-duck':>14}  {fit_never_duck:>9.1f}  "
              f"{info_never_duck['dry_fire_ticks']:>9}  "
              f"{info_never_duck['reload_correct_count']:>7}  "
              f"{info_never_duck['shots_fired']:>12}")
        print(f"{'reactive-duck':>14}  {fit_reactive:>9.1f}  "
              f"{info_reactive['dry_fire_ticks']:>9}  "
              f"{info_reactive['reload_correct_count']:>7}  "
              f"{info_reactive['shots_fired']:>12}")

        self.assertGreater(
            fit_reactive, fit_never_duck,
            "Ducking correctly on an empty clip should score higher than "
            "never ducking -- if not, the reward shaping isn't actually "
            "teaching the desired behavior.",
        )


# ---------------------------------------------------------------------------
# Aim trajectory diagnostics
# ---------------------------------------------------------------------------

class AimTrajectorySuite(unittest.TestCase):
    """Capture and print the per-tick aim trajectory so the reported
    "cursor traverses the screen in a curve" behavior can be inspected with
    concrete numbers instead of only watching BizHawk footage.

    These are diagnostic: whether a given trajectory shape is "fine" is a
    judgment call, so only basic sanity (finite, in-range) is asserted;
    range/std/step-delta/autocorrelation are printed for the user to read.
    """

    def test_random_theta_aim_trajectory_diagnostics(self):
        rng   = np.random.default_rng(1)
        theta = rng.normal(0.0, 0.1, PARAM_COUNT)
        xs, ys = _episode_aim_trajectory(theta, seed=1, max_ticks=300)
        _print_aim_stats("random theta", xs, ys)

        self.assertTrue(np.isfinite(xs).all() and np.isfinite(ys).all())
        self.assertTrue(bool(((xs >= -1.0) & (xs <= 1.0)).all()))
        self.assertTrue(bool(((ys >= -1.0) & (ys <= 1.0)).all()))

    def test_warm_start_theta_aim_trajectory_diagnostics(self):
        theta = _theta_warm_start()
        xs, ys = _episode_aim_trajectory(theta, seed=2, max_ticks=300)
        _print_aim_stats("warm-start theta", xs, ys)

        self.assertTrue(np.isfinite(xs).all() and np.isfinite(ys).all())
        self.assertTrue(bool(((xs >= -1.0) & (xs <= 1.0)).all()))
        self.assertTrue(bool(((ys >= -1.0) & (ys <= 1.0)).all()))
        if xs.std() < 1e-6 and ys.std() < 1e-6:
            print("  NOTE: aim never moved this run (std ~ 0) -- a frozen "
                  "aim is a distinct failure mode from a drifting \"curve\".")


# ---------------------------------------------------------------------------
# Extended mini-ES trend run
# ---------------------------------------------------------------------------

class ExtendedMiniESTrendSuite(unittest.TestCase):
    """Longer in-sim-only ES run (many more generations than
    MiniESTrainingSuite) tracking dry-fire/reload/aim metrics across
    generations.

    Not a strict behavioral gate -- it's exploratory, meant to show whether
    the current reward shaping resolves the two reported behaviors given
    enough training, or whether they persist even after far more generations
    than a partial real BizHawk run has seen so far.
    """

    POP   = 12
    GENS  = 80
    SIGMA = 0.05
    ALPHA = 0.02

    def _run_generation(self, theta, rng, gen_offset=0):
        half = self.POP // 2
        eps_half = rng.normal(0.0, 1.0, (half, PARAM_COUNT))
        eps      = np.concatenate([eps_half, -eps_half])
        candidates = theta + self.SIGMA * eps

        fitnesses, infos = [], []
        for i, c in enumerate(candidates):
            env = SimulatedTimeCrisisEnv(seed=gen_offset + i)
            fit, info = env.episode_fitness(c)
            fitnesses.append(fit)
            infos.append(info)
        return np.asarray(fitnesses), infos, eps

    def test_long_run_prints_trend_diagnostics(self):
        rng   = np.random.default_rng(42)
        theta = _theta_warm_start()

        print(f"\n{'gen':>4}  {'mean':>8}  {'std':>7}  {'best':>8}  "
              f"{'clear%':>7}  {'dryfire':>8}  {'reload':>7}  {'aimstd':>7}")
        print("-" * 68)

        for gen in range(self.GENS):
            fitnesses, infos, eps = self._run_generation(
                theta, rng, gen_offset=gen * self.POP
            )
            clear_rate = float(np.mean([1.0 if x["cleared"] else 0.0 for x in infos]))
            avg_dry    = float(np.mean([x["dry_fire_ticks"] for x in infos]))
            avg_reload = float(np.mean([x["reload_correct_count"] for x in infos]))

            xs, ys = _episode_aim_trajectory(theta, seed=gen)
            aim_std = float((xs.std() + ys.std()) / 2.0)

            if gen % 5 == 0 or gen == self.GENS - 1:
                print(f"{gen:>4}  {fitnesses.mean():>8.1f}  {fitnesses.std():>7.2f}  "
                      f"{fitnesses.max():>8.1f}  {clear_rate:>6.0%}  "
                      f"{avg_dry:>8.2f}  {avg_reload:>7.2f}  {aim_std:>7.3f}")

            shaped   = rank_transform(fitnesses)
            gradient = (eps.T @ shaped) / (self.POP * self.SIGMA)
            theta    = theta + self.ALPHA * gradient

        # Exploratory run: no strict pass/fail on the trend direction itself,
        # just confirm it completed cleanly with finite numbers throughout.
        self.assertTrue(np.isfinite(fitnesses).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
