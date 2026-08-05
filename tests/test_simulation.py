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
    SIGMA as CFG_SIGMA, STAGNATION_PATIENCE, STAGNATION_SIGMA_MULT,
    STD_STAGNATION_THRESHOLD,
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
      * a screen with MORE enemies than fit in one magazine
        (ENEMIES_TOTAL > AMMO_MAX_ROUNDS), spawned in waves of
        TARGET_POSITIONS at a time -- clearing a screen requires at least
        one real duck-to-reload cycle, unlike the earlier
        one-clip-clears-everything model.
      * shooting while peeking: a hit only registers if the aim lands near
        one of the current wave's alive target positions (Euclidean
        distance with a linear falloff, floored at a minimum chance) --
        not just "close to screen center" like the old model. This is a
        deliberately rough approximation of real enemy hitboxes ("3
        certain spots"), good enough to force the aim to actually move
        between distinct locations instead of camping one spot or
        exploiting a screen-centered heuristic.
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
    # 3 fixed on-screen target spots ("good enough", not real Time Crisis
    # coordinates) that enemies spawn at, one wave (<= 3 alive) at a time.
    TARGET_POSITIONS = [(0.3, 0.4), (0.5, 0.65), (0.7, 0.4)]
    HIT_PROB_PEAK       = 0.75   # hit chance with aim exactly on a target
    HIT_PROB_FLOOR      = 0.05   # hit chance floor once aim is far from every target
    HIT_FALLOFF_RADIUS  = 0.3    # distance at which hit_prob decays to the floor
    # More enemies than one magazine (AMMO_MAX_ROUNDS) can clear -- a real
    # clear needs at least one duck-to-reload cycle. Kept a multiple of
    # len(TARGET_POSITIONS) so every wave is a full 3-enemy wave.
    ENEMIES_TOTAL = 12
    DAMAGE_PER_HIT = 20        # player HP lost per enemy shot
    DAMAGE_PROB_PER_FRAME = 0.002  # ~12% / second when exposed while enemies remain

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)
        self.frame_count = 0
        self._reset_state()

    def _reset_state(self):
        self.shots_fired = 0
        self.shots_hit = 0
        self.timer = self.TIMER_START
        self.life = 100
        self.enemies_left = self.ENEMIES_TOTAL   # total kills still needed
        self.cleared = False
        self._shoot = False
        self._peek = False
        self._aim_x = 0.5
        self._aim_y = 0.5
        self.ammo = AMMO_MAX_ROUNDS
        self._wave_alive = [False] * len(self.TARGET_POSITIONS)
        self._spawn_wave()

    def _spawn_wave(self):
        """(Re)populate up to len(TARGET_POSITIONS) targets from the
        remaining pool. Called on reset and whenever a wave is fully
        cleared but enemies still remain."""
        n_active = min(len(self.TARGET_POSITIONS), self.enemies_left)
        self._wave_alive = [i < n_active for i in range(len(self.TARGET_POSITIONS))]

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

    def _nearest_alive_target(self):
        """Return (index, distance) of the alive target nearest the current
        aim, or (None, None) if no target is currently alive."""
        best_i, best_d = None, None
        for i, (pos, alive) in enumerate(zip(self.TARGET_POSITIONS, self._wave_alive)):
            if not alive:
                continue
            tx, ty = pos
            d = ((self._aim_x - tx) ** 2 + (self._aim_y - ty) ** 2) ** 0.5
            if best_d is None or d < best_d:
                best_i, best_d = i, d
        return best_i, best_d

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
            idx, dist = self._nearest_alive_target()
            if idx is not None:
                hit_prob = max(
                    self.HIT_PROB_FLOOR,
                    self.HIT_PROB_PEAK - (self.HIT_PROB_PEAK - self.HIT_PROB_FLOOR)
                    * dist / self.HIT_FALLOFF_RADIUS,
                )
                if self._rng.random() < hit_prob:
                    self.shots_hit = (self.shots_hit + 1) & 0xFFFF
                    self.enemies_left -= 1
                    self._wave_alive[idx] = False
                    if self.enemies_left == 0:
                        # Stage clear: jump timer upward so the Python heuristic fires.
                        self.cleared = True
                        self.timer   = (self.timer + 10_000) & 0xFFFF
                    elif not any(self._wave_alive):
                        self._spawn_wave()

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

    Bias values (+2.0 shoot / +1.0 peek) mirror es_train.py's actual
    warm-start exactly -- keep these two in sync (see the "+2 peek bias
    overcorrected" note in repo memory for why they're asymmetric).
    """
    rng = np.random.default_rng(seed)
    theta = rng.normal(0.0, 0.1, PARAM_COUNT)
    theta[_B2_OFFSET + 0] += 2.0  # shoot logit: P(shoot=True) >> 50 %
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


def _aim_bias_for(target_pos: float) -> float:
    """Return the b2 bias that makes ``tanh(b2) == target_pos - 0.5``, i.e.
    the raw bias whose saturation-free output makes act()'s aim_x/aim_y land
    exactly on ``target_pos`` (env maps aim = clip(0.5 + tanh(out), 0, 1))."""
    y = float(np.clip(target_pos - 0.5, -0.999, 0.999))
    return float(np.arctanh(y))


def _theta_always_aim(aim_x: float, aim_y: float) -> np.ndarray:
    """peek=True, shoot=True permanently, aim pinned exactly at
    (aim_x, aim_y) for the whole episode (no ammo/duck reactivity -- only
    good for isolating single-clip aim/hit behavior)."""
    theta = np.zeros(PARAM_COUNT)
    theta[_B2_OFFSET + 0] = 50.0  # shoot logit: always shoot
    theta[_B2_OFFSET + 1] = 50.0  # peek  logit: always exposed
    theta[_B2_OFFSET + 2] = _aim_bias_for(aim_x)
    theta[_B2_OFFSET + 3] = _aim_bias_for(aim_y)
    return theta


def _theta_reactive_ammo_duck_xy(aim_x: float, aim_y: float) -> np.ndarray:
    """Like ``_theta_reactive_ammo_duck`` (ducks/reloads reactively on
    ammo_norm), but pins aim at an independently-chosen (aim_x, aim_y)
    instead of applying the same bias to both axes."""
    theta = np.zeros(PARAM_COUNT)
    w1, b1, w2, b2 = _unpack_theta_view(theta)
    w1[_AMMO_OBS_INDEX, :] = 20.0
    w2[:, 1] = 5.0 / HIDDEN
    b2[1] = -1.0
    b2[0] = 50.0
    b2[2] = _aim_bias_for(aim_x)
    b2[3] = _aim_bias_for(aim_y)
    return theta


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
        clear_rates = []

        print(f"\n{'gen':>4}  {'mean':>8}  {'std':>7}  "
              f"{'best':>8}  {'clear%':>7}  {'shots':>6}")
        print("-" * 50)

        for gen in range(self.GENS):
            fitnesses, infos, eps = self._run_generation(
                theta, rng, gen_offset=gen * self.POP
            )
            clear_rate  = float(np.mean([1.0 if x["cleared"] else 0.0 for x in infos]))
            clear_rates.append(clear_rate)
            total_shots = int(sum(x["shots_fired"] for x in infos))

            print(f"{gen:>4}  {fitnesses.mean():>8.1f}  {fitnesses.std():>7.2f}  "
                  f"{fitnesses.max():>8.1f}  {clear_rate:>6.0%}  {total_shots:>6}")

            shaped   = rank_transform(fitnesses)
            gradient = (eps.T @ shaped) / (self.POP * self.SIGMA)
            theta    = theta + self.ALPHA * gradient

        # Weak sanity check. Since the env now hard-forces a duck-to-reload the
        # instant ammo hits 0 (see env_timecrisis.py step()), clears are common
        # from generation 0 and per-generation fitness (dominated by
        # CLEAR_BONUS - elapsed, which swings by thousands depending on how
        # quickly a candidate happens to clear) is far noisier than the old
        # mostly-failing dynamics this test's original +-200 absolute
        # tolerance was calibrated against -- that tolerance is meaningless at
        # this scale (std alone runs into the thousands here). clear_rate is
        # the stable signal instead: confirm the population is still clearing
        # most episodes by the final generation, i.e. the mini run didn't
        # collapse into never-clearing.
        self.assertGreaterEqual(
            clear_rates[-1], 0.5,
            f"Clear rate dropped to {clear_rates[-1]:.0%} by the final "
            f"generation (started at {clear_rates[0]:.0%}) -- population may "
            f"have collapsed.\nclear_rates={clear_rates}",
        )


# ---------------------------------------------------------------------------
# Dry-fire / reload behavioral tests
# ---------------------------------------------------------------------------

class DryFireBehaviorSuite(unittest.TestCase):
    """Confirm the empty-clip handling actually works as intended.

    env_timecrisis.py's step() now HARD-ENFORCES a duck-to-reload the instant
    ammo_left hits 0 (overrides the policy's own peek decision), rather than
    relying solely on the DRY_FIRE_PENALTY/RELOAD_BONUS reward shaping to
    teach it -- real training kept showing agents mag-dump and stay exposed
    even with that shaping in place. So "never duck" is no longer a
    real behavior a policy can express; these tests instead confirm the
    enforcement holds even for a policy that actively wants to stay exposed
    forever, and that reward shaping still correctly reflects it.

    Uses a fixed seed throughout: shot/ammo depletion timing is deterministic
    (only hit/miss and incoming damage are randomized), so these assertions
    are not seed-flaky.
    """

    SEED = 0

    def test_env_forces_duck_even_when_policy_always_wants_to_peek(self):
        """peek=always-True, shoot=always-True (never VOLUNTARILY ducks):
        the environment must still force a duck the instant ammo hits 0, so
        this policy should incur zero dry_fire_ticks and repeatedly earn the
        reload bonus anyway -- proving the enforcement doesn't depend on the
        policy cooperating.

        Aim is deliberately saturated off-center here so hits are rare and
        the episode runs long enough to exercise many forced duck/reload
        cycles rather than being decided by a single lucky clip.
        """
        theta = _theta_always(peek=True, shoot=True)
        theta[_B2_OFFSET + 2] = 50.0  # aim_x_bias -> saturates off-center
        theta[_B2_OFFSET + 3] = 50.0  # aim_y_bias -> saturates off-center

        env = SimulatedTimeCrisisEnv(seed=self.SEED)
        fitness, info = env.episode_fitness(theta)

        self.assertEqual(
            info["dry_fire_ticks"], 0,
            f"Environment should force a duck the instant ammo hits 0, "
            f"even for a policy whose own peek output never wants to duck."
            f"\ninfo={info}",
        )
        self.assertGreater(
            info["reload_correct_count"], 0,
            "Forced ducks on empty ammo should still count as correct "
            "reloads, even though the policy itself never chose to duck.",
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
        """A theta with centered aim should still score materially better
        than one with saturated off-center aim, even though the environment
        now forces BOTH of them through correct duck/reload cycles.

        Before the hard ammo-enforcement, this gap was mostly about duck
        *timing* (reactive vs. never). Now that duck timing is enforced
        environment-side for everyone, this gap is mostly a proxy for aim
        quality instead -- kept as a regression guard that the fitness
        gradient between "good aim" and "bad aim" thetas is still large and
        in the right direction.
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


class MultiSpotTargetingSuite(unittest.TestCase):
    """Verify the wave / 3-target-spot shooting model added to
    SimulatedGame: hits require the aim to actually be near one of the
    (<= 3) currently-alive target positions, and a full screen
    (ENEMIES_TOTAL=12) now needs more rounds than one magazine
    (AMMO_MAX_ROUNDS=6) holds -- so reload is genuinely mandatory for a
    clear, not just a nice-to-have.
    """

    SEED = 0

    def test_aim_on_a_target_hits_far_more_than_aim_off_all_targets(self):
        """Aiming exactly on a live target spot should need far fewer shots
        to clear the same number of enemies than aiming at a point far from
        every target -- confirms hit registration depends on proximity to a
        real target position, not just "shoot while exposed".

        Compares shots_fired (not shots_hit) needed to reach the same
        outcome: since the environment now hard-forces a duck-to-reload the
        instant ammo empties (env_timecrisis.py step()), ANY persistent
        shoot+peek theta eventually clears given enough ticks -- ammo is
        never a hard limiter any more, only aim quality changes how many
        rounds that takes.
        """
        tx, ty = SimulatedGame.TARGET_POSITIONS[1]
        on_target_theta  = _theta_always_aim(tx, ty)
        off_target_theta = _theta_always_aim(0.02, 0.98)  # far from all 3 spots

        env_on  = SimulatedTimeCrisisEnv(seed=self.SEED)
        _, info_on = env_on.episode_fitness(on_target_theta)
        env_off = SimulatedTimeCrisisEnv(seed=self.SEED)
        _, info_off = env_off.episode_fitness(off_target_theta)

        print(f"\n[on-target aim]  shots_fired={info_on['shots_fired']}  "
              f"shots_hit={info_on['shots_hit']}")
        print(f"[off-target aim] shots_fired={info_off['shots_fired']}  "
              f"shots_hit={info_off['shots_hit']}")

        self.assertLess(
            info_on["shots_fired"], info_off["shots_fired"],
            "Aiming directly on a live target spot should need fewer shots "
            "to land the same number of hits than aiming far from every "
            "target.",
        )

    def test_full_screen_requires_more_than_one_magazine(self):
        """A theta that ducks/reloads reactively and clears the whole screen
        must fire more shots -- and reload more than once -- than a single
        magazine allows, confirming ENEMIES_TOTAL > AMMO_MAX_ROUNDS actually
        forces multiple clips rather than being clearable in one burst."""
        theta = _theta_reactive_ammo_duck()  # centered aim, reacts to ammo
        env = SimulatedTimeCrisisEnv(seed=self.SEED)
        _, info = env.episode_fitness(theta)

        self.assertGreater(
            info["shots_fired"], AMMO_MAX_ROUNDS,
            f"Clearing the screen took only {info['shots_fired']} shots -- "
            f"one magazine ({AMMO_MAX_ROUNDS} rounds) should no longer be "
            f"enough.\ninfo={info}",
        )
        self.assertGreater(
            info["reload_correct_count"], 1,
            f"Clearing a {SimulatedGame.ENEMIES_TOTAL}-enemy, multi-wave "
            f"screen should take more than one duck-to-reload cycle.\n"
            f"info={info}",
        )

    def test_static_single_spot_aim_is_less_efficient_than_spreading_hits(self):
        """A theta that ducks/reloads correctly but keeps its aim pinned on
        just ONE of the three target spots forever should need at least as
        many shots to clear as one aimed where it has a moderate chance
        against all three simultaneously -- camping a single spot leaves 2
        of every 3 alive enemies (almost) untouchable."""
        pinned_theta   = _theta_reactive_ammo_duck_xy(*SimulatedGame.TARGET_POSITIONS[0])
        balanced_theta = _theta_reactive_ammo_duck()  # aim centered near all 3

        env_pinned = SimulatedTimeCrisisEnv(seed=self.SEED)
        _, info_pinned = env_pinned.episode_fitness(pinned_theta)
        env_balanced = SimulatedTimeCrisisEnv(seed=self.SEED)
        _, info_balanced = env_balanced.episode_fitness(balanced_theta)

        print(f"\n[pinned-on-one-spot]  shots_fired={info_pinned['shots_fired']}  "
              f"cleared={info_pinned['cleared']}")
        print(f"[centered-near-all-3] shots_fired={info_balanced['shots_fired']}  "
              f"cleared={info_balanced['cleared']}")

        self.assertGreaterEqual(
            info_pinned["shots_fired"], info_balanced["shots_fired"],
            "Camping aim on a single target spot should need at least as "
            "many shots as splitting attention across all three.",
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

    Mirrors es_train.py's ACTUAL hyperparameters (SIGMA, ALPHA, and the
    stagnation-kick escape mechanism) rather than a separate hardcoded set --
    otherwise this test silently drifts from what the live project really
    does and stops being a faithful predictor of real training behavior.

    Not a strict behavioral gate -- it's exploratory, meant to show whether
    the current reward shaping resolves the two reported behaviors given
    enough training, or whether they persist even after far more generations
    than a partial real BizHawk run has seen so far.
    """

    POP   = 12
    GENS  = 80
    SIGMA = CFG_SIGMA
    ALPHA = 0.02

    def _run_generation(self, theta, rng, sigma_this_gen, gen_offset=0):
        half = self.POP // 2
        eps_half = rng.normal(0.0, 1.0, (half, PARAM_COUNT))
        eps      = np.concatenate([eps_half, -eps_half])
        candidates = theta + sigma_this_gen * eps

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
        stagnant_gens = 0

        print(f"\n{'gen':>4}  {'mean':>8}  {'std':>7}  {'best':>8}  "
              f"{'clear%':>7}  {'dryfire':>8}  {'reload':>7}  {'aimstd':>7}  {'kick':>4}")
        print("-" * 76)

        for gen in range(self.GENS):
            kicking = stagnant_gens >= STAGNATION_PATIENCE
            sigma_this_gen = self.SIGMA * STAGNATION_SIGMA_MULT if kicking else self.SIGMA

            fitnesses, infos, eps = self._run_generation(
                theta, rng, sigma_this_gen, gen_offset=gen * self.POP
            )
            clear_rate = float(np.mean([1.0 if x["cleared"] else 0.0 for x in infos]))
            avg_dry    = float(np.mean([x["dry_fire_ticks"] for x in infos]))
            avg_reload = float(np.mean([x["reload_correct_count"] for x in infos]))

            xs, ys = _episode_aim_trajectory(theta, seed=gen)
            aim_std = float((xs.std() + ys.std()) / 2.0)

            if gen % 5 == 0 or gen == self.GENS - 1:
                print(f"{gen:>4}  {fitnesses.mean():>8.1f}  {fitnesses.std():>7.2f}  "
                      f"{fitnesses.max():>8.1f}  {clear_rate:>6.0%}  "
                      f"{avg_dry:>8.2f}  {avg_reload:>7.2f}  {aim_std:>7.3f}  "
                      f"{'Y' if kicking else '.':>4}")

            shaped   = rank_transform(fitnesses)
            gradient = (eps.T @ shaped) / (self.POP * sigma_this_gen)
            theta    = theta + self.ALPHA * gradient

            if float(fitnesses.std()) < STD_STAGNATION_THRESHOLD:
                stagnant_gens += 1
            else:
                stagnant_gens = 0

        # Exploratory run: no strict pass/fail on the trend direction itself,
        # just confirm it completed cleanly with finite numbers throughout.
        self.assertTrue(np.isfinite(fitnesses).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
