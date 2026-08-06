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
    ACT_DIM, AMMO_MAX_ROUNDS, CONTINUE_SCREEN_STALE_TICKS,
    CURSOR_X_MAX, CURSOR_X_MIN, CURSOR_Y_MAX, CURSOR_Y_MIN, HIDDEN, OBS_DIM,
    FRAME_SKIP, MAX_TICKS, PEEK_TRAVERSE_TICKS, RAM, TIMEOUT_TIMER_THRESHOLD,
    SIGMA as CFG_SIGMA, STAGNATION_PATIENCE, STAGNATION_SIGMA_MULT,
    STD_STAGNATION_THRESHOLD,
)
from env_timecrisis import (
    TimeCrisisEnv,
    compute_miss_correction_metrics,
    core_watchdog_snapshot,
    normalize_cursor,
    u16_delta,
)
from phase_inference import Phase, PhaseInferer, TickSignals
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
        if addr == RAM.cursor_x:
            return int(round(CURSOR_X_MIN + self._aim_x * (CURSOR_X_MAX - CURSOR_X_MIN)))
        if addr == RAM.cursor_y:
            return int(round(CURSOR_Y_MIN + self._aim_y * (CURSOR_Y_MAX - CURSOR_Y_MIN)))
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
        self.stale_core_ticks   = 0
        self.ammo_left          = AMMO_MAX_ROUNDS
        self.prev_aim_x_bias    = 0.0
        self.prev_aim_y_bias    = 0.0


class FrozenGame(SimulatedGame):
    """Game stub that freezes all core counters forever after reset.

    Mimics the non-playable continue/menu state where RAM values stop changing.
    """

    def step_frame(self):
        self.frame_count += 1
        # Intentionally do nothing: shots/timer/life remain frozen.


class FrozenStateEnv(SimulatedTimeCrisisEnv):
    """Simulated env wired to FrozenGame for watchdog regression testing."""

    def __init__(self, seed: int = 0):
        game = FrozenGame(seed=seed)
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
        self.stale_core_ticks   = 0
        self.ammo_left          = AMMO_MAX_ROUNDS
        self.prev_aim_x_bias    = 0.0
        self.prev_aim_y_bias    = 0.0


class TimedSpotGame:
    """Simulation variant with 5 random screen spots that become "good"
    one-at-a-time on a fixed time schedule.

    This is a better proxy for moving/running enemies than the static 3-spot
    model: the best aim point depends on *when* the shot is taken, not just
    where. The 5 spots are sampled once per env seed and then revisited on a
    repeating frame schedule.
    """

    TIMER_START = 5000
    TIMED_SPOT_COUNT = 5
    SPOT_HOLD_FRAMES = 30
    ENEMIES_TOTAL = 15
    HIT_PROB_PEAK = 1.0
    HIT_PROB_FLOOR = 0.02
    HIT_FALLOFF_RADIUS = 0.18
    DAMAGE_PER_HIT = 20
    DAMAGE_PROB_PER_FRAME = 0.0025
    _X_BOUNDS = (0.18, 0.82)
    _Y_BOUNDS = (0.18, 0.78)
    _MIN_SPOT_DIST = 0.14

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)
        self.spots = self._sample_spots()
        self.frame_count = 0
        self._reset_state()

    def _sample_spots(self):
        spots = []
        while len(spots) < self.TIMED_SPOT_COUNT:
            x = float(self._rng.uniform(*self._X_BOUNDS))
            y = float(self._rng.uniform(*self._Y_BOUNDS))
            if all(((x - px) ** 2 + (y - py) ** 2) ** 0.5 >= self._MIN_SPOT_DIST for px, py in spots):
                spots.append((x, y))
        return spots

    def _reset_state(self):
        self.shots_fired = 0
        self.shots_hit = 0
        self.timer = self.TIMER_START
        self.life = 100
        self.enemies_left = self.ENEMIES_TOTAL
        self.cleared = False
        self._shoot = False
        self._peek = False
        self._aim_x = 0.5
        self._aim_y = 0.5
        self.ammo = AMMO_MAX_ROUNDS
        self.last_success_frame = -1
        self.last_success_aim_x = 0.5
        self.last_success_aim_y = 0.5
        self.last_success_spot_index = -1

    def reset(self):
        self._reset_state()

    def set_input(self, shoot: bool, peek: bool,
                  aim_x: float = 0.5, aim_y: float = 0.5):
        peek = bool(peek)
        if self._peek and not peek:
            self.ammo = AMMO_MAX_ROUNDS
        self._shoot = bool(shoot)
        self._peek = peek
        self._aim_x = float(aim_x)
        self._aim_y = float(aim_y)

    def active_spot_index(self) -> int:
        return int((self.frame_count // self.SPOT_HOLD_FRAMES) % len(self.spots))

    def active_spot(self):
        return self.spots[self.active_spot_index()]

    def step_frame(self):
        self.frame_count += 1

        if not self.cleared:
            self.timer = max(0, self.timer - 1)

        if not self._peek or self.cleared or self.life <= 0:
            return

        if self._shoot and self.enemies_left > 0 and self.ammo > 0:
            self.ammo -= 1
            self.shots_fired = (self.shots_fired + 1) & 0xFFFF

            spot_i = self.active_spot_index()
            tx, ty = self.spots[spot_i]
            dist = ((self._aim_x - tx) ** 2 + (self._aim_y - ty) ** 2) ** 0.5
            hit_prob = max(
                self.HIT_PROB_FLOOR,
                self.HIT_PROB_PEAK - (self.HIT_PROB_PEAK - self.HIT_PROB_FLOOR)
                * dist / self.HIT_FALLOFF_RADIUS,
            )
            if self._rng.random() < hit_prob:
                self.shots_hit = (self.shots_hit + 1) & 0xFFFF
                self.enemies_left -= 1
                self.last_success_frame = self.frame_count
                self.last_success_aim_x = self._aim_x
                self.last_success_aim_y = self._aim_y
                self.last_success_spot_index = spot_i
                if self.enemies_left == 0:
                    self.cleared = True
                    self.timer = (self.timer + 10_000) & 0xFFFF

        if self._rng.random() < self.DAMAGE_PROB_PER_FRAME and self.enemies_left > 0:
            self.life = max(0, self.life - self.DAMAGE_PER_HIT)

    def read_u16(self, addr: int) -> int:
        if addr == RAM.shots_fired: return self.shots_fired
        if addr == RAM.shots_hit:   return self.shots_hit
        if addr == RAM.timer:       return self.timer
        if addr == RAM.life:        return self.life
        if addr == RAM.cursor_x:
            return int(round(CURSOR_X_MIN + self._aim_x * (CURSOR_X_MAX - CURSOR_X_MIN)))
        if addr == RAM.cursor_y:
            return int(round(CURSOR_Y_MIN + self._aim_y * (CURSOR_Y_MAX - CURSOR_Y_MIN)))
        return 0


class TimedSpotBaselineEnv(SimulatedTimeCrisisEnv):
    """Baseline observation stack on the timed 5-random-spot game."""

    def __init__(self, seed: int = 0):
        game = TimedSpotGame(seed=seed)
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
        self.stale_core_ticks   = 0
        self.ammo_left          = AMMO_MAX_ROUNDS
        self.prev_aim_x_bias    = 0.0
        self.prev_aim_y_bias    = 0.0


_SIM_MEMORY_EXTRA_DIMS = 3
_SIM_MEMORY_OBS_DIM = OBS_DIM + _SIM_MEMORY_EXTRA_DIMS
_SIM_MEMORY_PARAM_COUNT = _SIM_MEMORY_OBS_DIM * HIDDEN + HIDDEN + HIDDEN * ACT_DIM + ACT_DIM
_SIM_MEMORY_B2_OFFSET = _SIM_MEMORY_OBS_DIM * HIDDEN + HIDDEN + HIDDEN * ACT_DIM


def _act_with_obs_dim(theta: np.ndarray, obs: np.ndarray, obs_dim: int):
    i = 0
    w1 = theta[i:i + obs_dim * HIDDEN].reshape(obs_dim, HIDDEN); i += obs_dim * HIDDEN
    b1 = theta[i:i + HIDDEN];                                     i += HIDDEN
    w2 = theta[i:i + HIDDEN * ACT_DIM].reshape(HIDDEN, ACT_DIM);  i += HIDDEN * ACT_DIM
    b2 = theta[i:i + ACT_DIM]
    h = np.tanh(obs @ w1 + b1)
    out = h @ w2 + b2
    return bool(out[0] > 0.0), bool(out[1] > 0.0), float(np.tanh(out[2])), float(np.tanh(out[3]))


def _theta_memory_warm_start(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = rng.normal(0.0, 0.1, _SIM_MEMORY_PARAM_COUNT)
    theta[_SIM_MEMORY_B2_OFFSET + 0] += 2.0
    theta[_SIM_MEMORY_B2_OFFSET + 1] += 1.0
    return theta


def _theta_memory_always_aim(aim_x: float, aim_y: float) -> np.ndarray:
    theta = np.zeros(_SIM_MEMORY_PARAM_COUNT)
    theta[_SIM_MEMORY_B2_OFFSET + 0] = 50.0
    theta[_SIM_MEMORY_B2_OFFSET + 1] = 50.0
    theta[_SIM_MEMORY_B2_OFFSET + 2] = _aim_bias_for(aim_x)
    theta[_SIM_MEMORY_B2_OFFSET + 3] = _aim_bias_for(aim_y)
    return theta


class TimedSpotMemoryEnv(TimedSpotBaselineEnv):
    """Simulation-only v2 env: timed 5-spot game + memory of the last
    successful firing position and how long ago it worked."""

    MEMORY_HORIZON_TICKS = TimedSpotGame.TIMED_SPOT_COUNT * TimedSpotGame.SPOT_HOLD_FRAMES

    def reset(self) -> np.ndarray:
        self.client.load_state(self.state_slot)
        self.client.step_frames(2)
        self.prev = self._read_core()
        self.start_timer = self.prev["timer"]
        self.ticks = 0
        self.prev_peek = False
        self.peek_ticks = 0
        self.peek_lock = 0
        self.peek_locked_value = False
        self.stale_core_ticks = 0
        self.ammo_left = AMMO_MAX_ROUNDS
        self.prev_aim_x_bias = 0.0
        self.prev_aim_y_bias = 0.0
        self.last_success_aim_x = 0.5
        self.last_success_aim_y = 0.5
        self.ticks_since_success = self.MEMORY_HORIZON_TICKS
        self.phase_infer.reset()
        return self._build_obs_v2(self.prev, 0, 0, 0.0, self.ammo_left)

    def _build_obs_v2(self, cur, last_hit: int, last_miss: int,
                      peek_phase: float = 0.0,
                      ammo_left: int = AMMO_MAX_ROUNDS) -> np.ndarray:
        base = self._build_obs(
            cur,
            last_hit,
            last_miss,
            peek_phase,
            ammo_left,
            self.prev_aim_x_bias,
            self.prev_aim_y_bias,
        )
        since_norm = min(self.ticks_since_success / self.MEMORY_HORIZON_TICKS, 1.0)
        memory = np.array([
            self.last_success_aim_x,
            self.last_success_aim_y,
            since_norm,
        ], dtype=np.float32)
        return np.concatenate([base, memory], dtype=np.float32)

    def step(self, theta: np.ndarray):
        peek_phase = (self.peek_ticks / PEEK_TRAVERSE_TICKS) * (1.0 if self.prev_peek else -1.0)
        shoot, peek, aim_x_bias, aim_y_bias = _act_with_obs_dim(
            theta,
            self._build_obs_v2(self.prev, 0, 0, peek_phase, self.ammo_left),
            _SIM_MEMORY_OBS_DIM,
        )
        self.prev_aim_x_bias = float(aim_x_bias)
        self.prev_aim_y_bias = float(aim_y_bias)

        if self.ammo_left == 0:
            peek = False

        if self.peek_lock > 0:
            peek = self.peek_locked_value
            self.peek_lock -= 1
        elif peek != self.prev_peek:
            self.peek_lock = PEEK_TRAVERSE_TICKS - 1
            self.peek_locked_value = peek

        shoot_allowed = peek and self.prev_peek and self.peek_ticks >= PEEK_TRAVERSE_TICKS
        aim_x = min(1.0, max(0.0, 0.5 + float(aim_x_bias)))
        aim_y = min(1.0, max(0.0, 0.5 + float(aim_y_bias)))

        total_fired = total_hit = total_life_loss = 0
        cleared_guess = dead_guess = timed_out_guess = False
        continue_screen_guess = False
        tick_start_core = core_watchdog_snapshot(self.prev)
        timer_at_tick_start = self.prev["timer"]

        for f in range(FRAME_SKIP):
            self.client.set_input(
                shoot=bool(shoot and shoot_allowed and f < 2),
                peek=peek,
                aim_x=aim_x,
                aim_y=aim_y,
            )

            pre = self.prev
            self.client.step_frames(1)
            post = self._read_core()

            total_fired += max(0, u16_delta(post["shots_fired"], pre["shots_fired"]))
            total_hit   += max(0, u16_delta(post["shots_hit"],   pre["shots_hit"]))
            life_d       = u16_delta(post["life"], pre["life"])
            if life_d < 0:
                total_life_loss += -life_d

            lethal_wrap = (
                pre["life"] > 0
                and life_d < 0
                and post["life"] > pre["life"]
                and post["life"] >= 65000
            )
            if post["life"] == 0 or lethal_wrap:
                dead_guess = True
            if u16_delta(post["timer"], self.start_timer) > 100:
                cleared_guess = True
            timer_step = u16_delta(post["timer"], pre["timer"])
            if post["timer"] <= TIMEOUT_TIMER_THRESHOLD or (
                timer_step < 0 and pre["timer"] + timer_step <= TIMEOUT_TIMER_THRESHOLD
            ):
                timed_out_guess = True

            self.prev = post
            if dead_guess or cleared_guess or timed_out_guess:
                break

        if (
            not dead_guess
            and not cleared_guess
            and not timed_out_guess
            and core_watchdog_snapshot(self.prev) == tick_start_core
        ):
            self.stale_core_ticks += 1
        else:
            self.stale_core_ticks = 0
        if self.stale_core_ticks >= CONTINUE_SCREEN_STALE_TICKS:
            timed_out_guess = True
            continue_screen_guess = True

        ammo_before_tick = self.ammo_left
        dry_fire = bool(shoot_allowed and ammo_before_tick == 0)
        self.ammo_left = max(0, self.ammo_left - total_fired)
        ending_peek = (peek != self.prev_peek) and not peek
        reload_correct = False
        if ending_peek:
            reload_correct = self.ammo_left == 0
            self.ammo_left = AMMO_MAX_ROUNDS

        self.ticks += 1

        if total_hit > 0:
            self.last_success_aim_x = float(aim_x)
            self.last_success_aim_y = float(aim_y)
            self.ticks_since_success = 0
        else:
            self.ticks_since_success = min(self.ticks_since_success + 1, self.MEMORY_HORIZON_TICKS)

        phase = self.phase_infer.infer(TickSignals(
            shots_fired_delta=total_fired,
            shots_hit_delta=total_hit,
            life_delta=-total_life_loss,
            timer_delta=u16_delta(self.prev["timer"], timer_at_tick_start),
            cleared_guess=cleared_guess,
            dead_guess=dead_guess or timed_out_guess,
            can_fire_probe=(total_fired > 0),
        ))

        last_hit = 1 if total_hit > 0 else 0
        last_miss = 1 if (total_fired > 0 and total_hit == 0) else 0
        if peek == self.prev_peek:
            self.peek_ticks = min(self.peek_ticks + 1, PEEK_TRAVERSE_TICKS)
        else:
            self.peek_ticks = 1
        self.prev_peek = peek
        peek_phase_next = (self.peek_ticks / PEEK_TRAVERSE_TICKS) * (1.0 if peek else -1.0)
        obs = self._build_obs_v2(self.prev, last_hit, last_miss, peek_phase_next, self.ammo_left)

        game = self.client._game
        active_x, active_y = game.active_spot()

        done = (phase is Phase.TERMINAL) or (self.ticks >= MAX_TICKS)
        info = {
            "shots_fired_delta": total_fired,
            "shots_hit_delta": total_hit,
            "life_loss": total_life_loss,
            "cleared": bool(cleared_guess and not dead_guess and not timed_out_guess),
            "dead": dead_guess,
            "timed_out": timed_out_guess,
            "continue_screen": continue_screen_guess,
            "peek": bool(peek),
            "phase": phase.name,
            "dry_fire": dry_fire,
            "reload_correct": reload_correct,
            "ammo_left": self.ammo_left,
            "aim_x": float(aim_x),
            "aim_y": float(aim_y),
            "last_success_aim_x": float(self.last_success_aim_x),
            "last_success_aim_y": float(self.last_success_aim_y),
            "ticks_since_success": int(self.ticks_since_success),
            "active_spot_index": int(game.active_spot_index()),
            "active_spot_x": float(active_x),
            "active_spot_y": float(active_y),
        }
        return obs, done, info


# ---------------------------------------------------------------------------
# Hotspot-bank memory (v3): remember several recent hits, not just the last
# ---------------------------------------------------------------------------

HOTSPOT_BANK_CAPACITY = 5
HOTSPOT_MERGE_RADIUS = 0.12


class HotspotBank:
    """Fixed-capacity memory of recently-confirmed successful aim spots.

    Purely feedback-driven: only ever updated from the policy's own confirmed
    hits, never from the game's true target schedule, so it can't leak ground
    truth into the observation. Each slot tracks (x, y, ticks_since_hit,
    hit_count); a hit near an existing slot merges into it (EMA position
    update, ``hit_count += 1``, recency reset) instead of spawning a
    duplicate. Once the bank is full, a new, sufficiently distinct hit evicts
    whichever slot currently has the weakest recency+confidence score.
    """

    def __init__(self, capacity: int = HOTSPOT_BANK_CAPACITY,
                 merge_radius: float = HOTSPOT_MERGE_RADIUS,
                 recency_horizon: int = 150, ema_alpha: float = 0.3):
        self.capacity = capacity
        self.merge_radius = merge_radius
        self.recency_horizon = recency_horizon
        self.ema_alpha = ema_alpha
        self.slots: list[dict] = []

    def reset(self) -> None:
        self.slots = []

    def tick(self) -> None:
        """Age every remembered slot by one decision tick."""
        for slot in self.slots:
            slot["ticks_since_hit"] = min(slot["ticks_since_hit"] + 1, self.recency_horizon)

    def _slot_score(self, slot: dict) -> float:
        """Higher = fresher and/or more confirmed -- worth aiming at again."""
        recency = 1.0 - min(slot["ticks_since_hit"] / self.recency_horizon, 1.0)
        confidence = min(slot["hit_count"], 5) / 5.0
        return 0.5 * recency + 0.5 * confidence

    def record_hit(self, x: float, y: float) -> None:
        nearest, nearest_dist = None, None
        for slot in self.slots:
            d = ((slot["x"] - x) ** 2 + (slot["y"] - y) ** 2) ** 0.5
            if nearest_dist is None or d < nearest_dist:
                nearest, nearest_dist = slot, d

        if nearest is not None and nearest_dist <= self.merge_radius:
            nearest["x"] += self.ema_alpha * (x - nearest["x"])
            nearest["y"] += self.ema_alpha * (y - nearest["y"])
            nearest["ticks_since_hit"] = 0
            nearest["hit_count"] += 1
            return

        new_slot = {"x": float(x), "y": float(y), "ticks_since_hit": 0, "hit_count": 1}
        if len(self.slots) < self.capacity:
            self.slots.append(new_slot)
            return

        worst_i = min(range(len(self.slots)), key=lambda i: self._slot_score(self.slots[i]))
        self.slots[worst_i] = new_slot

    def best_candidate(self):
        """Return (x, y, confidence_norm, recency_norm) for the top-scoring
        slot (recency_norm: 0 = just hit, 1 = stale/at-or-past the horizon),
        or a neutral, maximally-stale default when the bank is empty."""
        if not self.slots:
            return 0.5, 0.5, 0.0, 1.0
        best = max(self.slots, key=self._slot_score)
        confidence_norm = min(best["hit_count"], 5) / 5.0
        recency_norm = min(best["ticks_since_hit"] / self.recency_horizon, 1.0)
        return best["x"], best["y"], confidence_norm, recency_norm


_SIM_HOTSPOT_EXTRA_DIMS = 4
_SIM_HOTSPOT_OBS_DIM = OBS_DIM + _SIM_HOTSPOT_EXTRA_DIMS
_SIM_HOTSPOT_PARAM_COUNT = _SIM_HOTSPOT_OBS_DIM * HIDDEN + HIDDEN + HIDDEN * ACT_DIM + ACT_DIM
_SIM_HOTSPOT_B2_OFFSET = _SIM_HOTSPOT_OBS_DIM * HIDDEN + HIDDEN + HIDDEN * ACT_DIM


def _theta_hotspot_warm_start(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = rng.normal(0.0, 0.1, _SIM_HOTSPOT_PARAM_COUNT)
    theta[_SIM_HOTSPOT_B2_OFFSET + 0] += 2.0  # shoot logit
    theta[_SIM_HOTSPOT_B2_OFFSET + 1] += 1.0  # peek  logit
    return theta


def _theta_hotspot_always_aim(aim_x: float, aim_y: float) -> np.ndarray:
    theta = np.zeros(_SIM_HOTSPOT_PARAM_COUNT)
    theta[_SIM_HOTSPOT_B2_OFFSET + 0] = 50.0  # shoot logit: always shoot
    theta[_SIM_HOTSPOT_B2_OFFSET + 1] = 50.0  # peek  logit: always exposed
    theta[_SIM_HOTSPOT_B2_OFFSET + 2] = _aim_bias_for(aim_x)
    theta[_SIM_HOTSPOT_B2_OFFSET + 3] = _aim_bias_for(aim_y)
    return theta


class TimedSpotHotspotMemoryEnv(TimedSpotBaselineEnv):
    """Simulation-only v3 env: timed 5-spot game + a small bank of recently
    confirmed hotspots (position, recency, confidence) instead of
    remembering only the single last successful shot (see
    ``TimedSpotMemoryEnv``). Feedback-driven only: the bank is built purely
    from the policy's own confirmed hits and never reads the game's true
    active-spot schedule, so it can't leak ground truth into the
    observation.
    """

    def __init__(self, seed: int = 0):
        game = TimedSpotGame(seed=seed)
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
        self.stale_core_ticks   = 0
        self.ammo_left          = AMMO_MAX_ROUNDS
        self.prev_aim_x_bias    = 0.0
        self.prev_aim_y_bias    = 0.0
        self.hotspots           = HotspotBank()

    def reset(self) -> np.ndarray:
        self.client.load_state(self.state_slot)
        self.client.step_frames(2)
        self.prev = self._read_core()
        self.start_timer = self.prev["timer"]
        self.ticks = 0
        self.prev_peek = False
        self.peek_ticks = 0
        self.peek_lock = 0
        self.peek_locked_value = False
        self.stale_core_ticks = 0
        self.ammo_left = AMMO_MAX_ROUNDS
        self.prev_aim_x_bias = 0.0
        self.prev_aim_y_bias = 0.0
        self.hotspots.reset()
        self.phase_infer.reset()
        return self._build_obs_v3(self.prev, 0, 0, 0.0, self.ammo_left)

    def _build_obs_v3(self, cur, last_hit: int, last_miss: int,
                      peek_phase: float = 0.0,
                      ammo_left: int = AMMO_MAX_ROUNDS) -> np.ndarray:
        base = self._build_obs(
            cur, last_hit, last_miss, peek_phase, ammo_left,
            self.prev_aim_x_bias, self.prev_aim_y_bias,
        )
        best_x, best_y, confidence_norm, recency_norm = self.hotspots.best_candidate()
        memory = np.array([best_x, best_y, confidence_norm, recency_norm], dtype=np.float32)
        return np.concatenate([base, memory], dtype=np.float32)

    def step(self, theta: np.ndarray):
        peek_phase = (self.peek_ticks / PEEK_TRAVERSE_TICKS) * (1.0 if self.prev_peek else -1.0)
        shoot, peek, aim_x_bias, aim_y_bias = _act_with_obs_dim(
            theta,
            self._build_obs_v3(self.prev, 0, 0, peek_phase, self.ammo_left),
            _SIM_HOTSPOT_OBS_DIM,
        )
        self.prev_aim_x_bias = float(aim_x_bias)
        self.prev_aim_y_bias = float(aim_y_bias)

        if self.ammo_left == 0:
            peek = False

        if self.peek_lock > 0:
            peek = self.peek_locked_value
            self.peek_lock -= 1
        elif peek != self.prev_peek:
            self.peek_lock = PEEK_TRAVERSE_TICKS - 1
            self.peek_locked_value = peek

        shoot_allowed = peek and self.prev_peek and self.peek_ticks >= PEEK_TRAVERSE_TICKS
        aim_x = min(1.0, max(0.0, 0.5 + float(aim_x_bias)))
        aim_y = min(1.0, max(0.0, 0.5 + float(aim_y_bias)))

        total_fired = total_hit = total_life_loss = 0
        cleared_guess = dead_guess = timed_out_guess = False
        continue_screen_guess = False
        tick_start_core = core_watchdog_snapshot(self.prev)
        timer_at_tick_start = self.prev["timer"]

        for f in range(FRAME_SKIP):
            self.client.set_input(
                shoot=bool(shoot and shoot_allowed and f < 2),
                peek=peek,
                aim_x=aim_x,
                aim_y=aim_y,
            )

            pre = self.prev
            self.client.step_frames(1)
            post = self._read_core()

            total_fired += max(0, u16_delta(post["shots_fired"], pre["shots_fired"]))
            total_hit   += max(0, u16_delta(post["shots_hit"],   pre["shots_hit"]))
            life_d       = u16_delta(post["life"], pre["life"])
            if life_d < 0:
                total_life_loss += -life_d

            lethal_wrap = (
                pre["life"] > 0
                and life_d < 0
                and post["life"] > pre["life"]
                and post["life"] >= 65000
            )
            if post["life"] == 0 or lethal_wrap:
                dead_guess = True
            if u16_delta(post["timer"], self.start_timer) > 100:
                cleared_guess = True
            timer_step = u16_delta(post["timer"], pre["timer"])
            if post["timer"] <= TIMEOUT_TIMER_THRESHOLD or (
                timer_step < 0 and pre["timer"] + timer_step <= TIMEOUT_TIMER_THRESHOLD
            ):
                timed_out_guess = True

            self.prev = post
            if dead_guess or cleared_guess or timed_out_guess:
                break

        if (
            not dead_guess
            and not cleared_guess
            and not timed_out_guess
            and core_watchdog_snapshot(self.prev) == tick_start_core
        ):
            self.stale_core_ticks += 1
        else:
            self.stale_core_ticks = 0
        if self.stale_core_ticks >= CONTINUE_SCREEN_STALE_TICKS:
            timed_out_guess = True
            continue_screen_guess = True

        ammo_before_tick = self.ammo_left
        dry_fire = bool(shoot_allowed and ammo_before_tick == 0)
        self.ammo_left = max(0, self.ammo_left - total_fired)
        ending_peek = (peek != self.prev_peek) and not peek
        reload_correct = False
        if ending_peek:
            reload_correct = self.ammo_left == 0
            self.ammo_left = AMMO_MAX_ROUNDS

        self.ticks += 1

        # Age the bank once per decision tick, then record this tick's hit
        # (if any) at the aim position actually used -- never at the game's
        # true active-spot position, which the policy has no access to.
        self.hotspots.tick()
        if total_hit > 0:
            self.hotspots.record_hit(aim_x, aim_y)

        phase = self.phase_infer.infer(TickSignals(
            shots_fired_delta=total_fired,
            shots_hit_delta=total_hit,
            life_delta=-total_life_loss,
            timer_delta=u16_delta(self.prev["timer"], timer_at_tick_start),
            cleared_guess=cleared_guess,
            dead_guess=dead_guess or timed_out_guess,
            can_fire_probe=(total_fired > 0),
        ))

        last_hit = 1 if total_hit > 0 else 0
        last_miss = 1 if (total_fired > 0 and total_hit == 0) else 0
        if peek == self.prev_peek:
            self.peek_ticks = min(self.peek_ticks + 1, PEEK_TRAVERSE_TICKS)
        else:
            self.peek_ticks = 1
        self.prev_peek = peek
        peek_phase_next = (self.peek_ticks / PEEK_TRAVERSE_TICKS) * (1.0 if peek else -1.0)
        obs = self._build_obs_v3(self.prev, last_hit, last_miss, peek_phase_next, self.ammo_left)

        game = self.client._game
        active_x, active_y = game.active_spot()
        best_x, best_y, confidence_norm, recency_norm = self.hotspots.best_candidate()

        done = (phase is Phase.TERMINAL) or (self.ticks >= MAX_TICKS)
        info = {
            "shots_fired_delta": total_fired,
            "shots_hit_delta": total_hit,
            "life_loss": total_life_loss,
            "cleared": bool(cleared_guess and not dead_guess and not timed_out_guess),
            "dead": dead_guess,
            "timed_out": timed_out_guess,
            "continue_screen": continue_screen_guess,
            "peek": bool(peek),
            "phase": phase.name,
            "dry_fire": dry_fire,
            "reload_correct": reload_correct,
            "ammo_left": self.ammo_left,
            "aim_x": float(aim_x),
            "aim_y": float(aim_y),
            "hotspot_best_x": float(best_x),
            "hotspot_best_y": float(best_y),
            "hotspot_confidence": float(confidence_norm),
            "hotspot_recency": float(recency_norm),
            "hotspot_count": len(self.hotspots.slots),
            "active_spot_index": int(game.active_spot_index()),
            "active_spot_x": float(active_x),
            "active_spot_y": float(active_y),
        }
        return obs, done, info


# ---------------------------------------------------------------------------
# Shot-index one-hot observation env (sim-only, idea #1 from 2026-08-06)
# ---------------------------------------------------------------------------
#
# Motivation: the memoryless MLP in policy.py, fed a smooth deterministic
# ammo_norm ramp (1.0 -> 0.17 across a clip) and a smooth prev_aim_bias
# feedback, naturally emits a smooth deterministic AIM arc across the 6
# shots of every clip -- the arc is essentially defined by the architecture,
# not chosen by the agent. ES then picks whichever random arc a lucky theta
# stumbled into. Adding a one-hot of the CURRENT shot index in the clip
# gives the network 6 independent input columns, one per shot, so ES can
# route each shot to a distinct aim output without disentangling it from a
# scalar ramp. This directly targets the "same 6-shot arc every clip"
# symptom without adding recurrence or changing the fitness formula.
#
# Shot index := AMMO_MAX_ROUNDS - ammo_left, clamped to [0, AMMO_MAX_ROUNDS-1].
# On the first shot of a clip ammo_left == 6 so shot_index == 0; on the last
# shot ammo_left == 1 so shot_index == 5. After the last shot ammo_left
# would drop to 0 but the hard duck-on-empty override kicks in and refills
# on the reload transition, so the index resets to 0 at the start of the
# next clip.

_SIM_SHOTIDX_EXTRA_DIMS = AMMO_MAX_ROUNDS
_SIM_SHOTIDX_OBS_DIM = OBS_DIM + _SIM_SHOTIDX_EXTRA_DIMS
_SIM_SHOTIDX_PARAM_COUNT = _SIM_SHOTIDX_OBS_DIM * HIDDEN + HIDDEN + HIDDEN * ACT_DIM + ACT_DIM
_SIM_SHOTIDX_B2_OFFSET = _SIM_SHOTIDX_OBS_DIM * HIDDEN + HIDDEN + HIDDEN * ACT_DIM


def _theta_shotidx_warm_start(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = rng.normal(0.0, 0.1, _SIM_SHOTIDX_PARAM_COUNT)
    theta[_SIM_SHOTIDX_B2_OFFSET + 0] += 2.0  # shoot logit
    theta[_SIM_SHOTIDX_B2_OFFSET + 1] += 1.0  # peek  logit
    return theta


def _shot_index_one_hot(ammo_left: int) -> np.ndarray:
    """Return a 6-dim one-hot of the current shot index within the clip."""
    idx = AMMO_MAX_ROUNDS - int(ammo_left)
    if idx < 0:
        idx = 0
    elif idx >= AMMO_MAX_ROUNDS:
        idx = AMMO_MAX_ROUNDS - 1
    onehot = np.zeros(AMMO_MAX_ROUNDS, dtype=np.float32)
    onehot[idx] = 1.0
    return onehot


class TimedSpotShotIndexEnv(TimedSpotBaselineEnv):
    """Sim-only env: timed 5-spot game + a 6-dim one-hot of the current
    shot index in the clip appended to the observation. See the module
    comment above for motivation. Fitness formula and step() bookkeeping
    are IDENTICAL to ``TimedSpotBaselineEnv``; only the observation stack
    differs, so any A/B result vs. baseline is attributable to the extra
    input dims (and the wider policy that consumes them) alone."""

    def reset(self) -> np.ndarray:
        obs = super().reset()
        return np.concatenate(
            [obs, _shot_index_one_hot(self.ammo_left)], dtype=np.float32
        )

    def step(self, theta: np.ndarray):
        peek_phase = (self.peek_ticks / PEEK_TRAVERSE_TICKS) * (1.0 if self.prev_peek else -1.0)
        base_obs = self._build_obs(
            self.prev, 0, 0, peek_phase, self.ammo_left,
            self.prev_aim_x_bias, self.prev_aim_y_bias,
        )
        aug_obs = np.concatenate(
            [base_obs, _shot_index_one_hot(self.ammo_left)], dtype=np.float32
        )
        shoot, peek, aim_x_bias, aim_y_bias = _act_with_obs_dim(
            theta, aug_obs, _SIM_SHOTIDX_OBS_DIM,
        )
        self.prev_aim_x_bias = float(aim_x_bias)
        self.prev_aim_y_bias = float(aim_y_bias)

        if self.ammo_left == 0:
            peek = False

        if self.peek_lock > 0:
            peek = self.peek_locked_value
            self.peek_lock -= 1
        elif peek != self.prev_peek:
            self.peek_lock = PEEK_TRAVERSE_TICKS - 1
            self.peek_locked_value = peek

        shoot_allowed = peek and self.prev_peek and self.peek_ticks >= PEEK_TRAVERSE_TICKS
        aim_x = min(1.0, max(0.0, 0.5 + float(aim_x_bias)))
        aim_y = min(1.0, max(0.0, 0.5 + float(aim_y_bias)))

        total_fired = total_hit = total_life_loss = 0
        cleared_guess = dead_guess = timed_out_guess = False
        continue_screen_guess = False
        tick_start_core = core_watchdog_snapshot(self.prev)
        timer_at_tick_start = self.prev["timer"]

        for f in range(FRAME_SKIP):
            self.client.set_input(
                shoot=bool(shoot and shoot_allowed and f < 2),
                peek=peek,
                aim_x=aim_x,
                aim_y=aim_y,
            )
            pre = self.prev
            self.client.step_frames(1)
            post = self._read_core()

            total_fired += max(0, u16_delta(post["shots_fired"], pre["shots_fired"]))
            total_hit   += max(0, u16_delta(post["shots_hit"],   pre["shots_hit"]))
            life_d       = u16_delta(post["life"], pre["life"])
            if life_d < 0:
                total_life_loss += -life_d

            lethal_wrap = (
                pre["life"] > 0
                and life_d < 0
                and post["life"] > pre["life"]
                and post["life"] >= 65000
            )
            if post["life"] == 0 or lethal_wrap:
                dead_guess = True
            if u16_delta(post["timer"], self.start_timer) > 100:
                cleared_guess = True
            timer_step = u16_delta(post["timer"], pre["timer"])
            if post["timer"] <= TIMEOUT_TIMER_THRESHOLD or (
                timer_step < 0 and pre["timer"] + timer_step <= TIMEOUT_TIMER_THRESHOLD
            ):
                timed_out_guess = True

            self.prev = post
            if dead_guess or cleared_guess or timed_out_guess:
                break

        if (
            not dead_guess
            and not cleared_guess
            and not timed_out_guess
            and core_watchdog_snapshot(self.prev) == tick_start_core
        ):
            self.stale_core_ticks += 1
        else:
            self.stale_core_ticks = 0
        if self.stale_core_ticks >= CONTINUE_SCREEN_STALE_TICKS:
            timed_out_guess = True
            continue_screen_guess = True

        ammo_before_tick = self.ammo_left
        dry_fire = bool(shoot_allowed and ammo_before_tick == 0)
        self.ammo_left = max(0, self.ammo_left - total_fired)
        ending_peek = (peek != self.prev_peek) and not peek
        reload_correct = False
        if ending_peek:
            reload_correct = self.ammo_left == 0
            self.ammo_left = AMMO_MAX_ROUNDS

        self.ticks += 1

        phase = self.phase_infer.infer(TickSignals(
            shots_fired_delta=total_fired,
            shots_hit_delta=total_hit,
            life_delta=-total_life_loss,
            timer_delta=u16_delta(self.prev["timer"], timer_at_tick_start),
            cleared_guess=cleared_guess,
            dead_guess=dead_guess or timed_out_guess,
            can_fire_probe=(total_fired > 0),
        ))

        last_hit = 1 if total_hit > 0 else 0
        last_miss = 1 if (total_fired > 0 and total_hit == 0) else 0
        if peek == self.prev_peek:
            self.peek_ticks = min(self.peek_ticks + 1, PEEK_TRAVERSE_TICKS)
        else:
            self.peek_ticks = 1
        self.prev_peek = peek
        peek_phase_next = (self.peek_ticks / PEEK_TRAVERSE_TICKS) * (1.0 if peek else -1.0)
        base_obs_next = self._build_obs(
            self.prev, last_hit, last_miss, peek_phase_next, self.ammo_left,
            self.prev_aim_x_bias, self.prev_aim_y_bias,
        )
        obs = np.concatenate(
            [base_obs_next, _shot_index_one_hot(self.ammo_left)], dtype=np.float32
        )

        done = (phase is Phase.TERMINAL) or (self.ticks >= MAX_TICKS)
        info = {
            "shots_fired_delta": total_fired,
            "shots_hit_delta": total_hit,
            "life_loss": total_life_loss,
            "cleared": bool(cleared_guess and not dead_guess and not timed_out_guess),
            "dead": dead_guess,
            "timed_out": timed_out_guess,
            "continue_screen": continue_screen_guess,
            "peek": bool(peek),
            "phase": phase.name,
            "dry_fire": dry_fire,
            "reload_correct": reload_correct,
            "ammo_left": self.ammo_left,
            "aim_x": float(aim_x),
            "aim_y": float(aim_y),
            "shot_index": int(AMMO_MAX_ROUNDS - ammo_before_tick),
        }
        return obs, done, info


# ---------------------------------------------------------------------------
# Shot-phase (sin/cos) observation env (sim-only, follow-up to shot-index)
# ---------------------------------------------------------------------------
#
# The 6-dim one-hot shot-index probe (`TimedSpotShotIndexEnv` above) DID
# loosen the arch on the X axis and increased overall aim spread, but
# hurt task performance in a fixed 30-gen budget -- 6 extra input dims add
# 6*HIDDEN=384 extra parameters for ES to random-walk over, which slows
# convergence even when the mechanism is sound.
#
# This variant tries the same idea with a much smaller signal:
#   shot_phase = 2 * pi * (AMMO_MAX_ROUNDS - ammo_left) / AMMO_MAX_ROUNDS
#   obs_extra  = (sin(shot_phase), cos(shot_phase))
# so only 2 extra dims (2*HIDDEN=128 extra params, 33% of the one-hot cost)
# still give a NON-SMOOTH per-shot signal (adjacent shot indices land at
# distinct 2-D positions on the unit circle, not on a monotonic ramp like
# ammo_norm) without the parameter blowup.
#
# The warm-start ALSO zero-initializes the two new input-column rows of
# `w1` (see `_theta_shotphase_warm_start` below), so at gen 0 the network
# behaves EXACTLY like the baseline warm-start: the new dims contribute
# nothing until ES's random perturbations discover them. This isolates the
# effect of the extra representational capacity from the effect of
# perturbing early behavior on day one.

_SIM_SHOTPHASE_EXTRA_DIMS = 2
_SIM_SHOTPHASE_OBS_DIM = OBS_DIM + _SIM_SHOTPHASE_EXTRA_DIMS
_SIM_SHOTPHASE_PARAM_COUNT = (
    _SIM_SHOTPHASE_OBS_DIM * HIDDEN + HIDDEN
    + HIDDEN * ACT_DIM + ACT_DIM
)
_SIM_SHOTPHASE_B2_OFFSET = (
    _SIM_SHOTPHASE_OBS_DIM * HIDDEN + HIDDEN + HIDDEN * ACT_DIM
)


def _shot_phase_features(ammo_left: int) -> np.ndarray:
    """Return (sin, cos) of the current shot-in-clip phase angle.

    Uses shot_idx = AMMO_MAX_ROUNDS - ammo_left so the angle advances by
    2*pi/AMMO_MAX_ROUNDS with each shot; when ammo refills back to the
    max at the reload transition, the angle wraps back to 0 (sin=0,
    cos=1) -- same value as the first shot of a clip.
    """
    idx = AMMO_MAX_ROUNDS - int(ammo_left)
    if idx < 0:
        idx = 0
    elif idx >= AMMO_MAX_ROUNDS:
        idx = AMMO_MAX_ROUNDS - 1
    angle = 2.0 * np.pi * idx / AMMO_MAX_ROUNDS
    return np.array([np.sin(angle), np.cos(angle)], dtype=np.float32)


def _theta_shotphase_warm_start(seed: int = 42) -> np.ndarray:
    """Warm-start for the shot-phase env, structured so that at gen 0 the
    network's output is IDENTICAL to a `_theta_warm_start` baseline theta
    with the same seed.

    Trick: draw the BASELINE theta first (13-dim obs). Its `w1` occupies
    the first OBS_DIM*HIDDEN entries. For the wider theta we need a
    (15, HIDDEN) `w1` block. We reshape the baseline's `w1` to (13, HIDDEN),
    concatenate a (2, HIDDEN) block of ZEROS for the new sin/cos input
    columns, flatten, and append the baseline's `b1`/`w2`/`b2` verbatim.
    So the extra input dims multiply against zero weights at gen 0 and
    contribute nothing to the hidden layer -- the net is behaviorally
    identical to baseline until ES perturbations kick in.
    """
    baseline_theta = _theta_warm_start(seed)
    i = 0
    w1_base = baseline_theta[i:i + OBS_DIM * HIDDEN].reshape(OBS_DIM, HIDDEN)
    i += OBS_DIM * HIDDEN
    b1 = baseline_theta[i:i + HIDDEN]
    i += HIDDEN
    w2 = baseline_theta[i:i + HIDDEN * ACT_DIM]
    i += HIDDEN * ACT_DIM
    b2 = baseline_theta[i:i + ACT_DIM]

    extra_rows = np.zeros((_SIM_SHOTPHASE_EXTRA_DIMS, HIDDEN), dtype=w1_base.dtype)
    w1_wide = np.concatenate([w1_base, extra_rows], axis=0)  # (15, HIDDEN)

    theta = np.concatenate([w1_wide.reshape(-1), b1, w2, b2]).astype(baseline_theta.dtype)
    assert theta.shape[0] == _SIM_SHOTPHASE_PARAM_COUNT
    return theta


class TimedSpotShotPhaseEnv(TimedSpotBaselineEnv):
    """Sim-only env: timed 5-spot game + 2-dim (sin, cos) shot-phase
    features appended to the obs. See the module comment above for
    motivation and the zero-init warm-start rationale. Fitness formula
    and step() bookkeeping are IDENTICAL to `TimedSpotBaselineEnv`; only
    the observation stack differs.
    """

    def reset(self) -> np.ndarray:
        obs = super().reset()
        return np.concatenate(
            [obs, _shot_phase_features(self.ammo_left)], dtype=np.float32
        )

    def step(self, theta: np.ndarray):
        peek_phase = (self.peek_ticks / PEEK_TRAVERSE_TICKS) * (1.0 if self.prev_peek else -1.0)
        base_obs = self._build_obs(
            self.prev, 0, 0, peek_phase, self.ammo_left,
            self.prev_aim_x_bias, self.prev_aim_y_bias,
        )
        aug_obs = np.concatenate(
            [base_obs, _shot_phase_features(self.ammo_left)], dtype=np.float32
        )
        shoot, peek, aim_x_bias, aim_y_bias = _act_with_obs_dim(
            theta, aug_obs, _SIM_SHOTPHASE_OBS_DIM,
        )
        self.prev_aim_x_bias = float(aim_x_bias)
        self.prev_aim_y_bias = float(aim_y_bias)

        if self.ammo_left == 0:
            peek = False

        if self.peek_lock > 0:
            peek = self.peek_locked_value
            self.peek_lock -= 1
        elif peek != self.prev_peek:
            self.peek_lock = PEEK_TRAVERSE_TICKS - 1
            self.peek_locked_value = peek

        shoot_allowed = peek and self.prev_peek and self.peek_ticks >= PEEK_TRAVERSE_TICKS
        aim_x = min(1.0, max(0.0, 0.5 + float(aim_x_bias)))
        aim_y = min(1.0, max(0.0, 0.5 + float(aim_y_bias)))

        total_fired = total_hit = total_life_loss = 0
        cleared_guess = dead_guess = timed_out_guess = False
        continue_screen_guess = False
        tick_start_core = core_watchdog_snapshot(self.prev)
        timer_at_tick_start = self.prev["timer"]

        for f in range(FRAME_SKIP):
            self.client.set_input(
                shoot=bool(shoot and shoot_allowed and f < 2),
                peek=peek,
                aim_x=aim_x,
                aim_y=aim_y,
            )
            pre = self.prev
            self.client.step_frames(1)
            post = self._read_core()

            total_fired += max(0, u16_delta(post["shots_fired"], pre["shots_fired"]))
            total_hit   += max(0, u16_delta(post["shots_hit"],   pre["shots_hit"]))
            life_d       = u16_delta(post["life"], pre["life"])
            if life_d < 0:
                total_life_loss += -life_d

            lethal_wrap = (
                pre["life"] > 0
                and life_d < 0
                and post["life"] > pre["life"]
                and post["life"] >= 65000
            )
            if post["life"] == 0 or lethal_wrap:
                dead_guess = True
            if u16_delta(post["timer"], self.start_timer) > 100:
                cleared_guess = True
            timer_step = u16_delta(post["timer"], pre["timer"])
            if post["timer"] <= TIMEOUT_TIMER_THRESHOLD or (
                timer_step < 0 and pre["timer"] + timer_step <= TIMEOUT_TIMER_THRESHOLD
            ):
                timed_out_guess = True

            self.prev = post
            if dead_guess or cleared_guess or timed_out_guess:
                break

        if (
            not dead_guess
            and not cleared_guess
            and not timed_out_guess
            and core_watchdog_snapshot(self.prev) == tick_start_core
        ):
            self.stale_core_ticks += 1
        else:
            self.stale_core_ticks = 0
        if self.stale_core_ticks >= CONTINUE_SCREEN_STALE_TICKS:
            timed_out_guess = True
            continue_screen_guess = True

        ammo_before_tick = self.ammo_left
        dry_fire = bool(shoot_allowed and ammo_before_tick == 0)
        self.ammo_left = max(0, self.ammo_left - total_fired)
        ending_peek = (peek != self.prev_peek) and not peek
        reload_correct = False
        if ending_peek:
            reload_correct = self.ammo_left == 0
            self.ammo_left = AMMO_MAX_ROUNDS

        self.ticks += 1

        phase = self.phase_infer.infer(TickSignals(
            shots_fired_delta=total_fired,
            shots_hit_delta=total_hit,
            life_delta=-total_life_loss,
            timer_delta=u16_delta(self.prev["timer"], timer_at_tick_start),
            cleared_guess=cleared_guess,
            dead_guess=dead_guess or timed_out_guess,
            can_fire_probe=(total_fired > 0),
        ))

        last_hit = 1 if total_hit > 0 else 0
        last_miss = 1 if (total_fired > 0 and total_hit == 0) else 0
        if peek == self.prev_peek:
            self.peek_ticks = min(self.peek_ticks + 1, PEEK_TRAVERSE_TICKS)
        else:
            self.peek_ticks = 1
        self.prev_peek = peek
        peek_phase_next = (self.peek_ticks / PEEK_TRAVERSE_TICKS) * (1.0 if peek else -1.0)
        base_obs_next = self._build_obs(
            self.prev, last_hit, last_miss, peek_phase_next, self.ammo_left,
            self.prev_aim_x_bias, self.prev_aim_y_bias,
        )
        obs = np.concatenate(
            [base_obs_next, _shot_phase_features(self.ammo_left)], dtype=np.float32
        )

        done = (phase is Phase.TERMINAL) or (self.ticks >= MAX_TICKS)
        info = {
            "shots_fired_delta": total_fired,
            "shots_hit_delta": total_hit,
            "life_loss": total_life_loss,
            "cleared": bool(cleared_guess and not dead_guess and not timed_out_guess),
            "dead": dead_guess,
            "timed_out": timed_out_guess,
            "continue_screen": continue_screen_guess,
            "peek": bool(peek),
            "phase": phase.name,
            "dry_fire": dry_fire,
            "reload_correct": reload_correct,
            "ammo_left": self.ammo_left,
            "aim_x": float(aim_x),
            "aim_y": float(aim_y),
            "shot_index": int(AMMO_MAX_ROUNDS - ammo_before_tick),
        }
        return obs, done, info


# ---------------------------------------------------------------------------
# Accuracy-shaped fitness (sim-only experiment, not in env_timecrisis.py)
# ---------------------------------------------------------------------------

# Production fitness (episode_fitness in env_timecrisis.py) only rewards raw
# hit COUNT (HIT_REWARD * total_hits), not hit RATE -- a policy that fires
# constantly and clears by attrition (guaranteed eventually, since duck-on-
# empty is hard-enforced) can score well without ever improving accuracy,
# which dilutes any accuracy-specific ranking signal. This weight is a
# sim-only experiment to test whether directly rewarding accuracy helps ES
# learn better aim -- NOT yet added to config.py/env_timecrisis.py.
ACCURACY_BONUS_WEIGHT = 1000.0


class AccuracyShapedFitnessMixin:
    """Mix into any Simulated*Env to add an explicit accuracy-based term on
    top of the existing production fitness formula (CLEAR_BONUS/elapsed/
    damage/HIT_REWARD/etc., unchanged), to test whether a stronger,
    accuracy-specific reward signal helps ES learn better aim than raw hit
    count alone. Sim-only: does not touch env_timecrisis.py."""

    ACCURACY_BONUS_WEIGHT = ACCURACY_BONUS_WEIGHT

    def episode_fitness(self, theta: np.ndarray):
        fitness, metrics = super().episode_fitness(theta)
        fitness += self.ACCURACY_BONUS_WEIGHT * metrics["accuracy"]
        return float(fitness), metrics


class TimedSpotBaselineAccuracyEnv(AccuracyShapedFitnessMixin, TimedSpotBaselineEnv):
    """Baseline observation stack on the timed 5-spot task, with the
    accuracy-shaped fitness on top (sim-only A/B arm)."""


class TimedSpotShotIndexAccuracyEnv(AccuracyShapedFitnessMixin, TimedSpotShotIndexEnv):
    """Shot-index one-hot observation stack on the timed 5-spot task, with
    the accuracy-shaped fitness on top -- direct A/B counterpart to
    ``TimedSpotBaselineAccuracyEnv`` where only the observation stack
    differs. See ``TimedSpotShotIndexEnv`` for motivation."""


class TimedSpotShotPhaseAccuracyEnv(AccuracyShapedFitnessMixin, TimedSpotShotPhaseEnv):
    """Shot-phase (sin, cos) observation stack on the timed 5-spot task,
    with the accuracy-shaped fitness on top -- follow-up variant to the
    one-hot arm above, using only 2 extra obs dims + zero-init on the new
    input columns. See ``TimedSpotShotPhaseEnv`` for motivation."""


def run_timed_spot_probe(env_cls, theta_init_fn, param_count: int,
                         seed: int = 0, gens: int = 25, pop: int = 12,
                         sigma: float = 0.1, alpha: float = 0.02,
                         use_stagnation_kick: bool = False,
                         episodes_per_candidate: int = 1):
    """Short ES probe for the timed 5-spot task. Returns per-generation metrics.

    ``use_stagnation_kick``: when True, mirrors es_train.py's stagnation-kick
    escape mechanism exactly, using the SAME config.py constants
    (STD_STAGNATION_THRESHOLD/STAGNATION_PATIENCE/STAGNATION_SIGMA_MULT) as
    the live project (see ExtendedMiniESTrendSuite for the same pattern
    applied to the original sim env). Once fitness std has stayed below
    STD_STAGNATION_THRESHOLD for STAGNATION_PATIENCE consecutive generations,
    that generation is sampled with sigma * STAGNATION_SIGMA_MULT instead of
    sigma (and the gradient normalization uses whichever sigma was actually
    used), reverting to the normal sigma as soon as std recovers for one
    generation. Default False keeps this probe's original fixed-sigma
    behavior unchanged.

    ``episodes_per_candidate``: when > 1, each candidate's fitness and
    accuracy/cleared/shots_hit metrics are averaged over this many episodes
    (each with a distinct seed) before rank_transform() ranks the
    population. This directly reduces single-episode ranking noise -- a
    single stochastic hit/damage roll can otherwise flip candidate order in
    rank-transform ES even when true policy quality differs. Default 1
    reproduces the original single-episode behavior and seeding scheme
    exactly (same per-candidate seed as before).
    """
    rng = np.random.default_rng(seed)
    theta = theta_init_fn(seed)
    history = []
    stagnant_gens = 0

    for gen in range(gens):
        kicking = use_stagnation_kick and stagnant_gens >= STAGNATION_PATIENCE
        sigma_this_gen = sigma * STAGNATION_SIGMA_MULT if kicking else sigma

        half = pop // 2
        eps_half = rng.normal(0.0, 1.0, (half, param_count))
        eps = np.concatenate([eps_half, -eps_half])
        candidates = theta + sigma_this_gen * eps

        fitnesses, infos = [], []
        for i, cand in enumerate(candidates):
            ep_fits, ep_cleared, ep_acc, ep_hits = [], [], [], []
            for e in range(episodes_per_candidate):
                ep_seed = (gen * pop * episodes_per_candidate
                           + i * episodes_per_candidate + e)
                env = env_cls(seed=ep_seed)
                fit, info = env.episode_fitness(cand)
                ep_fits.append(fit)
                ep_cleared.append(1.0 if info["cleared"] else 0.0)
                ep_acc.append(info["accuracy"])
                ep_hits.append(info["shots_hit"])
            fitnesses.append(float(np.mean(ep_fits)))
            infos.append({
                "cleared": float(np.mean(ep_cleared)),
                "accuracy": float(np.mean(ep_acc)),
                "shots_hit": float(np.mean(ep_hits)),
            })

        fitnesses = np.asarray(fitnesses, dtype=np.float64)
        history.append({
            "gen": gen,
            "mean": float(fitnesses.mean()),
            "std": float(fitnesses.std()),
            "best": float(fitnesses.max()),
            "clear_rate": float(np.mean([x["cleared"] for x in infos])),
            "mean_acc": float(np.mean([x["accuracy"] for x in infos])),
            "best_acc": float(np.max([x["accuracy"] for x in infos])),
            "mean_hits": float(np.mean([x["shots_hit"] for x in infos])),
            "kicking": bool(kicking),
            "sigma_used": float(sigma_this_gen),
            "episodes_per_candidate": int(episodes_per_candidate),
        })

        shaped = rank_transform(fitnesses)
        gradient = (eps.T @ shaped) / (pop * sigma_this_gen)
        theta = theta + alpha * gradient

        if use_stagnation_kick:
            if float(fitnesses.std()) < STD_STAGNATION_THRESHOLD:
                stagnant_gens += 1
            else:
                stagnant_gens = 0

    return history


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
#  prev_aim_x_bias, prev_aim_y_bias, cursor_x_norm, cursor_y_norm]
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

    def test_observation_includes_cursor_ram_position(self):
        env = SimulatedTimeCrisisEnv(seed=0)
        obs = env.reset()
        self.assertEqual(obs.shape[0], OBS_DIM)
        # obs tail layout is now [..., cursor_x, cursor_y, shot_phase_sin,
        # shot_phase_cos], so cursor is at -4/-3 rather than -2/-1.
        self.assertAlmostEqual(
            float(obs[-4]),
            normalize_cursor(env.prev["cursor_x"], CURSOR_X_MIN, CURSOR_X_MAX),
        )
        self.assertAlmostEqual(
            float(obs[-3]),
            normalize_cursor(env.prev["cursor_y"], CURSOR_Y_MIN, CURSOR_Y_MAX),
        )


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


class ContinueScreenWatchdogSuite(unittest.TestCase):
    """Regression tests for frozen-state (continue/menu) watchdog."""

    def test_frozen_state_terminates_within_threshold(self):
        theta = _theta_warm_start()
        env = FrozenStateEnv(seed=0)
        env.reset()

        done_tick = None
        saw_continue = False
        for t in range(CONTINUE_SCREEN_STALE_TICKS + 3):
            _, done, info = env.step(theta)
            if info.get("continue_screen", False):
                saw_continue = True
            if done:
                done_tick = t + 1
                break

        self.assertIsNotNone(done_tick, "Frozen state should terminate quickly")
        self.assertTrue(saw_continue, "Watchdog should label frozen termination as continue_screen")
        self.assertLessEqual(
            done_tick,
            CONTINUE_SCREEN_STALE_TICKS,
            f"Frozen termination took {done_tick} ticks, expected <= "
            f"CONTINUE_SCREEN_STALE_TICKS ({CONTINUE_SCREEN_STALE_TICKS})",
        )

    def test_episode_reports_continue_screen_count(self):
        theta = _theta_warm_start()
        env = FrozenStateEnv(seed=1)
        _, info = env.episode_fitness(theta)

        self.assertIn("continue_screen_count", info)
        self.assertGreaterEqual(info["continue_screen_count"], 1)
        self.assertTrue(info["timed_out"])


class TimedSpotMemorySuite(unittest.TestCase):
    """Simulation-only v2 checks: timed 5-spot task and hit-memory fields."""

    def test_timed_spots_advance_on_schedule(self):
        game = TimedSpotGame(seed=0)
        first = game.active_spot_index()
        for _ in range(TimedSpotGame.SPOT_HOLD_FRAMES):
            game.step_frame()
        self.assertEqual(len(game.spots), TimedSpotGame.TIMED_SPOT_COUNT)
        self.assertNotEqual(first, game.active_spot_index())

    def test_memory_env_remembers_where_and_when_a_hit_worked(self):
        env = TimedSpotMemoryEnv(seed=0)
        env.reset()
        target_x, target_y = env.client._game.active_spot()
        theta = _theta_memory_always_aim(target_x, target_y)

        hit_info = None
        for _ in range(PEEK_TRAVERSE_TICKS + 2):
            _, _, info = env.step(theta)
            if info["shots_hit_delta"] > 0:
                hit_info = info
                break

        self.assertIsNotNone(hit_info, "Exact active-spot aim should score a hit quickly")
        self.assertAlmostEqual(hit_info["last_success_aim_x"], target_x, places=3)
        self.assertAlmostEqual(hit_info["last_success_aim_y"], target_y, places=3)
        self.assertEqual(hit_info["ticks_since_success"], 0)


class HotspotMemorySuite(unittest.TestCase):
    """Regression checks for HotspotBank and TimedSpotHotspotMemoryEnv (v3):
    remembering several recent hits, not just the single last one."""

    def test_bank_merges_nearby_hits_into_one_slot(self):
        bank = HotspotBank(capacity=5, merge_radius=0.12)
        bank.record_hit(0.30, 0.40)
        bank.record_hit(0.32, 0.41)  # within merge radius of the first hit
        self.assertEqual(len(bank.slots), 1)
        self.assertEqual(bank.slots[0]["hit_count"], 2)

    def test_bank_creates_new_slots_for_distinct_hits_up_to_capacity(self):
        bank = HotspotBank(capacity=3, merge_radius=0.05)
        bank.record_hit(0.1, 0.1)
        bank.record_hit(0.5, 0.5)
        bank.record_hit(0.9, 0.9)
        self.assertEqual(len(bank.slots), 3)

    def test_bank_evicts_weakest_slot_once_full(self):
        bank = HotspotBank(capacity=2, merge_radius=0.05, recency_horizon=10)
        bank.record_hit(0.1, 0.1)   # slot A: 1 hit, about to go stale
        bank.record_hit(0.5, 0.5)   # slot B: reinforced below, stays fresh/confident
        for _ in range(5):
            bank.tick()
        bank.record_hit(0.5, 0.5)
        bank.record_hit(0.5, 0.5)

        bank.record_hit(0.9, 0.9)   # a new, distinct hit -- bank is full
        positions = {(round(s["x"], 2), round(s["y"], 2)) for s in bank.slots}
        self.assertIn((0.5, 0.5), positions)
        self.assertIn((0.9, 0.9), positions)
        self.assertNotIn((0.1, 0.1), positions)

    def test_recency_resets_on_hit_and_grows_each_tick(self):
        bank = HotspotBank(capacity=5, merge_radius=0.1, recency_horizon=50)
        bank.record_hit(0.3, 0.3)
        self.assertEqual(bank.slots[0]["ticks_since_hit"], 0)
        bank.tick()
        bank.tick()
        bank.tick()
        self.assertEqual(bank.slots[0]["ticks_since_hit"], 3)

    def test_env_resets_hotspot_bank_between_episodes(self):
        env = TimedSpotHotspotMemoryEnv(seed=0)
        env.reset()
        target_x, target_y = env.client._game.active_spot()
        theta = _theta_hotspot_always_aim(target_x, target_y)

        for _ in range(PEEK_TRAVERSE_TICKS + 2):
            _, _, info = env.step(theta)
            if info["shots_hit_delta"] > 0:
                break

        self.assertGreater(len(env.hotspots.slots), 0)
        env.reset()
        self.assertEqual(len(env.hotspots.slots), 0)

    def test_env_remembers_a_confirmed_hit_as_best_candidate(self):
        env = TimedSpotHotspotMemoryEnv(seed=0)
        env.reset()
        target_x, target_y = env.client._game.active_spot()
        theta = _theta_hotspot_always_aim(target_x, target_y)

        hit_info = None
        for _ in range(PEEK_TRAVERSE_TICKS + 2):
            _, _, info = env.step(theta)
            if info["shots_hit_delta"] > 0:
                hit_info = info
                break

        self.assertIsNotNone(hit_info, "Exact active-spot aim should score a hit quickly")
        self.assertAlmostEqual(hit_info["hotspot_best_x"], target_x, places=3)
        self.assertAlmostEqual(hit_info["hotspot_best_y"], target_y, places=3)
        self.assertEqual(hit_info["hotspot_recency"], 0.0)
        self.assertGreater(hit_info["hotspot_count"], 0)


class AccuracyShapedFitnessSuite(unittest.TestCase):
    """Regression check for the sim-only accuracy-shaped fitness experiment."""

    def test_shaped_fitness_adds_accuracy_bonus_on_top_of_base_fitness(self):
        rng = np.random.default_rng(3)
        theta = rng.normal(0.0, 0.1, PARAM_COUNT)
        theta[_B2_OFFSET + 0] += 2.0
        theta[_B2_OFFSET + 1] += 1.0

        base_fitness, base_metrics = TimedSpotBaselineEnv(seed=5).episode_fitness(theta)
        shaped_fitness, shaped_metrics = TimedSpotBaselineAccuracyEnv(seed=5).episode_fitness(theta)

        self.assertEqual(base_metrics["shots_fired"], shaped_metrics["shots_fired"])
        self.assertEqual(base_metrics["shots_hit"], shaped_metrics["shots_hit"])
        self.assertAlmostEqual(
            shaped_fitness,
            base_fitness + ACCURACY_BONUS_WEIGHT * base_metrics["accuracy"],
            places=3,
        )


class ShotIndexObservationSuite(unittest.TestCase):
    """Regression checks for the shot-index one-hot obs env (idea #1)."""

    def test_reset_obs_has_extra_one_hot_dims(self):
        env = TimedSpotShotIndexEnv(seed=0)
        obs = env.reset()
        self.assertEqual(obs.shape[0], _SIM_SHOTIDX_OBS_DIM)
        onehot = obs[OBS_DIM:]
        # Ammo starts full, so shot_index == 0 -> onehot[0] == 1.
        self.assertEqual(onehot.shape[0], AMMO_MAX_ROUNDS)
        self.assertEqual(float(onehot[0]), 1.0)
        self.assertEqual(float(onehot[1:].sum()), 0.0)

    def test_one_hot_advances_with_each_shot_in_clip(self):
        env = TimedSpotShotIndexEnv(seed=0)
        env.reset()
        # Aim at the first spot so shots register.
        target_x, target_y = env.client._game.spots[0]
        theta = np.zeros(_SIM_SHOTIDX_PARAM_COUNT)
        theta[_SIM_SHOTIDX_B2_OFFSET + 0] = 50.0  # shoot logit
        theta[_SIM_SHOTIDX_B2_OFFSET + 1] = 50.0  # peek logit
        theta[_SIM_SHOTIDX_B2_OFFSET + 2] = _aim_bias_for(target_x)
        theta[_SIM_SHOTIDX_B2_OFFSET + 3] = _aim_bias_for(target_y)

        # Walk enough ticks to burn one full clip (peek traverse + up to 6 shots).
        seen_indices = []
        for _ in range(PEEK_TRAVERSE_TICKS + AMMO_MAX_ROUNDS + 4):
            obs, _, info = env.step(theta)
            onehot = obs[OBS_DIM:]
            active_idx = int(np.argmax(onehot))
            seen_indices.append((info["ammo_left"], active_idx, float(onehot.sum())))
        # Every observation is a proper one-hot (exactly one dim == 1.0).
        for _, _, s in seen_indices:
            self.assertEqual(s, 1.0)
        # We should have seen at least 3 distinct one-hot positions across
        # a clip (i.e. the index actually advances, not stuck at 0).
        distinct = {idx for _, idx, _ in seen_indices}
        self.assertGreaterEqual(len(distinct), 3)

    def test_param_count_matches_wider_obs_dim(self):
        expected = (
            _SIM_SHOTIDX_OBS_DIM * HIDDEN + HIDDEN
            + HIDDEN * ACT_DIM + ACT_DIM
        )
        self.assertEqual(_SIM_SHOTIDX_PARAM_COUNT, expected)
        # Warm-start theta is the right shape and the shoot/peek biases
        # were bumped like the baseline warm-start.
        theta = _theta_shotidx_warm_start(seed=42)
        self.assertEqual(theta.shape[0], _SIM_SHOTIDX_PARAM_COUNT)
        self.assertGreater(theta[_SIM_SHOTIDX_B2_OFFSET + 0], 1.5)
        self.assertGreater(theta[_SIM_SHOTIDX_B2_OFFSET + 1], 0.5)


class ShotPhaseObservationSuite(unittest.TestCase):
    """Regression checks for the shot-phase (sin/cos) obs env variant."""

    def test_reset_obs_has_two_extra_phase_dims(self):
        env = TimedSpotShotPhaseEnv(seed=0)
        obs = env.reset()
        self.assertEqual(obs.shape[0], _SIM_SHOTPHASE_OBS_DIM)
        # At reset, ammo is full so shot_idx=0 -> sin=0, cos=1.
        self.assertAlmostEqual(float(obs[OBS_DIM + 0]), 0.0, places=6)
        self.assertAlmostEqual(float(obs[OBS_DIM + 1]), 1.0, places=6)

    def test_phase_features_advance_on_the_unit_circle(self):
        # As shot_idx goes 0..5, sin/cos should land at 6 distinct points on
        # the unit circle. This is what makes the signal non-smooth even
        # though it's only 2 dims: adjacent shots aren't monotonically close.
        pts = []
        for ammo in range(AMMO_MAX_ROUNDS, 0, -1):
            feat = _shot_phase_features(ammo)
            pts.append((round(float(feat[0]), 6), round(float(feat[1]), 6)))
        # 6 distinct points -> all shots have unique phase features.
        self.assertEqual(len(set(pts)), AMMO_MAX_ROUNDS)
        # sin^2 + cos^2 == 1 for every point.
        for s, c in pts:
            self.assertAlmostEqual(s * s + c * c, 1.0, places=5)

    def test_warm_start_matches_baseline_behavior_exactly_at_gen0(self):
        # The whole point of zero-initing the extra input-column rows: at
        # gen 0 the phase env's warm-start theta should produce IDENTICAL
        # (shoot, peek, aim_x, aim_y) as the baseline warm-start theta on
        # the same obs (with the extra sin/cos dims appended).
        rng = np.random.default_rng(0)
        obs_base = rng.normal(0.0, 1.0, OBS_DIM).astype(np.float32)
        obs_phase = np.concatenate([obs_base, _shot_phase_features(AMMO_MAX_ROUNDS)])

        theta_base = _theta_warm_start(seed=42)
        theta_phase = _theta_shotphase_warm_start(seed=42)

        # Use the same _act_with_obs_dim helper both arms use in step().
        out_base = _act_with_obs_dim(theta_base, obs_base, OBS_DIM)
        out_phase = _act_with_obs_dim(theta_phase, obs_phase, _SIM_SHOTPHASE_OBS_DIM)

        self.assertEqual(out_base[0], out_phase[0])  # shoot
        self.assertEqual(out_base[1], out_phase[1])  # peek
        self.assertAlmostEqual(out_base[2], out_phase[2], places=6)  # aim_x
        self.assertAlmostEqual(out_base[3], out_phase[3], places=6)  # aim_y

    def test_param_count_matches_wider_obs_dim(self):
        expected = (
            _SIM_SHOTPHASE_OBS_DIM * HIDDEN + HIDDEN
            + HIDDEN * ACT_DIM + ACT_DIM
        )
        self.assertEqual(_SIM_SHOTPHASE_PARAM_COUNT, expected)
        theta = _theta_shotphase_warm_start(seed=42)
        self.assertEqual(theta.shape[0], _SIM_SHOTPHASE_PARAM_COUNT)


class StagnationKickProbeSuite(unittest.TestCase):
    """Regression checks that run_timed_spot_probe's optional stagnation-kick
    mirrors es_train.py's mechanism (config.py's STD_STAGNATION_THRESHOLD/
    STAGNATION_PATIENCE/STAGNATION_SIGMA_MULT, same as ExtendedMiniESTrendSuite
    for the original sim env) when opted into, and is a true no-op otherwise.
    """

    def test_kick_disabled_by_default_uses_fixed_sigma(self):
        hist = run_timed_spot_probe(
            TimedSpotBaselineEnv, _theta_warm_start, PARAM_COUNT,
            seed=0, gens=3, pop=6, sigma=0.1, alpha=0.02,
        )
        self.assertTrue(all(g["sigma_used"] == 0.1 for g in hist))
        self.assertTrue(all(g["kicking"] is False for g in hist))

    def test_kick_enabled_escalates_sigma_after_patience_stagnant_gens(self):
        # A theta whose peek logit is saturated hard-negative never exposes
        # (out = b2 exactly, since every other weight is 0), so every
        # mirrored candidate scores identically (-FAIL_PENALTY) regardless of
        # SIGMA=0.1 perturbation -- deterministically reproducing the "flat
        # std" stagnation trigger without depending on RNG luck.
        theta = np.zeros(PARAM_COUNT)
        theta[_B2_OFFSET + 1] = -50.0  # peek logit: always False

        def _frozen_init(seed):
            return theta.copy()

        gens = STAGNATION_PATIENCE + 2
        hist = run_timed_spot_probe(
            TimedSpotBaselineEnv, _frozen_init, PARAM_COUNT,
            seed=0, gens=gens, pop=6, sigma=0.1, alpha=0.02,
            use_stagnation_kick=True,
        )
        self.assertTrue(hist[-1]["kicking"])
        self.assertAlmostEqual(hist[-1]["sigma_used"], 0.1 * STAGNATION_SIGMA_MULT)


class EpisodeAveragingProbeSuite(unittest.TestCase):
    """Regression checks that run_timed_spot_probe's optional
    episodes_per_candidate averaging is a true no-op at its default (1),
    and genuinely averages over distinct-seeded episodes when > 1."""

    def test_default_episode_count_matches_pre_change_single_episode_behavior(self):
        # With episodes_per_candidate=1, per-candidate seeds must reduce to
        # exactly gen * pop + i (the original scheme), so history is
        # unaffected for every existing caller that doesn't pass the new arg.
        hist_default = run_timed_spot_probe(
            TimedSpotBaselineEnv, _theta_warm_start, PARAM_COUNT,
            seed=0, gens=3, pop=6, sigma=0.1, alpha=0.02,
        )
        hist_explicit = run_timed_spot_probe(
            TimedSpotBaselineEnv, _theta_warm_start, PARAM_COUNT,
            seed=0, gens=3, pop=6, sigma=0.1, alpha=0.02,
            episodes_per_candidate=1,
        )
        for g_default, g_explicit in zip(hist_default, hist_explicit):
            self.assertEqual(g_default["mean"], g_explicit["mean"])
            self.assertEqual(g_default["clear_rate"], g_explicit["clear_rate"])
            self.assertEqual(g_default["mean_acc"], g_explicit["mean_acc"])
        self.assertTrue(all(g["episodes_per_candidate"] == 1 for g in hist_default))

    def test_multi_episode_fitness_equals_mean_of_distinct_seeded_episodes(self):
        # Reconstruct generation 0's per-candidate averaged fitness by hand
        # from N direct episode_fitness() calls using the SAME seed formula
        # the probe uses internally, and confirm they match exactly -- this
        # also proves each episode uses a distinct seed (not one episode
        # replayed N times), since TimedSpotBaselineEnv's spot schedule/
        # enemy behavior depends on its seed.
        pop = 6
        episodes = 3
        theta = _theta_warm_start(0)
        rng = np.random.default_rng(0)
        half = pop // 2
        eps_half = rng.normal(0.0, 1.0, (half, PARAM_COUNT))
        eps = np.concatenate([eps_half, -eps_half])
        candidates = theta + 0.1 * eps

        expected_fits = []
        for i, cand in enumerate(candidates):
            ep_fits = []
            for e in range(episodes):
                seed = 0 * pop * episodes + i * episodes + e
                env = TimedSpotBaselineEnv(seed=seed)
                fit, _info = env.episode_fitness(cand)
                ep_fits.append(fit)
            expected_fits.append(float(np.mean(ep_fits)))

        hist = run_timed_spot_probe(
            TimedSpotBaselineEnv, _theta_warm_start, PARAM_COUNT,
            seed=0, gens=1, pop=pop, sigma=0.1, alpha=0.02,
            episodes_per_candidate=episodes,
        )
        self.assertAlmostEqual(hist[0]["mean"], float(np.mean(expected_fits)), places=6)
        self.assertEqual(hist[0]["episodes_per_candidate"], episodes)


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


class MissCorrectionMetricsSuite(unittest.TestCase):
    """Verify the miss-correction reward diagnostics are computed as intended."""

    def test_metrics_detect_corrective_and_repeated_miss_patterns(self):
        shots = [
            {"aim_x": 0.50, "aim_y": 0.50, "hit": False},
            {"aim_x": 0.535, "aim_y": 0.50, "hit": False},
            {"aim_x": 0.540, "aim_y": 0.50, "hit": False},
            {"aim_x": 0.60, "aim_y": 0.50, "hit": True},
            {"aim_x": 0.62, "aim_y": 0.50, "hit": True},
        ]

        metrics = compute_miss_correction_metrics(shots)

        self.assertGreater(metrics["corrected"], 0.0)
        self.assertGreater(metrics["repeated"], 0.0)
        self.assertGreater(metrics["center_camp"], 0.0)

    def test_clip_shift_rewards_aim_variation_between_magazines(self):
        """Two 6-shot magazines aimed at very different spots should score a
        much higher clip_shift than two magazines aimed at the same spot --
        this is the metric that directly targets the "same arc every reload"
        symptom, now wired into production fitness via CLIP_SHIFT_BONUS."""
        same_spot_shots = [
            {"aim_x": 0.3, "aim_y": 0.3, "hit": False} for _ in range(12)
        ]
        shifted_shots = (
            [{"aim_x": 0.2, "aim_y": 0.2, "hit": False} for _ in range(6)]
            + [{"aim_x": 0.8, "aim_y": 0.8, "hit": False} for _ in range(6)]
        )

        same_metrics = compute_miss_correction_metrics(same_spot_shots)
        shifted_metrics = compute_miss_correction_metrics(shifted_shots)

        self.assertGreater(shifted_metrics["clip_shift"], same_metrics["clip_shift"])

    def test_clip_shift_penalizes_shift_once_then_repeat_pattern(self):
        """Regression for the live-reported symptom (2026-08-06): arc1 shifts
        to a different arc2, but arc3+ then keep repeating arc2 unchanged.
        A mean-of-raw-distances metric let this pattern still score a decent
        clip_shift (one big jump dilutes across several zero-shift pairs but
        doesn't zero out the average) -- the fix takes the MIN across all
        consecutive clip pairs, so any single repeated pair should drag the
        whole score down close to 0, regardless of how many clips came before."""
        shift_once_then_repeat = (
            [{"aim_x": 0.2, "aim_y": 0.2, "hit": False} for _ in range(6)]      # arc1
            + [{"aim_x": 0.8, "aim_y": 0.8, "hit": False} for _ in range(6)]    # arc2 (differs)
            + [{"aim_x": 0.8, "aim_y": 0.8, "hit": False} for _ in range(6)]    # arc3 (repeats arc2)
            + [{"aim_x": 0.8, "aim_y": 0.8, "hit": False} for _ in range(6)]    # arc4 (repeats arc2)
        )

        metrics = compute_miss_correction_metrics(shift_once_then_repeat)

        self.assertLess(metrics["clip_shift"], 0.1)

    def test_shot_slot_diversity_rewards_slot_variation_across_clips(self):
        """If the same shot slot lands at different coordinates across clips,
        shot_slot_diversity should be higher than for repeated identical arcs."""
        repeated_arc = (
            [{"aim_x": 0.20, "aim_y": 0.20, "hit": False} for _ in range(6)]
            + [{"aim_x": 0.20, "aim_y": 0.20, "hit": False} for _ in range(6)]
        )
        varied_slots = [
            # clip 1
            {"aim_x": 0.20, "aim_y": 0.20, "hit": False},
            {"aim_x": 0.30, "aim_y": 0.20, "hit": False},
            {"aim_x": 0.40, "aim_y": 0.20, "hit": False},
            {"aim_x": 0.50, "aim_y": 0.20, "hit": False},
            {"aim_x": 0.60, "aim_y": 0.20, "hit": False},
            {"aim_x": 0.70, "aim_y": 0.20, "hit": False},
            # clip 2 (slot-wise shifted)
            {"aim_x": 0.25, "aim_y": 0.25, "hit": False},
            {"aim_x": 0.35, "aim_y": 0.25, "hit": False},
            {"aim_x": 0.45, "aim_y": 0.25, "hit": False},
            {"aim_x": 0.55, "aim_y": 0.25, "hit": False},
            {"aim_x": 0.65, "aim_y": 0.25, "hit": False},
            {"aim_x": 0.75, "aim_y": 0.25, "hit": False},
        ]

        repeated_metrics = compute_miss_correction_metrics(repeated_arc)
        varied_metrics = compute_miss_correction_metrics(varied_slots)

        self.assertGreater(
            varied_metrics["shot_slot_diversity"],
            repeated_metrics["shot_slot_diversity"],
        )

    def test_shot_slot_diversity_is_zero_with_fewer_than_two_clips(self):
        one_clip = [{"aim_x": 0.4, "aim_y": 0.5, "hit": False} for _ in range(6)]
        metrics = compute_miss_correction_metrics(one_clip)
        self.assertEqual(metrics["shot_slot_diversity"], 0.0)


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
