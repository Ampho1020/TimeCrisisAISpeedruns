"""Scalar-only environment wrapper (v1: no vision yet)."""

import numpy as np

from bridge_client import BridgeClient
from config import (
    AMMO_MAX_ROUNDS, CLEAR_BONUS, PEEK_TRAVERSE_TICKS, DAMAGE_PENALTY,
    CONTINUE_SCREEN_STALE_TICKS,
    CURSOR_X_MAX, CURSOR_X_MIN, CURSOR_Y_MAX, CURSOR_Y_MIN,
    DRY_FIRE_PENALTY, FAIL_PENALTY, FRAME_SKIP, HOST, HIT_REWARD, MAX_TICKS,
    PORT, RAM, RELOAD_BONUS, STATE_SLOT, TIMEOUT_TIMER_THRESHOLD,
)
from phase_inference import Phase, PhaseInferer, TickSignals
from policy import act


def u16_delta(new_v: int, old_v: int) -> int:
    """Signed delta between two u16 reads, wrap-around safe."""
    d = new_v - old_v
    if d < -32768:
        d += 65536
    elif d > 32768:
        d -= 65536
    return d


def normalize_cursor(raw_value: int, lo: int, hi: int) -> float:
    """Map inclusive cursor RAM coordinates to normalized screen space [0, 1]."""
    if hi <= lo:
        return 0.0
    clipped = min(max(int(raw_value), lo), hi)
    return float((clipped - lo) / (hi - lo))


def core_watchdog_snapshot(cur: dict[str, int]) -> tuple[int, int, int, int]:
    """Return the menu-watchdog counters only, excluding aim/cursor RAM."""
    return cur["shots_fired"], cur["shots_hit"], cur["timer"], cur["life"]


def peek_hold_reward(
    peek_flags,
    traverse_ticks: int = PEEK_TRAVERSE_TICKS,
    reward: float = 1.0,
) -> float:
    """Reward for holding the PEEK button (A) long enough for the exit
    animation to complete.

    ``peek=True`` means the A button IS PRESSED: the character is stepping
    out of cover (exposed, can shoot, can be hit).
    ``peek=False`` means A is NOT pressed: the character stays in cover.

    Only True (A-pressed = peeking out) runs are counted:
      * a single-tick tap earns nothing,
      * each additional tick exposed, up to traverse_ticks, adds ``reward``,
      * runs longer than traverse_ticks are capped.
    """
    def run_value(run: int) -> float:
        return reward * min(max(run - 1, 0), traverse_ticks - 1)

    total = 0.0
    run = 0
    for held in peek_flags:
        if held:              # A pressed = character peeking out (exposed)
            run += 1
        else:                 # A released = back to cover: finalise the peek run
            total += run_value(run)
            run = 0
    total += run_value(run)   # finalise last run
    return total


class TimeCrisisEnv:
    def __init__(self, host=HOST, port=PORT, state_slot=STATE_SLOT):
        self.client = BridgeClient(host, port)
        self.state_slot = state_slot
        self.phase_infer = PhaseInferer(vote_window=3)
        self.prev: dict[str, int] = {}
        self.start_timer = 0
        self.ticks = 0
        self.prev_peek: bool = False
        self.peek_ticks: int = 0
        self.peek_lock: int = 0   # minimum hold: any transition holds for PEEK_TRAVERSE_TICKS
        self.peek_locked_value: bool = False   # what state the lock is holding
        self.stale_core_ticks: int = 0  # consecutive ticks with identical core RAM snapshot
        self.ammo_left: int = AMMO_MAX_ROUNDS
        self.prev_aim_x_bias: float = 0.0   # last tick's aim_x_bias, fed back as obs
        self.prev_aim_y_bias: float = 0.0   # last tick's aim_y_bias, fed back as obs

    # -- lifecycle ------------------------------------------------------

    def connect(self):
        self.client.connect()

    def start_listening(self):
        self.client.start_listening()

    def finish_connect(self):
        self.client.finish_connect()

    def close(self):
        self.client.close()

    # -- RAM ------------------------------------------------------------

    def _read_core(self):
        return {
            "shots_fired": self.client.read_u16(RAM.shots_fired),
            "shots_hit":   self.client.read_u16(RAM.shots_hit),
            "timer":       self.client.read_u16(RAM.timer),
            "life":        self.client.read_u16(RAM.life),
            "cursor_x":    self.client.read_u16(RAM.cursor_x),
            "cursor_y":    self.client.read_u16(RAM.cursor_y),
        }

    @staticmethod
    def _build_obs(cur, last_hit: int, last_miss: int, peek_phase: float = 0.0,
                   ammo_left: int = AMMO_MAX_ROUNDS,
                   prev_aim_x_bias: float = 0.0, prev_aim_y_bias: float = 0.0) -> np.ndarray:
        fired = max(cur["shots_fired"], 1)
        return np.array([
            cur["timer"] / 10000.0,
            cur["life"] / 100.0,
            cur["shots_fired"] / 1000.0,
            cur["shots_hit"] / 1000.0,
            cur["shots_hit"] / fired,
            float(last_hit),
            float(last_miss),
            peek_phase,
            ammo_left / AMMO_MAX_ROUNDS,
            prev_aim_x_bias,
            prev_aim_y_bias,
            normalize_cursor(cur.get("cursor_x", CURSOR_X_MIN), CURSOR_X_MIN, CURSOR_X_MAX),
            normalize_cursor(cur.get("cursor_y", CURSOR_Y_MIN), CURSOR_Y_MIN, CURSOR_Y_MAX),
        ], dtype=np.float32)

    # -- episode --------------------------------------------------------

    def reset(self) -> np.ndarray:
        self.client.load_state(self.state_slot)
        self.client.step_frames(2)
        self.prev = self._read_core()
        self.start_timer = self.prev["timer"]
        self.ticks = 0
        # Character always starts in cover at the top of each screen; the peek
        # button (A) is released and the traverse animation hasn't begun yet.
        self.prev_peek = False
        self.peek_ticks = 0
        self.peek_lock = 0
        self.peek_locked_value = False
        self.stale_core_ticks = 0
        self.ammo_left = AMMO_MAX_ROUNDS
        self.prev_aim_x_bias = 0.0
        self.prev_aim_y_bias = 0.0
        self.phase_infer.reset()
        return self._build_obs(self.prev, 0, 0, 0.0, self.ammo_left, self.prev_aim_x_bias, self.prev_aim_y_bias)

    def step(self, theta: np.ndarray):
        peek_phase = (self.peek_ticks / PEEK_TRAVERSE_TICKS) * (1.0 if self.prev_peek else -1.0)
        shoot, peek, aim_x_bias, aim_y_bias = act(
            theta, self._build_obs(
                self.prev, 0, 0, peek_phase, self.ammo_left,
                self.prev_aim_x_bias, self.prev_aim_y_bias,
            )
        )
        # Feed this tick's aim decision back as next tick's "previous aim" obs.
        # The policy is a plain feedforward net with no recurrence of its own;
        # without this it can't tell what it last aimed at and has no signal
        # to shift away from a spot that isn't working. Storing the raw
        # [-1, 1] bias (not the clamped screen position) keeps it in the same
        # scale the network already outputs/consumes.
        self.prev_aim_x_bias = float(aim_x_bias)
        self.prev_aim_y_bias = float(aim_y_bias)
        # peek=True  -> A button PRESSED  -> character EXITS cover (exposed, can shoot)
        # peek=False -> A button RELEASED -> character STAYS in cover (protected)

        # Hard enforcement, not just reward shaping: with an empty clip there is
        # NEVER a reason to stay exposed -- the trigger can't fire and the only
        # possible outcome is taking free damage. DRY_FIRE_PENALTY/RELOAD_BONUS
        # were relied on to teach this via ES alone, but real training kept
        # showing agents mag-dump and stay exposed anyway (a difficult, easy-to-
        # miss local optimum for evolution to escape on its own -- see repo
        # memory). So this overrides the policy's own peek decision the instant
        # ammo runs out, forcing the duck-to-reload transition to start
        # immediately. The policy is still free to choose exactly when to peek
        # out and when to duck early (e.g. before emptying the clip); this only
        # removes the strictly-dominated "stay out with 0 ammo" option.
        if self.ammo_left == 0:
            peek = False

        # Minimum hold lock: BOTH transitions (into cover and out of cover) have
        # to be held for PEEK_TRAVERSE_TICKS ticks so the traverse animation can
        # complete. Previously only the False→True transition (leaving cover) was
        # locked, which let the policy re-enter cover for just a single tick
        # before being forced back out -- the "1 tick cover in-out" flicker
        # observed during training. Locking symmetrically kills that oscillation.
        if self.peek_lock > 0:
            peek = self.peek_locked_value
            self.peek_lock -= 1
        elif peek != self.prev_peek:
            # Any transition: lock the new state
            self.peek_lock = PEEK_TRAVERSE_TICKS - 1  # -1 because this tick counts
            self.peek_locked_value = peek

        # Gate the trigger: shots only register when the character is FULLY out of
        # cover (A held for at least PEEK_TRAVERSE_TICKS consecutive ticks).  Firing
        # during the transition animation silently fails in-game, so we block it here
        # to avoid wasting the edge-trigger on a guaranteed miss.
        shoot_allowed = peek and self.prev_peek and self.peek_ticks >= PEEK_TRAVERSE_TICKS
        # Full-range mapping: tanh bias [-1, 1] spans the full screen [0, 1].
        # Using 0.5× previously kept the cursor in [0.17, 0.83] with typical
        # small initial weights; 1.0× lets early exploration reach the edges.
        aim_x = min(1.0, max(0.0, 0.5 + float(aim_x_bias)))
        aim_y = min(1.0, max(0.0, 0.5 + float(aim_y_bias)))

        total_fired = total_hit = total_life_loss = 0
        cleared_guess = dead_guess = timed_out_guess = False
        continue_screen_guess = False
        tick_start_core = core_watchdog_snapshot(self.prev)
        timer_at_tick_start = self.prev["timer"]

        for f in range(FRAME_SKIP):
            # Edge-trigger the shot: press briefly, release. Holding the
            # button for all 5 frames makes fire rate uncontrollable.
            # shoot_allowed ensures the trigger only fires when fully exposed.
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

            # Death detection: normally life reaches exactly 0, but under
            # frame-skip / u16 sampling we can also observe a lethal underflow
            # wrap (e.g. 1 -> 65535) instead of an exact 0 sample. Treat that
            # as terminal too so we don't drift into the continue menu.
            lethal_wrap = (
                pre["life"] > 0
                and life_d < 0
                and post["life"] > pre["life"]
                and post["life"] >= 65000
            )
            if post["life"] == 0 or lethal_wrap:
                dead_guess = True
            # Heuristic clear detection: timer jumps discontinuously upward.
            # Replace with a real flag if you ever find one.
            if u16_delta(post["timer"], self.start_timer) > 100:
                cleared_guess = True
            # Timeout: the countdown reached zero -> "continue?" screen. Detect
            # the zero-cross here (a large downward step across the tick also
            # counts, in case the timer skips the exact zero sample).
            timer_step = u16_delta(post["timer"], pre["timer"])
            if post["timer"] <= TIMEOUT_TIMER_THRESHOLD or (
                timer_step < 0 and pre["timer"] + timer_step <= TIMEOUT_TIMER_THRESHOLD
            ):
                timed_out_guess = True

            self.prev = post
            if dead_guess or cleared_guess or timed_out_guess:
                break

        # Fallback continue/menu watchdog: if all core counters were frozen
        # across the entire decision tick, count it. Several consecutive frozen
        # ticks indicate we've likely landed on a non-playable menu (e.g.
        # continue prompt) that escaped direct life/timer terminal detection.
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

        # Wasted exposure: penalise ticks where the agent is fully exposed with
        # an EMPTY clip (ammo_left was already 0 at the start of this tick)
        # instead of ducking back into cover to reload. This is exact now that
        # ammo_left is tracked, rather than the old total_fired == 0 proxy
        # which also (wrongly) fired whenever the policy simply chose not to
        # shoot with ammo still available.
        ammo_before_tick = self.ammo_left
        dry_fire = bool(shoot_allowed and ammo_before_tick == 0)

        # Ammo bookkeeping: consume rounds fired this tick (only ever nonzero
        # while shoot_allowed, i.e. fully exposed), then -- on the exact tick
        # the character ducks back into cover -- award a flat, count-
        # independent RELOAD_BONUS if the clip was empty, and refill to a
        # full clip. Using a flat bonus (not scaled by shots fired) avoids
        # incentivising magdumping just to inflate the reload reward.
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

        last_hit  = 1 if total_hit > 0 else 0
        last_miss = 1 if (total_fired > 0 and total_hit == 0) else 0
        if peek == self.prev_peek:
            self.peek_ticks = min(self.peek_ticks + 1, PEEK_TRAVERSE_TICKS)
        else:
            self.peek_ticks = 1
        self.prev_peek = peek
        peek_phase_next = (self.peek_ticks / PEEK_TRAVERSE_TICKS) * (1.0 if peek else -1.0)
        obs = self._build_obs(
            self.prev, last_hit, last_miss, peek_phase_next, self.ammo_left,
            self.prev_aim_x_bias, self.prev_aim_y_bias,
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
        }
        return obs, done, info

    def episode_fitness(self, theta: np.ndarray):
        """Run one full episode. Returns (fitness, metrics)."""
        self.reset()
        total_hits = total_fired = total_life_loss = 0
        dry_fire_ticks = 0
        reload_correct_count = 0
        cleared = False
        timed_out = dead = False
        continue_screen_count = 0
        # Record (peek, shots_fired) per tick for post-episode diagnostics.
        peek_flags = []
        shots_per_tick = []
        hits_per_tick = []
        aim_x_per_tick = []
        aim_y_per_tick = []

        while True:
            _, done, info = self.step(theta)
            total_hits      += info["shots_hit_delta"]
            total_fired     += info["shots_fired_delta"]
            total_life_loss += info["life_loss"]
            cleared = cleared or info["cleared"]
            timed_out = timed_out or info["timed_out"]
            dead = dead or info["dead"]
            continue_screen_count += int(info.get("continue_screen", False))
            peek_flags.append(info["peek"])
            shots_per_tick.append(info["shots_fired_delta"])
            hits_per_tick.append(info["shots_hit_delta"])
            aim_x_per_tick.append(info["aim_x"])
            aim_y_per_tick.append(info["aim_y"])
            dry_fire_ticks += int(info["dry_fire"])
            reload_correct_count += int(info["reload_correct"])
            if done:
                break

        elapsed = u16_delta(self.start_timer, self.prev["timer"])

        if cleared:
            fitness = CLEAR_BONUS - elapsed - DAMAGE_PENALTY * total_life_loss
        else:
            fitness = -FAIL_PENALTY
        # Diagnostics only (NOT added to fitness): peek_hold_score, peek_flips
        # and ticks_in_cover used to feed reward shaping (COVER_HOLD_REWARD,
        # COVER_FLIP_PENALTY, COVER_TIME_PENALTY); that noisy shaping was
        # removed so raw ES only optimizes the actual outcome. Kept here purely
        # for CSV logging / plotting.
        hold_score = 0.0
        _run_ticks = 0
        _run_shot = False
        for _pk, _sf in zip(peek_flags, shots_per_tick):
            if _pk:                        # peeking out (exposed)
                _run_ticks += 1
                if _sf > 0:
                    _run_shot = True
            else:                          # returned to cover
                if _run_shot:
                    hold_score += min(
                        max(_run_ticks - 1, 0), PEEK_TRAVERSE_TICKS - 1
                    )
                _run_ticks = 0
                _run_shot = False
        if _run_shot:                      # episode ended while still peeking
            hold_score += min(
                max(_run_ticks - 1, 0), PEEK_TRAVERSE_TICKS - 1
            )
        peek_flips = sum(
            1 for i in range(1, len(peek_flags))
            if peek_flags[i] != peek_flags[i - 1]
        )
        ticks_in_cover = sum(1 for f in peek_flags if not f)  # A NOT pressed = protected

        ax = np.asarray(aim_x_per_tick, dtype=np.float64)
        ay = np.asarray(aim_y_per_tick, dtype=np.float64)
        if len(ax) > 0:
            aim_x_std = float(ax.std())
            aim_y_std = float(ay.std())
            aim_span_x = float(ax.max() - ax.min())
            aim_span_y = float(ay.max() - ay.min())
        else:
            aim_x_std = aim_y_std = aim_span_x = aim_span_y = 0.0
        if len(ax) > 1:
            mean_abs_aim_dx = float(np.abs(np.diff(ax)).mean())
            mean_abs_aim_dy = float(np.abs(np.diff(ay)).mean())
        else:
            mean_abs_aim_dx = mean_abs_aim_dy = 0.0

        # Shot/hit location diagnostics by aim_x lane: left/mid/right.
        shots_left = shots_mid = shots_right = 0
        hits_left = hits_mid = hits_right = 0
        for x, s, h in zip(aim_x_per_tick, shots_per_tick, hits_per_tick):
            if x < (1.0 / 3.0):
                shots_left += int(s)
                hits_left += int(h)
            elif x < (2.0 / 3.0):
                shots_mid += int(s)
                hits_mid += int(h)
            else:
                shots_right += int(s)
                hits_right += int(h)
        total_shots = max(shots_left + shots_mid + shots_right, 1)
        shot_left_frac = float(shots_left / total_shots)
        shot_mid_frac = float(shots_mid / total_shots)
        shot_right_frac = float(shots_right / total_shots)
        hit_rate_left = float(hits_left / max(shots_left, 1))
        hit_rate_mid = float(hits_mid / max(shots_mid, 1))
        hit_rate_right = float(hits_right / max(shots_right, 1))

        fitness += HIT_REWARD * total_hits
        fitness -= DRY_FIRE_PENALTY * dry_fire_ticks
        fitness += RELOAD_BONUS * reload_correct_count

        # Hygiene reset: if this episode ended in a failed terminal state
        # (death/timeout), proactively reload now so BizHawk does not linger on
        # the "Continue?" UI between evaluations. The next episode still calls
        # reset() as usual; this just prevents visible spillover screens.
        if dead or timed_out:
            try:
                self.client.load_state(self.state_slot)
                self.client.step_frames(1)
            except Exception:
                pass

        return float(fitness), {
            "cleared": cleared,
            "timed_out": bool(timed_out),
            "dead": bool(dead),
            "elapsed": float(elapsed),
            "damage": float(total_life_loss),
            "accuracy": float(total_hits / max(total_fired, 1)),
            "shots_fired": int(total_fired),
            "shots_hit": int(total_hits),
            "peek_flips": int(peek_flips),
            "peek_hold_score": float(hold_score),
            "cover_time": int(ticks_in_cover),
            "dry_fire_ticks": int(dry_fire_ticks),
            "reload_correct_count": int(reload_correct_count),
            "continue_screen_count": int(continue_screen_count),
            "aim_x_std": aim_x_std,
            "aim_y_std": aim_y_std,
            "aim_span_x": aim_span_x,
            "aim_span_y": aim_span_y,
            "mean_abs_aim_dx": mean_abs_aim_dx,
            "mean_abs_aim_dy": mean_abs_aim_dy,
            "shots_left": int(shots_left),
            "shots_mid": int(shots_mid),
            "shots_right": int(shots_right),
            "shot_left_frac": shot_left_frac,
            "shot_mid_frac": shot_mid_frac,
            "shot_right_frac": shot_right_frac,
            "hit_rate_left": hit_rate_left,
            "hit_rate_mid": hit_rate_mid,
            "hit_rate_right": hit_rate_right,
        }
