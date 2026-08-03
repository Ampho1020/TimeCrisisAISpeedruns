"""Scalar-only environment wrapper (v1: no vision yet)."""

import numpy as np

from bridge_client import BridgeClient
from config import (
    AMMO_MAX_ROUNDS, CLEAR_BONUS, COVER_TRAVERSE_TICKS, DAMAGE_PENALTY,
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


def cover_hold_reward(
    cover_flags,
    traverse_ticks: int = COVER_TRAVERSE_TICKS,
    reward: float = 1.0,
) -> float:
    """Reward for holding the EXPOSE button (A) long enough for the exit
    animation to complete.

    NOTE ON NAMING: ``cover=True`` in code means the A button IS PRESSED,
    which makes the character EXIT cover (exposed, can shoot, can be hit).
    ``cover=False`` means A is NOT pressed = character stays IN cover.
    This is the opposite of what the variable name suggests.

    Only True (A-pressed = exposed) runs are counted:
      * a single-tick tap earns nothing,
      * each additional tick exposed, up to traverse_ticks, adds ``reward``,
      * runs longer than traverse_ticks are capped.
    """
    def run_value(run: int) -> float:
        return reward * min(max(run - 1, 0), traverse_ticks - 1)

    total = 0.0
    run = 0
    for held in cover_flags:
        if held:              # A pressed = character exiting/out of cover
            run += 1
        else:                 # A released = back to cover: finalise the exposed run
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
        self.prev_cover: bool = False
        self.cover_ticks: int = 0
        self.cover_lock: int = 0   # minimum hold: any transition holds for COVER_TRAVERSE_TICKS
        self.cover_locked_value: bool = False   # what state the lock is holding
        self.ammo_left: int = AMMO_MAX_ROUNDS

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
        }

    @staticmethod
    def _build_obs(cur, last_hit: int, last_miss: int, cover_phase: float = 0.0,
                   ammo_left: int = AMMO_MAX_ROUNDS) -> np.ndarray:
        fired = max(cur["shots_fired"], 1)
        return np.array([
            cur["timer"] / 10000.0,
            cur["life"] / 100.0,
            cur["shots_fired"] / 1000.0,
            cur["shots_hit"] / 1000.0,
            cur["shots_hit"] / fired,
            float(last_hit),
            float(last_miss),
            cover_phase,
            ammo_left / AMMO_MAX_ROUNDS,
        ], dtype=np.float32)

    # -- episode --------------------------------------------------------

    def reset(self) -> np.ndarray:
        self.client.load_state(self.state_slot)
        self.client.step_frames(2)
        self.prev = self._read_core()
        self.start_timer = self.prev["timer"]
        self.ticks = 0
        self.prev_cover = False
        self.cover_ticks = 0
        self.cover_lock = 0
        self.cover_locked_value = False
        self.ammo_left = AMMO_MAX_ROUNDS
        self.phase_infer.reset()
        return self._build_obs(self.prev, 0, 0, 0.0, self.ammo_left)

    def step(self, theta: np.ndarray):
        cover_phase = (self.cover_ticks / COVER_TRAVERSE_TICKS) * (1.0 if self.prev_cover else -1.0)
        shoot, cover, aim_x_bias, aim_y_bias = act(
            theta, self._build_obs(self.prev, 0, 0, cover_phase, self.ammo_left)
        )
        # cover=True  -> A button PRESSED  -> character EXITS cover (exposed, can shoot)
        # cover=False -> A button RELEASED -> character STAYS in cover (protected)
        # The name is inverted vs. the game state; see cover_hold_reward docstring.

        # Minimum hold lock: BOTH transitions (into cover and out of cover) have
        # to be held for COVER_TRAVERSE_TICKS ticks so the traverse animation can
        # complete. Previously only the False→True transition (leaving cover) was
        # locked, which let the policy re-enter cover for just a single tick
        # before being forced back out -- the "1 tick cover in-out" flicker
        # observed during training. Locking symmetrically kills that oscillation.
        if self.cover_lock > 0:
            cover = self.cover_locked_value
            self.cover_lock -= 1
        elif cover != self.prev_cover:
            # Any transition: lock the new state
            self.cover_lock = COVER_TRAVERSE_TICKS - 1  # -1 because this tick counts
            self.cover_locked_value = cover

        # Gate the trigger: shots only register when the character is FULLY out of
        # cover (A held for at least COVER_TRAVERSE_TICKS consecutive ticks).  Firing
        # during the transition animation silently fails in-game, so we block it here
        # to avoid wasting the edge-trigger on a guaranteed miss.
        shoot_allowed = cover and self.prev_cover and self.cover_ticks >= COVER_TRAVERSE_TICKS
        # Full-range mapping: tanh bias [-1, 1] spans the full screen [0, 1].
        # Using 0.5× previously kept the cursor in [0.17, 0.83] with typical
        # small initial weights; 1.0× lets early exploration reach the edges.
        aim_x = min(1.0, max(0.0, 0.5 + float(aim_x_bias)))
        aim_y = min(1.0, max(0.0, 0.5 + float(aim_y_bias)))

        total_fired = total_hit = total_life_loss = 0
        cleared_guess = dead_guess = timed_out_guess = False
        timer_at_tick_start = self.prev["timer"]

        for f in range(FRAME_SKIP):
            # Edge-trigger the shot: press briefly, release. Holding the
            # button for all 5 frames makes fire rate uncontrollable.
            # shoot_allowed ensures the trigger only fires when fully exposed.
            self.client.set_input(
                shoot=bool(shoot and shoot_allowed and f < 2),
                cover=cover,
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

            if post["life"] == 0:
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
        entering_cover = (cover != self.prev_cover) and not cover
        reload_correct = False
        if entering_cover:
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
        if cover == self.prev_cover:
            self.cover_ticks = min(self.cover_ticks + 1, COVER_TRAVERSE_TICKS)
        else:
            self.cover_ticks = 1
        self.prev_cover = cover
        cover_phase_next = (self.cover_ticks / COVER_TRAVERSE_TICKS) * (1.0 if cover else -1.0)
        obs = self._build_obs(self.prev, last_hit, last_miss, cover_phase_next, self.ammo_left)

        done = (phase is Phase.TERMINAL) or (self.ticks >= MAX_TICKS)
        info = {
            "shots_fired_delta": total_fired,
            "shots_hit_delta": total_hit,
            "life_loss": total_life_loss,
            "cleared": bool(cleared_guess and not dead_guess and not timed_out_guess),
            "dead": dead_guess,
            "timed_out": timed_out_guess,
            "cover": bool(cover),
            "phase": phase.name,
            "dry_fire": dry_fire,
            "reload_correct": reload_correct,
            "ammo_left": self.ammo_left,
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
        # Record (cover, shots_fired) per tick for post-episode reward shaping.
        cover_flags = []
        shots_per_tick = []

        while True:
            _, done, info = self.step(theta)
            total_hits      += info["shots_hit_delta"]
            total_fired     += info["shots_fired_delta"]
            total_life_loss += info["life_loss"]
            cleared = cleared or info["cleared"]
            timed_out = timed_out or info["timed_out"]
            dead = dead or info["dead"]
            cover_flags.append(info["cover"])
            shots_per_tick.append(info["shots_fired_delta"])
            dry_fire_ticks += int(info["dry_fire"])
            reload_correct_count += int(info["reload_correct"])
            if done:
                break

        elapsed = u16_delta(self.start_timer, self.prev["timer"])

        if cleared:
            fitness = CLEAR_BONUS - elapsed - DAMAGE_PENALTY * total_life_loss
        else:
            fitness = -FAIL_PENALTY
        # Diagnostics only (NOT added to fitness): cover_hold_score, cover_flips
        # and ticks_in_cover used to feed reward shaping (COVER_HOLD_REWARD,
        # COVER_FLIP_PENALTY, COVER_TIME_PENALTY); that noisy shaping was
        # removed so raw ES only optimizes the actual outcome. Kept here purely
        # for CSV logging / plotting.
        hold_score = 0.0
        _run_ticks = 0
        _run_shot = False
        for _cv, _sf in zip(cover_flags, shots_per_tick):
            if _cv:                        # exposed
                _run_ticks += 1
                if _sf > 0:
                    _run_shot = True
            else:                          # ducked back into cover
                if _run_shot:
                    hold_score += min(
                        max(_run_ticks - 1, 0), COVER_TRAVERSE_TICKS - 1
                    )
                _run_ticks = 0
                _run_shot = False
        if _run_shot:                      # episode ended while still exposed
            hold_score += min(
                max(_run_ticks - 1, 0), COVER_TRAVERSE_TICKS - 1
            )
        cover_flips = sum(
            1 for i in range(1, len(cover_flags))
            if cover_flags[i] != cover_flags[i - 1]
        )
        ticks_in_cover = sum(1 for f in cover_flags if not f)  # A NOT pressed = protected

        fitness += HIT_REWARD * total_hits
        fitness -= DRY_FIRE_PENALTY * dry_fire_ticks
        fitness += RELOAD_BONUS * reload_correct_count

        return float(fitness), {
            "cleared": cleared,
            "timed_out": bool(timed_out),
            "dead": bool(dead),
            "elapsed": float(elapsed),
            "damage": float(total_life_loss),
            "accuracy": float(total_hits / max(total_fired, 1)),
            "shots_fired": int(total_fired),
            "shots_hit": int(total_hits),
            "cover_flips": int(cover_flips),
            "cover_hold_score": float(hold_score),
            "cover_time": int(ticks_in_cover),
            "dry_fire_ticks": int(dry_fire_ticks),
            "reload_correct_count": int(reload_correct_count),
        }
