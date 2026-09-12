

import time

import numpy as np

from bridge_client import BridgeClient
from config import (
    ACCURACY_BONUS_WEIGHT, AMMO_MAX_ROUNDS, AMMO_SHOT_COST,
    CLEAR_ACCURACY_GATE_FLOOR, CLEAR_ACCURACY_GATE_TARGET,
    CLEAR_BONUS, CONTINUE_SCREEN_FALLBACK_TICKS,
    CONTINUE_SCREEN_STALE_TICKS, CURSOR_X_MAX,
    CURSOR_X_MIN, CURSOR_Y_MAX, CURSOR_Y_MIN, COVER_HESITATION_PENALTY,
    DAMAGE_PENALTY,
    FAIL_PENALTY,
    EXPOSED_NO_SHOT_PENALTY,
    FRAME_SKIP, HIT_DELTA_NORM_FRAMES, HIT_DELTA_PENALTY, HOST, HIT_REWARD,
    MAX_TICKS, MISS_PENALTY, MULTI_CLEAR_BONUS,
    PEEK_LOCK_IN_TICKS, PEEK_LOCK_OUT_TICKS, PEEK_TRAVERSE_TICKS,
    POLICY_MODE, PORT, RAM, REACTION_LATENCY_NORM_TICKS, REACTION_LATENCY_PENALTY,
    RELOAD_BONUS,
    SCREEN_CLEAR_TIMER_BUMP, SHOOT_PULSE_EVERY_N_FRAMES,
    STATE_SLOT, TIMEOUT_TIMER_THRESHOLD,
    VISION_CAPTURE_EVERY_N_TICKS, VISION_DETECTOR_DEVICE, VISION_ONNX_MODEL_PATH,
    VISION_PROFILE, VISION_PROFILE_PRINT_EVERY, VISION_TORCH_MODEL_PATH,
)
from phase_inference import Phase, PhaseInferer, TickSignals
from policy import act, act_schedule, act_vision_schedule


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


def shot_phase_features(ammo_left: int) -> tuple[float, float]:
    """Return (sin, cos) of the current shot-in-clip phase angle.

    Uses shot_idx = AMMO_MAX_ROUNDS - ammo_left so the angle advances by
    2*pi/AMMO_MAX_ROUNDS with each shot; when ammo refills back to the max
    at the reload transition, the angle wraps back to 0 (sin=0, cos=1) --
    same value as the first shot of a clip. Non-smooth per-shot signal
    (see config.py OBS_DIM comment) intended to break the "same 6-shot arc
    every clip" symptom by giving the policy an explicit cue for which
    shot in the clip is currently loaded, independent of the smooth
    ammo_norm ramp already in the observation.
    """
    idx = AMMO_MAX_ROUNDS - int(ammo_left)
    if idx < 0:
        idx = 0
    elif idx >= AMMO_MAX_ROUNDS:
        idx = AMMO_MAX_ROUNDS - 1
    angle = 2.0 * np.pi * idx / AMMO_MAX_ROUNDS
    return float(np.sin(angle)), float(np.cos(angle))


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
    def __init__(self, host=HOST, port=PORT, state_slot=STATE_SLOT, per_frame_vision: bool = False):
        self.client = BridgeClient(host, port)
        self.state_slot = state_slot
        # When True (set by run_eval.py, never by training/worker_pool.py),
        # the vision_schedule branch below re-captures a screenshot and
        # re-runs the detector + aim blend on EVERY raw emulator frame
        # inside the FRAME_SKIP inner loop, instead of once per decision
        # tick (cached across VISION_CAPTURE_EVERY_N_TICKS ticks). This makes
        # standalone evaluation react to the freshest possible frame each
        # 1/60s, like a human player watching the screen continuously,
        # instead of the training-time batched cadence tuned for ES
        # wall-clock cost. shoot/peek/base-aim still come from the fixed
        # per-tick theta row (that granularity is inherent to the trained
        # table and unaffected by this flag) -- only the vision-informed aim
        # blend gets refreshed every frame. Substantially slower (many more
        # detector calls per episode) so it's opt-in and only used for eval.
        self.per_frame_vision = per_frame_vision
        self.phase_infer = PhaseInferer(vote_window=3)
        self.prev: dict[str, int] = {}
        self.start_timer = 0
        self.ticks = 0
        self.prev_peek: bool = False
        self.peek_ticks: int = 0
        self.peek_lock: int = 0   # minimum hold: any transition holds for PEEK_TRAVERSE_TICKS
        self.peek_locked_value: bool = False   # what state the lock is holding
        self.stale_core_ticks: int = 0  # consecutive ticks with identical core RAM snapshot
        self.stale_shots_life_ticks: int = 0  # consecutive ticks with frozen shots/life only (timer-independent)
        self.ammo_left: int = AMMO_MAX_ROUNDS
        # Confirmed-empty latch: True only once RAM.ammo has been observed
        # counting DOWN to 0 through firing (a real >0 -> 0 transition), and
        # cleared again the moment any >0 value is read back. The hard cover
        # override below keys off THIS, not the raw live value, so a resting 0
        # (freshly loaded savestate still in cover, mid reload-animation frames,
        # or a transient u16 sample) can't force the agent into cover -- that
        # false-positive trapped the entire ES population from generation 0.
        self.clip_empty: bool = False
        self.hit_delta: int = 0  # frames since the last confirmed hit
        self.reaction_no_shot_streak: int = 0  # ticks a target has been visible with no shot fired since
        self.prev_aim_x_bias: float = 0.0   # last tick's aim_x_bias, fed back as obs
        self.prev_aim_y_bias: float = 0.0   # last tick's aim_y_bias, fed back as obs
        # Multi-screen tracking (see reset() for the full comment).
        self.screens_cleared: int = 0
        # Set to a directory path (via ``run_eval.py --dump-frames <dir>``) to
        # save one PNG per decision tick during ``episode_fitness()``. Purely
        # diagnostic / used to produce the labelling corpus for the Phase 2
        # offline YOLO fine-tune workflow documented at the bottom of
        # detector.py -- has NO effect on training or fitness. Default None
        # (no capture, zero per-tick overhead).
        self.dump_frames_dir: str | None = None
        self._dump_frame_counter: int = 0
        # Vision-conditioned schedule mode (POLICY_MODE="vision_schedule"):
        # build the detector once (ONNX if VISION_ONNX_MODEL_PATH points at a
        # real file, else the classical CV baseline -- see
        # detector.build_detector). Every VISION_CAPTURE_EVERY_N_TICKS the
        # env captures a fresh screenshot + refreshes ``last_detections``;
        # between captures the cached detections are reused so we don't
        # pay a full BMP + inference cost on every single tick. Under the
        # other POLICY_MODEs (mlp / schedule) the detector is not built at
        # all so the schedule/mlp code paths pay zero import/init cost.
        if POLICY_MODE == "vision_schedule":
            from detector import build_detector
            self.detector = build_detector(
                VISION_ONNX_MODEL_PATH or None,
                torch_model_path=VISION_TORCH_MODEL_PATH or None,
                device=VISION_DETECTOR_DEVICE,
            )
        else:
            self.detector = None
        self.last_detections: list | None = None
        # VISION_PROFILE diagnostics (config.py): rolling per-tick timing
        # buffers, only ever appended to when VISION_PROFILE is True (see
        # _profile_reset/_profile_maybe_print below). Zero cost otherwise.
        self._profile_shot_ms: list[float] = []
        self._profile_detect_ms: list[float] = []
        self._profile_tick_ms: list[float] = []
        self._profile_setinput_ms: list[float] = []
        self._profile_stepframes_ms: list[float] = []
        self._profile_readcore_ms: list[float] = []

    def _profile_reset(self) -> None:
        self._profile_shot_ms = []
        self._profile_detect_ms = []
        self._profile_tick_ms = []
        self._profile_setinput_ms = []
        self._profile_stepframes_ms = []
        self._profile_readcore_ms = []

    def _profile_maybe_print(self) -> None:
        """Print a rolling mean/max timing summary every
        VISION_PROFILE_PRINT_EVERY ticks, then clear the buffers so each
        printed window reflects only the ticks since the last print (a live
        rolling view, not a cumulative episode-long average)."""
        if len(self._profile_tick_ms) < VISION_PROFILE_PRINT_EVERY:
            return

        def _stats(vals: list[float]) -> str:
            if not vals:
                return "n/a"
            return f"mean={sum(vals) / len(vals):.1f}ms max={max(vals):.1f}ms"

        print(
            f"[vision_profile] tick={self.ticks} n={len(self._profile_tick_ms)} "
            f"| screenshot {_stats(self._profile_shot_ms)} "
            f"| detect {_stats(self._profile_detect_ms)} "
            f"| set_input {_stats(self._profile_setinput_ms)} "
            f"| step_frames {_stats(self._profile_stepframes_ms)} "
            f"| read_core {_stats(self._profile_readcore_ms)} "
            f"| full_tick {_stats(self._profile_tick_ms)}",
            flush=True,
        )
        self._profile_reset()

    def connect(self):
        self.client.connect()

    def start_listening(self):
        self.client.start_listening()

    def finish_connect(self):
        self.client.finish_connect()

    def close(self):
        self.client.close()

    def _dump_current_frame(self) -> None:
        """Capture one frame via the bridge and save it as a PNG under
        ``self.dump_frames_dir``. Guarded by a broad try/except so a
        transient screenshot failure never aborts the surrounding episode
        or training run (this is a diagnostic path, not a fitness input).

        File name pattern: ``frame_XXXXXX.png`` (six-digit zero-padded
        tick counter) so the natural sort matches the tick order the
        frames were captured in.
        """
        try:
            import os
            frame = self.client.get_screenshot()
            os.makedirs(self.dump_frames_dir, exist_ok=True)
            path = os.path.join(
                self.dump_frames_dir,
                f"frame_{self._dump_frame_counter:06d}.png",
            )
            # cv2 expects BGR; convert once.
            import cv2
            bgr = frame[:, :, [2, 1, 0]]
            cv2.imwrite(path, bgr)
            self._dump_frame_counter += 1
        except Exception as exc:  # pragma: no cover -- diagnostic path
            print(
                f"[env] frame dump failed at tick {self.ticks}: {exc!r}",
                flush=True,
            )

    # -- RAM ------------------------------------------------------------

    def _read_core(self):
        # Prefer a single batched round trip (BridgeClient.read_u16_multi,
        # added 2026-09-05 -- see repo memory) over six sequential
        # read_u16() calls: each command pays a fixed Lua-loop round-trip
        # cost regardless of payload, so batching collapses ~75-90ms/frame
        # down to ~13ms/frame. Falls back to individual read_u16() calls for
        # any client that doesn't implement the batched method (e.g. the
        # in-process simulation fakes in tests/test_simulation.py).
        batch = getattr(self.client, "read_u16_multi", None)
        if batch is not None:
            vals = batch([
                RAM.shots_fired, RAM.shots_hit, RAM.timer,
                RAM.life, RAM.cursor_x, RAM.cursor_y, RAM.ammo,
            ])
            return {
                "shots_fired": vals[0],
                "shots_hit":   vals[1],
                "timer":       vals[2],
                "life":        vals[3],
                "cursor_x":    vals[4],
                "cursor_y":    vals[5],
                "ammo":        vals[6],
            }
        return {
            "shots_fired": self.client.read_u16(RAM.shots_fired),
            "shots_hit":   self.client.read_u16(RAM.shots_hit),
            "timer":       self.client.read_u16(RAM.timer),
            "life":        self.client.read_u16(RAM.life),
            "cursor_x":    self.client.read_u16(RAM.cursor_x),
            "cursor_y":    self.client.read_u16(RAM.cursor_y),
            "ammo":        self.client.read_u16(RAM.ammo),
        }

    @staticmethod
    def _build_obs(cur, last_hit: int, last_miss: int, peek_phase: float = 0.0,
                   ammo_left: int = AMMO_MAX_ROUNDS,
                   prev_aim_x_bias: float = 0.0, prev_aim_y_bias: float = 0.0,
                   hit_delta: int = 0) -> np.ndarray:
        fired = max(cur["shots_fired"], 1)
        shot_sin, shot_cos = shot_phase_features(ammo_left)
        hit_delta_norm = float(np.clip(hit_delta / HIT_DELTA_NORM_FRAMES, 0.0, 1.0))
        return np.array([
            cur["timer"] / 10000.0,
            cur["life"] / 100.0,
            cur["shots_fired"] / 1000.0,
            cur["shots_hit"] / 1000.0,
            cur["shots_hit"] / fired,
            float(last_hit),
            float(last_miss),
            hit_delta_norm,
            peek_phase,
            ammo_left / AMMO_MAX_ROUNDS,
            prev_aim_x_bias,
            prev_aim_y_bias,
            normalize_cursor(cur.get("cursor_x", CURSOR_X_MIN), CURSOR_X_MIN, CURSOR_X_MAX),
            normalize_cursor(cur.get("cursor_y", CURSOR_Y_MIN), CURSOR_Y_MIN, CURSOR_Y_MAX),
            shot_sin,
            shot_cos,
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
        self.stale_shots_life_ticks = 0
        self.ammo_left = self.prev["ammo"]
        # Start every screen NOT confirmed-empty regardless of what RAM.ammo
        # happens to read at the load_state instant (see __init__): the clip is
        # only ever latched empty by an observed >0 -> 0 firing transition.
        self.clip_empty = False
        self.hit_delta = 0
        self.reaction_no_shot_streak = 0
        self.prev_aim_x_bias = 0.0
        self.prev_aim_y_bias = 0.0
        self.phase_infer.reset()
        # Multi-screen tracking (added 2026-08-10, reworked 2026-08-15). We
        # count DISTINCT screen-clear events across the episode by comparing
        # the timer at the END of each decision tick to what it was at the
        # START of that same tick (see step() for why this is tick-granular,
        # not per-frame/stateful).
        self.screens_cleared = 0
        # Vision state: discard the previous episode's background model
        # (MOG2 in ClassicalDetector accumulates its own history across
        # calls) and the cached detections. Guarded via getattr so sim
        # subclasses that skip TimeCrisisEnv.__init__ still work.
        detector = getattr(self, "detector", None)
        if detector is not None:
            detector.reset()
        self.last_detections = None
        self._profile_reset()
        return self._build_obs(
            self.prev, 0, 0, 0.0, self.ammo_left,
            self.prev_aim_x_bias, self.prev_aim_y_bias,
            hit_delta=self.hit_delta,
        )

    def step(self, theta: np.ndarray):
        _tick_t0 = time.perf_counter() if VISION_PROFILE else 0.0
        peek_phase = (self.peek_ticks / PEEK_TRAVERSE_TICKS) * (1.0 if self.prev_peek else -1.0)
        enemy_visible = False
        if POLICY_MODE == "schedule":
            # Open-loop: action is read directly from theta[self.ticks],
            # no observation consumed for action selection. `obs` is still
            # built via self._build_obs(...) below purely for return-value/
            # diagnostic parity with the closed-loop path -- not used here.
            shoot, peek, aim_x_bias, aim_y_bias = act_schedule(theta, self.ticks)
        elif POLICY_MODE == "vision_schedule":
            # Vision-conditioned schedule: refresh detections on the capture
            # cadence, then blend into the base aim via act_vision_schedule.
            # In per_frame_vision mode (eval only) we always capture fresh
            # here too -- this first capture backs the shoot/peek gating
            # decision below; the frame loop further down re-captures again
            # for every subsequent raw frame within this same tick.
            _is_new_capture = (
                getattr(self, "per_frame_vision", False)
                or self.last_detections is None
                or (self.ticks % VISION_CAPTURE_EVERY_N_TICKS) == 0
            )
            if _is_new_capture:
                try:
                    if VISION_PROFILE:
                        _t0 = time.perf_counter()
                        frame = self.client.get_screenshot()
                        self._profile_shot_ms.append((time.perf_counter() - _t0) * 1000.0)
                        _t0 = time.perf_counter()
                        self.last_detections = self.detector.detect(frame)
                        self._profile_detect_ms.append((time.perf_counter() - _t0) * 1000.0)
                    else:
                        frame = self.client.get_screenshot()
                        self.last_detections = self.detector.detect(frame)
                except Exception as exc:  # pragma: no cover - defensive
                    if self.last_detections is None:
                        self.last_detections = []
                    print(f"[env] vision capture failed: {exc!r}", flush=True)

            shoot, peek, aim_x_bias, aim_y_bias = act_vision_schedule(
                theta,
                self.ticks,
                self.last_detections or [],
                cursor_x_norm=normalize_cursor(
                    self.prev.get("cursor_x", CURSOR_X_MIN), CURSOR_X_MIN, CURSOR_X_MAX,
                ),
                cursor_y_norm=normalize_cursor(
                    self.prev.get("cursor_y", CURSOR_Y_MIN), CURSOR_Y_MIN, CURSOR_Y_MAX,
                ),
                ammo_left_norm=self.ammo_left / AMMO_MAX_ROUNDS,
            )
            enemy_visible = any(
                int(det.class_id) == 0 for det in (self.last_detections or [])
            )
        else:
            shoot, peek, aim_x_bias, aim_y_bias = act(
                theta, self._build_obs(
                    self.prev, 0, 0, peek_phase, self.ammo_left,
                    self.prev_aim_x_bias, self.prev_aim_y_bias,
                    hit_delta=self.hit_delta,
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
        #
        # 2026-09-12: this now keys off self.clip_empty (a confirmed >0 -> 0
        # firing transition) instead of the raw live RAM.ammo value. Gating on
        # the raw value made the override fire whenever RAM.ammo momentarily
        # read 0 for a NON-firing reason -- a savestate loaded while still in
        # cover, the frames during the real reload animation, or a noisy u16
        # sample -- which pinned the whole population in a cover<->peek flicker
        # from generation 0 (flat, negative fitness, no ES progress at all).
        # The confirmed-empty latch only forces cover when the clip was truly
        # shot dry, restoring the good-run behaviour.
        if self.clip_empty:
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
            lock_ticks = PEEK_LOCK_OUT_TICKS if peek else PEEK_LOCK_IN_TICKS
            self.peek_lock = max(0, int(lock_ticks) - 1)  # -1 because this tick counts
            self.peek_locked_value = peek

        # Gate the trigger: only attempt to fire once we're not mid-transition
        # (this tick AND last tick both chose "peek"). We used to also require
        # self.peek_ticks >= PEEK_TRAVERSE_TICKS (a fixed, rough estimate of
        # the ~12-frame traverse-out animation) before even sending the shoot
        # button -- but that's an artificial delay we don't need: real
        # in-game success/failure is decided by the emulator itself, not by
        # us, and is read back ground-truth via shots_fired (RAM) regardless
        # of what we assume here. Firing before the real animation completes
        # just silently fails in-game (total_fired stays 0, no reward, no
        # ammo consumed) -- removing the estimate only lets the agent attempt
        # shots as early as the actual game allows instead of waiting out our
        # guess, i.e. a "frame perfect" reload/re-expose cycle bounded by real
        # game state (peek + ammo_left) rather than a hardcoded tick count.
        # NOTE: this is independent from peek_lock above, which must stay --
        # that lock prevents the AGENT from reversing the peek button mid
        # traverse (a real in-game animation-reversal bug, confirmed live),
        # not from firing too early.
        shoot_allowed = peek
        # Full-range mapping: tanh bias [-1, 1] spans the full screen [0, 1].
        # Using 0.5× previously kept the cursor in [0.17, 0.83] with typical
        # small initial weights; 1.0× lets early exploration reach the edges.
        aim_x = min(1.0, max(0.0, 0.5 + float(aim_x_bias)))
        aim_y = min(1.0, max(0.0, 0.5 + float(aim_y_bias)))

        total_fired = total_hit = total_life_loss = 0
        dead_guess = timed_out_guess = False
        continue_screen_guess = False
        tick_start_core = core_watchdog_snapshot(self.prev)
        tick_start_shots_life = (
            self.prev["shots_fired"], self.prev["shots_hit"], self.prev["life"],
        )
        timer_at_tick_start = self.prev["timer"]
        # Snapshot the CONFIRMED-empty state as of the START of this tick
        # (before the frame loop below updates self.clip_empty from live
        # RAM.ammo transitions) for the post-loop "was the clip actually empty
        # when the agent decided to duck" diagnostics further down. Keyed off
        # the latch rather than a raw RAM.ammo == 0 sample so those
        # penalties/bonuses can't fire on a spurious 0 either.
        clip_empty_at_tick_start = self.clip_empty

        for f in range(FRAME_SKIP):
            # Per-frame vision refresh (eval only, see __init__): re-capture
            # + re-blend the aim on every raw frame after the first (which
            # already got a fresh capture above) so aim tracks the latest
            # frame instead of being frozen for the whole FRAME_SKIP-frame
            # tick. shoot/peek/base-aim are NOT recomputed here -- they come
            # from the fixed per-tick theta row and would be identical.
            if getattr(self, "per_frame_vision", False) and POLICY_MODE == "vision_schedule" and f > 0:
                try:
                    if VISION_PROFILE:
                        _t0 = time.perf_counter()
                        frame_img = self.client.get_screenshot()
                        self._profile_shot_ms.append((time.perf_counter() - _t0) * 1000.0)
                        _t0 = time.perf_counter()
                        self.last_detections = self.detector.detect(frame_img)
                        self._profile_detect_ms.append((time.perf_counter() - _t0) * 1000.0)
                    else:
                        frame_img = self.client.get_screenshot()
                        self.last_detections = self.detector.detect(frame_img)
                except Exception as exc:  # pragma: no cover - defensive
                    print(f"[env] per-frame vision capture failed: {exc!r}", flush=True)
                _, _, aim_x_bias, aim_y_bias = act_vision_schedule(
                    theta,
                    self.ticks,
                    self.last_detections or [],
                    cursor_x_norm=normalize_cursor(
                        self.prev.get("cursor_x", CURSOR_X_MIN), CURSOR_X_MIN, CURSOR_X_MAX,
                    ),
                    cursor_y_norm=normalize_cursor(
                        self.prev.get("cursor_y", CURSOR_Y_MIN), CURSOR_Y_MIN, CURSOR_Y_MAX,
                    ),
                    ammo_left_norm=self.ammo_left / AMMO_MAX_ROUNDS,
                )
                aim_x = min(1.0, max(0.0, 0.5 + float(aim_x_bias)))
                aim_y = min(1.0, max(0.0, 0.5 + float(aim_y_bias)))
                self.prev_aim_x_bias = float(aim_x_bias)
                self.prev_aim_y_bias = float(aim_y_bias)

            # Edge-trigger the shot: press briefly, release. Holding the
            # button for all 5 frames makes fire rate uncontrollable.
            # shoot_allowed ensures the trigger only fires when fully exposed.
            pulse_every = max(2, int(SHOOT_PULSE_EVERY_N_FRAMES))
            if VISION_PROFILE:
                _t0 = time.perf_counter()
            self.client.set_input(
                shoot=bool(
                    shoot
                    and shoot_allowed
                    and (f % pulse_every == 0)
                ),
                peek=peek,
                aim_x=aim_x,
                aim_y=aim_y,
            )
            if VISION_PROFILE:
                self._profile_setinput_ms.append((time.perf_counter() - _t0) * 1000.0)

            pre = self.prev
            if VISION_PROFILE:
                _t0 = time.perf_counter()
            self.client.step_frames(1)
            if VISION_PROFILE:
                self._profile_stepframes_ms.append((time.perf_counter() - _t0) * 1000.0)
                _t0 = time.perf_counter()
            post = self._read_core()
            if VISION_PROFILE:
                self._profile_readcore_ms.append((time.perf_counter() - _t0) * 1000.0)

            total_fired += max(0, u16_delta(post["shots_fired"], pre["shots_fired"]))
            frame_hits = max(0, u16_delta(post["shots_hit"], pre["shots_hit"]))
            total_hit += frame_hits
            if frame_hits > 0:
                self.hit_delta = 0
            else:
                self.hit_delta += 1
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
            # Timeout: the countdown reached zero -> "continue?" screen. Detect
            # the zero-cross here (a large downward step across the tick also
            # counts, in case the timer skips the exact zero sample).
            timer_step = u16_delta(post["timer"], pre["timer"])
            if post["timer"] <= TIMEOUT_TIMER_THRESHOLD or (
                timer_step < 0 and pre["timer"] + timer_step <= TIMEOUT_TIMER_THRESHOLD
            ):
                timed_out_guess = True

            self.prev = post
            # Update the confirmed-empty latch from this frame's RAM.ammo
            # transition (see self.clip_empty in __init__): any observed >0
            # value clears it (a real refill / full clip), a >0 -> 0 step
            # latches it (the round count actually counted down through
            # firing), and a resting 0 -> 0 is left untouched so a savestate or
            # reload-animation frame that simply reads 0 never latches empty.
            if post["ammo"] > 0:
                self.clip_empty = False
            elif pre["ammo"] > 0:
                self.clip_empty = True
            self.ammo_left = post["ammo"]
            # Only bail out of the inner frame loop for TERMINAL outcomes
            # (death or timeout). A clear no longer breaks: we want the
            # remaining frames to run so the game can start rendering the
            # NEXT screen this same tick, giving vision a fresh frame to
            # capture next tick.
            if dead_guess or timed_out_guess:
                break

        # Screen-clear detection (added 2026-08-10, reworked 2026-08-15):
        # compare the timer at the END of this whole decision tick to what it
        # was at the START of the tick -- tick-granular, not per-frame. Time
        # Crisis' timer counts DOWN a few units/tick in normal play AND
        # throughout the screen-to-screen transition itself (there is no
        # "frozen" phase to key off), so a per-frame stateful "are we still
        # inside a clear transition" flag is both unnecessary and unreliable
        # (the timer's continuous countdown during the transition made a
        # naive "re-arm once timer_step < 0" check fire far too early,
        # letting a single clear's bonus roll get double-counted). The bonus
        # roll conversion itself completes within a single tick, so one
        # tick-level comparison against SCREEN_CLEAR_TIMER_BUMP is enough --
        # this is also the single source of truth ``info["cleared"]`` derives
        # from below, replacing the old separate cleared_guess heuristic.
        tick_timer_delta = u16_delta(self.prev["timer"], timer_at_tick_start)
        clear_this_tick = tick_timer_delta > SCREEN_CLEAR_TIMER_BUMP
        if clear_this_tick:
            self.screens_cleared += 1

        # Fallback continue/menu watchdog: if all core counters were frozen
        # across the entire decision tick, count it. Several consecutive frozen
        # ticks indicate we've likely landed on a non-playable menu (e.g.
        # continue prompt) that escaped direct life/timer terminal detection.
        if (
            not dead_guess
            and not clear_this_tick
            and not timed_out_guess
            and core_watchdog_snapshot(self.prev) == tick_start_core
        ):
            self.stale_core_ticks += 1
        else:
            self.stale_core_ticks = 0
        if self.stale_core_ticks >= CONTINUE_SCREEN_STALE_TICKS:
            timed_out_guess = True
            continue_screen_guess = True

        # Second, slower fallback that ignores ``timer`` entirely (see
        # CONTINUE_SCREEN_FALLBACK_TICKS in config.py): catches the case where
        # the continue-prompt countdown keeps the timer RAM address moving,
        # which would otherwise prevent the watchdog above from ever firing.
        current_shots_life = (
            self.prev["shots_fired"], self.prev["shots_hit"], self.prev["life"],
        )
        if (
            not dead_guess
            and not clear_this_tick
            and current_shots_life == tick_start_shots_life
        ):
            self.stale_shots_life_ticks += 1
        else:
            self.stale_shots_life_ticks = 0
        if self.stale_shots_life_ticks >= CONTINUE_SCREEN_FALLBACK_TICKS:
            timed_out_guess = True
            continue_screen_guess = True
            print(
                "[env_timecrisis] shots/life-stale fallback fired "
                f"({self.stale_shots_life_ticks} ticks) -- likely stuck on a "
                "continue/menu screen the timer-based watchdog missed.",
                flush=True,
            )

        # Wasted exposure: penalise ticks where the agent is fully exposed with
        # an EMPTY clip (ammo_left was already 0 at the start of this tick)
        # instead of ducking back into cover to reload. This is exact now that
        # ammo_left is tracked, rather than the old total_fired == 0 proxy
        # which also (wrongly) fired whenever the policy simply chose not to
        # shoot with ammo still available.
        # Empty-clip checks key off the CONFIRMED-empty latch as of tick start
        # (clip_empty_at_tick_start) rather than a raw RAM.ammo == 0 sample, so
        # they can't mis-fire on a spurious 0 -- same rationale as the hard
        # peek override above.
        dry_fire = bool(shoot_allowed and clip_empty_at_tick_start)
        no_shot_exposed = bool(
            shoot_allowed and (not clip_empty_at_tick_start) and enemy_visible and total_fired == 0
        )
        hesitated_cover = bool((not peek) and (not clip_empty_at_tick_start) and enemy_visible)

        # Reaction latency: how many ticks a target has been visible without
        # a shot being fired at it yet (see REACTION_LATENCY_PENALTY in
        # config.py). Resets to 0 the instant a shot is fired while a target
        # is visible -- reaction_latency below then reports the streak length
        # right BEFORE that reset (0 means "fired the same tick it appeared").
        # Also resets to 0 whenever no target is visible (nothing to react to).
        reaction_latency = None
        if enemy_visible:
            if total_fired > 0:
                reaction_latency = self.reaction_no_shot_streak
                self.reaction_no_shot_streak = 0
            else:
                self.reaction_no_shot_streak += 1
        else:
            self.reaction_no_shot_streak = 0

        # Ammo bookkeeping (2026-09-11: switched to real RAM.ammo instead of
        # software-simulating consumption/reload -- see AMMO_MAX_ROUNDS in
        # config.py). self.ammo_left is now kept continuously in sync with
        # RAM.ammo inside the frame loop above (self.ammo_left = post["ammo"]),
        # so it already reflects the TRUE in-game clip state here -- no manual
        # decrement, no instant "refill to full the moment peek->False" hack,
        # and no separate free-reload-on-screen-clear special case needed:
        # the real game's own RAM naturally reflects a full clip once it
        # actually reloads or starts a new screen, however long that takes.
        ending_peek = (peek != self.prev_peek) and not peek
        reload_correct = bool(ending_peek and clip_empty_at_tick_start)

        self.ticks += 1

        phase = self.phase_infer.infer(TickSignals(
            shots_fired_delta=total_fired,
            shots_hit_delta=total_hit,
            life_delta=-total_life_loss,
            timer_delta=tick_timer_delta,
            # Multi-screen: a clear no longer forces Phase.TERMINAL (which is
            # absorbing) -- we want the episode to keep running so ES can
            # LEARN to chain screens for the quadratic MULTI_CLEAR_BONUS.
            # Only death/timeout still end the episode; the "was there any
            # clear" signal is captured by ``self.screens_cleared`` for the
            # fitness formula (see info["cleared"] below).
            cleared_guess=False,
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
        # Same free-transition reset as the ammo refill above: the new
        # screen also always starts fully in cover, so the next tick's
        # peek-lock/traverse bookkeeping must reflect that rather than
        # whatever exposure state carried over from the moment of the clear.
        if clear_this_tick:
            self.prev_peek = False
            self.peek_lock = 0
            self.peek_ticks = 0
        peek_phase_next = (self.peek_ticks / PEEK_TRAVERSE_TICKS) * (1.0 if peek else -1.0)
        obs = self._build_obs(
            self.prev, last_hit, last_miss, peek_phase_next, self.ammo_left,
            self.prev_aim_x_bias, self.prev_aim_y_bias,
            hit_delta=self.hit_delta,
        )

        done = (phase is Phase.TERMINAL) or (self.ticks >= MAX_TICKS)
        info = {
            "shots_fired_delta": total_fired,
            "shots_hit_delta": total_hit,
            "life_loss": total_life_loss,
            # Single source of truth: derived directly from screens_cleared
            # (no separate/second clear heuristic -- see the screen-clear
            # detection comment above, 2026-08-15). "At least one screen
            # cleared so far this episode", excluding a tick where the SAME
            # tick also ended in death/timeout.
            "cleared": bool(self.screens_cleared > 0 and not dead_guess and not timed_out_guess),
            # Cumulative screen-clear count so far this episode.
            "screens_cleared": int(self.screens_cleared),
            "dead": dead_guess,
            "timed_out": timed_out_guess,
            "continue_screen": continue_screen_guess,
            "peek": bool(peek),
            "phase": phase.name,
            "dry_fire": dry_fire,
            "no_shot_exposed": no_shot_exposed,
            "hesitated_cover": hesitated_cover,
            "reaction_latency": reaction_latency,
            "reload_correct": reload_correct,
            "ammo_left": self.ammo_left,
            "hit_delta": int(self.hit_delta),
            "aim_x": float(aim_x),
            "aim_y": float(aim_y),
        }
        if VISION_PROFILE:
            self._profile_tick_ms.append((time.perf_counter() - _tick_t0) * 1000.0)
            self._profile_maybe_print()
        return obs, done, info

    def episode_fitness(self, theta: np.ndarray):
        """Run one full episode. Returns (fitness, metrics)."""
        self.reset()
        total_hits = total_fired = total_life_loss = 0
        dry_fire_ticks = 0
        no_shot_exposed_ticks = 0
        hesitated_cover_ticks = 0
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
        hit_delta_per_tick = []
        reaction_latencies = []

        while True:
            _, done, info = self.step(theta)
            if getattr(self, "dump_frames_dir", None) is not None:
                # Optional per-tick frame dump for the offline YOLO fine-tune
                # workflow (see detector.py footer). No-op unless
                # dump_frames_dir was set on the env; getattr fallback keeps
                # sim subclasses that skip TimeCrisisEnv.__init__ working.
                self._dump_current_frame()
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
            hit_delta_per_tick.append(float(info.get("hit_delta", self.hit_delta)))
            dry_fire_ticks += int(info["dry_fire"])
            no_shot_exposed_ticks += int(info.get("no_shot_exposed", False))
            hesitated_cover_ticks += int(info.get("hesitated_cover", False))
            if info.get("reaction_latency") is not None:
                reaction_latencies.append(int(info["reaction_latency"]))
            reload_correct_count += int(info["reload_correct"])
            if done:
                break

        # An episode ending mid-streak (a target still visible, no shot
        # fired at it yet) would otherwise never contribute a sample --
        # closing that loophole so a policy can't dodge this metric by
        # simply never firing at a visible target.
        if self.reaction_no_shot_streak > 0:
            reaction_latencies.append(int(self.reaction_no_shot_streak))

        elapsed = u16_delta(self.start_timer, self.prev["timer"])
        # Read the cumulative screen-clear count from the env itself (not
        # accumulated across ticks) -- self.screens_cleared is monotone within
        # an episode and info["screens_cleared"] on the LAST tick already
        # holds the final total. This is now the SOLE clear signal (info
        # ["cleared"]/``cleared`` above is just a derived view of it -- see
        # step()'s 2026-08-15 comment), so the branch below only needs to
        # check screens_cleared.
        screens_cleared = int(getattr(self, "screens_cleared", 0))

        # Accuracy of the whole episode, needed up-front to gate the clear
        # reward (see CLEAR_ACCURACY_GATE_FLOOR/TARGET in config.py). Computed
        # here rather than lower down so the dominant clear term can scale by
        # it; the later diagnostics block reuses this same value.
        accuracy = float(total_hits / max(total_fired, 1))
        # Gate ramps FLOOR -> 1.0 as accuracy climbs to TARGET, then saturates.
        clear_accuracy_gate = CLEAR_ACCURACY_GATE_FLOOR + (
            1.0 - CLEAR_ACCURACY_GATE_FLOOR
        ) * min(accuracy / max(CLEAR_ACCURACY_GATE_TARGET, 1e-9), 1.0)

        if screens_cleared > 0:
            fitness = CLEAR_BONUS - elapsed - DAMAGE_PENALTY * total_life_loss
        else:
            fitness = -FAIL_PENALTY
        # QUADRATIC multi-screen bonus (added 2026-08-10 alongside
        # vision_schedule). Rewards clearing more screens strictly-more per
        # extra screen: gap between (N+1)-clear and N-clear fitness is
        # (2N+1) * MULTI_CLEAR_BONUS -- so ES has an increasing marginal
        # incentive to push for one more screen every time. See MULTI_CLEAR_BONUS
        # in config.py for the full "why quadratic" rationale.
        fitness += MULTI_CLEAR_BONUS * screens_cleared * screens_cleared
        # Accuracy-gate the DOMINANT clear reward so a sloppy clear scores
        # strictly less than a clean one (2026-09-13 Strategy A fix). Applied
        # only on cleared episodes -- the accuracy portion of the reward earned
        # so far (CLEAR_BONUS + MULTI_CLEAR_BONUS*screens**2, minus the elapsed/
        # damage costs already netted in) is scaled down when accuracy is low.
        # We scale only the positive clear component to avoid perversely
        # *reducing* the elapsed/damage penalties at low accuracy.
        if screens_cleared > 0:
            clear_component = (
                CLEAR_BONUS + MULTI_CLEAR_BONUS * screens_cleared * screens_cleared
            )
            fitness -= clear_component * (1.0 - clear_accuracy_gate)
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

        accuracy = float(total_hits / max(total_fired, 1))  # already gated above
        mean_hit_delta = float(np.mean(hit_delta_per_tick)) if hit_delta_per_tick else 0.0
        mean_hit_delta_norm = mean_hit_delta / HIT_DELTA_NORM_FRAMES
        mean_reaction_latency = float(np.mean(reaction_latencies)) if reaction_latencies else 0.0
        mean_reaction_latency_norm = mean_reaction_latency / REACTION_LATENCY_NORM_TICKS

        fitness += ACCURACY_BONUS_WEIGHT * accuracy
        fitness -= HIT_DELTA_PENALTY * mean_hit_delta_norm
        fitness -= REACTION_LATENCY_PENALTY * mean_reaction_latency_norm

        fitness += HIT_REWARD * total_hits
        # Per-miss penalty (see MISS_PENALTY in config.py): charge each wasted
        # shot so accuracy is directly selected for, countering the
        # spray-and-survive optimum where misses were free.
        fitness -= MISS_PENALTY * max(0, total_fired - total_hits)
        # Per-shot ammo cost (see AMMO_SHOT_COST in config.py): every round
        # fired costs fitness, hit or miss, so bullets are scarce and the agent
        # must value trigger discipline -- the mechanical complement to the
        # accuracy gate that caps the spray volume the RAM-ammo change enabled.
        fitness -= AMMO_SHOT_COST * total_fired
        fitness -= EXPOSED_NO_SHOT_PENALTY * no_shot_exposed_ticks
        fitness -= COVER_HESITATION_PENALTY * hesitated_cover_ticks
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
            "screens_cleared": screens_cleared,
            "timed_out": bool(timed_out),
            "dead": bool(dead),
            "elapsed": float(elapsed),
            "damage": float(total_life_loss),
            "accuracy": accuracy,
            "shots_fired": int(total_fired),
            "shots_hit": int(total_hits),
            "peek_flips": int(peek_flips),
            "peek_hold_score": float(hold_score),
            "cover_time": int(ticks_in_cover),
            "dry_fire_ticks": int(dry_fire_ticks),
            "no_shot_exposed_ticks": int(no_shot_exposed_ticks),
            "hesitated_cover_ticks": int(hesitated_cover_ticks),
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
            "mean_hit_delta": mean_hit_delta,
            "mean_hit_delta_norm": mean_hit_delta_norm,
            "mean_reaction_latency": mean_reaction_latency,
            "mean_reaction_latency_norm": mean_reaction_latency_norm,
        }
