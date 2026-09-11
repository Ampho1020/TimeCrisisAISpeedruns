"""Tiny feed-forward policy stored as one flat parameter vector, plus an
open-loop per-tick action-schedule alternative (see config.POLICY_MODE)."""

import numpy as np

from config import (
    ACT_DIM, AMMO_CONSERVE_SCALE, HIDDEN, MAX_TICKS, NUM_ENEMY_CLASSES, OBS_DIM,
    PEEK_DETECTION_SCALE,
    SCHEDULE_BLOCK_TICKS,
    SHOOT_DETECTION_SCALE,
    VISION_DRIFT_EDGE_START,
    VISION_FORCE_SHOOT_CONFIDENCE,
    VISION_MIN_BLEND_GAIN,
)

PARAM_COUNT = OBS_DIM * HIDDEN + HIDDEN + HIDDEN * ACT_DIM + ACT_DIM

# Open-loop schedule: theta is a flat (MAX_TICKS, 4) per-tick action table
# instead of MLP weights -- one (shoot_logit, peek_logit, aim_x_bias,
# aim_y_bias) row per decision tick, indexed directly by the tick counter.
# Sim-validated in tests/test_simulation.py's ScheduleSearchSuite / repo
# memory "Open-loop schedule search (2026-08-09): POSITIVE result".
SCHEDULE_PARAM_COUNT = MAX_TICKS * 4

# Vision-conditioned schedule: per-tick 4-tuple table (shoot_logit,
# peek_logit, base_aim_x_bias, base_aim_y_bias) PLUS a global
# NUM_ENEMY_CLASSES-vector holding class priorities that feed a softmax over
# detector.EnemyClass, PLUS one single global vision_gain_logit scalar at
# the very end. See act_vision_schedule for how these are consumed.
#
# vision_gain used to be a 5th per-tick column (one independent value per
# of the MAX_TICKS rows). That was changed to a single shared scalar
# (2026-08-15) because "how much do I trust vision over my base aim" is a
# stable question across the whole episode -- unlike aim position, which
# legitimately differs tick to tick since enemies differ tick to tick.
# Splitting it 900 ways meant each generation's gradient estimate for
# vision_gain was diluted across 900 nearly-independent parameters instead
# of pooling all 900 ticks' fitness signal into one number -- confirmed
# empirically: population MEAN vision_gain sat flat for 20 generations
# while the per-tick values underneath had actually moved substantially
# (many rows shifted by >0.01, std growing 5x), because roughly half moved
# up and half moved down and cancelled out in the aggregate. A single
# shared scalar makes that signal directly visible and should converge far
# faster. Class priorities stay global for the same "roughly stable across
# a savestate" reason (see repo memory: shot-index-one-hot probe hit the
# analogous parameter-blowup problem when tried per-tick).
#
# NOTE: this is a breaking layout change -- existing theta_*.npy checkpoints
# saved under the old 5-column-per-tick layout are NOT compatible and must
# be discarded/retrained.
# 2026-08-17: added a second global scalar, shoot_gain_logit, appended
# after vision_gain_logit. Root cause: shoot/peek are read purely from the
# fixed per-tick row -- an open-loop timing table completely uninformed by
# whether a target is actually visible right now. Once per_frame_vision
# (env_timecrisis.py) made AIM track a fresh detection every single raw
# frame, the mismatch became obvious/reported live: the crosshair snaps
# onto an enemy almost instantly, but the trigger only fires whenever that
# tick's fixed shoot_logit happens to be positive -- which can be many
# ticks (observed: multiple seconds) away from the moment a target became
# visible. shoot_gain lets ES learn how much a CURRENT detection's presence
# should nudge the trigger, on top of the open-loop baseline -- see
# act_vision_schedule for the exact blend. With shoot_gain=0 this collapses
# byte-identical to the old open-loop-only shoot decision.
#
# NOTE: this is a breaking layout change (PARAM_COUNT grew by 1) --
# existing theta_*.npy checkpoints are NOT compatible and must be
# discarded/retrained.
#
# 2026-09-09: added a fourth global scalar, peek_gain_logit, appended after
# drift_gain_logit. Root cause: peek (cover <-> exposed) was read PURELY
# from the fixed per-tick row, completely blind to detections -- unlike
# shoot, which already blends detection presence via shoot_gain and can
# even force-fire on a confident detection (VISION_FORCE_SHOOT_CONFIDENCE).
# That force-shoot override was silently a no-op whenever the open-loop
# schedule's peek for that tick happened to be False, since
# env_timecrisis.py gates the trigger with shoot_allowed = peek -- the
# agent could be staring at a clean, high-confidence target and still not
# fire, purely because the absolute-tick schedule hadn't scheduled an
# exposure window at that moment. peek_gain lets ES learn how much a
# CURRENT detection's presence should nudge the agent to come OUT of cover,
# on top of the open-loop baseline, and reuses the same
# VISION_FORCE_SHOOT_CONFIDENCE bar to force peek=True alongside shoot=True
# (see act_vision_schedule). With peek_gain=0 this collapses byte-identical
# to the old open-loop-only peek decision. ammo_left==0 still hard-forces
# peek=False afterward in env_timecrisis.py, so this cannot make the agent
# expose with an empty clip.
#
# NOTE: this is a breaking layout change (PARAM_COUNT grew by 1 again) --
# existing theta_*.npy checkpoints are NOT compatible and must be
# discarded/retrained.
#
# 2026-09-09 (later same day): split the old 4-column-per-tick row into TWO
# separately-resolved tables (recommendation #2 from the peek_gain
# follow-up). Root cause: now that shoot/peek are BOTH reactive to live
# detections (shoot_gain/peek_gain above), the per-tick shoot_logit/
# peek_logit columns mostly just need to supply a sane DEFAULT for "nothing
# visible yet" -- they no longer need to independently re-derive 900
# separate exposure/firing windows from scratch, which was a needlessly
# large, sparse search space for ES (POP_SIZE=30) to optimize. aim_x/aim_y
# bias genuinely DO need per-tick resolution (enemy position legitimately
# differs tick to tick), so those stay as-is. shoot_logit/peek_logit now
# share ONE value per SCHEDULE_BLOCK_TICKS-tick block instead of one value
# per tick -- at the default SCHEDULE_BLOCK_TICKS=30 that's 30 blocks
# instead of 900 rows for those two columns, shrinking the table from
# MAX_TICKS*4=3600 to MAX_TICKS*2 + NUM_SCHEDULE_BLOCKS*2 = 1860 (~48% of
# the old size) while leaving aim's per-tick granularity untouched.
#
# NOTE: this is a breaking layout change (the per-tick row shrank from 4
# columns to 2, and a new block table was inserted) -- existing
# theta_*.npy checkpoints are NOT compatible and must be discarded/
# retrained.
VISION_SCHEDULE_AIM_DIM = 2      # per-tick: (base_aim_x_bias, base_aim_y_bias)
VISION_SCHEDULE_BLOCK_DIM = 2    # per-block: (shoot_logit, peek_logit)
NUM_SCHEDULE_BLOCKS = (MAX_TICKS + SCHEDULE_BLOCK_TICKS - 1) // SCHEDULE_BLOCK_TICKS
VISION_SCHEDULE_AIM_TABLE_SIZE = MAX_TICKS * VISION_SCHEDULE_AIM_DIM
VISION_SCHEDULE_BLOCK_TABLE_SIZE = NUM_SCHEDULE_BLOCKS * VISION_SCHEDULE_BLOCK_DIM
# 2026-09-10: added a fifth global scalar, ammo_gain_logit, appended after
# peek_gain_logit -- see the "ammo-awareness" block in
# act_vision_schedule's docstring below. Zero-init-safe (gain=0 reproduces
# old behaviour byte-identically), but this is still a breaking layout
# change (PARAM_COUNT grew by 1) -- existing theta_*.npy checkpoints are
# NOT compatible and must be discarded/retrained.
#
# 2026-09-11: a SIXTH scalar, switch_gain_logit, was added alongside
# ammo_gain to bias target selection toward the runner-up detection
# instead of the best one ("ammo-aware target allocation"). It was
# reverted the same day after a live 80-gen retrain showed a clear
# regression (clear_rate/mean_acc/best fitness all noticeably worse than
# the pre-change baseline) -- root cause: the runner-up detection is not
# restricted to the same EnemyClass as best_det, so the switch could (and
# did) redirect aim onto a GRENADE/PROJECTILE detection instead of another
# ENEMY, which is not a sane "spread ammo across enemies" behaviour. A
# fixed version (same-class-only switching) may be revisited later; for
# now only ammo_gain (shoot suppression/encouragement from ammo scarcity,
# which showed no such flaw) remains.
VISION_SCHEDULE_PARAM_COUNT = (
    VISION_SCHEDULE_AIM_TABLE_SIZE + VISION_SCHEDULE_BLOCK_TABLE_SIZE
    + NUM_ENEMY_CLASSES + 5
)
# Indexes of global scalars in tail order:
# vision_gain, shoot_gain, drift_gain, peek_gain, ammo_gain.
VISION_SCHEDULE_GAIN_IDX = VISION_SCHEDULE_PARAM_COUNT - 5
SHOOT_GAIN_IDX = VISION_SCHEDULE_PARAM_COUNT - 4
DRIFT_GAIN_IDX = VISION_SCHEDULE_PARAM_COUNT - 3
PEEK_GAIN_IDX = VISION_SCHEDULE_PARAM_COUNT - 2
AMMO_GAIN_IDX = VISION_SCHEDULE_PARAM_COUNT - 1


def _unpack(theta: np.ndarray):
    i = 0
    w1 = theta[i:i + OBS_DIM * HIDDEN].reshape(OBS_DIM, HIDDEN); i += OBS_DIM * HIDDEN
    b1 = theta[i:i + HIDDEN];                                    i += HIDDEN
    w2 = theta[i:i + HIDDEN * ACT_DIM].reshape(HIDDEN, ACT_DIM); i += HIDDEN * ACT_DIM
    b2 = theta[i:i + ACT_DIM]
    return w1, b1, w2, b2


def act(theta: np.ndarray, obs: np.ndarray):
    """
    Deterministic action selection -- ES explores via weight noise,
    not action noise, so argmax/threshold is correct here.

    Returns (shoot: bool, peek: bool, aim_x_bias: float in [-1, 1], aim_y_bias: float in [-1, 1])
    """
    w1, b1, w2, b2 = _unpack(theta)
    h = np.tanh(obs @ w1 + b1)
    out = h @ w2 + b2
    return bool(out[0] > 0.0), bool(out[1] > 0.0), float(np.tanh(out[2])), float(np.tanh(out[3]))


def act_schedule(theta: np.ndarray, tick: int):
    """Open-loop action selection: read the fixed action row for `tick`.

    `theta` is reshaped as (MAX_TICKS, 4); ticks past MAX_TICKS clip to the
    last row. Uses the SAME decode as `act()` (threshold 0.0 for the bools,
    tanh for the aim biases) so behaviour/units match the closed-loop policy
    exactly -- only the SOURCE of the action differs (indexed by tick, not
    computed from an observation).

    Returns (shoot: bool, peek: bool, aim_x_bias: float in [-1, 1], aim_y_bias: float in [-1, 1])
    """
    idx = min(int(tick), MAX_TICKS - 1)
    row = theta[idx * 4:idx * 4 + 4]
    return bool(row[0] > 0.0), bool(row[1] > 0.0), float(np.tanh(row[2])), float(np.tanh(row[3]))


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1D float array. Used to turn the
    raw class-priority parameters into a normalized weighting -- keeps
    ES-perturbed priorities in a well-defined [0, 1] simplex without an
    extra normalization term in the fitness formula."""
    shifted = x - float(np.max(x))
    exp = np.exp(shifted)
    total = float(exp.sum())
    if total <= 0.0:
        # Degenerate but recoverable -- return uniform (matches all-zeros
        # warm-start behaviour, i.e. every detected class is equally
        # prioritized).
        return np.full_like(x, 1.0 / len(x))
    return exp / total


def act_vision_schedule(
    theta: np.ndarray,
    tick: int,
    detections,
    cursor_x_norm: float | None = None,
    cursor_y_norm: float | None = None,
    ammo_left_norm: float | None = None,
):
    """Vision-conditioned open-loop action selection.

    Theta layout (see ``VISION_SCHEDULE_PARAM_COUNT`` above):
      * first ``VISION_SCHEDULE_AIM_TABLE_SIZE`` entries: (base_aim_x_bias,
        base_aim_y_bias) rows indexed by ``tick`` directly -- one row per
        tick, since enemy position genuinely varies tick to tick.
      * next ``VISION_SCHEDULE_BLOCK_TABLE_SIZE`` entries: (shoot_logit,
        peek_logit) rows indexed by ``tick // SCHEDULE_BLOCK_TICKS`` --
        ONE shared row per block of ``SCHEDULE_BLOCK_TICKS`` ticks instead
        of per-tick (added 2026-09-09 as recommendation #2 of the
        peek_gain follow-up: now that shoot/peek both react to live
        detections via shoot_gain/peek_gain, the open-loop columns mostly
        just need to supply a sane default for "nothing visible yet", so a
        much coarser table is enough and shrinks ES's search space).
      * next ``NUM_ENEMY_CLASSES`` entries: raw class priority scores
        (softmax'd here before use).
            * ``VISION_SCHEDULE_GAIN_IDX``: one shared ``vision_gain_logit``
                scalar used for every tick, blending the detected centroid
                into the aim.
            * ``SHOOT_GAIN_IDX``: one shared ``shoot_gain_logit`` scalar,
                blending detection PRESENCE into the shoot decision.
            * ``DRIFT_GAIN_IDX``: one shared ``drift_gain_logit`` scalar used
                to learn edge drift correction from live cursor error.
            * ``PEEK_GAIN_IDX``: one shared ``peek_gain_logit`` scalar,
                blending detection PRESENCE into the peek (exposure) decision
                -- same idea as shoot_gain, but for whether to come OUT of
                cover at all.
            * ``AMMO_GAIN_IDX``: one shared ``ammo_gain_logit`` scalar,
                letting ammo scarcity suppress (or encourage) firing.

    ``detections`` is a list of ``detector.Detection`` objects (may be
    empty). With zero-init ``vision_gain``/``shoot_gain``/``peek_gain``/
    ``ammo_gain``/``class_priority`` this reproduces plain schedule-mode
    behaviour exactly on generation 0.

    If detections are present, we score each one by
    ``confidence * softmax(class_priority)[class_id]`` and rank them --
    ``best_det`` is the top scorer. That drives FOUR independent blends:

    1. Shoot -- a +1/-1 nudge (present/absent) scaled by shoot_gain, added
       to the tick's base shoot_logit before the threshold:

           shoot_gain = tanh(shoot_gain_logit)                      # in [-1, 1]
           shoot = (base_shoot_logit + shoot_gain * (+1 if best_det else -1)) > 0

       This is what lets the trigger react to "is a target actually in
       view right now" instead of firing purely on the open-loop tick
       schedule (see the 2026-08-17 note above VISION_SCHEDULE_BLOCK_DIM).
       On top of this, an ammo-conservation term (see 4. below) can
       additionally suppress or encourage firing based on ammo scarcity.

    2. Peek -- the SAME shape of nudge, scaled by peek_gain, added to the
       tick's base peek_logit before the threshold:

           peek_gain = tanh(peek_gain_logit)                        # in [-1, 1]
           peek = (base_peek_logit + peek_gain * (+1 if best_det else -1)) > 0

       This lets the agent learn to come OUT of cover when a target is
       actually visible instead of only exposing on the open-loop tick
       schedule (see the 2026-09-09 note above VISION_SCHEDULE_BLOCK_DIM).
       Without this, a confident detection could force shoot=True while
       peek stayed False that tick, and env_timecrisis.py's
       ``shoot_allowed = peek`` gate would silently swallow the shot.

    3. Aim -- best_det's normalized centroid blended into the base aim in
       [0,1] screen space:

        gain = tanh(vision_gain_logit)                 # in [-1, 1]
        base_x_01 = clip(0.5 + base_aim_x_bias, 0, 1)  # env's own decoder
        blended_x_01 = clip(base_x_01 + gain * (det.cx_norm - base_x_01), 0, 1)
        aim_x_bias = blended_x_01 - 0.5                # env re-adds 0.5

    4. Ammo-awareness (added 2026-09-10) -- lets the agent ration its
       magazine based on ammo scarcity instead of always firing at the
       same rate regardless of rounds remaining:

           ammo_gain = tanh(ammo_gain_logit)                        # in [-1, 1]
           ammo_scarcity = 1.0 - ammo_left_norm                     # in [0, 1], 0 = full clip
           shoot_logit += AMMO_CONSERVE_SCALE * ammo_gain * ammo_scarcity

       lets ES learn to hold fire (negative ammo_gain) as the clip empties,
       reserving rounds for a better shot, or -- if ES finds it more
       effective -- fire more freely as ammo drops (positive ammo_gain,
       "use it or lose it" before a reload). Zero when ``ammo_left_norm``
       is not supplied (e.g. older call sites/tests), matching the
       zero-init safety pattern used by every other gain here.

       NOTE: a companion ``switch_gain`` scalar (bias target selection
       toward the runner-up detection when ammo is scarce) was added
       alongside this on 2026-09-10 and reverted on 2026-09-11 after a
       live retrain showed a regression -- the runner-up detection wasn't
       restricted to the same EnemyClass as best_det, so it could (and
       did) redirect aim onto a GRENADE/PROJECTILE instead of another
       ENEMY. See policy.py git history / repo memory if revisiting this.

    Returning the biases in [-1, 1] keeps the contract identical to
    ``act()`` / ``act_schedule()`` so ``env_timecrisis.step`` doesn't need a
    special case for aim conversion. Negative gain lets ES also learn to
    avoid a detected point when that class has low task priority.

    Returns (shoot: bool, peek: bool, aim_x_bias: float in [-1, 1], aim_y_bias: float in [-1, 1])
    """
    idx = min(int(tick), MAX_TICKS - 1)
    aim_row_start = idx * VISION_SCHEDULE_AIM_DIM
    aim_row = theta[aim_row_start:aim_row_start + VISION_SCHEDULE_AIM_DIM]
    base_ax_bias = float(np.tanh(aim_row[0]))
    base_ay_bias = float(np.tanh(aim_row[1]))
    block_idx = idx // SCHEDULE_BLOCK_TICKS
    block_row_start = VISION_SCHEDULE_AIM_TABLE_SIZE + block_idx * VISION_SCHEDULE_BLOCK_DIM
    block_row = theta[block_row_start:block_row_start + VISION_SCHEDULE_BLOCK_DIM]
    base_shoot_logit = float(block_row[0])
    base_peek_logit = float(block_row[1])
    gain = float(np.tanh(theta[VISION_SCHEDULE_GAIN_IDX]))
    shoot_gain = float(np.tanh(theta[SHOOT_GAIN_IDX]))
    drift_gain = float(np.tanh(theta[DRIFT_GAIN_IDX]))
    peek_gain = float(np.tanh(theta[PEEK_GAIN_IDX]))
    ammo_gain = float(np.tanh(theta[AMMO_GAIN_IDX]))

    # Score every detection (if any) up front -- best_det feeds BOTH the
    # shoot decision below and the aim blend, so "no valid target" is
    # handled identically (best_det is None) whether detections was empty
    # or every entry had an out-of-range class_id.
    best_det = None
    best_conf = 0.0
    if detections:
        priority_start = VISION_SCHEDULE_AIM_TABLE_SIZE + VISION_SCHEDULE_BLOCK_TABLE_SIZE
        priority_raw = theta[priority_start:priority_start + NUM_ENEMY_CLASSES]
        priority = _softmax(np.asarray(priority_raw, dtype=np.float64))
        best_score = -np.inf
        for det in detections:
            cid = int(det.class_id)
            if cid < 0 or cid >= NUM_ENEMY_CLASSES:
                continue
            score = float(det.confidence) * float(priority[cid])
            if score > best_score:
                best_score = score
                best_det = det
                best_conf = float(det.confidence)

    ammo_scarcity = 0.0
    if ammo_left_norm is not None:
        ammo_scarcity = float(np.clip(1.0 - ammo_left_norm, 0.0, 1.0))

    # Shoot: base open-loop logit plus a confidence-shaped additive nudge.
    # Safety behavior for existing checkpoints:
    #   * no detection -> 0.0 (preserve baseline schedule firing)
    #   * detection    -> [0, 1] additive encouragement only
    # This avoids the failure mode where low/no detections suppress trigger
    # output below the old baseline and the policy appears to stop firing.
    detection_term = 0.0
    if best_det is not None:
        detection_term = float(np.clip(best_conf, 0.0, 1.0))
    shoot_logit = (
        base_shoot_logit
        + SHOOT_DETECTION_SCALE * shoot_gain * detection_term
        + AMMO_CONSERVE_SCALE * ammo_gain * ammo_scarcity
    )
    shoot = bool(shoot_logit > 0.0)

    # Peek: same shape of confidence-shaped additive nudge as shoot above,
    # so a visible target can pull the agent OUT of cover instead of
    # waiting for the open-loop schedule's fixed exposure window. Shares
    # the same force-override bar as shoot -- if we're confident enough to
    # force-fire, we're confident enough to force-expose (ammo_left == 0
    # still hard-overrides peek back to False afterward, in
    # env_timecrisis.py).
    peek_logit = base_peek_logit + PEEK_DETECTION_SCALE * peek_gain * detection_term
    peek = bool(peek_logit > 0.0)
    if best_det is not None and best_conf >= VISION_FORCE_SHOOT_CONFIDENCE:
        shoot = True
        peek = True

    if best_det is None:
        # No usable target this tick -- fall back to base aim.
        return shoot, peek, base_ax_bias, base_ay_bias

    # Blend in [0, 1] screen space, then convert back to the [-1, 1] bias
    # contract env.step consumes (env re-adds 0.5 and clips to [0, 1]).
    base_x_01 = min(1.0, max(0.0, 0.5 + base_ax_bias))
    base_y_01 = min(1.0, max(0.0, 0.5 + base_ay_bias))
    target_x_norm = float(getattr(best_det, "aim_x_norm", best_det.cx_norm))
    target_y_norm = float(getattr(best_det, "aim_y_norm", best_det.cy_norm))

    if cursor_x_norm is not None and cursor_y_norm is not None:
        edge_mag_x = max(0.0, abs(target_x_norm - 0.5) * 2.0 - VISION_DRIFT_EDGE_START)
        edge_gain_x = min(1.0, edge_mag_x / max(1e-6, 1.0 - VISION_DRIFT_EDGE_START))
        target_x_norm = float(np.clip(
            target_x_norm + drift_gain * edge_gain_x * (target_x_norm - float(cursor_x_norm)),
            0.0,
            1.0,
        ))

    blend_gain = gain
    if best_conf >= VISION_FORCE_SHOOT_CONFIDENCE:
        blend_gain = max(blend_gain, VISION_MIN_BLEND_GAIN)
    blended_x_01 = min(
        1.0, max(0.0, base_x_01 + blend_gain * (target_x_norm - base_x_01)),
    )
    blended_y_01 = min(
        1.0, max(0.0, base_y_01 + blend_gain * (target_y_norm - base_y_01)),
    )
    return shoot, peek, blended_x_01 - 0.5, blended_y_01 - 0.5
