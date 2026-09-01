"""Tiny feed-forward policy stored as one flat parameter vector, plus an
open-loop per-tick action-schedule alternative (see config.POLICY_MODE)."""

import numpy as np

from config import (
    ACT_DIM, HIDDEN, MAX_TICKS, NUM_ENEMY_CLASSES, OBS_DIM,
    SHOOT_DETECTION_SCALE,
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
VISION_SCHEDULE_ROW_DIM = 4
VISION_SCHEDULE_PARAM_COUNT = MAX_TICKS * VISION_SCHEDULE_ROW_DIM + NUM_ENEMY_CLASSES + 2
# Index of the single global vision_gain_logit scalar (second-to-last entry).
VISION_SCHEDULE_GAIN_IDX = VISION_SCHEDULE_PARAM_COUNT - 2
# Index of the single global shoot_gain_logit scalar -- the last entry.
SHOOT_GAIN_IDX = VISION_SCHEDULE_PARAM_COUNT - 1


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


def act_vision_schedule(theta: np.ndarray, tick: int, detections):
    """Vision-conditioned open-loop action selection.

    Theta layout (see ``VISION_SCHEDULE_PARAM_COUNT`` above):
      * first ``MAX_TICKS * 4`` entries: (shoot, peek, base_aim_x_bias,
        base_aim_y_bias) rows indexed by ``tick``.
      * next ``NUM_ENEMY_CLASSES`` entries: raw class priority scores
        (softmax'd here before use).
      * penultimate entry (``VISION_SCHEDULE_GAIN_IDX``): one shared
        ``vision_gain_logit`` scalar used for every tick, blending the
        detected centroid into the aim.
      * final entry (``SHOOT_GAIN_IDX``): one shared ``shoot_gain_logit``
        scalar, blending detection PRESENCE into the shoot decision.

    ``detections`` is a list of ``detector.Detection`` objects (may be
    empty). With zero-init ``vision_gain``/``shoot_gain``/``class_priority``
    this reproduces plain schedule-mode behaviour exactly on generation 0.

    If detections are present, we score each one by
    ``confidence * softmax(class_priority)[class_id]`` and pick the top
    scorer (``best_det``). That drives TWO independent blends:

    1. Shoot -- a +1/-1 nudge (present/absent) scaled by shoot_gain, added
       to the tick's base shoot_logit before the threshold:

           shoot_gain = tanh(shoot_gain_logit)                      # in [-1, 1]
           shoot = (base_shoot_logit + shoot_gain * (+1 if best_det else -1)) > 0

       This is what lets the trigger react to "is a target actually in
       view right now" instead of firing purely on the open-loop tick
       schedule (see the 2026-08-17 note above VISION_SCHEDULE_ROW_DIM).

    2. Aim -- best_det's normalized centroid blended into the base aim in
       [0,1] screen space:

        gain = tanh(vision_gain_logit)                 # in [-1, 1]
        base_x_01 = clip(0.5 + base_aim_x_bias, 0, 1)  # env's own decoder
        blended_x_01 = clip(base_x_01 + gain * (det.cx_norm - base_x_01), 0, 1)
        aim_x_bias = blended_x_01 - 0.5                # env re-adds 0.5

    Returning the biases in [-1, 1] keeps the contract identical to
    ``act()`` / ``act_schedule()`` so ``env_timecrisis.step`` doesn't need a
    special case for aim conversion. Negative gain lets ES also learn to
    avoid a detected point when that class has low task priority.

    Returns (shoot: bool, peek: bool, aim_x_bias: float in [-1, 1], aim_y_bias: float in [-1, 1])
    """
    idx = min(int(tick), MAX_TICKS - 1)
    row_start = idx * VISION_SCHEDULE_ROW_DIM
    row = theta[row_start:row_start + VISION_SCHEDULE_ROW_DIM]
    base_shoot_logit = float(row[0])
    peek = bool(row[1] > 0.0)
    base_ax_bias = float(np.tanh(row[2]))
    base_ay_bias = float(np.tanh(row[3]))
    gain = float(np.tanh(theta[VISION_SCHEDULE_GAIN_IDX]))
    shoot_gain = float(np.tanh(theta[SHOOT_GAIN_IDX]))

    # Score every detection (if any) up front -- best_det feeds BOTH the
    # shoot decision below and the aim blend, so "no valid target" is
    # handled identically (best_det is None) whether detections was empty
    # or every entry had an out-of-range class_id.
    best_det = None
    best_conf = 0.0
    if detections:
        priority_start = MAX_TICKS * VISION_SCHEDULE_ROW_DIM
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

    # Shoot: base open-loop logit plus a confidence-shaped additive nudge.
    # Safety behavior for existing checkpoints:
    #   * no detection -> 0.0 (preserve baseline schedule firing)
    #   * detection    -> [0, 1] additive encouragement only
    # This avoids the failure mode where low/no detections suppress trigger
    # output below the old baseline and the policy appears to stop firing.
    detection_term = 0.0
    if best_det is not None:
        detection_term = float(np.clip(best_conf, 0.0, 1.0))
    shoot = bool(
        base_shoot_logit + SHOOT_DETECTION_SCALE * shoot_gain * detection_term > 0.0
    )

    if best_det is None:
        # No usable target this tick -- fall back to base aim.
        return shoot, peek, base_ax_bias, base_ay_bias

    # Blend in [0, 1] screen space, then convert back to the [-1, 1] bias
    # contract env.step consumes (env re-adds 0.5 and clips to [0, 1]).
    base_x_01 = min(1.0, max(0.0, 0.5 + base_ax_bias))
    base_y_01 = min(1.0, max(0.0, 0.5 + base_ay_bias))
    target_x_norm = float(getattr(best_det, "aim_x_norm", best_det.cx_norm))
    target_y_norm = float(getattr(best_det, "aim_y_norm", best_det.cy_norm))
    blended_x_01 = min(
        1.0, max(0.0, base_x_01 + gain * (target_x_norm - base_x_01)),
    )
    blended_y_01 = min(
        1.0, max(0.0, base_y_01 + gain * (target_y_norm - base_y_01)),
    )
    return shoot, peek, blended_x_01 - 0.5, blended_y_01 - 0.5
