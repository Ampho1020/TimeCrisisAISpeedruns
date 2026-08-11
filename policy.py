"""Tiny feed-forward policy stored as one flat parameter vector, plus an
open-loop per-tick action-schedule alternative (see config.POLICY_MODE)."""

import numpy as np

from config import ACT_DIM, HIDDEN, MAX_TICKS, NUM_ENEMY_CLASSES, OBS_DIM

PARAM_COUNT = OBS_DIM * HIDDEN + HIDDEN + HIDDEN * ACT_DIM + ACT_DIM

# Open-loop schedule: theta is a flat (MAX_TICKS, 4) per-tick action table
# instead of MLP weights -- one (shoot_logit, peek_logit, aim_x_bias,
# aim_y_bias) row per decision tick, indexed directly by the tick counter.
# Sim-validated in tests/test_simulation.py's ScheduleSearchSuite / repo
# memory "Open-loop schedule search (2026-08-09): POSITIVE result".
SCHEDULE_PARAM_COUNT = MAX_TICKS * 4

# Vision-conditioned schedule: per-tick 5-tuple table (shoot_logit,
# peek_logit, base_aim_x_bias, base_aim_y_bias, vision_gain_logit) PLUS a
# global NUM_ENEMY_CLASSES-vector at the end holding class priorities that
# feed a softmax over detector.EnemyClass. See act_vision_schedule for how
# these are consumed. This layout was chosen (over per-tick class priorities,
# which would add MAX_TICKS*NUM_ENEMY_CLASSES more params) because "which
# enemy type is the highest-value shot" is roughly stable across a savestate
# -- letting ES tune it once globally, rather than per tick, keeps the
# parameter count comparable to the plain schedule mode instead of ballooning
# it (see repo memory: shot-index-one-hot probe hit exactly this parameter-
# blowup problem).
VISION_SCHEDULE_ROW_DIM = 5
VISION_SCHEDULE_PARAM_COUNT = MAX_TICKS * VISION_SCHEDULE_ROW_DIM + NUM_ENEMY_CLASSES


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
      * first ``MAX_TICKS * 5`` entries: (shoot, peek, base_aim_x_bias,
        base_aim_y_bias, vision_gain_logit) rows indexed by ``tick``.
      * final ``NUM_ENEMY_CLASSES`` entries: raw class priority scores
        (softmax'd here before use).

    ``detections`` is a list of ``detector.Detection`` objects (may be
    empty). If empty, this returns the base action untouched -- so a
    zero-init ``vision_gain`` and ``class_priority`` reproduce plain
    schedule-mode behaviour exactly on generation 0.

    If detections are present, we score each one by
    ``confidence * softmax(class_priority)[class_id]``, pick the top
    scorer, and blend its normalized centroid into the base aim in [0,1]
    screen space:

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
    shoot = bool(row[0] > 0.0)
    peek = bool(row[1] > 0.0)
    base_ax_bias = float(np.tanh(row[2]))
    base_ay_bias = float(np.tanh(row[3]))
    gain = float(np.tanh(row[4]))

    if not detections:
        # Vision blend has no target this tick -- fall back to base aim.
        return shoot, peek, base_ax_bias, base_ay_bias

    priority_raw = theta[MAX_TICKS * VISION_SCHEDULE_ROW_DIM:
                         MAX_TICKS * VISION_SCHEDULE_ROW_DIM + NUM_ENEMY_CLASSES]
    priority = _softmax(np.asarray(priority_raw, dtype=np.float64))
    # Score each detection; ignore any with class_id outside the priority
    # range so a misconfigured detector can never index-out-of-range here
    # (a bad detection is treated the same as no detection).
    best_det = None
    best_score = -np.inf
    for det in detections:
        cid = int(det.class_id)
        if cid < 0 or cid >= NUM_ENEMY_CLASSES:
            continue
        score = float(det.confidence) * float(priority[cid])
        if score > best_score:
            best_score = score
            best_det = det
    if best_det is None:
        return shoot, peek, base_ax_bias, base_ay_bias

    # Blend in [0, 1] screen space, then convert back to the [-1, 1] bias
    # contract env.step consumes (env re-adds 0.5 and clips to [0, 1]).
    base_x_01 = min(1.0, max(0.0, 0.5 + base_ax_bias))
    base_y_01 = min(1.0, max(0.0, 0.5 + base_ay_bias))
    blended_x_01 = min(
        1.0, max(0.0, base_x_01 + gain * (float(best_det.cx_norm) - base_x_01)),
    )
    blended_y_01 = min(
        1.0, max(0.0, base_y_01 + gain * (float(best_det.cy_norm) - base_y_01)),
    )
    return shoot, peek, blended_x_01 - 0.5, blended_y_01 - 0.5
