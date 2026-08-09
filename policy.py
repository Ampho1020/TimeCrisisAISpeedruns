"""Tiny feed-forward policy stored as one flat parameter vector, plus an
open-loop per-tick action-schedule alternative (see config.POLICY_MODE)."""

import numpy as np

from config import ACT_DIM, HIDDEN, MAX_TICKS, OBS_DIM

PARAM_COUNT = OBS_DIM * HIDDEN + HIDDEN + HIDDEN * ACT_DIM + ACT_DIM

# Open-loop schedule: theta is a flat (MAX_TICKS, 4) per-tick action table
# instead of MLP weights -- one (shoot_logit, peek_logit, aim_x_bias,
# aim_y_bias) row per decision tick, indexed directly by the tick counter.
# Sim-validated in tests/test_simulation.py's ScheduleSearchSuite / repo
# memory "Open-loop schedule search (2026-08-09): POSITIVE result".
SCHEDULE_PARAM_COUNT = MAX_TICKS * 4


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
