"""Tiny feed-forward policy stored as one flat parameter vector."""

import numpy as np

from config import OBS_DIM, HIDDEN, ACT_DIM

PARAM_COUNT = OBS_DIM * HIDDEN + HIDDEN + HIDDEN * ACT_DIM + ACT_DIM


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
