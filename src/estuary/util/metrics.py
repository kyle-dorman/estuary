import numpy as np


def index_of_agreement(obs, pred):
    """
    Willmott (1981) Index of Agreement (IOA)
    obs: array-like of observed values
    pred: array-like of predicted values
    Returns float in [0, 1]
    """
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)

    o_mean = np.mean(obs)

    numerator = np.sum((pred - obs) ** 2)
    denominator = np.sum((np.abs(pred - o_mean) + np.abs(obs - o_mean)) ** 2)

    # If denominator is zero (e.g., constant obs), define IOA = 1 if perfect match, else 0
    if denominator == 0:
        return 1.0 if np.allclose(obs, pred) else 0.0

    return 1 - numerator / denominator


def ioa_modified(obs, pred):
    """
    Modified Index of Agreement (Willmott et al. 2012).
    Less sensitive to extreme values; recommended for binary or bounded data.

    Returns float in [0, 1].
    """
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)

    o_mean = np.mean(obs)

    numerator = np.sum(np.abs(pred - obs))
    denominator = np.sum(np.abs(pred - o_mean) + np.abs(obs - o_mean))

    if denominator == 0:
        return 1.0 if np.allclose(obs, pred) else 0.0

    return 1 - numerator / denominator
