"""Internal evaluation metrics for model selection.

The official challenge metric is macro-averaged ST-RAE (soft-threshold relative
absolute error), computed server-side in a Lambda we cannot run locally. Its
defining property is public: error is measured as distance to the experimental
confidence-interval bounds — a prediction landing inside a compound's CI incurs
ZERO penalty — and low-activity compounds (pIC50 < 4) are downweighted.

We reproduce that *shape* here for ranking models internally. Absolute values will
not match the leaderboard; only relative ordering matters for selection.
"""

from __future__ import annotations

import numpy as np

from .vendored import official_scoring


def soft_threshold_abs_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    conf_low: np.ndarray | None = None,
    conf_high: np.ndarray | None = None,
) -> np.ndarray:
    """Per-sample |error| with the region inside [conf_low, conf_high] set to zero.

    If a prediction falls inside the CI, error is 0; otherwise it is the distance
    to the nearer CI bound. Falls back to plain |y_pred - y_true| where CI bounds
    are missing.
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    if conf_low is None or conf_high is None:
        return np.abs(y_pred - y_true)
    lo = np.asarray(conf_low, float)
    hi = np.asarray(conf_high, float)
    # where CI is missing, treat the bound as the point estimate (→ plain abs error)
    lo = np.where(np.isnan(lo), y_true, lo)
    hi = np.where(np.isnan(hi), y_true, hi)
    below = np.clip(lo - y_pred, 0, None)   # prediction under the lower bound
    above = np.clip(y_pred - hi, 0, None)   # prediction over the upper bound
    return below + above


def st_rae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    conf_low: np.ndarray | None = None,
    conf_high: np.ndarray | None = None,
) -> float:
    """Soft-Threshold Relative Absolute Error — delegates to the OFFICIAL scorer
    vendored from OpenADMET/CYP-Challenge-Tutorial (ported from the challenge
    backend; matches the leaderboard).

    CRITICAL: the denominator is the mean-baseline error put through the SAME
    soft-thresholding, NOT the plain |y - mean| deviation. ST-RAE = 1.0 means "as
    good as predicting the global mean"; > 1 means WORSE than the mean. An earlier
    hand-reconstruction here used a plain-deviation denominator, which made internal
    CV scores optimistic by a large, systematic factor.
    """
    return official_scoring.rae_soft_threshold_absolute_error(
        np.asarray(y_true, float), np.asarray(y_pred, float),
        y_true_upper=None if conf_high is None else np.asarray(conf_high, float),
        y_true_lower=None if conf_low is None else np.asarray(conf_low, float),
    )


def constant_baseline_st_rae(
    y_true: np.ndarray,
    conf_low: np.ndarray | None = None,
    conf_high: np.ndarray | None = None,
) -> float:
    """ST-RAE of the best single-constant predictor (the mean of y_true) — the
    'no-skill' floor. On the real leaderboard a no-skill entry scored ~0.49, so this
    quantifies how much of our score is free CI-landing vs. genuine predictive skill."""
    y_true = np.asarray(y_true, float)
    c = np.full_like(y_true, y_true.mean())
    return st_rae(y_true, c, conf_low, conf_high)


def low_activity_weights(y_true: np.ndarray, threshold: float = 4.0,
                         floor: float = 0.25) -> np.ndarray:
    """Optional TRAINING sample weights that de-emphasize low-activity compounds
    (not used in the metric, which handles this via wide CIs). Linear ramp from
    `floor` at pIC50=0 to 1.0 at `threshold`, flat above."""
    y = np.asarray(y_true, float)
    return floor + (1.0 - floor) * np.clip(y / threshold, 0, 1)


def report(y_true, y_pred, conf_low=None, conf_high=None) -> dict[str, float]:
    """Bundle of internal metrics for a single isoform."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return {
        "n": int(len(y_true)),
        "mae": float(np.mean(np.abs(y_pred - y_true))),
        "rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "st_rae": st_rae(y_true, y_pred, conf_low, conf_high),
        "const_st_rae": constant_baseline_st_rae(y_true, conf_low, conf_high),
    }
