import numpy as np
from sklearn.metrics import (
    log_loss,
    brier_score_loss,
    roc_auc_score,
)


def ece(y, p, bins=10):
    y = np.asarray(y)
    p = np.asarray(p)

    idx = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)) - 1, 0, bins - 1)

    total = 0.0

    for b in range(bins):
        mask = idx == b

        if mask.any():
            total += mask.mean() * abs(y[mask].mean() - p[mask].mean())

    return total


def evaluate(y, p):
    y = np.asarray(y)
    p = np.asarray(p)

    if y.ndim != 1 or p.ndim != 1:
        raise ValueError("y and p must be 1-dimensional")

    if len(y) != len(p):
        raise ValueError("y and p must have the same length")

    if len(y) == 0:
        raise ValueError("y and p cannot be empty")

    if not np.isin(y, [0, 1]).all():
        raise ValueError("y must contain only 0 and 1")

    if not np.isfinite(p).all():
        raise ValueError("p must contain only finite values")

    if ((p < 0) | (p > 1)).any():
        raise ValueError("p must contain probabilities between 0 and 1")

    p = np.clip(p, 1e-6, 1 - 1e-6)

    return {
        "logloss": log_loss(y, p),
        "brier": brier_score_loss(y, p),
        "auc": roc_auc_score(y, p),
        "ece": ece(y, p),
    }
