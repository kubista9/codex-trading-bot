from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics for out-of-sample predictions."""
    true = pd.Series(y_true).astype(float)
    pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(true, pred))),
        "r2": float(r2_score(true, pred)),
    }


def classification_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Compute classification metrics for out-of-sample direction predictions."""
    true = pd.Series(y_true).astype(int)
    pred = np.asarray(y_pred, dtype=int)
    accuracy = float(accuracy_score(true, pred))
    return {
        "accuracy": accuracy,
        "hit_rate": accuracy,
    }
