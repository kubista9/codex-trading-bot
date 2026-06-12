from __future__ import annotations

import pandas as pd


def signals_from_expected_returns(
    predictions: pd.DataFrame,
    *,
    prediction_col: str = "prediction",
    buy_threshold: float = 0.02,
    sell_threshold: float = -0.02,
) -> pd.DataFrame:
    """Convert expected-return predictions into buy/hold/sell signals."""
    if buy_threshold <= sell_threshold:
        msg = "buy_threshold must be greater than sell_threshold"
        raise ValueError(msg)
    if prediction_col not in predictions.columns:
        msg = f"Predictions missing {prediction_col}"
        raise ValueError(msg)

    frame = predictions.copy()
    frame["signal"] = "hold"
    frame.loc[frame[prediction_col] >= buy_threshold, "signal"] = "buy"
    frame.loc[frame[prediction_col] <= sell_threshold, "signal"] = "sell"
    return frame
