from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalHoldout:
    """Train/test frames split only by time."""

    train: pd.DataFrame
    test: pd.DataFrame
    cutoff_date: pd.Timestamp


def make_temporal_holdout(
    frame: pd.DataFrame,
    *,
    date_col: str = "as_of_date",
    target_col: str,
    test_size_days: int = 252,
    min_train_days: int = 252,
) -> TemporalHoldout:
    """Create a time-ordered train/test split with no random shuffling."""
    if date_col not in frame.columns or target_col not in frame.columns:
        msg = f"Frame must include {date_col} and {target_col}"
        raise ValueError(msg)

    data = frame.loc[frame[target_col].notna()].copy()
    if data.empty:
        msg = f"No rows with non-null target {target_col}"
        raise ValueError(msg)

    data[date_col] = pd.to_datetime(data[date_col]).dt.normalize()
    data = data.sort_values([date_col, "ticker" if "ticker" in data.columns else date_col])
    dates = pd.Index(data[date_col].drop_duplicates().sort_values())
    if len(dates) < 3:
        msg = "At least three target dates are required for temporal holdout"
        raise ValueError(msg)

    max_test_days = max(1, len(dates) - min(min_train_days, len(dates) - 1))
    effective_test_size = min(test_size_days, max_test_days)
    cutoff_date = pd.Timestamp(dates[-effective_test_size])

    train = data[data[date_col] < cutoff_date].copy()
    test = data[data[date_col] >= cutoff_date].copy()
    if train.empty or test.empty:
        msg = "Temporal split produced an empty train or test set"
        raise ValueError(msg)

    return TemporalHoldout(train=train, test=test, cutoff_date=cutoff_date)
