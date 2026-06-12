from __future__ import annotations

import pandas as pd


def target_columns_for_horizons(horizons: tuple[int, ...] | list[int]) -> list[str]:
    """Return regression, classification, and signal target columns for horizons."""
    columns: list[str] = []
    for horizon in horizons:
        columns.extend(
            [
                f"forward_return_{int(horizon)}d",
                f"direction_{int(horizon)}d",
                f"signal_target_{int(horizon)}d",
            ]
        )
    return columns


def make_complete_modeling_frame(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    horizons: tuple[int, ...] | list[int],
    context_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Build a training/export frame with no missing model features or targets."""
    context = context_columns or ["as_of_date", "ticker", "adj_close", "volume"]
    target_columns = target_columns_for_horizons(horizons)
    required_columns = list(dict.fromkeys([*context, *feature_columns, *target_columns]))
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        msg = f"Cannot build modeling frame; missing columns: {missing}"
        raise ValueError(msg)

    modeling = frame[required_columns].copy()
    modeling = modeling.dropna(subset=[*feature_columns, *target_columns]).reset_index(drop=True)
    if modeling.empty:
        msg = "Complete modeling frame is empty after dropping missing features/targets"
        raise ValueError(msg)
    return modeling
