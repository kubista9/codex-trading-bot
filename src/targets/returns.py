from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TargetConfig:
    """Configuration for forward return, direction, and signal-label targets."""

    horizons: tuple[int, ...] = (1, 5, 20, 60)
    price_col: str = "adj_close"
    positive_threshold: float = 0.0
    buy_threshold: float = 0.02
    sell_threshold: float = -0.02
    drop_incomplete_targets: bool = False


def build_return_targets(panel: pd.DataFrame, config: TargetConfig | None = None) -> pd.DataFrame:
    """Add horizon-specific forward return, direction, and signal targets.

    Target columns intentionally use future prices. They must be created after
    feature assembly and kept out of model input columns.
    """
    cfg = config or TargetConfig()
    required = {"ticker", "as_of_date", cfg.price_col}
    missing = required.difference(panel.columns)
    if missing:
        msg = f"Panel missing required columns for target generation: {sorted(missing)}"
        raise ValueError(msg)

    if cfg.buy_threshold <= cfg.sell_threshold:
        msg = "buy_threshold must be greater than sell_threshold"
        raise ValueError(msg)

    frame = panel.copy()
    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"]).dt.normalize()
    frame = frame.sort_values(["ticker", "as_of_date"]).reset_index(drop=True)
    grouped = frame.groupby("ticker", group_keys=False, sort=False)

    target_columns: list[str] = []
    for horizon in cfg.horizons:
        if horizon <= 0:
            msg = f"Horizon must be positive, got {horizon}"
            raise ValueError(msg)

        future_price = grouped[cfg.price_col].shift(-horizon)
        return_col = f"forward_return_{horizon}d"
        direction_col = f"direction_{horizon}d"
        signal_col = f"signal_target_{horizon}d"

        frame[return_col] = future_price / frame[cfg.price_col] - 1.0
        frame[direction_col] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
        valid = frame[return_col].notna()
        frame.loc[valid, direction_col] = (
            frame.loc[valid, return_col] > cfg.positive_threshold
        ).astype("int64")
        frame[signal_col] = pd.Series(pd.NA, index=frame.index, dtype="string")
        frame.loc[valid & (frame[return_col] >= cfg.buy_threshold), signal_col] = "buy"
        frame.loc[valid & (frame[return_col] <= cfg.sell_threshold), signal_col] = "sell"
        frame.loc[valid & frame[signal_col].isna(), signal_col] = "hold"
        target_columns.extend([return_col, direction_col, signal_col])

    if cfg.drop_incomplete_targets:
        required_targets = [column for column in target_columns if column.startswith("forward_return_")]
        frame = frame.dropna(subset=required_targets).reset_index(drop=True)

    return frame
