from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.features.base import FeatureBuildResult, feature_metadata


@dataclass(frozen=True)
class TechnicalFeatureConfig:
    """Configuration for price and volume feature generation."""

    price_col: str = "adj_close"
    volume_col: str = "volume"
    return_windows: tuple[int, ...] = (1, 5, 20, 60)
    volatility_windows: tuple[int, ...] = (5, 20, 60)
    moving_average_windows: tuple[int, ...] = (5, 20, 60)
    min_periods_by_window: dict[int, int] = field(default_factory=dict)

    def min_periods(self, window: int) -> int:
        """Return configured min periods, defaulting to a full window."""
        return self.min_periods_by_window.get(window, window)


def build_technical_features(
    panel: pd.DataFrame,
    config: TechnicalFeatureConfig | None = None,
) -> FeatureBuildResult:
    """Build leakage-safe price and volume features from a PIT panel.

    Features are computed independently per ticker using current and prior rows.
    The function assumes each `as_of_date` snapshot is observed after the market
    close for that date.
    """
    cfg = config or TechnicalFeatureConfig()
    required = {"ticker", "as_of_date", cfg.price_col}
    missing = required.difference(panel.columns)
    if missing:
        msg = f"Panel missing required columns for technical features: {sorted(missing)}"
        raise ValueError(msg)

    frame = panel.copy()
    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"]).dt.normalize()
    frame = frame.sort_values(["ticker", "as_of_date"]).reset_index(drop=True)
    grouped = frame.groupby("ticker", group_keys=False, sort=False)
    features: list[str] = []

    frame["return_1d"] = grouped[cfg.price_col].pct_change()
    features.append("return_1d")

    previous_price = grouped[cfg.price_col].shift(1)
    frame["log_return_1d"] = np.log(frame[cfg.price_col] / previous_price)
    frame.loc[~np.isfinite(frame["log_return_1d"]), "log_return_1d"] = np.nan
    features.append("log_return_1d")

    for window in cfg.return_windows:
        column = f"return_{window}d" if window != 1 else "return_1d"
        if window == 1:
            continue
        frame[column] = frame[cfg.price_col] / grouped[cfg.price_col].shift(window) - 1.0
        features.append(column)

    for window in cfg.volatility_windows:
        column = f"volatility_{window}d"
        frame[column] = grouped["return_1d"].transform(
            lambda series, window=window: series.rolling(
                window=window,
                min_periods=cfg.min_periods(window),
            ).std()
        )
        features.append(column)

    for window in cfg.moving_average_windows:
        ma_column = f"ma_{window}d"
        distance_column = f"price_to_ma_{window}d"
        frame[ma_column] = grouped[cfg.price_col].transform(
            lambda series, window=window: series.rolling(
                window=window,
                min_periods=cfg.min_periods(window),
            ).mean()
        )
        frame[distance_column] = frame[cfg.price_col] / frame[ma_column] - 1.0
        features.extend([ma_column, distance_column])

        if cfg.volume_col in frame.columns:
            volume_ma_column = f"volume_ma_{window}d"
            volume_ratio_column = f"volume_to_ma_{window}d"
            frame[volume_ma_column] = grouped[cfg.volume_col].transform(
                lambda series, window=window: series.rolling(
                    window=window,
                    min_periods=cfg.min_periods(window),
                ).mean()
            )
            frame[volume_ratio_column] = frame[cfg.volume_col] / frame[volume_ma_column] - 1.0
            features.extend([volume_ma_column, volume_ratio_column])

    metadata = feature_metadata(features, group="technical_price_volume", source="price")
    return FeatureBuildResult(frame=frame, feature_columns=features, metadata=metadata)
