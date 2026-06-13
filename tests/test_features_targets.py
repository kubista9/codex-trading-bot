from __future__ import annotations

import pandas as pd

from src.features.technical import TechnicalFeatureConfig, build_technical_features
from src.targets.returns import TargetConfig, build_return_targets


def _sample_price_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=4)
    return pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA", "AAA", "BBB", "BBB", "BBB", "BBB"],
            "as_of_date": list(dates) + list(dates),
            "adj_close": [100.0, 101.0, 103.0, 102.0, 50.0, 49.0, 51.0, 52.0],
            "volume": [1000, 1100, 1200, 1150, 500, 550, 600, 650],
        }
    )


def test_technical_features_are_grouped_by_ticker() -> None:
    result = build_technical_features(
        _sample_price_panel(),
        TechnicalFeatureConfig(return_windows=(1, 2), moving_average_windows=(2,), volatility_windows=(2,)),
    )
    frame = result.frame

    first_bbb = frame[(frame["ticker"] == "BBB")].iloc[0]
    assert pd.isna(first_bbb["return_1d"])
    assert pd.isna(first_bbb["return_2d"])
    assert "price_to_ma_2d" in result.feature_columns


def test_technical_features_use_only_current_and_prior_prices() -> None:
    result = build_technical_features(
        _sample_price_panel(),
        TechnicalFeatureConfig(return_windows=(1, 2), moving_average_windows=(2,), volatility_windows=()),
    )
    aaa = result.frame[result.frame["ticker"] == "AAA"].reset_index(drop=True)

    assert round(float(aaa.loc[2, "return_2d"]), 6) == 0.03
    assert aaa.loc[1, "ma_2d"] == 100.5
    assert pd.isna(aaa.loc[0, "ma_2d"])


def test_forward_return_targets_do_not_cross_tickers() -> None:
    frame = build_return_targets(_sample_price_panel(), TargetConfig(horizons=(1,)))

    aaa = frame[frame["ticker"] == "AAA"].reset_index(drop=True)
    bbb = frame[frame["ticker"] == "BBB"].reset_index(drop=True)
    assert round(float(aaa.loc[0, "forward_return_1d"]), 6) == 0.01
    assert pd.isna(aaa.loc[3, "forward_return_1d"])
    assert round(float(bbb.loc[0, "forward_return_1d"]), 6) == -0.02


def test_direction_and_signal_targets_respect_thresholds() -> None:
    frame = build_return_targets(
        _sample_price_panel(),
        TargetConfig(horizons=(1,), buy_threshold=0.015, sell_threshold=-0.015),
    )
    aaa = frame[frame["ticker"] == "AAA"].reset_index(drop=True)
    bbb = frame[frame["ticker"] == "BBB"].reset_index(drop=True)

    assert aaa.loc[0, "direction_1d"] == 1
    assert aaa.loc[0, "signal_target_1d"] == "hold"
    assert aaa.loc[1, "signal_target_1d"] == "buy"
    assert bbb.loc[0, "direction_1d"] == 0
    assert bbb.loc[0, "signal_target_1d"] == "sell"
