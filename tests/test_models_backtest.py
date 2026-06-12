from __future__ import annotations

import pandas as pd

from src.backtest.simple import run_signal_backtest
from src.models.dataset import make_complete_modeling_frame
from src.models.splits import make_temporal_holdout
from src.signals.rules import signals_from_expected_returns


def test_temporal_holdout_splits_by_date_without_shuffle() -> None:
    frame = pd.DataFrame(
        {
            "as_of_date": pd.bdate_range("2024-01-01", periods=10),
            "ticker": ["AAA"] * 10,
            "target": range(10),
        }
    )

    split = make_temporal_holdout(
        frame,
        target_col="target",
        test_size_days=3,
        min_train_days=5,
    )

    assert split.train["as_of_date"].max() < split.test["as_of_date"].min()
    assert len(split.test["as_of_date"].drop_duplicates()) == 3


def test_signal_rules_apply_expected_return_thresholds() -> None:
    predictions = pd.DataFrame({"prediction": [0.03, 0.0, -0.03]})

    signals = signals_from_expected_returns(predictions, buy_threshold=0.02, sell_threshold=-0.02)

    assert signals["signal"].tolist() == ["buy", "hold", "sell"]


def test_backtest_reports_trading_metrics() -> None:
    signals = pd.DataFrame(
        {
            "as_of_date": pd.bdate_range("2024-01-01", periods=3),
            "ticker": ["AAA", "AAA", "AAA"],
            "model_name": ["linear_regression"] * 3,
            "horizon": [1, 1, 1],
            "signal": ["buy", "hold", "sell"],
            "actual": [0.01, 0.02, -0.03],
        }
    )

    metrics = run_signal_backtest(signals, transaction_cost_bps=0)

    assert metrics.loc[0, "cumulative_return"] > 0
    assert metrics.loc[0, "turnover"] > 0


def test_complete_modeling_frame_drops_incomplete_edges() -> None:
    frame = pd.DataFrame(
        {
            "as_of_date": pd.bdate_range("2024-01-01", periods=4),
            "ticker": ["AAA"] * 4,
            "adj_close": [100.0, 101.0, 102.0, 103.0],
            "volume": [1000, 1100, 1200, 1300],
            "return_1d": [None, 0.01, 0.01, 0.01],
            "forward_return_1d": [0.01, 0.01, 0.01, None],
            "direction_1d": [1, 1, 1, None],
            "signal_target_1d": ["hold", "hold", "hold", None],
        }
    )

    complete = make_complete_modeling_frame(
        frame,
        feature_columns=["return_1d"],
        horizons=(1,),
    )

    assert complete["as_of_date"].tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]
    assert not complete.isna().any().any()
