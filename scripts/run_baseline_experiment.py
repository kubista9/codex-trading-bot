from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.backtest.simple import run_signal_backtest
from src.data.pit import PITBuildConfig, PointInTimeDatasetBuilder
from src.data.storage import LocalDataStore
from src.data.yahoo import YahooFinanceAdapter
from src.features.technical import TechnicalFeatureConfig, build_technical_features
from src.models.dataset import make_complete_modeling_frame
from src.models.training import train_classification_models, train_regression_models
from src.signals.rules import signals_from_expected_returns
from src.targets.returns import TargetConfig, build_return_targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a price-only baseline stock experiment.")
    parser.add_argument("--universe", default="configs/universe.sample.yml")
    parser.add_argument("--horizons", default="configs/horizons.yml")
    parser.add_argument("--start-date", default="2010-01-01")
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--artifact-dir", default="artifacts/baseline_experiment")
    parser.add_argument("--test-size-days", type=int, default=252)
    parser.add_argument("--min-train-days", type=int, default=756)
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        msg = f"Expected YAML mapping in {path}"
        raise ValueError(msg)
    return data


def load_universe(path: str | Path, tickers: list[str] | None) -> list[dict[str, Any]]:
    config = load_yaml(path)
    members = list(config.get("universe", {}).get("members", []))
    if not members:
        msg = f"No universe members found in {path}"
        raise ValueError(msg)
    if tickers:
        selected = {ticker.upper() for ticker in tickers}
        members = [member for member in members if str(member.get("ticker", "")).upper() in selected]
    if not members:
        msg = "Ticker filter removed all universe members"
        raise ValueError(msg)
    return members


def load_horizons(path: str | Path) -> tuple[int, ...]:
    config = load_yaml(path)
    horizons = config.get("horizons", {}).get("trading_days", [])
    if not horizons:
        msg = f"No horizons configured in {path}"
        raise ValueError(msg)
    return tuple(int(horizon) for horizon in horizons)


def write_csv(frame: pd.DataFrame, artifact_dir: Path, name: str) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / name
    frame.to_csv(path, index=False)
    return path


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    store = LocalDataStore()

    members = load_universe(args.universe, args.tickers)
    tickers = [str(member["ticker"]).upper() for member in members]
    horizons = load_horizons(args.horizons)

    yahoo = YahooFinanceAdapter(store=store)
    price_result = yahoo.fetch(tickers, args.start_date, args.end_date)
    prices = price_result.frame.dropna(subset=["adj_close"]).copy()
    write_csv(prices, artifact_dir, "prices.csv")

    builder = PointInTimeDatasetBuilder(
        PITBuildConfig(
            start_date=args.start_date,
            end_date=args.end_date,
            feature_regime="core_long_history",
        )
    )
    pit = builder.build(members, prices=prices)
    panel = pit.panel.dropna(subset=["adj_close"]).copy()
    write_csv(panel, artifact_dir, "pit_price_panel.csv")
    write_csv(pit.coverage, artifact_dir, "dataset_coverage.csv")

    feature_result = build_technical_features(
        panel,
        TechnicalFeatureConfig(
            return_windows=(1, 5, 20, 60),
            volatility_windows=(5, 20, 60),
            moving_average_windows=(5, 20, 60),
        ),
    )
    feature_frame = feature_result.frame
    target_frame = build_return_targets(
        feature_frame,
        TargetConfig(
            horizons=horizons,
            buy_threshold=0.02,
            sell_threshold=-0.02,
        ),
    )
    write_csv(target_frame, artifact_dir, "feature_target_panel.csv")
    write_csv(feature_result.metadata, artifact_dir, "feature_metadata.csv")

    metrics: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    importances: list[pd.DataFrame] = []
    signal_frames: list[pd.DataFrame] = []
    backtests: list[pd.DataFrame] = []

    feature_columns = feature_result.feature_columns
    model_frame = make_complete_modeling_frame(
        target_frame,
        feature_columns=feature_columns,
        horizons=horizons,
        context_columns=["as_of_date", "ticker", "adj_close", "volume"],
    )
    write_csv(model_frame, artifact_dir, "model_dataset.csv")

    for horizon in horizons:
        regression_results = train_regression_models(
            model_frame,
            feature_columns=feature_columns,
            target_col=f"forward_return_{horizon}d",
            horizon=horizon,
            test_size_days=args.test_size_days,
            min_train_days=args.min_train_days,
        )
        classification_results = train_classification_models(
            model_frame,
            feature_columns=feature_columns,
            target_col=f"direction_{horizon}d",
            horizon=horizon,
            test_size_days=args.test_size_days,
            min_train_days=args.min_train_days,
        )

        for result in [*regression_results, *classification_results]:
            metrics.append(result.metrics)
            predictions.append(result.predictions)
            if not result.feature_importance.empty:
                importances.append(result.feature_importance)

        for result in regression_results:
            signals = signals_from_expected_returns(
                result.predictions,
                buy_threshold=0.02,
                sell_threshold=-0.02,
            )
            signal_frames.append(signals)
            backtests.append(
                run_signal_backtest(
                    signals,
                    transaction_cost_bps=args.transaction_cost_bps,
                )
            )

    metrics_frame = pd.DataFrame.from_records(metrics)
    predictions_frame = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    importances_frame = pd.concat(importances, ignore_index=True) if importances else pd.DataFrame()
    signals_frame = pd.concat(signal_frames, ignore_index=True) if signal_frames else pd.DataFrame()
    backtest_frame = pd.concat(backtests, ignore_index=True) if backtests else pd.DataFrame()

    write_csv(metrics_frame, artifact_dir, "metrics.csv")
    write_csv(predictions_frame, artifact_dir, "predictions.csv")
    write_csv(importances_frame, artifact_dir, "feature_importance.csv")
    write_csv(signals_frame, artifact_dir, "signals.csv")
    write_csv(backtest_frame, artifact_dir, "backtest.csv")

    print("Baseline experiment complete")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Horizons: {', '.join(str(horizon) for horizon in horizons)}")
    print(f"Artifacts: {artifact_dir}")
    if not metrics_frame.empty:
        print(metrics_frame.to_string(index=False))
    if not backtest_frame.empty:
        print(backtest_frame.to_string(index=False))


if __name__ == "__main__":
    main()
