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
from src.data.missingness import (
    drop_missingness_flag_columns,
    is_missingness_flag,
    summarize_missingness_flags,
)
from src.data.pit import PITBuildConfig, PointInTimeDatasetBuilder
from src.data.fred import FredAdapter
from src.data.sec import SECEdgarAdapter, company_facts_to_wide
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
    parser.add_argument("--data-sources", default="configs/data_sources.yml")
    parser.add_argument("--start-date", default="2010-01-01")
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--artifact-dir", default="artifacts/baseline_experiment")
    parser.add_argument("--test-size-days", type=int, default=252)
    parser.add_argument("--min-train-days", type=int, default=756)
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument(
        "--use-existing-prices",
        action="store_true",
        help="Use artifact-dir/prices.csv when present instead of refetching Yahoo prices.",
    )
    parser.add_argument("--include-fred", action="store_true", help="Fetch and join FRED macro data.")
    parser.add_argument("--include-sec", action="store_true", help="Fetch and join SEC company facts.")
    parser.add_argument(
        "--use-existing-fred",
        action="store_true",
        help="Use artifact-dir/fred_macro.csv when present instead of refetching FRED.",
    )
    parser.add_argument(
        "--use-existing-sec",
        action="store_true",
        help="Use SEC artifact CSVs when present instead of refetching SEC company facts.",
    )
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


def configured_fred_series(path: str | Path) -> list[str]:
    config = load_yaml(path)
    return [str(series) for series in config.get("fred", {}).get("series", [])]


def configured_fred_lags(path: str | Path) -> dict[str, int]:
    config = load_yaml(path)
    lags = config.get("fred", {}).get("availability_lag_days", {})
    return {str(series): int(days) for series, days in lags.items()}


def configured_sec_concepts(path: str | Path) -> list[str]:
    config = load_yaml(path)
    concepts = [str(concept) for concept in config.get("sec", {}).get("concepts", [])]
    return list(dict.fromkeys(concepts))


def cik_to_ticker_map(members: list[dict[str, Any]]) -> dict[str, str]:
    """Build a normalized CIK-to-ticker map from configured universe members."""
    mapping: dict[str, str] = {}
    for member in members:
        cik = str(member.get("cik", "")).lstrip("0")
        ticker = str(member.get("ticker", "")).upper()
        if cik and ticker:
            mapping[f"{int(cik):010d}"] = ticker
    return mapping


def apply_fred_availability_lags(macro: pd.DataFrame, lags: dict[str, int]) -> pd.DataFrame:
    """Apply conservative availability lags to FRED observations."""
    frame = macro.copy()
    frame["observation_date"] = pd.to_datetime(frame["observation_date"]).dt.normalize()
    frame["series_id"] = frame["series_id"].astype(str)
    lag_days = frame["series_id"].map(lags).fillna(31).astype(int)
    frame["available_date"] = frame["observation_date"] + pd.to_timedelta(lag_days, unit="D")
    return frame


def infer_numeric_source_features(
    frame: pd.DataFrame,
    *,
    prefixes: tuple[str, ...],
    exclude: set[str] | None = None,
) -> list[str]:
    """Infer numeric feature columns from joined source data."""
    excluded = exclude or set()
    features: list[str] = []
    for column in frame.columns:
        if column in excluded or not column.startswith(prefixes):
            continue
        if is_missingness_flag(column):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]) and frame[column].notna().any():
            features.append(column)
    return features


def choose_complete_feature_set(
    frame: pd.DataFrame,
    *,
    candidate_features: list[str],
    horizons: tuple[int, ...],
    min_rows: int,
) -> list[str]:
    """Keep features that still leave enough complete training rows."""
    target_columns = []
    for horizon in horizons:
        target_columns.extend(
            [
                f"forward_return_{horizon}d",
                f"direction_{horizon}d",
                f"signal_target_{horizon}d",
            ]
        )

    selected: list[str] = []
    for feature in candidate_features:
        trial = [*selected, feature]
        rows = frame.dropna(subset=[*trial, *target_columns])
        if len(rows) >= min_rows:
            selected.append(feature)
    return selected


def train_and_export(
    *,
    frame: pd.DataFrame,
    feature_columns: list[str],
    horizons: tuple[int, ...],
    artifact_dir: Path,
    prefix: str,
    test_size_days: int,
    min_train_days: int,
    transaction_cost_bps: float,
) -> dict[str, pd.DataFrame]:
    """Train all baseline models for one dataset regime and write artifacts."""
    metrics: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    importances: list[pd.DataFrame] = []
    signal_frames: list[pd.DataFrame] = []
    backtests: list[pd.DataFrame] = []

    for horizon in horizons:
        regression_results = train_regression_models(
            frame,
            feature_columns=feature_columns,
            target_col=f"forward_return_{horizon}d",
            horizon=horizon,
            test_size_days=test_size_days,
            min_train_days=min_train_days,
        )
        classification_results = train_classification_models(
            frame,
            feature_columns=feature_columns,
            target_col=f"direction_{horizon}d",
            horizon=horizon,
            test_size_days=test_size_days,
            min_train_days=min_train_days,
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
                    transaction_cost_bps=transaction_cost_bps,
                )
            )

    metrics_frame = pd.DataFrame.from_records(metrics)
    predictions_frame = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    importances_frame = pd.concat(importances, ignore_index=True) if importances else pd.DataFrame()
    signals_frame = pd.concat(signal_frames, ignore_index=True) if signal_frames else pd.DataFrame()
    backtest_frame = pd.concat(backtests, ignore_index=True) if backtests else pd.DataFrame()

    write_csv(metrics_frame, artifact_dir, f"metrics_{prefix}.csv")
    write_csv(predictions_frame, artifact_dir, f"predictions_{prefix}.csv")
    write_csv(importances_frame, artifact_dir, f"feature_importance_{prefix}.csv")
    write_csv(signals_frame, artifact_dir, f"signals_{prefix}.csv")
    write_csv(backtest_frame, artifact_dir, f"backtest_{prefix}.csv")
    return {
        "metrics": metrics_frame,
        "predictions": predictions_frame,
        "feature_importance": importances_frame,
        "signals": signals_frame,
        "backtest": backtest_frame,
    }


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    store = LocalDataStore()

    members = load_universe(args.universe, args.tickers)
    tickers = [str(member["ticker"]).upper() for member in members]
    horizons = load_horizons(args.horizons)
    ciks = [str(member["cik"]) for member in members if member.get("cik")]
    ticker_by_cik = cik_to_ticker_map(members)

    prices_path = artifact_dir / "prices.csv"
    if args.use_existing_prices and prices_path.exists():
        prices = pd.read_csv(prices_path, parse_dates=["as_of_date"])
    else:
        yahoo = YahooFinanceAdapter(store=store)
        price_result = yahoo.fetch(tickers, args.start_date, args.end_date)
        prices = price_result.frame.dropna(subset=["adj_close"]).copy()
        write_csv(prices, artifact_dir, "prices.csv")

    macro = None
    if args.include_fred:
        macro_path = artifact_dir / "fred_macro.csv"
        fred_lags = configured_fred_lags(args.data_sources)
        if args.use_existing_fred and macro_path.exists():
            macro = pd.read_csv(macro_path, parse_dates=["observation_date", "available_date"])
        else:
            series = configured_fred_series(args.data_sources)
            fred = FredAdapter()
            macro_result = fred.fetch(series, args.start_date, args.end_date)
            macro = macro_result.frame.dropna(subset=["value"]).copy()
        macro = apply_fred_availability_lags(macro, fred_lags)
        write_csv(macro, artifact_dir, "fred_macro.csv")

    fundamentals = None
    if args.include_sec:
        facts_long_path = artifact_dir / "sec_company_facts_long.csv"
        facts_wide_path = artifact_dir / "sec_company_facts_wide.csv"
        if args.use_existing_sec and facts_wide_path.exists():
            fundamentals = pd.read_csv(facts_wide_path, parse_dates=["available_date"])
        else:
            concepts = configured_sec_concepts(args.data_sources)
            sec = SECEdgarAdapter()
            facts_result = sec.fetch(ciks, concepts)
            facts_long = facts_result.frame.dropna(subset=["value"]).copy()
            facts_long["cik"] = facts_long["cik"].astype(str).str.zfill(10)
            facts_long["ticker"] = facts_long["ticker"].fillna(facts_long["cik"].map(ticker_by_cik))
            fundamentals = company_facts_to_wide(facts_long)
            write_csv(facts_long, artifact_dir, "sec_company_facts_long.csv")
            write_csv(fundamentals, artifact_dir, "sec_company_facts_wide.csv")

    builder = PointInTimeDatasetBuilder(
        PITBuildConfig(
            start_date=args.start_date,
            end_date=args.end_date,
            feature_regime="expanded" if args.include_fred or args.include_sec else "core_long_history",
        )
    )
    pit = builder.build(members, prices=prices, macro=macro, fundamentals=fundamentals)
    panel = pit.panel.dropna(subset=["adj_close"]).copy()
    missingness_summaries = [summarize_missingness_flags(panel, table_name="pit_price_panel")]
    write_csv(drop_missingness_flag_columns(panel), artifact_dir, "pit_price_panel.csv")
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
    missingness_summaries.append(
        summarize_missingness_flags(target_frame, table_name="feature_target_panel")
    )
    write_csv(drop_missingness_flag_columns(target_frame), artifact_dir, "feature_target_panel.csv")
    write_csv(drop_missingness_flag_columns(target_frame), artifact_dir, "research_dataset_2010.csv")
    missingness_summary = pd.concat(missingness_summaries, ignore_index=True)
    write_csv(missingness_summary, artifact_dir, "missingness_summary.csv")
    write_csv(feature_result.metadata, artifact_dir, "feature_metadata.csv")

    source_feature_columns = infer_numeric_source_features(
        target_frame,
        prefixes=("macro_", "fund_"),
        exclude={"macro_available_date", "fundamentals_available_date"},
    )
    macro_feature_columns = [
        column for column in source_feature_columns if column.startswith("macro_")
    ]
    fundamental_feature_columns = [
        column for column in source_feature_columns if column.startswith("fund_")
    ]

    core_feature_columns = choose_complete_feature_set(
        target_frame,
        candidate_features=[*feature_result.feature_columns, *macro_feature_columns],
        horizons=horizons,
        min_rows=max(args.min_train_days + args.test_size_days, 1000),
    )
    expanded_feature_columns = choose_complete_feature_set(
        target_frame,
        candidate_features=[*core_feature_columns, *fundamental_feature_columns],
        horizons=horizons,
        min_rows=max(args.min_train_days + args.test_size_days, 1000),
    )
    write_csv(
        pd.DataFrame({"feature": core_feature_columns}),
        artifact_dir,
        "selected_features_core_2010.csv",
    )
    write_csv(
        pd.DataFrame({"feature": expanded_feature_columns}),
        artifact_dir,
        "selected_features_expanded_financials.csv",
    )

    core_model_frame = make_complete_modeling_frame(
        target_frame,
        feature_columns=core_feature_columns,
        horizons=horizons,
        context_columns=["as_of_date", "ticker", "adj_close", "volume"],
    )
    expanded_model_frame = make_complete_modeling_frame(
        target_frame,
        feature_columns=expanded_feature_columns,
        horizons=horizons,
        context_columns=["as_of_date", "ticker", "adj_close", "volume"],
    )
    write_csv(core_model_frame, artifact_dir, "model_dataset_core_2010.csv")
    write_csv(expanded_model_frame, artifact_dir, "model_dataset_expanded_financials.csv")
    write_csv(core_model_frame, artifact_dir, "model_dataset.csv")

    core_results = train_and_export(
        frame=core_model_frame,
        feature_columns=core_feature_columns,
        horizons=horizons,
        artifact_dir=artifact_dir,
        prefix="core_2010",
        test_size_days=args.test_size_days,
        min_train_days=args.min_train_days,
        transaction_cost_bps=args.transaction_cost_bps,
    )
    expanded_results = train_and_export(
        frame=expanded_model_frame,
        feature_columns=expanded_feature_columns,
        horizons=horizons,
        artifact_dir=artifact_dir,
        prefix="expanded_financials",
        test_size_days=args.test_size_days,
        min_train_days=args.min_train_days,
        transaction_cost_bps=args.transaction_cost_bps,
    )

    metrics_frame = pd.concat(
        [
            core_results["metrics"].assign(dataset_regime="core_2010"),
            expanded_results["metrics"].assign(dataset_regime="expanded_financials"),
        ],
        ignore_index=True,
    )
    backtest_frame = pd.concat(
        [
            core_results["backtest"].assign(dataset_regime="core_2010"),
            expanded_results["backtest"].assign(dataset_regime="expanded_financials"),
        ],
        ignore_index=True,
    )
    predictions_frame = pd.concat(
        [
            core_results["predictions"].assign(dataset_regime="core_2010"),
            expanded_results["predictions"].assign(dataset_regime="expanded_financials"),
        ],
        ignore_index=True,
    )
    importances_frame = pd.concat(
        [
            core_results["feature_importance"].assign(dataset_regime="core_2010"),
            expanded_results["feature_importance"].assign(dataset_regime="expanded_financials"),
        ],
        ignore_index=True,
    )
    signals_frame = pd.concat(
        [
            core_results["signals"].assign(dataset_regime="core_2010"),
            expanded_results["signals"].assign(dataset_regime="expanded_financials"),
        ],
        ignore_index=True,
    )

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
