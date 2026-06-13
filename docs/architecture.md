# Architecture Plan

## Goals

Build a modular research platform that can produce daily point-in-time datasets, train return and direction models, generate buy/sell/hold signals, evaluate backtests, and expose saved results to notebooks and a React dashboard.

The platform starts with a small seed universe and S&P 400-oriented design, then expands toward broader US equity coverage once point-in-time membership and delisting support are available.

## Core Principles

- Public data only: Yahoo Finance style price data, FRED macro series, and SEC EDGAR/company facts.
- Point-in-time safety: rows are keyed by `as_of_date`; features can only appear once historically available.
- No global intersection filtering: preserve long-history rows even when richer features start later.
- Modular adapters: source quirks stay inside `src/data`.
- Reproducibility: configs, cached raw data, curated outputs, metrics, and artifacts are written separately.
- Experiment traceability: models, settings, feature lists, metrics, and signal/backtest outputs are stored as artifacts.

## Data Layers

`data/raw`
: Unmodified source payloads, API responses, and downloaded quote files.

`data/staged`
: Normalized source-specific tables with identifiers, dates, provenance, and source refresh timestamps.

`data/curated`
: Point-in-time panels, feature matrices, target tables, and exportable CSV snapshots.

`artifacts`
: Predictions, metrics, model files, feature importance, charts, backtest summaries, monitoring reports, and dashboard-ready JSON/CSV.

## Module Plan

`src/data`
: Source adapters, ticker/CIK identifiers, market calendar, local storage, and point-in-time panel builder.

`src/features`
: Technical, macro, fundamental, rolling, lagged, and missingness feature builders. Feature groups are driven by config.

`src/targets`
: Forward-return, direction, and signal target generation for configurable horizons.

`src/models`
: Regression and classification pipelines, model registry, feature selection, tuning, and artifact persistence.

`src/evaluation`
: Regression/classification metrics, fold summaries, charts, and model comparison tables.

`src/signals`
: Forecast-to-signal conversion rules and threshold configuration.

`src/backtest`
: Transparent signal backtesting with transaction cost and turnover hooks.

`src/monitoring`
: Rolling out-of-sample monitoring, degradation checks, drift summaries, and retraining/tuning triggers.

`src/agent_review`
: Optional review module that proposes changes from metrics/configs without silently mutating the system.

`dashboard`
: React app that consumes artifacts or a lightweight API layer.

## Point-In-Time Dataset Contract

Each curated row represents one `ticker` at one `as_of_date`.

Required rules:

- Universe rows start no earlier than listing date and membership start.
- Macro data joins by availability date, not observation date.
- Fundamentals join by filing/public availability date, not fiscal period end.
- No source feature is backfilled into dates before its first valid availability.
- Forward-fill is only allowed after a value is available and only where economically valid.
- Missingness flags distinguish missing values from unavailable history.
- Provenance columns retain source and refresh timestamps where practical.

## Feature Regimes

Core long-history
: price, volume, and long-running macro features.

Expanded
: core features plus selected fundamentals and ratios.

Full
: expanded features plus broader fundamentals, XBRL-derived fields, and richer valuation/profitability metrics.

## Modeling Roadmap

Regression models:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- optional XGBoost Regressor

Classification models:

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- optional XGBoost Classifier

All training uses time-series validation only. No random shuffling is allowed.

## Dashboard Roadmap

The dashboard should remain artifact-first at the beginning. It can read saved CSV/JSON/Parquet-derived summaries and later move to FastAPI if interactive filtering or larger artifacts require it.

Primary views:

- model overview
- predictions
- signals
- feature importance
- backtest performance
- dataset coverage and missingness
- experiment tracking
