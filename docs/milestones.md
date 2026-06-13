# Feature Branches and PR Milestones

## PR 1: Data Layer and Point-In-Time Scaffold

Branch: `feature/data-layer-pit-scaffold`

Scope:

- architecture docs and repository structure
- config samples
- data adapter interfaces for Yahoo Finance, FRED, and SEC EDGAR
- local storage conventions
- point-in-time panel builder
- tests for leakage-sensitive joins

## PR 2: Yahoo and Calendar Ingestion

Branch: `feature/data-yahoo-ingestion`

Scope:

- harden Yahoo adapter
- add retry/logging policy
- staged price writer
- market calendar coverage metadata

## PR 3: FRED Macro Ingestion

Branch: `feature/data-fred-ingestion`

Scope:

- FRED observations client
- release-aware availability handling
- macro staged and curated outputs
- macro feature examples

## PR 4: SEC Company Facts

Branch: `feature/sec-companyfacts`

Scope:

- SEC ticker-to-CIK mapping
- company facts ingestion
- concept extraction
- filing-date availability alignment
- fundamentals staged outputs

## PR 5: Features and Targets

Branches:

- `feature/features-technical`
- `feature/features-macro-fundamental`
- `feature/targets-forward-returns`

Scope:

- technical features
- macro lag/rolling features
- fundamental ratios
- configurable feature regimes
- horizon-specific regression and direction targets

## PR 6: Modeling Pipelines

Branches:

- `feature/model-regression-pipeline`
- `feature/model-classification-pipeline`

Scope:

- train/evaluate per horizon and task
- time-series split only
- model registry and artifact saving
- feature importance exports

## PR 7: Evaluation and Backtesting

Branch: `feature/backtest-engine`

Scope:

- regression/classification metrics
- signal generation
- transparent backtest assumptions
- trading metrics and charts

## PR 8: Monitoring and Experiment Tracking

Branch: `feature/monitoring-experiment-tracking`

Scope:

- experiment manifests
- rolling out-of-sample monitoring
- degradation triggers
- optional agent review reports

## PR 9: Dashboard

Branches:

- `feature/dashboard-scaffold`
- `feature/dashboard-predictions`
- `feature/dashboard-backtests`

Scope:

- React scaffold
- artifact/API data loading
- filters for ticker, horizon, model, and date range
- overview, predictions, signals, feature importance, backtest, coverage, and experiment views
