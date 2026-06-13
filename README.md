# Stock Research Platform

Production-style scaffold for public-data stock forecasting, signal generation, backtesting, and experiment tracking.

The first milestone focuses on correctness-sensitive foundations:

- isolated public data adapters for Yahoo Finance, FRED, and SEC EDGAR
- a point-in-time daily panel builder
- configuration-driven universe, source, horizon, and feature-regime design
- tests for alignment rules that commonly cause leakage

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Set `SEC_USER_AGENT` before using SEC endpoints. Set `FRED_API_KEY` before using FRED API ingestion.

## Run Tests

```bash
pytest
```

## Run Baseline Experiment

After installing dependencies, run a price-only baseline over the seed universe:

```bash
python scripts/run_baseline_experiment.py \
  --start-date 2010-01-01 \
  --end-date 2026-06-12
```

Outputs are written to `artifacts/baseline_experiment/`.

To include FRED macro data and SEC company facts:

```bash
python scripts/run_baseline_experiment.py \
  --start-date 2010-01-01 \
  --end-date 2026-06-12 \
  --include-fred \
  --include-sec
```

Important exported tables:

- `research_dataset_2010.csv`: full 2010-start research panel with price, macro, and SEC financial columns where point-in-time available.
- `feature_target_panel.csv`: full audit panel, including natural edge missingness from rolling lookbacks and future targets.
- `model_dataset_core_2010.csv`: strict no-null long-history training dataset using technical and FRED macro features.
- `model_dataset_expanded_financials.csv`: strict no-null expanded training dataset using technical, FRED macro, and SEC financial features.
- `model_dataset.csv`: compatibility alias for `model_dataset_core_2010.csv`, so the default modeling file starts in 2010.
- `missingness_summary.csv`: compact counts for generated missingness flags.
- `feature_importance.csv`: model coefficients or tree feature importances by task, horizon, and model.
- `feature_importance_core_2010.csv` and `feature_importance_expanded_financials.csv`: regime-specific importance exports.
- `fred_macro.csv`: FRED observations when `--include-fred` is used.
- `sec_company_facts_long.csv` and `sec_company_facts_wide.csv`: SEC XBRL company facts when `--include-sec` is used.

## Current Branch

Initial scaffold work is intended to happen on:

```bash
feature/data-layer-pit-scaffold
```

Do not commit directly to `main`. Keep future milestones on focused feature branches and open PRs into `main`.

## Architecture Docs

- [Architecture Plan](docs/architecture.md)
- [Repository Tree](docs/repository_tree.md)
- [Milestones](docs/milestones.md)
