# Proposed Repository Tree

```text
.
├── artifacts/
│   └── .gitkeep
├── configs/
│   ├── data_sources.yml
│   ├── feature_regimes.yml
│   ├── horizons.yml
│   ├── training.yml
│   └── universe.sample.yml
├── dashboard/
│   └── README.md
├── data/
│   ├── curated/
│   │   └── .gitkeep
│   ├── raw/
│   │   └── .gitkeep
│   └── staged/
│       └── .gitkeep
├── docs/
│   ├── architecture.md
│   ├── milestones.md
│   └── repository_tree.md
├── notebooks/
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── calendar.py
│   │   ├── fred.py
│   │   ├── identifiers.py
│   │   ├── pit.py
│   │   ├── sec.py
│   │   ├── storage.py
│   │   └── yahoo.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── selection.py
│   │   └── technical.py
│   ├── targets/
│   │   ├── __init__.py
│   │   └── returns.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       └── logging.py
└── tests/
    ├── test_features_targets.py
    └── test_pit_builder.py
```

Later milestones add:

```text
src/models/
src/evaluation/
src/signals/
src/backtest/
src/monitoring/
src/agent_review/
dashboard/src/
dashboard/package.json
```
