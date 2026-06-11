from __future__ import annotations

import pandas as pd

from src.data.pit import PITBuildConfig, PointInTimeDatasetBuilder


def test_universe_rows_start_at_listing_date() -> None:
    builder = PointInTimeDatasetBuilder(PITBuildConfig(start_date="2024-01-01", end_date="2024-01-10"))
    dataset = builder.build(
        [
            {
                "ticker": "AAA",
                "listing_date": "2024-01-05",
                "membership_start": "2024-01-01",
            }
        ]
    )

    assert dataset.panel["as_of_date"].min() == pd.Timestamp("2024-01-05")
    assert set(dataset.panel["ticker"]) == {"AAA"}


def test_macro_uses_available_date_without_backfill() -> None:
    builder = PointInTimeDatasetBuilder(PITBuildConfig(start_date="2024-01-01", end_date="2024-01-08"))
    macro = pd.DataFrame(
        {
            "series_id": ["DGS10"],
            "observation_date": ["2023-12-29"],
            "available_date": ["2024-01-05"],
            "value": [4.1],
        }
    )

    panel = builder.build([{"ticker": "AAA", "listing_date": "2020-01-01"}], macro=macro).panel

    before = panel.loc[panel["as_of_date"] < pd.Timestamp("2024-01-05"), "macro_dgs10"]
    after = panel.loc[panel["as_of_date"] >= pd.Timestamp("2024-01-05"), "macro_dgs10"]
    assert before.isna().all()
    assert (after == 4.1).all()
    assert panel.loc[panel["as_of_date"] < pd.Timestamp("2024-01-05"), "macro_dgs10__is_structurally_missing"].all()


def test_fundamentals_align_on_filing_availability_not_period_end() -> None:
    builder = PointInTimeDatasetBuilder(PITBuildConfig(start_date="2024-01-02", end_date="2024-01-08"))
    fundamentals = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "available_date": ["2024-01-05"],
            "period_end": ["2023-09-30"],
            "filing_date": ["2024-01-05"],
            "fund_revenues": [100.0],
        }
    )

    panel = builder.build([{"ticker": "AAA", "listing_date": "2020-01-01"}], fundamentals=fundamentals).panel

    before = panel.loc[panel["as_of_date"] < pd.Timestamp("2024-01-05"), "fund_revenues"]
    after = panel.loc[panel["as_of_date"] >= pd.Timestamp("2024-01-05"), "fund_revenues"]
    assert before.isna().all()
    assert (after == 100.0).all()


def test_fundamentals_do_not_leak_across_tickers() -> None:
    builder = PointInTimeDatasetBuilder(PITBuildConfig(start_date="2024-01-02", end_date="2024-01-08"))
    fundamentals = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "available_date": ["2024-01-03"],
            "fund_assets": [250.0],
        }
    )

    panel = builder.build(
        [
            {"ticker": "AAA", "listing_date": "2020-01-01"},
            {"ticker": "BBB", "listing_date": "2020-01-01"},
        ],
        fundamentals=fundamentals,
    ).panel

    bbb = panel[panel["ticker"] == "BBB"]
    assert bbb["fund_assets"].isna().all()
    assert bbb["fund_assets__is_structurally_missing"].all()
