from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd

from src.data.calendar import build_market_calendar
from src.data.identifiers import members_to_frame, normalize_ticker


@dataclass(frozen=True)
class PITBuildConfig:
    """Configuration for point-in-time panel assembly."""

    start_date: str
    end_date: str
    feature_regime: str = "core_long_history"
    add_missingness_flags: bool = True


@dataclass(frozen=True)
class PITDataset:
    """Point-in-time panel plus coverage and feature metadata."""

    panel: pd.DataFrame
    coverage: pd.DataFrame
    feature_metadata: pd.DataFrame


class PointInTimeDatasetBuilder:
    """Build daily point-in-time panels from source-specific staged tables."""

    def __init__(
        self,
        config: PITBuildConfig,
        *,
        holidays: Iterable[str | pd.Timestamp] | None = None,
    ) -> None:
        self.config = config
        self.calendar = build_market_calendar(config.start_date, config.end_date, holidays=holidays)

    def build(
        self,
        universe_members: list[dict[str, Any]] | pd.DataFrame,
        *,
        prices: pd.DataFrame | None = None,
        macro: pd.DataFrame | None = None,
        fundamentals: pd.DataFrame | None = None,
    ) -> PITDataset:
        """Assemble a point-in-time daily panel.

        `prices` should be keyed by `as_of_date` and `ticker`.
        `macro` should include `available_date`; long FRED format with
        `series_id`/`value` is accepted.
        `fundamentals` should include `ticker` and `available_date`.
        """
        panel = self._build_universe_panel(universe_members)
        feature_sources: dict[str, str] = {}

        if prices is not None and not prices.empty:
            panel, price_features = self._merge_prices(panel, prices)
            feature_sources.update({column: "price" for column in price_features})

        if macro is not None and not macro.empty:
            panel, macro_features = self._merge_macro(panel, macro)
            feature_sources.update({column: "macro" for column in macro_features})

        if fundamentals is not None and not fundamentals.empty:
            panel, fundamental_features = self._merge_fundamentals(panel, fundamentals)
            feature_sources.update({column: "fundamentals" for column in fundamental_features})

        panel = panel.sort_values(["as_of_date", "ticker"]).reset_index(drop=True)
        if self.config.add_missingness_flags:
            panel = self._add_missingness_flags(panel, feature_sources)

        coverage = self._build_coverage(panel, feature_sources)
        metadata = self._build_feature_metadata(panel, feature_sources)
        return PITDataset(panel=panel, coverage=coverage, feature_metadata=metadata)

    def _build_universe_panel(self, universe_members: list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
        members = members_to_frame(universe_members)
        members = members[members["active"]].copy()
        if members.empty:
            msg = "No active universe members"
            raise ValueError(msg)

        calendar = pd.DataFrame({"as_of_date": self.calendar})
        members["_join_key"] = 1
        calendar["_join_key"] = 1
        panel = calendar.merge(members, on="_join_key", how="inner").drop(columns="_join_key")

        default_start = pd.Timestamp(self.config.start_date).normalize()
        default_end = pd.Timestamp(self.config.end_date).normalize()
        effective_start = panel[["listing_date", "membership_start"]].max(axis=1).fillna(default_start)
        effective_end = panel[["delisting_date", "membership_end"]].min(axis=1).fillna(default_end)

        panel = panel[(panel["as_of_date"] >= effective_start) & (panel["as_of_date"] <= effective_end)]
        panel["feature_regime"] = self.config.feature_regime
        return panel[
            [
                "as_of_date",
                "ticker",
                "cik",
                "listing_date",
                "membership_start",
                "membership_end",
                "delisting_date",
                "feature_regime",
            ]
        ].reset_index(drop=True)

    def _merge_prices(self, panel: pd.DataFrame, prices: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        price_frame = prices.copy()
        if "date" in price_frame.columns and "as_of_date" not in price_frame.columns:
            price_frame = price_frame.rename(columns={"date": "as_of_date"})
        required = {"as_of_date", "ticker"}
        missing = required.difference(price_frame.columns)
        if missing:
            msg = f"Prices missing required columns: {sorted(missing)}"
            raise ValueError(msg)

        price_frame["as_of_date"] = pd.to_datetime(price_frame["as_of_date"]).dt.normalize()
        price_frame["ticker"] = price_frame["ticker"].map(normalize_ticker)
        skip = {"as_of_date", "ticker", "source_refresh_time"}
        feature_cols = [column for column in price_frame.columns if column not in skip]
        merged = panel.merge(price_frame, on=["as_of_date", "ticker"], how="left")
        return merged, feature_cols

    def _merge_macro(self, panel: pd.DataFrame, macro: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        macro_frame = normalize_macro_frame(macro)
        feature_cols = [column for column in macro_frame.columns if column != "available_date"]
        if not feature_cols:
            return panel, []

        left = panel.sort_values("as_of_date")
        right = macro_frame.sort_values("available_date")
        merged = pd.merge_asof(
            left,
            right,
            left_on="as_of_date",
            right_on="available_date",
            direction="backward",
        )
        merged = merged.rename(columns={"available_date": "macro_available_date"})
        return merged, feature_cols

    def _merge_fundamentals(
        self,
        panel: pd.DataFrame,
        fundamentals: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[str]]:
        facts = fundamentals.copy()
        required = {"ticker", "available_date"}
        missing = required.difference(facts.columns)
        if missing:
            msg = f"Fundamentals missing required columns: {sorted(missing)}"
            raise ValueError(msg)

        facts["ticker"] = facts["ticker"].map(normalize_ticker)
        facts["available_date"] = pd.to_datetime(facts["available_date"]).dt.normalize()
        feature_cols = [
            column
            for column in facts.columns
            if column
            not in {
                "ticker",
                "cik",
                "available_date",
                "filing_date",
                "period_end",
                "source_refresh_time",
            }
        ]
        if not feature_cols:
            return panel, []

        chunks: list[pd.DataFrame] = []
        right_cols = ["ticker", "available_date", *feature_cols]
        if "filing_date" in facts.columns:
            right_cols.append("filing_date")
        right = facts[right_cols].sort_values(["ticker", "available_date"])

        for ticker, left_group in panel.groupby("ticker", sort=False):
            left_sorted = left_group.sort_values("as_of_date")
            right_group = right[right["ticker"] == ticker].sort_values("available_date")
            if right_group.empty:
                chunks.append(left_sorted.copy())
                continue

            right_group = right_group.rename(
                columns={
                    "available_date": "fundamentals_available_date",
                    "filing_date": "fundamentals_filing_date",
                }
            )
            merged = pd.merge_asof(
                left_sorted,
                right_group.drop(columns=["ticker"]),
                left_on="as_of_date",
                right_on="fundamentals_available_date",
                direction="backward",
            )
            chunks.append(merged)

        merged_panel = pd.concat(chunks, ignore_index=True)
        if "fundamentals_available_date" not in merged_panel.columns:
            merged_panel["fundamentals_available_date"] = pd.NaT
        for column in feature_cols:
            if column not in merged_panel.columns:
                merged_panel[column] = pd.NA
        if "filing_date" in right_cols and "fundamentals_filing_date" not in merged_panel.columns:
            merged_panel["fundamentals_filing_date"] = pd.NaT
        return merged_panel, feature_cols

    def _add_missingness_flags(self, panel: pd.DataFrame, feature_sources: dict[str, str]) -> pd.DataFrame:
        output = panel.copy()
        for feature, source in feature_sources.items():
            if feature not in output.columns:
                continue
            output[f"{feature}__is_missing"] = output[feature].isna()
            structural = self._structural_missing_mask(output, feature, source)
            output[f"{feature}__is_structurally_missing"] = structural
        return output

    def _structural_missing_mask(self, panel: pd.DataFrame, feature: str, source: str) -> pd.Series:
        if source == "price":
            return pd.Series(False, index=panel.index)

        if source == "macro":
            first_available = panel.loc[panel[feature].notna(), "macro_available_date"].min()
            if pd.isna(first_available):
                return pd.Series(True, index=panel.index)
            return panel["as_of_date"] < first_available

        if source == "fundamentals":
            available = panel.loc[panel[feature].notna(), ["ticker", "fundamentals_available_date"]]
            first_by_ticker = available.groupby("ticker")["fundamentals_available_date"].min()
            mapped = panel["ticker"].map(first_by_ticker)
            return mapped.isna() | (panel["as_of_date"] < mapped)

        return pd.Series(False, index=panel.index)

    def _build_coverage(self, panel: pd.DataFrame, feature_sources: dict[str, str]) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        for ticker, group in panel.groupby("ticker"):
            record: dict[str, Any] = {
                "ticker": ticker,
                "row_count": len(group),
                "first_as_of_date": group["as_of_date"].min(),
                "last_as_of_date": group["as_of_date"].max(),
            }
            for feature in feature_sources:
                if feature in group.columns:
                    record[f"{feature}__coverage"] = float(group[feature].notna().mean())
            records.append(record)
        return pd.DataFrame.from_records(records)

    def _build_feature_metadata(
        self,
        panel: pd.DataFrame,
        feature_sources: dict[str, str],
    ) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        for feature, source in feature_sources.items():
            if feature not in panel.columns:
                continue
            present = panel.loc[panel[feature].notna(), "as_of_date"]
            records.append(
                {
                    "feature": feature,
                    "source": source,
                    "first_available_as_of_date": present.min() if not present.empty else pd.NaT,
                    "last_available_as_of_date": present.max() if not present.empty else pd.NaT,
                    "feature_regime": self.config.feature_regime,
                }
            )
        return pd.DataFrame.from_records(records)


def normalize_macro_frame(macro: pd.DataFrame) -> pd.DataFrame:
    """Normalize macro observations to wide `available_date` form."""
    frame = macro.copy()
    if "available_date" not in frame.columns:
        msg = "Macro data must include available_date"
        raise ValueError(msg)
    frame["available_date"] = pd.to_datetime(frame["available_date"]).dt.normalize()

    if {"series_id", "value"}.issubset(frame.columns):
        frame["feature"] = "macro_" + frame["series_id"].astype(str).str.lower()
        wide = frame.pivot_table(
            index="available_date",
            columns="feature",
            values="value",
            aggfunc="last",
        ).reset_index()
        wide.columns.name = None
        wide = wide.sort_values("available_date").reset_index(drop=True)
        feature_cols = [column for column in wide.columns if column != "available_date"]
        wide[feature_cols] = wide[feature_cols].ffill()
        return wide

    return frame.sort_values("available_date").reset_index(drop=True)
