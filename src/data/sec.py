from __future__ import annotations

import os
import re
from time import sleep
from typing import Any, Iterable

import pandas as pd
import requests

from src.data.base import DataSourceError, IngestionResult, SourceMetadata, utc_now
from src.data.identifiers import normalize_cik, normalize_ticker


def snake_case(value: str) -> str:
    """Convert SEC concept names to snake_case feature suffixes."""
    value = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", value)
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


class SECEdgarAdapter:
    """SEC EDGAR/data.sec.gov adapter for ticker mapping and company facts."""

    source_name = "sec"

    def __init__(
        self,
        *,
        user_agent: str | None = None,
        user_agent_env: str = "SEC_USER_AGENT",
        timeout_seconds: int = 30,
        requests_per_second: float = 5,
        session: requests.Session | None = None,
    ) -> None:
        self.user_agent = user_agent or os.getenv(user_agent_env)
        if not self.user_agent:
            msg = "SEC_USER_AGENT must be set for SEC EDGAR requests"
            raise DataSourceError(msg)

        self.timeout_seconds = timeout_seconds
        self.delay_seconds = 1 / requests_per_second if requests_per_second > 0 else 0
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept-Encoding": "gzip, deflate",
                "Host": "data.sec.gov",
            }
        )

    def fetch(self, ciks: Iterable[str | int], concepts: Iterable[str]) -> IngestionResult:
        """Fetch selected company fact concepts for several CIKs."""
        frames: list[pd.DataFrame] = []
        failures: list[str] = []
        concept_list = list(concepts)
        for cik in ciks:
            normalized_cik = normalize_cik(cik)
            if normalized_cik is None:
                continue
            try:
                facts = self.fetch_company_facts(normalized_cik)
                frames.append(extract_company_fact_concepts(facts, concept_list))
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{normalized_cik}: {exc}")
            sleep(self.delay_seconds)

        if not frames:
            msg = "No SEC company facts were downloaded"
            raise DataSourceError(msg)

        metadata = SourceMetadata(
            source=self.source_name,
            query={"ciks": list(ciks), "concepts": concept_list},
            notes=failures,
        )
        return IngestionResult(pd.concat(frames, ignore_index=True), metadata)

    def fetch_company_tickers(self) -> pd.DataFrame:
        """Fetch SEC company ticker-to-CIK mapping."""
        response = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers={"User-Agent": self.user_agent},
            timeout=self.timeout_seconds,
        )
        if response.status_code >= 400:
            msg = f"SEC ticker mapping returned HTTP {response.status_code}"
            raise DataSourceError(msg)
        frame = pd.DataFrame(response.json()).T
        frame["ticker"] = frame["ticker"].map(normalize_ticker)
        frame["cik"] = frame["cik_str"].map(normalize_cik)
        frame["source_refresh_time"] = utc_now()
        return frame[["ticker", "cik", "title", "source_refresh_time"]]

    def fetch_company_facts(self, cik: str | int) -> dict[str, Any]:
        """Fetch raw company facts JSON for a CIK."""
        normalized = normalize_cik(cik)
        if normalized is None:
            msg = "CIK is required"
            raise DataSourceError(msg)
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{normalized}.json"
        response = self.session.get(url, timeout=self.timeout_seconds)
        if response.status_code >= 400:
            msg = f"SEC company facts returned HTTP {response.status_code} for CIK {normalized}"
            raise DataSourceError(msg)
        return response.json()


def extract_company_fact_concepts(
    facts: dict[str, Any],
    concepts: Iterable[str],
    *,
    taxonomy: str = "us-gaap",
) -> pd.DataFrame:
    """Extract selected SEC company fact concepts in long form.

    The `filed` date is treated as the first public availability date in this
    scaffold. Later work can refine this with acceptance timestamps where
    available.
    """
    cik = normalize_cik(facts.get("cik"))
    ticker = normalize_ticker(facts.get("tickers", [""])[0]) if facts.get("tickers") else None
    taxonomy_facts = facts.get("facts", {}).get(taxonomy, {})
    records: list[dict[str, Any]] = []

    for concept in concepts:
        concept_payload = taxonomy_facts.get(concept)
        if not concept_payload:
            continue
        units = concept_payload.get("units", {})
        for unit, observations in units.items():
            for observation in observations:
                filed = observation.get("filed")
                if filed is None:
                    continue
                records.append(
                    {
                        "ticker": ticker,
                        "cik": cik,
                        "concept": concept,
                        "feature": f"fund_{snake_case(concept)}",
                        "unit": unit,
                        "value": observation.get("val"),
                        "period_end": observation.get("end"),
                        "fiscal_year": observation.get("fy"),
                        "fiscal_period": observation.get("fp"),
                        "form": observation.get("form"),
                        "accession": observation.get("accn"),
                        "filing_date": filed,
                        "available_date": filed,
                        "source_refresh_time": utc_now(),
                    }
                )

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "cik",
                "concept",
                "feature",
                "unit",
                "value",
                "period_end",
                "fiscal_year",
                "fiscal_period",
                "form",
                "accession",
                "filing_date",
                "available_date",
                "source_refresh_time",
            ]
        )

    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    for column in ["period_end", "filing_date", "available_date"]:
        frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.normalize()
    return frame


def company_facts_to_wide(facts: pd.DataFrame) -> pd.DataFrame:
    """Pivot long SEC company facts into PIT-builder-ready wide form."""
    if facts.empty:
        return pd.DataFrame(columns=["ticker", "cik", "available_date"])

    frame = facts.copy()
    required = {"ticker", "cik", "available_date", "feature", "value"}
    missing = required.difference(frame.columns)
    if missing:
        msg = f"Company facts missing required columns: {sorted(missing)}"
        raise ValueError(msg)

    frame["ticker"] = frame["ticker"].map(normalize_ticker)
    frame["cik"] = frame["cik"].map(normalize_cik)
    frame["available_date"] = pd.to_datetime(frame["available_date"]).dt.normalize()
    wide = frame.pivot_table(
        index=["ticker", "cik", "available_date"],
        columns="feature",
        values="value",
        aggfunc="last",
    ).reset_index()
    wide.columns.name = None
    return wide.sort_values(["ticker", "available_date"]).reset_index(drop=True)
