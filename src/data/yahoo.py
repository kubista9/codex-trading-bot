from __future__ import annotations

import json
from pathlib import Path
from time import sleep
from typing import Iterable

import pandas as pd
import requests

from src.data.base import DataSourceError, IngestionResult, SourceMetadata
from src.data.identifiers import normalize_ticker
from src.data.storage import LocalDataStore
from src.data.base import utc_now
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


class YahooFinanceAdapter:
    """Isolated adapter for Yahoo Finance style historical quote downloads."""

    source_name = "yahoo"

    def __init__(
        self,
        store: LocalDataStore | None = None,
        *,
        timeout_seconds: int = 30,
        rate_limit_seconds: float = 0.5,
        session: requests.Session | None = None,
    ) -> None:
        self.store = store or LocalDataStore()
        self.timeout_seconds = timeout_seconds
        self.rate_limit_seconds = rate_limit_seconds
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": "stock-research-platform/0.1"})

    def fetch(
        self,
        tickers: Iterable[str],
        start_date: str,
        end_date: str,
        *,
        interval: str = "1d",
    ) -> IngestionResult:
        """Fetch daily historical OHLCV data for multiple tickers."""
        frames: list[pd.DataFrame] = []
        failures: list[str] = []
        raw_paths: list[str] = []

        for ticker in tickers:
            normalized = normalize_ticker(ticker)
            try:
                frame, raw_path = self._fetch_one(normalized, start_date, end_date, interval)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Yahoo download failed for %s: %s", normalized, exc)
                failures.append(f"{normalized}: {exc}")
                continue
            frames.append(frame)
            raw_paths.append(str(raw_path))
            sleep(self.rate_limit_seconds)

        if not frames:
            msg = "No Yahoo Finance price data was downloaded"
            raise DataSourceError(msg)

        metadata = SourceMetadata(
            source=self.source_name,
            query={"tickers": list(tickers), "start_date": start_date, "end_date": end_date},
            raw_path=";".join(raw_paths),
            notes=failures,
        )
        return IngestionResult(pd.concat(frames, ignore_index=True), metadata)

    def _fetch_one(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str,
    ) -> tuple[pd.DataFrame, Path]:
        start_ts = int(pd.Timestamp(start_date, tz="UTC").timestamp())
        end_ts = int((pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)).timestamp())
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {
            "period1": start_ts,
            "period2": end_ts,
            "interval": interval,
            "events": "history",
            "includeAdjustedClose": "true",
        }
        response = self.session.get(url, params=params, timeout=self.timeout_seconds)
        if response.status_code >= 400:
            msg = f"Yahoo returned HTTP {response.status_code} for {ticker}"
            raise DataSourceError(msg)

        raw_dir = self.store.source_raw_dir(self.source_name)
        raw_path = raw_dir / f"{ticker}_{start_date}_{end_date}.json"
        raw_path.write_text(response.text, encoding="utf-8")

        payload = response.json()
        frame = parse_chart_payload(payload, ticker)
        if frame.empty:
            msg = f"Yahoo returned no rows for {ticker}"
            raise DataSourceError(msg)

        frame["source_refresh_time"] = utc_now()
        return frame[
            [
                "as_of_date",
                "ticker",
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "volume",
                "source_refresh_time",
            ]
        ], raw_path


def parse_chart_payload(payload: dict[str, object], ticker: str) -> pd.DataFrame:
    """Parse Yahoo chart JSON into normalized OHLCV rows."""
    chart = payload.get("chart", {})
    if not isinstance(chart, dict):
        msg = "Yahoo chart payload missing chart object"
        raise DataSourceError(msg)
    error = chart.get("error")
    if error:
        msg = f"Yahoo chart error for {ticker}: {json.dumps(error)}"
        raise DataSourceError(msg)
    results = chart.get("result")
    if not isinstance(results, list) or not results:
        return pd.DataFrame()

    result = results[0]
    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators", {})
    quote = (indicators.get("quote") or [{}])[0]
    adjclose = (indicators.get("adjclose") or [{}])[0].get("adjclose")
    frame = pd.DataFrame(
        {
            "as_of_date": pd.to_datetime(timestamps, unit="s", utc=True).tz_localize(None).normalize(),
            "ticker": normalize_ticker(ticker),
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "adj_close": adjclose if adjclose is not None else quote.get("close"),
            "volume": quote.get("volume"),
        }
    )
    return frame.dropna(subset=["as_of_date"]).sort_values("as_of_date").reset_index(drop=True)
