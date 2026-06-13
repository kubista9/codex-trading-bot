from __future__ import annotations

import os
from time import sleep
from typing import Iterable

import pandas as pd
import requests

from src.data.base import DataSourceError, IngestionResult, SourceMetadata, utc_now
from src.utils.env import load_dotenv
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


class FredAdapter:
    """FRED observations adapter with realtime availability fields."""

    source_name = "fred"
    base_url = "https://api.stlouisfed.org/fred"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_key_env: str = "FRED_API_KEY",
        timeout_seconds: int = 30,
        rate_limit_seconds: float = 0.2,
        session: requests.Session | None = None,
    ) -> None:
        load_dotenv()
        self.api_key = api_key or os.getenv(api_key_env)
        self.timeout_seconds = timeout_seconds
        self.rate_limit_seconds = rate_limit_seconds
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": "stock-research-platform/0.1"})

    def fetch(
        self,
        series_ids: Iterable[str],
        start_date: str,
        end_date: str,
    ) -> IngestionResult:
        """Fetch FRED observations in normalized long form."""
        if not self.api_key:
            msg = "FRED_API_KEY is required for FRED API ingestion"
            raise DataSourceError(msg)

        frames: list[pd.DataFrame] = []
        failures: list[str] = []
        for series_id in series_ids:
            try:
                frames.append(self._fetch_series(series_id, start_date, end_date))
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("FRED download failed for %s: %s", series_id, exc)
                failures.append(f"{series_id}: {exc}")
            sleep(self.rate_limit_seconds)

        if not frames:
            msg = "No FRED series were downloaded"
            raise DataSourceError(msg)

        metadata = SourceMetadata(
            source=self.source_name,
            query={"series_ids": list(series_ids), "start_date": start_date, "end_date": end_date},
            notes=failures,
        )
        return IngestionResult(pd.concat(frames, ignore_index=True), metadata)

    def _fetch_series(self, series_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        url = f"{self.base_url}/series/observations"
        params = {
            "series_id": series_id,
            "observation_start": start_date,
            "observation_end": end_date,
            "api_key": self.api_key,
            "file_type": "json",
        }
        response = self.session.get(url, params=params, timeout=self.timeout_seconds)
        if response.status_code >= 400:
            msg = f"FRED returned HTTP {response.status_code} for {series_id}"
            raise DataSourceError(msg)
        payload = response.json()
        observations = payload.get("observations", [])
        frame = pd.DataFrame(observations)
        if frame.empty:
            return pd.DataFrame(
                columns=[
                    "series_id",
                    "observation_date",
                    "available_date",
                    "value",
                    "realtime_start",
                    "realtime_end",
                    "source_refresh_time",
                ]
            )

        frame["series_id"] = series_id
        frame["observation_date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame["realtime_start"] = pd.to_datetime(frame["realtime_start"]).dt.normalize()
        frame["realtime_end"] = pd.to_datetime(frame["realtime_end"]).dt.normalize()
        frame["available_date"] = frame["realtime_start"]
        frame["value"] = pd.to_numeric(frame["value"].replace(".", pd.NA), errors="coerce")
        frame["source_refresh_time"] = utc_now()
        return frame[
            [
                "series_id",
                "observation_date",
                "available_date",
                "value",
                "realtime_start",
                "realtime_end",
                "source_refresh_time",
            ]
        ]
