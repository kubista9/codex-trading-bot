from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

import pandas as pd


class DataSourceError(RuntimeError):
    """Raised when a source adapter cannot fetch or normalize data."""


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class SourceMetadata:
    """Metadata attached to a source download or staged table."""

    source: str
    retrieval_time: datetime = field(default_factory=utc_now)
    query: dict[str, Any] = field(default_factory=dict)
    raw_path: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class IngestionResult:
    """Normalized output from an ingestion adapter."""

    frame: pd.DataFrame
    metadata: SourceMetadata


class SourceAdapter(Protocol):
    """Protocol implemented by source-specific ingestion adapters."""

    source_name: str

    def fetch(self, *args: Any, **kwargs: Any) -> IngestionResult:
        """Fetch source data and return a normalized DataFrame."""
        ...
