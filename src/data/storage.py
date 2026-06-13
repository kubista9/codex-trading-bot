from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.base import SourceMetadata


class LocalDataStore:
    """Path helper for raw, staged, curated, and artifact layers."""

    def __init__(self, data_root: str | Path = "data", artifact_root: str | Path = "artifacts") -> None:
        self.data_root = Path(data_root)
        self.artifact_root = Path(artifact_root)
        self.raw_dir = self.data_root / "raw"
        self.staged_dir = self.data_root / "staged"
        self.curated_dir = self.data_root / "curated"
        for path in [self.raw_dir, self.staged_dir, self.curated_dir, self.artifact_root]:
            path.mkdir(parents=True, exist_ok=True)

    def source_raw_dir(self, source: str) -> Path:
        """Return and create the raw cache directory for a source."""
        path = self.raw_dir / source
        path.mkdir(parents=True, exist_ok=True)
        return path

    def table_path(self, layer: str, name: str, suffix: str) -> Path:
        """Build a path for a named table in a data layer."""
        layer_map = {
            "raw": self.raw_dir,
            "staged": self.staged_dir,
            "curated": self.curated_dir,
            "artifact": self.artifact_root,
        }
        if layer not in layer_map:
            msg = f"Unknown layer: {layer}"
            raise ValueError(msg)
        return layer_map[layer] / f"{name}.{suffix.lstrip('.')}"

    def write_dataframe(
        self,
        frame: pd.DataFrame,
        layer: str,
        name: str,
        *,
        include_csv: bool = True,
        include_parquet: bool = True,
        index: bool = False,
    ) -> dict[str, Path]:
        """Write a DataFrame to Parquet and/or CSV, returning created paths."""
        written: dict[str, Path] = {}
        if include_parquet:
            parquet_path = self.table_path(layer, name, "parquet")
            frame.to_parquet(parquet_path, index=index)
            written["parquet"] = parquet_path
        if include_csv:
            csv_path = self.table_path(layer, name, "csv")
            frame.to_csv(csv_path, index=index)
            written["csv"] = csv_path
        return written

    def write_metadata(self, metadata: SourceMetadata | dict[str, Any], layer: str, name: str) -> Path:
        """Write source metadata as JSON."""
        payload = asdict(metadata) if isinstance(metadata, SourceMetadata) else dict(metadata)
        if "retrieval_time" in payload and hasattr(payload["retrieval_time"], "isoformat"):
            payload["retrieval_time"] = payload["retrieval_time"].isoformat()
        path = self.table_path(layer, name, "metadata.json")
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path
