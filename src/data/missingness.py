from __future__ import annotations

import pandas as pd


def is_missingness_flag(column: str) -> bool:
    """Return true when a column is a generated missingness audit flag."""
    return column.endswith("__is_missing") or column.endswith("__is_structurally_missing")


def missingness_flag_columns(frame: pd.DataFrame) -> list[str]:
    """Return generated missingness flag columns in frame order."""
    return [column for column in frame.columns if is_missingness_flag(column)]


def drop_missingness_flag_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop generated missingness flag columns from a user-facing wide table."""
    return frame.drop(columns=missingness_flag_columns(frame), errors="ignore")


def summarize_missingness_flags(frame: pd.DataFrame, *, table_name: str) -> pd.DataFrame:
    """Summarize generated missingness flags without exporting long FALSE columns."""
    records: list[dict[str, object]] = []
    for column in missingness_flag_columns(frame):
        values = frame[column].fillna(False).astype(bool)
        true_count = int(values.sum())
        records.append(
            {
                "table_name": table_name,
                "flag_column": column,
                "true_count": true_count,
                "false_count": int((~values).sum()),
                "true_rate": float(true_count / len(values)) if len(values) else 0.0,
                "any_true": bool(true_count > 0),
            }
        )
    return pd.DataFrame.from_records(records)
