from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd


def normalize_ticker(ticker: str) -> str:
    """Normalize a public equity ticker for internal joins."""
    return ticker.strip().upper().replace(".", "-")


def normalize_cik(cik: str | int | None) -> str | None:
    """Normalize CIK values to SEC zero-padded ten-character strings."""
    if cik is None or pd.isna(cik):
        return None
    return f"{int(cik):010d}"


@dataclass(frozen=True)
class UniverseMember:
    """Security membership record with point-in-time-ready dates."""

    ticker: str
    cik: str | int | None = None
    listing_date: str | date | None = None
    membership_start: str | date | None = None
    membership_end: str | date | None = None
    delisting_date: str | date | None = None
    active: bool = True

    def to_record(self) -> dict[str, Any]:
        """Convert the member to a normalized dictionary."""
        return {
            "ticker": normalize_ticker(self.ticker),
            "cik": normalize_cik(self.cik),
            "listing_date": self.listing_date,
            "membership_start": self.membership_start,
            "membership_end": self.membership_end,
            "delisting_date": self.delisting_date,
            "active": self.active,
        }


def members_to_frame(members: list[UniverseMember | dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    """Normalize universe members into a DataFrame."""
    if isinstance(members, pd.DataFrame):
        frame = members.copy()
    else:
        records: list[dict[str, Any]] = []
        for member in members:
            if isinstance(member, UniverseMember):
                records.append(member.to_record())
            else:
                records.append(dict(member))
        frame = pd.DataFrame.from_records(records)

    if frame.empty:
        msg = "Universe members cannot be empty"
        raise ValueError(msg)

    frame["ticker"] = frame["ticker"].map(normalize_ticker)
    if "cik" in frame.columns:
        frame["cik"] = frame["cik"].map(normalize_cik)
    else:
        frame["cik"] = None

    for column in ["listing_date", "membership_start", "membership_end", "delisting_date"]:
        if column not in frame.columns:
            frame[column] = pd.NaT
        frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.normalize()

    if "active" not in frame.columns:
        frame["active"] = True
    frame["active"] = frame["active"].fillna(True).astype(bool)

    return frame
