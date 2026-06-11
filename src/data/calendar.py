from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def build_market_calendar(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    holidays: Iterable[str | pd.Timestamp] | None = None,
) -> pd.DatetimeIndex:
    """Build a daily weekday market calendar index.

    This is a conservative scaffold, not a full exchange calendar. A later
    milestone should replace or augment it with a vetted exchange calendar.
    """
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end < start:
        msg = "end_date must be on or after start_date"
        raise ValueError(msg)

    calendar = pd.bdate_range(start=start, end=end, name="as_of_date")
    if holidays is None:
        return calendar

    holiday_index = pd.DatetimeIndex(pd.to_datetime(list(holidays))).normalize()
    return calendar.difference(holiday_index)
