"""The MODIS year-completeness rule the window script warns with."""

from __future__ import annotations

import pandas as pd

from glace_production.snow import MODIS_LATENCY_DAYS, year_is_complete


def test_a_past_year_is_complete() -> None:
    assert year_is_complete(2015)


def test_the_current_year_is_not() -> None:
    assert not year_is_complete(pd.Timestamp.now(tz="UTC").year)


def test_last_year_completes_after_the_latency() -> None:
    year = pd.Timestamp.now(tz="UTC").year - 1
    settles = pd.Timestamp(year=year + 1, month=1, day=1, tz="UTC") + pd.Timedelta(
        days=MODIS_LATENCY_DAYS,
    )
    assert year_is_complete(year) == (pd.Timestamp.now(tz="UTC") >= settles)
