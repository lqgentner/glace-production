"""Time-series helpers for finding low-value windows in a yearly index."""

from __future__ import annotations

from typing import cast

import pandas as pd


def find_lowest_consecutive_window(
    series: pd.Series,
    window_days: int = 60,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Find the consecutive time window with the lowest sum of values.

    Resamples the series to daily frequency, interpolates missing values,
    and identifies the window with minimum rolling sum.

    Parameters
    ----------
    series : pd.Series
        Time series with pd.Timestamp index.
    window_days : int, default=60
        Size of the rolling window in days.

    Returns
    -------
    start_date : pd.Timestamp
        First day of the window (inclusive).
    end_date : pd.Timestamp
        Last day of the window (inclusive).
    """
    series = series.asfreq("D").interpolate()
    rolling_sum = series.rolling(window=window_days).sum()
    end_date = cast("pd.Timestamp", rolling_sum.idxmin())
    # A rolling sum labelled at ``end_date`` covers that day and the preceding
    # ``window_days - 1``, so both bounds are inclusive and span ``window_days``.
    start_date = end_date - pd.Timedelta(days=window_days - 1)
    return start_date, end_date
