"""Snow extent determination from MODIS.

The public entry point is :func:`find_modis_snowfree_window`, which returns the
consecutive window with the lowest regional snow cover. Every call queries Earth
Engine: the result is a compositing window that ``compute-snow-window.py`` writes
into a committed windows file, and that file — not an on-disk series cache — is
where a value is reused from.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
from typing import TYPE_CHECKING, Literal

import ee
import matplotlib.pyplot as plt
import pandas as pd
import shapely

from glace_production.gee import (
    MODIS_PIXEL_SIZE,
    SupportsGeoInterface,
    ensure_initialized,
    get_statistics_from_collection,
    shapely_to_ee_geometry,
)
from glace_production.timeseries import find_lowest_consecutive_window

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

MODIS_TERRA_DAILY = "MODIS/061/MOD10A1"
MODIS_AQUA_DAILY = "MODIS/061/MYD10A1"

#: Chunk size of a single Earth Engine query. Chunking keeps individual
#: server-side computations small enough to avoid backend timeouts on large AOIs.
_CHUNK_MONTHS = 1

#: Maximum number of concurrent Earth Engine requests. Kept below the underlying
#: HTTP connection pool size (10) so that connections are not discarded and rebuilt.
_MAX_WORKERS = 8

#: MODIS products are published with a few days of latency, and near-real-time
#: values are revised after, so a calendar year is only considered complete once
#: this much time has passed. See :func:`year_is_complete`.
MODIS_LATENCY_DAYS = 14

#: Tolerance for simplifying the ROI before it is sent to Earth Engine, in metres.
#: The reducers run at ``MODIS_PIXEL_SIZE``, so outline detail far below that does
#: not change the result.
_SIMPLIFY_M = 100

_M_PER_DEG = 111_320


# ---------------------------------------------------------------------------
# Server-side reducers
# ---------------------------------------------------------------------------


def ndsi_metrics(
    img: ee.Image,
    roi: ee.Geometry,
    *,
    band_name: str = "NDSI_Snow_Cover",
    qa_band_name: str = "NDSI_Snow_Cover_Basic_QA",
) -> ee.Feature:
    """Calculate NDSI snow metrics for daily MODIS products."""
    # Catagorical values (above 100) are already masked out
    ndsi = img.select(band_name).rename("snow")
    valid = ndsi.mask().multiply(100).rename("valid")
    night = img.select(qa_band_name).eq(211).multiply(100).rename("night")

    pct_mean = ee.Image.cat([ndsi, night, valid]).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=roi,
        scale=MODIS_PIXEL_SIZE,
        maxPixels=int(1e9),
        bestEffort=True,
    )

    return ee.Feature(
        None,
        {
            "date": img.date().millis(),
            "snow": pct_mean.get("snow"),
            "night": pct_mean.get("night"),
            "valid": pct_mean.get("valid"),
        },
    )


def merge_modis(
    terra: ee.ImageCollection,
    aqua: ee.ImageCollection,
    precedence: Literal["terra", "aqua"] = "terra",
) -> ee.ImageCollection:
    """
    Combine MODIS Terra and Aqua products.

    ``precedence`` sets the primary product to use. If the primary product is not available,
    the secondary product will be used to fill values.
    """
    if precedence == "terra":
        primary = terra
        secondary = aqua
    elif precedence == "aqua":
        primary = aqua
        secondary = terra
    else:
        msg = f"`precedence` must be 'terra' or 'aqua', received '{precedence}'."
        raise ValueError(msg)

    # Merge collections and use mosaic() for same-day combination
    combined = secondary.merge(primary)

    # Group by date and mosaic (Terra (last) on top)
    def daily_mosaic(date: ee.Date) -> ee.Image:
        daily_images = combined.filterDate(date, date.advance(1, "day"))
        return daily_images.mosaic().set(
            {"system:time_start": date.millis(), "system:index": date.format("YYYY_MM_dd")},
        )

    # Unique dates, sorted: distinct() keeps first occurrences, so a date only the
    # primary product has would otherwise land after all of the secondary's dates
    dates = combined.aggregate_array("system:time_start").distinct().sort()

    return ee.ImageCollection.fromImages(dates.map(lambda d: daily_mosaic(ee.Date(d))))


def backfill_modis(col: ee.ImageCollection, n_days: int = 8) -> ee.ImageCollection:
    """
    Fill missing/invalid pixels in current image using number of specified previous days.

    Strategy: Start with current image, then fill gaps with most recent valid data
    from the previous 8 days, working backwards in time.
    """

    def backfill_single_image(img: ee.Image) -> ee.Image:
        current_date = img.date()

        # Current and previous days, sorted: lastNonNull follows collection order
        coll_8days = col.filterDate(
            current_date.advance(-n_days + 1, "day"),
            current_date.advance(1, "day"),
        ).sort("system:time_start")
        return (
            coll_8days.reduce(ee.Reducer.lastNonNull())
            .rename(img.bandNames())
            .set("system:time_start", img.date().millis())
        )

    return col.map(backfill_single_image)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def _simplify_roi(geom: shapely.geometry.base.BaseGeometry) -> shapely.geometry.base.BaseGeometry:
    """Reduce the vertex count of a WGS-84 ROI to keep Earth Engine requests small.

    Earth Engine embeds the ROI in every request body, so a detailed AOI can exceed
    the 10 MB payload limit on its own.
    """
    simplified = geom.simplify(_SIMPLIFY_M / _M_PER_DEG)
    logger.debug(
        "Simplified ROI from %d to %d vertices",
        shapely.get_num_coordinates(geom),
        shapely.get_num_coordinates(simplified),
    )
    return simplified


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def _slice(df: pd.DataFrame, start: pd.Timestamp, stop: pd.Timestamp) -> pd.DataFrame:
    """Restrict a UTC-indexed frame to ``[start, stop)``, accepting naive bounds."""
    index = df.index
    lo, hi = (t.tz_localize("UTC") if t.tzinfo is None else t for t in (start, stop))
    return df[(index >= lo) & (index < hi)]


def _month_chunks(
    start: pd.Timestamp,
    stop: pd.Timestamp,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split ``[start, stop)`` into chunks of ``_CHUNK_MONTHS`` months."""
    chunks = []
    chunk_start = start.normalize()
    while chunk_start < stop:
        chunk_stop = min(chunk_start + pd.DateOffset(months=_CHUNK_MONTHS), stop)
        chunks.append((chunk_start, chunk_stop))
        chunk_start = chunk_stop
    return chunks


def _fetch_ndsi(
    roi: ee.Geometry,
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> pd.DataFrame:
    """Query daily Terra/Aqua NDSI metrics for the given date ranges.

    The ranges are split into monthly chunks that are fetched concurrently; a
    year of data costs roughly one round-trip instead of twelve.
    """
    chunks = [c for start, stop in ranges for c in _month_chunks(start, stop)]
    if not chunks:
        return pd.DataFrame()

    span_start = min(c[0] for c in chunks)
    span_stop = max(c[1] for c in chunks)
    terra = ee.ImageCollection(MODIS_TERRA_DAILY).filterDate(span_start, span_stop)
    aqua = ee.ImageCollection(MODIS_AQUA_DAILY).filterDate(span_start, span_stop)
    col = backfill_modis(merge_modis(terra, aqua, precedence="terra"))

    def fetch_chunk(rng: tuple[pd.Timestamp, pd.Timestamp]) -> pd.DataFrame:
        return get_statistics_from_collection(
            col.filterDate(rng[0], rng[1]),
            ndsi_metrics,
            roi=roi,
        )

    logger.info("Querying MODIS NDSI in %d chunks...", len(chunks))
    with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(chunks))) as pool:
        frames = list(pool.map(fetch_chunk, chunks))
    return pd.concat(frames).sort_index()


def year_is_complete(year: int) -> bool:
    """Whether MODIS coverage for ``year`` can be considered final.

    A window computed before this is a provisional value: the NRT products the
    query reads are revised for a while after acquisition, and a later query can
    move it by a few days.
    """
    year_stop = pd.Timestamp(year=year + 1, month=1, day=1, tz="UTC")
    return pd.Timestamp.now(tz="UTC") >= year_stop + pd.Timedelta(days=MODIS_LATENCY_DAYS)


def _ndsi_series(
    roi: SupportsGeoInterface,
    start: pd.Timestamp,
    stop: pd.Timestamp,
) -> pd.DataFrame:
    """Return daily NDSI metrics for ``[start, stop)``."""
    geom = _simplify_roi(shapely.geometry.shape(roi.__geo_interface__))
    return _slice(_fetch_ndsi(shapely_to_ee_geometry(geom), [(start, stop)]), start, stop)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_modis_snowfree_window(
    roi: SupportsGeoInterface,
    start: pd.Timestamp,
    stop: pd.Timestamp,
    *,
    window_days: int,
    plot_file: Path | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Find the consecutive window with the lowest snow cover using MODIS Terra/Aqua NDSI.

    Parameters
    ----------
    roi : SupportsGeoInterface
        Region of interest as a Shapely geometry in WGS-84. Simplified to a
        tolerance well below the MODIS pixel size before it is queried.
    start : pd.Timestamp
        Start of the search period.
    stop : pd.Timestamp
        End of the search period (exclusive).
    window_days : int
        Length of the snow-free window in days.
    plot_file : Path | None, default: None
        Saves a plot of the timeseries to this path if provided.

    Returns
    -------
    tuple[pd.Timestamp, pd.Timestamp]
        First and last day of the lowest-snow window, both inclusive.
    """
    ensure_initialized()
    df_ndsi = _ndsi_series(roi, start, stop)
    if df_ndsi.empty:
        msg = f"No MODIS NDSI observations between {start.date()} and {stop.date()}."
        raise ValueError(msg)

    logger.info("Finding %d-day snow-free window...", window_days)
    snowfree_start, snowfree_stop = find_lowest_consecutive_window(
        df_ndsi.snow,
        window_days=window_days,
    )
    logger.info("Snow-free period: %s to %s", snowfree_start.date(), snowfree_stop.date())

    if plot_file:
        _plot_ndsi_timeseries(
            ndsi_series=df_ndsi.snow,
            start=snowfree_start,
            stop=snowfree_stop,
            output_path=plot_file,
        )

    return snowfree_start, snowfree_stop


def _plot_ndsi_timeseries(
    ndsi_series: pd.Series,
    start: pd.Timestamp,
    stop: pd.Timestamp,
    output_path: Path,
) -> None:
    """
    Plot MODIS NDSI time series with highlighted snow-free window.

    Parameters
    ----------
    ndsi_series : pd.Series
        NDSI time series data
    start : pd.Timestamp
        Start of snow-free window
    stop : pd.Timestamp
        End of snow-free window
    output_path : Path
        Path to save the plot
    """
    lowest_window = ndsi_series.loc[start:stop]
    window_size = (stop - start).days + 1  # both bounds inclusive

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ndsi_series.index, ndsi_series, linewidth=1)
    ax.fill_between(
        ndsi_series.index,
        ndsi_series,
        color="tab:blue",
        alpha=0.2,
        label="Full time series",
    )
    ax.fill_between(
        lowest_window.index,
        lowest_window,
        alpha=0.7,
        color="tab:orange",
        label=f"Lowest {window_size}-day window",
    )

    ax.set(
        xlabel="Date",
        ylabel="NDSI",
        ylim=(0, 110),
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved NDSI plot to '%s'", output_path)
