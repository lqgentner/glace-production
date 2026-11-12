"""Google Earth Engine plumbing: authentication, retries, and generic reducers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

import ee
import pandas as pd
import shapely
import tenacity

from glace_production.config import settings

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

EE_HIGH_VOLUME_URL = "https://earthengine-highvolume.googleapis.com"

MODIS_PIXEL_SIZE = 1000

# Substrings of transient Earth Engine backend errors that are worth retrying.
# These are server-side/overload conditions (often triggered when many jobs hit
# the backend concurrently), as opposed to deterministic query errors.
_EE_TRANSIENT_MESSAGES = (
    "computation timed out",
    "timed out",
    "too many concurrent aggregations",
)


class SupportsGeoInterface(Protocol):
    """Anything exposing the ``__geo_interface__`` mapping, such as a shapely geometry."""

    @property
    def __geo_interface__(self) -> dict[str, Any]:
        """The GeoJSON-like mapping describing the geometry."""
        ...


def _is_transient_ee_error(exc: BaseException) -> bool:
    """Return ``True`` for Earth Engine errors that are safe to retry."""
    return isinstance(exc, ee.EEException) and any(
        token in str(exc).lower() for token in _EE_TRANSIENT_MESSAGES
    )


def get_info(obj: ee.computedobject.ComputedObject) -> Any:
    """Call ``obj.getInfo()`` with exponential backoff on transient EE errors.

    Earth Engine's ``getInfo()`` materialises a server-side computation and can
    fail with ``EEException: Computation timed out.`` when the backend is
    overloaded (e.g. many VMs issuing concurrent requests). Retries the call a
    few times with jittered exponential backoff; non-transient errors (bad
    asset, syntax, quota) are re-raised immediately.
    """
    retryer = tenacity.Retrying(
        reraise=True,
        stop=tenacity.stop_after_attempt(5),
        wait=tenacity.wait_exponential(multiplier=2, min=4, max=60) + tenacity.wait_random(0, 5),
        retry=tenacity.retry_if_exception(_is_transient_ee_error),
        # tenacity's LoggerProtocol promises `log(..., **kwargs)`, which the stdlib
        # Logger does not accept; it is only ever called as `log(level, msg)`.
        # pyrefly: ignore[bad-argument-type]
        before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
    )
    return retryer(obj.getInfo)


def initialize_ee(url: str = EE_HIGH_VOLUME_URL) -> None:
    """Initialize Earth Engine using service account credentials from settings.

    Reads the key from the file ``EE_SERVICE_ACC_KEY_FILE`` points at, so the
    private key stays out of the process environment.
    """
    acc = settings.ee_service_acc
    if not acc.key_file:
        msg = "Set EE_SERVICE_ACC_KEY_FILE"
        raise RuntimeError(msg)
    credentials = ee.ServiceAccountCredentials(email=acc.email, key_file=acc.key_file)
    ee.Initialize(credentials=credentials, opt_url=url)


def ensure_initialized(url: str = EE_HIGH_VOLUME_URL) -> None:
    """Initialize Earth Engine unless this process already did so."""
    if not ee.data.is_initialized():
        initialize_ee(url)


def shapely_to_ee_geometry(shapely_geom: SupportsGeoInterface) -> ee.Geometry:
    """Convert a Shapely geometry to an Earth Engine geometry."""
    geojson = shapely.geometry.mapping(shapely_geom)
    return ee.Geometry(geojson)


def get_statistics_from_collection(
    col: ee.ImageCollection,
    agg_func: Callable[..., ee.Feature],
    **func_kwargs: Any,
) -> pd.DataFrame:
    """Reduce every image of a collection to a feature and materialise it as a DataFrame.

    Parameters
    ----------
    col : ee.ImageCollection
        Collection to aggregate over.
    agg_func : Callable[..., ee.Feature]
        Per-image reducer returning an ``ee.Feature`` with a ``date`` property
        holding milliseconds since epoch.
    **func_kwargs : Any
        Extra keyword arguments forwarded to ``agg_func``.

    Returns
    -------
    pd.DataFrame
        Feature properties indexed by UTC date, sorted ascending.
    """

    def aggregate(img: ee.Image) -> ee.Feature:
        return agg_func(img, **func_kwargs)

    tseries = col.map(aggregate)
    info = get_info(tseries)
    records = [f["properties"] for f in info["features"]]
    if not records:
        # An empty collection (e.g. a date range beyond the product's coverage)
        # still has to yield a correctly indexed frame so callers can concatenate.
        return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC", name="date"))
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True)
    # EE returns None for masked/missing pixels; cast all non-date columns to
    # float so None → NaN and downstream interpolation works correctly.
    numeric_cols = [c for c in df.columns if c != "date"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    return df.sort_values("date").set_index("date")
