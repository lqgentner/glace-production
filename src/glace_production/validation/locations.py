"""Validation locations: every pixel of the scope of each label mask.

The scope of a (tile, year) is its glacier pixels plus the ring of
``RING_M`` outside the outlines. Distances come from the outlines rasterised
with a halo, so a glacier cut by the tile edge has no false boundary there.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
import re
from typing import TYPE_CHECKING, cast

from freezebase.mgrs import MGRSGeoBox
import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.enums import Resampling
from scipy.ndimage import distance_transform_edt

from glace_production.config import settings
from glace_production.datasets import SGI2016, SGI2023
from glace_production.geo import MGRSInventory
from glace_production.labels import IGNORE_VALUE
from glace_production.validation.bands import RING_M
from glace_production.validation.config import PIXEL_SIZE_M
from glace_production.validation.rasters import read_label, warp_to_geobox

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray
    from upath import UPath

    from glace_production.validation.config import LocationsConfig
    from glace_production.validation.store import ValidationStore

logger = logging.getLogger(__name__)

LOCATION_COLUMNS = [
    "tile",
    "year",
    "row",
    "col",
    "label",
    "dist_m",
    "gid",
    "near_gid",
    "debris",
    "elevation",
]
"""Columns of every locations file, in order."""

_LABEL_RE = re.compile(
    r"^LABELS_(?P<inv>[A-Z0-9]+)_(?P<year>\d{4})_(?P<tile>\d{2}[A-Z]{3}\d{2})\.tif$"
)
_INVENTORIES = {"SGI2016": SGI2016, "SGI2023": SGI2023}


class LabelMismatchError(RuntimeError):
    """The label mask and the rasterised inventory disagree inside the tile."""


def parse_label_name(name: str) -> tuple[str, int, str] | None:
    """Split ``LABELS_{inventory}_{year}_{tile}.tif`` into its parts, or ``None``."""
    match = _LABEL_RE.match(name)
    if match is None:
        return None
    return match["inv"], int(match["year"]), match["tile"]


def select_tile_locations(
    *,
    tile: str,
    year: int,
    label: NDArray[np.uint8],
    outlines: NDArray[np.bool_],
    gids: NDArray[np.int32],
    debris: NDArray[np.bool_],
    elevation: NDArray[np.float32],
    config: LocationsConfig,
) -> tuple[pd.DataFrame, dict[str, int | str]]:
    """Select the scope rows of one (tile, year).

    Parameters
    ----------
    label
        The tile's label mask (1 glacier, 0 background, 255 ignore). Ignore
        pixels never enter the scope.
    outlines
        The year's outlines rasterised onto the tile padded by ``config.pad_px``.
    gids
        Glacier ids on the same padded grid, -1 outside.
    debris
        Tile-sized flag: debris-covered pixel.
    elevation
        Tile-sized elevation.

    Returns
    -------
    tuple[pandas.DataFrame, dict[str, int | str]]
        The rows (:data:`LOCATION_COLUMNS`) and the pixel-count report.
    """
    pad = config.pad_px
    inside = outlines[pad:-pad, pad:-pad]
    glacier = label == 1
    known = label != IGNORE_VALUE
    # Glacier exactly where the outlines are, unless the mask makes no claim there.
    if not np.array_equal(glacier, inside & known):
        n_bad = int((glacier != (inside & known)).sum())
        msg = f"{tile} {year}: mask and inventory disagree on {n_bad} pixels"
        raise LabelMismatchError(msg)

    dist_m, near_gid = _distances(outlines, gids, pad, config.pad_m)
    ring = known & ~glacier & (dist_m > 0) & (dist_m <= RING_M)
    rows, cols = np.nonzero(glacier | ring)

    frame = pd.DataFrame(
        {
            "tile": tile,
            "year": np.int16(year),
            "row": rows.astype(np.uint16),
            "col": cols.astype(np.uint16),
            "label": label[rows, cols],
            "dist_m": dist_m[rows, cols].astype(np.float32),
            "gid": np.where(glacier, gids[pad:-pad, pad:-pad], -1).astype(np.int32)[rows, cols],
            "near_gid": near_gid[rows, cols].astype(np.int32),
            "debris": debris[rows, cols],
            "elevation": elevation[rows, cols].astype(np.float32),
        },
        columns=LOCATION_COLUMNS,
    )
    report = {
        "tile": tile,
        "year": year,
        "total": int(label.size),
        "ignore": int((~known).sum()),
        "glacier": int(glacier.sum()),
        "ring": int(ring.sum()),
        "selected": len(frame),
    }
    return frame, report


def _distances(
    outlines: NDArray[np.bool_], gids: NDArray[np.int32], pad: int, pad_m: float
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """Signed distance to the outline (m, censored at ``pad_m``) and nearest glacier id."""
    shape = (outlines.shape[0] - 2 * pad, outlines.shape[1] - 2 * pad)
    if not outlines.any():
        return np.full(shape, pad_m, np.float32), np.full(shape, -1, np.int32)
    if outlines.all():
        return np.full(shape, -pad_m, np.float32), gids[pad:-pad, pad:-pad].copy()
    dist_in = distance_transform_edt(outlines)
    dist_out, (ri, ci) = distance_transform_edt(~outlines, return_indices=True)
    dist = np.where(outlines, -dist_in, dist_out) * PIXEL_SIZE_M
    near = np.where(outlines, gids, gids[ri, ci])
    dist = np.clip(dist, -pad_m, pad_m)
    # A pixel clipped to +pad_m lies beyond the halo, so no nearest glacier is
    # known for it.
    near = np.where(dist >= pad_m, -1, near)
    crop = (slice(pad, -pad), slice(pad, -pad))
    return dist[crop].astype(np.float32), near[crop].astype(np.int32)


def _label_files(
    labels_dir: Path | UPath, config: LocationsConfig
) -> list[tuple[str, int, Path | UPath]]:
    files = []
    for path in sorted(labels_dir.glob("LABELS_*.tif")):
        parsed = parse_label_name(path.name)
        if parsed is None:
            continue
        inventory, year, tile = parsed
        if inventory == config.inventory and year in config.years:
            files.append((tile, year, path))
    if not files:
        msg = f"no {config.inventory} label masks for years {config.years} under {labels_dir}"
        raise FileNotFoundError(msg)
    return files


def write_locations(
    config: LocationsConfig,
    store: ValidationStore,
    *,
    overwrite: bool = False,
    max_workers: int = 4,
) -> pd.DataFrame:
    """Select the scope rows of every label mask of the config into the store.

    Returns
    -------
    pandas.DataFrame
        One row per (tile, year) with its pixel counts.
    """
    out_dir = store.locations_dir
    store.check_config(out_dir, config, overwrite=overwrite)
    labels_dir = settings.resolve_data_path(config.labels_dir)
    files = _label_files(labels_dir, config)
    dem = settings.resolve_data_path(config.dem)

    inventory = _INVENTORIES[config.inventory]().get_data(download=True)
    debris_all = _INVENTORIES[config.inventory](product="D").get_data(download=True)
    years = inventory[config.year_column]
    outlines_by_year = {
        y: MGRSInventory(gpd.GeoDataFrame(inventory[years == y])) for y in config.years
    }
    debris_years = debris_all[config.year_column]
    debris_by_year = {
        y: MGRSInventory(gpd.GeoDataFrame(debris_all[debris_years == y])) for y in config.years
    }

    # MGRSInventory's reprojection/spatial-index caches are an unsynchronised
    # check-then-populate; build them single-threaded for every EPSG code the
    # label tiles use before `one` shares these instances across the pool.
    epsgs: set[int] = set()
    for tile, _, _ in files:
        crs = MGRSGeoBox.from_mgrs(tile, resolution=PIXEL_SIZE_M).crs
        if crs is not None and crs.epsg is not None:
            epsgs.add(crs.epsg)
    for inventory_by_epsg in (*outlines_by_year.values(), *debris_by_year.values()):
        for epsg in epsgs:
            inventory_by_epsg.prepare(epsg)

    def one(item: tuple[str, int, Path | UPath]) -> list[dict[str, object]]:
        tile, year, path = item
        out_path = store.pair_path(out_dir, tile, year)
        if out_path.exists() and not overwrite:
            logger.info("%s %d: exists, skipped", tile, year)
            return []
        geobox = MGRSGeoBox.from_mgrs(tile, resolution=PIXEL_SIZE_M)
        padded = geobox.pad(config.pad_px)
        label = read_label(path, geobox, year=year)
        frame, report = select_tile_locations(
            tile=tile,
            year=year,
            label=label,
            outlines=outlines_by_year[year].rasterize(padded).astype(bool),
            gids=outlines_by_year[year].rasterize_values(padded, "gid"),
            debris=debris_by_year[year].rasterize(geobox).astype(bool),
            elevation=warp_to_geobox(dem, geobox, resampling=Resampling.bilinear),
            config=config,
        )
        store.write_parquet(frame, out_path)
        logger.info(
            "%s %d: %d rows (%d glacier)", tile, year, report["selected"], report["glacier"]
        )
        # dict[str, int | str] is narrower than the report row type; dict is
        # invariant in its value type, so a cast is needed although every
        # row is a valid dict[str, object] at runtime.
        return [cast("dict[str, object]", report)]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        store.collect_reports(out_dir, [pool.submit(one, item) for item in files])
    return store.read_report(out_dir)
