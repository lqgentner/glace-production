"""Read a dataset's layers at the stored sample locations."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
import re
import threading
from typing import TYPE_CHECKING, cast

from freezebase.mgrs import MGRSGeoBox
import numpy as np
from rasterio.enums import Resampling
from rasterio.windows import Window
from scipy.ndimage import minimum_filter, uniform_filter

from glace_production.config import settings
from glace_production.validation.config import PIXEL_SIZE_M
from glace_production.validation.rasters import read_tile, warp_to_geobox
from glace_production.validation.store import KEY_COLUMNS, ValidationStore

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray
    from odc.geo.geobox import GeoBox
    import pandas as pd
    from upath import UPath

    from glace_production.validation.config import DatasetConfig, LayerSpec

logger = logging.getLogger(__name__)

EXTRACTION_VERSION = "2"
"""Bumped whenever the extraction changes what it writes for an unchanged dataset config
(2: window statistics on a halo from the neighbouring tiles)."""

_TILE_RE = r"(?P<tile>\d{2}[A-Z]{3}\d{2})"
_tiles_cache: dict[tuple[str, str], dict[str, Path | UPath]] = {}
_tiles_lock = threading.Lock()


def layer_tiles(comp_dir: Path | UPath, spec: LayerSpec) -> dict[str, Path | UPath]:
    """Return every tile a layer has a file for under ``comp_dir``, keyed by MGRS code.

    Listed once per (directory, pattern) and cached for the process, since the
    halo read of every tile of a year consults the same listing. A tile with
    several matching files is rejected, as :func:`resolve_layer_path` does.

    Parameters
    ----------
    comp_dir
        Directory holding the layer's tiles.
    spec
        Layer whose file pattern to list.

    Returns
    -------
    dict[str, Path | UPath]
        File per MGRS tile code.
    """
    key = (str(comp_dir), spec.pattern)
    # The lock is held across the listing itself, so concurrent first callers
    # wait for one listing instead of each running their own.
    with _tiles_lock:
        if key in _tiles_cache:
            return _tiles_cache[key]
        name_re = re.compile(
            "^" + re.escape(spec.pattern).replace(r"\{tile\}", _TILE_RE).replace(r"\*", ".*") + "$"
        )
        tiles: dict[str, Path | UPath] = {}
        for path in sorted(comp_dir.glob(spec.pattern.format(tile="*"))):
            match = name_re.match(path.name)
            if not match:
                continue
            if match["tile"] in tiles:
                msg = f"several files for tile {match['tile']} match {comp_dir / spec.pattern}"
                raise ValueError(msg)
            tiles[match["tile"]] = path
        _tiles_cache[key] = tiles
        return tiles


def resolve_layer_path(comp_dir: Path | UPath, spec: LayerSpec, tile: str) -> Path | UPath:
    """Return the one file matching a layer pattern for a tile; raise on zero or several."""
    pattern = spec.pattern.format(tile=tile)
    matches = sorted(comp_dir.glob(pattern))
    if not matches:
        msg = f"no file matches {comp_dir / pattern}"
        raise FileNotFoundError(msg)
    if len(matches) > 1:
        msg = f"{len(matches)} matches for {comp_dir / pattern}: {[m.name for m in matches]}"
        raise ValueError(msg)
    return matches[0]


def window_stats(
    data: NDArray[np.floating], window: int
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Mean and (population) standard deviation over a square window, NaN-aware.

    A pixel gets a value only where every pixel of its window is finite, so
    the statistics are NaN next to nodata and within ``window // 2`` pixels
    of the array edge.
    """
    finite = np.isfinite(data)
    values = np.where(finite, data, 0.0).astype(np.float64)
    # A window is complete when its minimum over the finite mask is 1, with
    # out-of-bounds pixels counted as missing; exact for any window size.
    full = minimum_filter(finite.astype(np.uint8), window, mode="constant", cval=0) == 1
    mean = uniform_filter(values, window, mode="constant")
    sq_mean = uniform_filter(values * values, window, mode="constant")
    # Only complete windows are kept, and those average over finite values
    # alone, so the means need no renormalisation.
    var = np.clip(sq_mean - mean * mean, 0.0, None)
    mean_out = np.where(full, mean, np.nan).astype(np.float32)
    std_out = np.where(full, np.sqrt(var), np.nan).astype(np.float32)
    return mean_out, std_out


def _to_measurement(data: NDArray[np.float32], spec: LayerSpec) -> NDArray[np.float32]:
    """Backscatter stored linear is converted to dB; everything else is returned as is."""
    if spec.kind == "backscatter" and not spec.decibel:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(data > 0, 10.0 * np.log10(data), np.nan).astype(np.float32)
    return data


def read_layer(path: Path | UPath, geobox: GeoBox, spec: LayerSpec) -> NDArray[np.float32]:
    """Read one layer onto the tile grid, backscatter converted to dB."""
    # read_tile's return type also covers raw=True (any dtype); called with the
    # default raw=False here, the array is always float32.
    return _to_measurement(cast("NDArray[np.float32]", read_tile(path, geobox)), spec)


def neighbour_tiles(
    comp_dir: Path | UPath, spec: LayerSpec, tile: str, padded: GeoBox
) -> list[tuple[str, Path | UPath, MGRSGeoBox]]:
    """Return the layer's other tiles on the same grid that overlap ``padded``.

    Tiles of another UTM zone are ignored: MGRS zones overlap, so a tile at a
    zone boundary is covered by its own zone's neighbours, and a cross-zone
    resampling is never needed.

    Parameters
    ----------
    comp_dir
        Directory holding the layer's tiles.
    spec
        Layer whose tiles to consider.
    tile
        MGRS code of the tile being read; excluded from the result.
    padded
        The tile's grid padded by the halo.

    Returns
    -------
    list[tuple[str, Path | UPath, MGRSGeoBox]]
        Code, file and grid of every overlapping neighbour, in fill order.
    """
    found: list[tuple[str, Path | UPath, MGRSGeoBox]] = []
    for other, path in sorted(layer_tiles(comp_dir, spec).items()):
        if other == tile:
            continue
        other_box = MGRSGeoBox.from_mgrs(other, resolution=PIXEL_SIZE_M)
        if other_box.crs != padded.crs:
            continue
        overlap = padded.extent.intersection(other_box.extent)
        if not overlap.is_empty and overlap.area > 0:
            found.append((other, path, other_box))
    return found


def _ring_windows(shape: tuple[int, int], halo: int) -> list[Window]:
    """Return the four windows that make up a ``halo``-wide ring of a ``shape`` array."""
    ny, nx = shape
    return [
        Window(0, 0, nx, halo),
        Window(0, ny - halo, nx, halo),
        Window(0, halo, halo, ny - 2 * halo),
        Window(nx - halo, halo, halo, ny - 2 * halo),
    ]


def read_layer_padded(
    comp_dir: Path | UPath, spec: LayerSpec, tile: str, halo: int
) -> NDArray[np.float32]:
    """Read a layer's tile with a ``halo``-pixel ring filled from the neighbouring tiles.

    The tile itself is read exactly and is never altered, nodata included.
    The ring is filled from the same-grid neighbours of :func:`neighbour_tiles`,
    each read only over the ring windows (an exact copy, since the grids are
    aligned). Ring pixels no neighbour covers stay NaN.

    Parameters
    ----------
    comp_dir
        Directory holding the layer's tiles.
    spec
        Layer to read.
    tile
        MGRS code of the tile.
    halo
        Ring width in pixels; ``0`` reads the tile alone.

    Returns
    -------
    NDArray[np.float32]
        The tile padded by ``halo`` on every side.
    """
    geobox = MGRSGeoBox.from_mgrs(tile, resolution=PIXEL_SIZE_M)
    own = read_layer(resolve_layer_path(comp_dir, spec, tile), geobox, spec)
    if halo == 0:
        return own
    padded = geobox.pad(halo)
    out = np.full(padded.shape.yx, np.nan, dtype=np.float32)
    out[halo:-halo, halo:-halo] = own
    ring = np.ones(out.shape, dtype=bool)
    ring[halo:-halo, halo:-halo] = False
    windows = _ring_windows(out.shape, halo)
    for _other, path, other_box in neighbour_tiles(comp_dir, spec, tile, padded):
        if not (ring & np.isnan(out)).any():
            break
        strip = _to_measurement(
            warp_to_geobox(
                path,
                padded,
                resampling=Resampling.nearest,
                windows=windows,
                source_geobox=other_box,
            ),
            spec,
        )
        fill = ring & np.isnan(out) & np.isfinite(strip)
        out[fill] = strip[fill]
    return out


def layer_columns(
    comp_dir: Path | UPath, spec: LayerSpec, tile: str, name: str
) -> dict[str, NDArray[np.float32]]:
    """Return full-tile arrays of a layer and of every statistic it derives, keyed by column.

    The statistics are computed on the tile padded by half the largest window
    from the neighbouring tiles (:func:`read_layer_padded`), so they are
    continuous across tile borders; they are NaN where the window holds nodata
    or reaches past a tile edge that has no neighbour.

    Parameters
    ----------
    comp_dir
        Directory holding the layer's tiles.
    spec
        Layer to read.
    tile
        MGRS code of the tile.
    name
        Column name of the layer; derived columns are named from it.

    Returns
    -------
    dict[str, NDArray[np.float32]]
        Arrays of the tile's shape, the layer first, then its statistics.
    """
    halo = max((d.window // 2 for d in spec.derive), default=0)
    padded = read_layer_padded(comp_dir, spec, tile, halo)
    crop = (slice(halo, -halo), slice(halo, -halo)) if halo else (slice(None), slice(None))
    out = {name: padded[crop]}
    stats: dict[int, tuple[NDArray[np.float32], NDArray[np.float32]]] = {}
    for derived in spec.derive:
        if derived.window not in stats:
            mean, std = window_stats(padded, derived.window)
            stats[derived.window] = (mean[crop], std[crop])
        mean, std = stats[derived.window]
        out[derived.column(name)] = mean if derived.stat == "mean" else std
    return out


def _comp_dir(config: DatasetConfig, year: int, data_root: Path | UPath | None) -> Path | UPath:
    raw = config.comp_dir.format(year=year)
    if data_root is not None:
        return data_root / raw
    return settings.resolve_data_path(raw)


def extract_pair(
    config: DatasetConfig,
    tile: str,
    year: int,
    keys: pd.DataFrame,
    *,
    data_root: Path | UPath | None = None,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Features of every layer at the key rows of one (tile, year)."""
    comp_dir = _comp_dir(config, year, data_root)
    rows = keys["row"].to_numpy()
    cols = keys["col"].to_numpy()
    out = keys[KEY_COLUMNS].copy()
    report: list[dict[str, object]] = []
    for name, spec in config.layers.items():
        path = resolve_layer_path(comp_dir, spec, tile)
        for column, data in layer_columns(comp_dir, spec, tile, name).items():
            values = data[rows, cols]
            out[column] = values
            report.append(
                {
                    "tile": tile,
                    "year": year,
                    "layer": column,
                    "source": str(path),
                    "n_nan": int(np.isnan(values).sum()),
                    "missing": False,
                }
            )
    return out, report


def write_samples(
    config: DatasetConfig,
    store: ValidationStore,
    *,
    allow_missing: bool = False,
    overwrite: bool = False,
    max_workers: int = 8,
    data_root: Path | UPath | None = None,
) -> pd.DataFrame:
    """Extract every location pair for a dataset into the store; return the report."""
    out_dir = store.samples_dir(config.name)
    pairs = store.list_pairs(store.locations_dir)
    if not pairs:
        msg = f"no locations under {store.locations_dir}"
        raise FileNotFoundError(msg)
    upstream = {
        "locations": store.generation(store.locations_dir),
        "extraction_version": EXTRACTION_VERSION,
    }
    store.check_config(out_dir, config, overwrite=overwrite, upstream=upstream)

    def one(pair: tuple[str, int]) -> list[dict[str, object]]:
        tile, year = pair
        out_path = store.pair_path(out_dir, tile, year)
        if out_path.exists() and not overwrite:
            return []
        keys = store.read_parquet(store.pair_path(store.locations_dir, tile, year), KEY_COLUMNS)
        try:
            frame, report = extract_pair(config, tile, year, keys, data_root=data_root)
        except FileNotFoundError as exc:
            if not allow_missing:
                msg = f"{tile} {year}: {exc}"
                raise FileNotFoundError(msg) from exc
            logger.warning("%s %d: skipped, %s", tile, year, exc)
            if out_path.exists():
                out_path.unlink()  # a previous generation's file must not stand in
            return [
                {
                    "tile": tile,
                    "year": year,
                    "layer": None,
                    "source": None,
                    "n_nan": None,
                    "missing": True,
                }
            ]
        store.write_parquet(frame, out_path)
        return report

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        store.collect_reports(out_dir, [pool.submit(one, pair) for pair in pairs])
    return store.read_report(out_dir)
