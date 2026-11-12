"""The composite tiles of `alps-total`, shared by the paper-plot notebooks.

The notebooks read the composite trees under `alps-total/{year}` (the published GLACE
tiles are byte-identical copies of `s1-comp`), so the tiles are taken from the MGRS
grid of the scope rather than from the release's item table.
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import TYPE_CHECKING

from freezebase.mgrs import MGRSGrid
from freezebase.raster import rasterio_open
import geopandas as gpd
import numpy as np
from rasterio.merge import merge
from shapely import box

from glace_production.composites import resolve_aoi

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from upath import UPath

PIXEL_SIZE_M = 10


def scope_tiles(scope: str) -> gpd.GeoDataFrame:
    """Every MGRS tile of a composite scope: its code, UTM EPSG code and WGS 84 footprint."""
    grid = MGRSGrid(resolve_aoi(scope), resolution=PIXEL_SIZE_M)
    return gpd.GeoDataFrame(
        {"tile": [g.mgrs_code for g in grid], "epsg": [g.crs.epsg for g in grid]},
        geometry=[g.geographic_extent.geom for g in grid],
        crs=4326,
    )


def window_tiles(bounds: tuple[float, float, float, float], epsg: int) -> list[str]:
    """Return the tiles of UTM zone ``epsg`` that overlap ``bounds`` (left, bottom, right, top).

    A tile that only touches the window's edge is left out.
    """
    window = box(*bounds)
    aoi = gpd.GeoSeries([window], crs=epsg).to_crs(4326)
    return sorted(
        g.mgrs_code
        for g in MGRSGrid(aoi, resolution=PIXEL_SIZE_M)
        if g.crs.epsg == epsg and g.extent.geom.intersection(window).area > 0
    )


def tile_file(comp_dir: Path | UPath, pattern: str, tile: str) -> Path | UPath:
    """Return the one file of ``tile`` matching ``pattern``, ``{tile}`` standing for its code."""
    (path,) = comp_dir.glob(pattern.format(tile=tile))
    return path


def read_window(
    files: Iterable[Path | UPath], bounds: tuple[float, float, float, float], epsg: int
) -> np.ndarray:
    """Mosaic band 1 of ``files`` inside ``bounds`` (left, bottom, right, top), nodata as NaN."""
    with ExitStack() as stack:
        sources = [stack.enter_context(rasterio_open(f)) for f in files]
        for src in sources:
            if src.crs.to_epsg() != epsg:
                msg = f"{src.name}: expected EPSG:{epsg}, got {src.crs}"
                raise ValueError(msg)
        data, _ = merge(sources, bounds=bounds, nodata=np.nan, dtype="float32", indexes=[1])
    return data[0]
