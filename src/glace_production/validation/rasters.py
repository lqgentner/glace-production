"""Reading rasters onto a tile geobox.

Every read validates the file's grid against the geobox first: matching a
filename and shape does not prove matching CRS or transform, and a shifted
raster would yield plausible but wrong features without this check.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from freezebase.raster import rasterio_open
import numpy as np
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from glace_production.labels import check_label_tags

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from numpy.typing import NDArray
    from odc.geo.geobox import GeoBox
    from rasterio.io import DatasetReader
    from rasterio.windows import Window
    from upath import UPath

_TRANSFORM_TOL = 1e-6


def validate_header(src: DatasetReader, geobox: GeoBox) -> None:
    """Raise ``ValueError`` unless ``src`` sits exactly on ``geobox``."""
    problems = []
    if src.count != 1:
        problems.append(f"{src.count} bands")
    if tuple(src.shape) != tuple(geobox.shape.yx):
        problems.append(f"shape {src.shape} != {tuple(geobox.shape.yx)}")
    if src.crs is None or geobox.crs is None or src.crs.to_epsg() != geobox.crs.epsg:
        problems.append(f"crs {src.crs} != {geobox.crs}")
    if any(
        not math.isclose(a, b, rel_tol=0.0, abs_tol=_TRANSFORM_TOL)
        for a, b in zip(src.transform[:6], geobox.affine[:6], strict=True)
    ):
        problems.append(f"transform {src.transform} != {geobox.affine}")
    if problems:
        msg = f"{src.name}: grid mismatch ({'; '.join(problems)})"
        raise ValueError(msg)


def read_band(
    src: DatasetReader, band: int = 1, *, window: Window | None = None
) -> NDArray[np.float32]:
    """Read one band as float32 with nodata and infinities set to NaN and scales applied."""
    data = src.read(band, window=window).astype(np.float32)
    nodata = src.nodatavals[band - 1]
    invalid = ~np.isfinite(data)
    if nodata is not None and not (isinstance(nodata, float) and math.isnan(nodata)):
        invalid |= data == nodata
    scale, offset = src.scales[band - 1], src.offsets[band - 1]
    if scale != 1.0 or offset != 0.0:
        data = data * np.float32(scale) + np.float32(offset)
    data[invalid] = np.nan
    return data


def read_tile(
    path: Path | UPath, geobox: GeoBox, *, raw: bool = False
) -> NDArray[np.float32] | NDArray[np.generic]:
    """Read a single-band tile raster that must sit on ``geobox``.

    With ``raw`` the band is returned as stored (dtype and nodata untouched),
    which is what label masks need: their nodata value 255 is the ignore class.
    """
    with rasterio_open(path) as src:
        validate_header(src, geobox)
        return src.read(1) if raw else read_band(src)


def read_label(
    path: Path | UPath, geobox: GeoBox, *, year: int | None = None
) -> NDArray[np.uint8]:
    """Read a label mask that must sit on ``geobox`` and carry the label contract's tags.

    The mask's 255 is the ignore class (other years' glaciers, outside the
    domain); a mask without the tags is refused, since its ignore value may
    not cover the domain.
    """
    with rasterio_open(path) as src:
        validate_header(src, geobox)
        check_label_tags(src.tags(), year=year)
        return np.asarray(src.read(1), dtype=np.uint8)


def warp_to_geobox(
    path: Path | UPath,
    geobox: GeoBox,
    *,
    resampling: Resampling = Resampling.bilinear,
    windows: Sequence[Window] | None = None,
    source_geobox: GeoBox | None = None,
) -> NDArray[np.float32]:
    """Reproject and resample a raster onto ``geobox`` (NaN outside the source).

    Parameters
    ----------
    path
        Raster to read.
    geobox
        Target grid.
    resampling
        Warp resampling method.
    windows
        Pixel windows of ``geobox`` to fill; the rest of the returned array
        stays NaN. ``None`` reads the whole grid.
    source_geobox
        Grid the source file must sit on exactly (checked with
        :func:`validate_header` before warping); ``None`` skips the check.

    Returns
    -------
    NDArray[np.float32]
        Array of ``geobox.shape``.
    """
    ny, nx = geobox.shape.yx
    with rasterio_open(path) as src:
        if source_geobox is not None:
            validate_header(src, source_geobox)
        with WarpedVRT(
            src,
            crs=geobox.crs.to_wkt() if geobox.crs is not None else None,
            transform=geobox.affine,
            width=nx,
            height=ny,
            resampling=resampling,
            nodata=np.nan,
            dtype="float32",
        ) as vrt:
            if windows is None:
                return read_band(vrt)
            out = np.full((ny, nx), np.nan, dtype=np.float32)
            for window in windows:
                out[window.toslices()] = read_band(vrt, window=window)
            return out
