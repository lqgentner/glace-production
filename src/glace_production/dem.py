"""Digital elevation models: the GLO-30 download config and the conversion to ellipsoidal heights.

National and global DEMs come as orthometric heights above a geoid (EGM2008 for
Copernicus GLO-30, CHGeo2004 for SwissALTI3D), while the SAR processing expects
heights above the WGS84 ellipsoid.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from freezebase import vrt
from freezebase.vrt import VRT
from pydantic import BaseModel, model_validator
import pyproj
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
import yaml

from glace_production.config import settings

if TYPE_CHECKING:
    from pathlib import Path

    from freezebase.raster import AnyPath
    from upath import UPath

# A compound CRS carries exactly a horizontal and a vertical component.
_COMPOUND_PARTS = 2


class Glo30DownloadConfig(BaseModel):
    """A committed Copernicus GLO-30 download: where it goes and what it covers.

    ``output`` is relative to ``settings.data_dir``, or an absolute path or URI
    taken as given. ``bbox`` is ``(min_lon, min_lat, max_lon, max_lat)`` in degrees.
    """

    output: str
    bbox: tuple[float, float, float, float]
    vertical_transform: bool = True

    @model_validator(mode="after")
    def _check_bbox(self) -> Glo30DownloadConfig:
        min_lon, min_lat, max_lon, max_lat = self.bbox
        if not (min_lon < max_lon and min_lat < max_lat):
            msg = f"bbox must be (min_lon, min_lat, max_lon, max_lat), got {self.bbox}"
            raise ValueError(msg)
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> Glo30DownloadConfig:
        """Read a download config from a YAML file.

        Parameters
        ----------
        path : Path
            The YAML file.

        Returns
        -------
        Glo30DownloadConfig
            The validated config.
        """
        return cls.model_validate(yaml.safe_load(path.read_text()))


def resolve_download_output(
    config: Glo30DownloadConfig,
    output: Path | None,
    *,
    overwrite: bool,
) -> Path | UPath:
    """Return where a GLO-30 download is written, refusing to replace an existing DEM.

    Parameters
    ----------
    config : Glo30DownloadConfig
        The download config, whose ``output`` applies when ``output`` is not given.
    output : Path or None
        An explicit output path, which takes precedence over the config.
    overwrite : bool
        Allow replacing an existing file.

    Returns
    -------
    Path or UPath
        The output path, local or in an object store.

    Raises
    ------
    FileExistsError
        If the output exists and ``overwrite`` is false.
    """
    target = output if output is not None else settings.resolve_data_path(config.output)
    if not overwrite and target.exists():
        msg = f"{target} already exists; pass --overwrite to replace it"
        raise FileExistsError(msg)
    return target


def ellipsoidal_heights(source: AnyPath | VRT, *, orthometric_crs: str) -> VRT:
    """Convert a raster of orthometric heights to heights above its CRS's ellipsoid.

    The geoid undulation is evaluated by PROJ at every pixel centre and added on
    the source's own grid, so no pixel moves. Nothing is written.

    Parameters
    ----------
    source : str | Path | UPath | VRT
        Single-band raster of orthometric heights.
    orthometric_crs : str
        Compound CRS of the heights: the source's horizontal CRS and the vertical
        datum they refer to, e.g. ``"EPSG:4326+3855"`` for EGM2008.

    Returns
    -------
    VRT
        A warped VRT on the source's grid, with its CRS, dtype and NODATA.

    Raises
    ------
    ValueError
        If ``orthometric_crs`` is not compound or its horizontal part is not the
        source's CRS, or if PROJ has no exact transformation, e.g. because the geoid
        grid is unavailable.
    """
    src = source if isinstance(source, VRT) else vrt.from_raster(source)
    if src.crs is None:
        msg = "The source has no CRS."
        raise ValueError(msg)
    horizontal = pyproj.CRS.from_wkt(src.crs.to_wkt())
    compound = pyproj.CRS.from_user_input(orthometric_crs)
    if len(compound.sub_crs_list) != _COMPOUND_PARTS:
        msg = f"orthometric_crs must be a compound (horizontal + vertical) CRS, got '{compound}'."
        raise ValueError(msg)
    if not compound.sub_crs_list[0].equals(horizontal, ignore_axis_order=True):
        msg = (
            f"The horizontal part of '{orthometric_crs}' is not the source's CRS "
            f"'{src.crs}'; reproject the source first."
        )
        raise ValueError(msg)

    transform = src.transform
    shifted = vrt.warp(
        source,
        src_crs=orthometric_crs,
        crs=CRS.from_wkt(horizontal.to_3d().to_wkt()),
        resolution=(transform.a, -transform.e),
        bounds=array_bounds(src.height, src.width, transform),
        resampling=Resampling.nearest,
        snap=False,
    )
    # The heights are now ellipsoidal; the file keeps the plain horizontal CRS.
    shifted.crs = src.crs
    return shifted
