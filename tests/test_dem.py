"""Orthometric-to-ellipsoidal height conversion and the GLO-30 download config.

The conversion is the one step of the DEM pipeline that rewrites pixel values
rather than moving them, so a silent failure -- a missing geoid grid falling
back to a no-op -- would shift a whole DEM by tens of metres without any error.
The geoids here are synthetic GTX grids, so the tests need no PROJ download.
"""

from __future__ import annotations

from pathlib import Path
import struct
from typing import TYPE_CHECKING

import numpy as np
from pydantic import ValidationError
import pytest
import rasterio
from rasterio.transform import from_origin

from glace_production.composites import default_config_dir
from glace_production.config import settings
from glace_production.dem import (
    Glo30DownloadConfig,
    ellipsoidal_heights,
    resolve_download_output,
)

if TYPE_CHECKING:
    from collections.abc import Callable

_ARCSEC = 1.0 / 3600.0
# The test DEM: 8 x 6 one-arcsecond pixels with their top-left corner here.
_WEST, _NORTH = 8.0, 46.5
# The synthetic geoids: one-degree nodes over 7-9 E, 45-48 N. PROJ discards GTX
# values beyond about +-1000 m, so the slopes below stay small.
_GRID_SOUTH, _GRID_WEST, _GRID_ROWS, _GRID_COLS = 45.0, 7.0, 4, 3


def _write_geoid(path: Path, undulation: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> str:
    """Write a GTX geoid grid and return a compound CRS that refers to it."""
    lats, lons = np.meshgrid(
        _GRID_SOUTH + np.arange(_GRID_ROWS), _GRID_WEST + np.arange(_GRID_COLS), indexing="ij"
    )
    with path.open("wb") as f:
        f.write(struct.pack(">ddddii", _GRID_SOUTH, _GRID_WEST, 1.0, 1.0, *lats.shape))
        f.write(undulation(lons, lats).astype(">f4").tobytes())
    return f"+proj=longlat +datum=WGS84 +geoidgrids={path} +vunits=m +type=crs"


def _write_dem(path: Path, value: float, *, nodata: float | None = None) -> None:
    data = np.full((6, 8), value, dtype="float32")
    if nodata is not None:
        data[0, 0] = nodata
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=8,
        height=6,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(_WEST, _NORTH, _ARCSEC, _ARCSEC),
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)


def test_ellipsoidal_heights_adds_the_undulation(tmp_path: Path) -> None:
    geoid = _write_geoid(tmp_path / "geoid.gtx", lambda lons, _: np.full_like(lons, 10.0))
    _write_dem(tmp_path / "dem.tif", 1000.0)

    shifted = ellipsoidal_heights(tmp_path / "dem.tif", orthometric_crs=geoid)

    with rasterio.open(tmp_path / "dem.tif") as before, shifted.open() as after:
        assert after.crs == before.crs
        assert after.transform == before.transform
        assert after.shape == before.shape
        assert after.dtypes == before.dtypes
        np.testing.assert_array_equal(after.read(1), np.full((6, 8), 1010.0, dtype="float32"))


def test_ellipsoidal_heights_evaluates_the_geoid_at_every_pixel_centre(tmp_path: Path) -> None:
    """A geoid rising 0.1 m per arcsecond eastwards shifts column ``c`` by ``0.1 (c + 0.5)``."""
    geoid = _write_geoid(tmp_path / "geoid.gtx", lambda lons, _: (lons - _WEST) * 360.0)
    _write_dem(tmp_path / "dem.tif", 0.0)

    with ellipsoidal_heights(tmp_path / "dem.tif", orthometric_crs=geoid).open() as after:
        shifted = after.read(1)

    expected = np.broadcast_to(0.1 * (np.arange(8) + 0.5), (6, 8))
    np.testing.assert_allclose(shifted, expected, atol=1e-5)


def test_ellipsoidal_heights_leaves_nodata_alone(tmp_path: Path) -> None:
    geoid = _write_geoid(tmp_path / "geoid.gtx", lambda lons, _: np.full_like(lons, 10.0))
    _write_dem(tmp_path / "dem.tif", 1000.0, nodata=-9999.0)

    with ellipsoidal_heights(tmp_path / "dem.tif", orthometric_crs=geoid).open() as after:
        shifted = after.read(1)
        assert after.nodata == -9999.0
    assert shifted[0, 0] == -9999.0
    assert np.all(shifted.ravel()[1:] == 1010.0)


def test_ellipsoidal_heights_refuses_a_missing_geoid(tmp_path: Path) -> None:
    """Without its grid the heights cannot be shifted; that must raise, not pass them through.

    GDAL raises while building the warp, or, when PROJ may look for the grid online,
    on the first read.
    """
    geoid = f"+proj=longlat +datum=WGS84 +geoidgrids={tmp_path / 'absent.gtx'} +vunits=m +type=crs"
    _write_dem(tmp_path / "dem.tif", 1000.0)

    # GDAL's error types are private to rasterio, hence the match on its messages.
    with (
        pytest.raises(Exception, match=r"vgridshift|exact transformation|Read failed"),
        ellipsoidal_heights(tmp_path / "dem.tif", orthometric_crs=geoid).open() as after,
    ):
        after.read(1)


def test_ellipsoidal_heights_refuses_another_horizontal_crs(tmp_path: Path) -> None:
    """The heights are shifted in place; a compound CRS on another grid would also move them."""
    _write_dem(tmp_path / "dem.tif", 1000.0)

    with pytest.raises(ValueError, match="horizontal"):
        ellipsoidal_heights(tmp_path / "dem.tif", orthometric_crs="EPSG:2056+5728")


def test_ellipsoidal_heights_refuses_a_crs_without_heights(tmp_path: Path) -> None:
    _write_dem(tmp_path / "dem.tif", 1000.0)

    with pytest.raises(ValueError, match="compound"):
        ellipsoidal_heights(tmp_path / "dem.tif", orthometric_crs="EPSG:4326")


def test_glo30_config_reads_committed_file() -> None:
    config = Glo30DownloadConfig.from_yaml(default_config_dir() / "dem" / "glo30-alps.yml")

    assert config.bbox == (4.3, 42.9, 17.0, 48.9)
    assert config.output == "dem/glo30-alps_ellipsoid.tif"
    assert config.vertical_transform is True


def test_glo30_config_rejects_inverted_bbox() -> None:
    with pytest.raises(ValidationError, match="bbox"):
        Glo30DownloadConfig(output="dem.tif", bbox=(17.0, 42.9, 4.3, 48.9))


def test_download_output_resolves_under_data_dir() -> None:
    config = Glo30DownloadConfig(output="dem/x.tif", bbox=(7.0, 46.0, 9.0, 47.0))

    # overwrite skips the existence check, so the test never reaches a real store
    output = resolve_download_output(config, None, overwrite=True)

    assert str(output) == str(settings.data_dir / "dem" / "x.tif")


def test_download_output_cli_path_wins(tmp_path: Path) -> None:
    config = Glo30DownloadConfig(output="dem/x.tif", bbox=(7.0, 46.0, 9.0, 47.0))

    assert (
        resolve_download_output(config, tmp_path / "y.tif", overwrite=False) == tmp_path / "y.tif"
    )


def test_download_output_refuses_existing_file(tmp_path: Path) -> None:
    existing = tmp_path / "dem.tif"
    existing.write_bytes(b"published")
    config = Glo30DownloadConfig(output=str(existing), bbox=(7.0, 46.0, 9.0, 47.0))

    with pytest.raises(FileExistsError, match=r"dem\.tif"):
        resolve_download_output(config, None, overwrite=False)
    assert existing.read_bytes() == b"published"
    assert resolve_download_output(config, None, overwrite=True) == existing
