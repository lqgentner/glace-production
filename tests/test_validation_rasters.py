"""Raster reads against a tile geobox: header contract and value normalisation."""

from __future__ import annotations

from pathlib import Path

from affine import Affine
import numpy as np
from odc.geo.geobox import GeoBox
import pytest
import rasterio
from rasterio.enums import Resampling

from glace_production.validation.rasters import read_tile, warp_to_geobox

CRS = "EPSG:32632"


def _geobox(n: int = 8, left: float = 0.0, top: float = 80.0, res: float = 10.0) -> GeoBox:
    return GeoBox((n, n), Affine(res, 0, left, 0, -res, top), CRS)


def _write(
    path: Path,
    data: np.ndarray,
    *,
    transform: Affine,
    crs: str = CRS,
    nodata: float | None = None,
    scales: float | None = None,
) -> Path:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[1],
        height=data.shape[0],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data.astype("float32"), 1)
        if scales is not None:
            dst.scales = (scales,)
    return path


def test_read_tile_normalises_nodata_and_inf(tmp_path: Path) -> None:
    gb = _geobox()
    data = np.full((8, 8), 2.0, dtype=np.float32)
    data[0, 0] = -9999.0
    data[0, 1] = np.inf
    path = _write(tmp_path / "t.tif", data, transform=gb.affine, nodata=-9999.0)
    out = read_tile(path, gb)
    assert np.isnan(out[0, 0])
    assert np.isnan(out[0, 1])
    assert out[1, 1] == 2.0
    assert out.dtype == np.float32


def test_read_tile_applies_scale(tmp_path: Path) -> None:
    gb = _geobox()
    path = _write(tmp_path / "s.tif", np.full((8, 8), 50.0), transform=gb.affine, scales=0.01)
    assert read_tile(path, gb)[3, 3] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("transform", "crs", "n"),
    [
        (Affine(10, 0, 5, 0, -10, 80), CRS, 8),  # shifted half a pixel
        (Affine(10, 0, 0, 0, -10, 80), "EPSG:32633", 8),
        (Affine(10, 0, 0, 0, -10, 80), CRS, 9),
    ],
)
def test_read_tile_rejects_grid_mismatch(
    tmp_path: Path, transform: Affine, crs: str, n: int
) -> None:
    path = _write(tmp_path / "bad.tif", np.zeros((n, n)), transform=transform, crs=crs)
    with pytest.raises(ValueError, match=r"bad\.tif"):
        read_tile(path, _geobox())


def test_read_tile_raw_keeps_nodata(tmp_path: Path) -> None:
    gb = _geobox()
    data = np.full((8, 8), 1, dtype=np.uint8)
    data[0, 0] = 255
    with rasterio.open(
        tmp_path / "l.tif",
        "w",
        driver="GTiff",
        width=8,
        height=8,
        count=1,
        dtype="uint8",
        crs=CRS,
        transform=gb.affine,
        nodata=255,
    ) as dst:
        dst.write(data, 1)
    out = read_tile(tmp_path / "l.tif", gb, raw=True)
    assert out.dtype == np.uint8
    assert out[0, 0] == 255


def test_warp_to_geobox_resamples_coarser_source(tmp_path: Path) -> None:
    gb = _geobox(n=8)
    coarse = np.array(
        [[1.0, 3.0], [5.0, 7.0]], dtype=np.float32
    )  # 40 m pixels over the same extent
    path = _write(tmp_path / "coarse.tif", coarse, transform=Affine(40, 0, 0, 0, -40, 80))
    out = warp_to_geobox(path, gb, resampling=Resampling.nearest)
    assert out.shape == (8, 8)
    assert out[0, 0] == 1.0
    assert out[7, 7] == 7.0
