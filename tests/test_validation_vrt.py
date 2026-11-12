"""Relocatable MGRS tile VRTs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from freezebase.mgrs import MGRSGeoBox
from freezebase.raster import rasterio_open
import numpy as np
import pytest
import rasterio

from glace_production.validation.vrt import (
    build_relative_vrt,
    default_vrt_name,
    expand_tile_pattern,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_tile(path: Path, tile: str, value: int) -> None:
    gb = MGRSGeoBox.from_mgrs(tile)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=gb.shape.x,
        height=gb.shape.y,
        count=1,
        dtype="uint8",
        crs="EPSG:32632",
        transform=gb.affine,
        nodata=255,
    ) as dst:
        data = np.full(gb.shape.yx, value, np.uint8)
        if value == 1:  # the later year is nodata in its northern half
            data[:500] = 255
        dst.write(data, 1)


def test_build_relative_vrt_orders_years_and_relativizes(tmp_path: Path) -> None:
    tiles = tmp_path / "maps"
    _write_tile(tiles / "PRED_32TMS55_2021.tif", "32TMS55", 7)
    _write_tile(tiles / "PRED_32TMS55_2022.tif", "32TMS55", 1)
    _write_tile(tiles / "PRED_32TMS65_2022.tif", "32TMS65", 3)
    out = tiles / "all.vrt"
    # handed over unsorted: the later year must still be added last
    files = [
        tiles / "PRED_32TMS55_2022.tif",
        tiles / "PRED_32TMS65_2022.tif",
        tiles / "PRED_32TMS55_2021.tif",
    ]
    assert build_relative_vrt(files, out) == 3
    assert 'relativeToVRT="0"' not in out.read_text()
    with rasterio_open(out) as src:
        data = src.read(1)
    assert data[100, 100] == 7  # 2022 is nodata here, 2021 shows through
    assert data[900, 100] == 1  # 2022 on top of 2021
    assert data[900, 1500] == 3


def test_build_relative_vrt_refuses_empty() -> None:
    with pytest.raises(ValueError, match="no tiles"):
        build_relative_vrt([], "x.vrt")


def test_expand_tile_pattern_excludes_qa_layers(tmp_path: Path) -> None:
    for name in ("S1_COH12_LRW_32TMS55_VV_x.tif", "S1_COH12_LRW_QA-NUM_32TMS55_VV_x.tif"):
        (tmp_path / name).touch()
    hits = sorted(
        p.name for p in tmp_path.glob(expand_tile_pattern("S1_COH12_LRW_{tile}_VV_*.tif"))
    )
    assert hits == ["S1_COH12_LRW_32TMS55_VV_x.tif"]


def test_default_vrt_name() -> None:
    assert default_vrt_name("S1_COH12_LRW_*_VV_*.tif") == "S1_COH12_LRW_VV.vrt"
    assert default_vrt_name("S1_COH12_LRW_{tile}_VV_*.tif") == "S1_COH12_LRW_VV.vrt"
    assert default_vrt_name("CLS_*_2023.tif") == "CLS_2023.vrt"
    assert default_vrt_name("LABELS_SGI2023_2022_*.tif") == "LABELS_SGI2023_2022.vrt"
    with pytest.raises(ValueError, match="derive"):
        default_vrt_name("*.tif")
