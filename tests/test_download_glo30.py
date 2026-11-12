"""The COG ``scripts/dem/download-glo30.py`` writes from GLO-30 tiles.

The tiles follow GLO-30's layout, with their origin half a pixel off the whole
degree. Reads run multi-threaded, the setting under which GDAL once placed
fractional source windows one pixel off.
"""

from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path
import struct
from typing import TYPE_CHECKING

import numpy as np
import rasterio
from rasterio.transform import from_origin

if TYPE_CHECKING:
    from types import ModuleType

_SCRIPT = Path(__file__).parents[1] / "scripts" / "dem" / "download-glo30.py"
_TILE_PX = 360
_TILES = 3


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("download_glo30", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_tiles(directory: Path) -> dict[str, tuple[int, int, np.ndarray]]:
    """Write a 3x3 block of one-degree tiles; return each tile's mosaic offset and data."""
    res = 1 / _TILE_PX
    tiles = {}
    for row, col in itertools.product(range(_TILES), range(_TILES)):
        data = np.random.default_rng(row * _TILES + col).random(
            (_TILE_PX, _TILE_PX), dtype=np.float32
        )
        path = directory / f"tile_{row}_{col}.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=_TILE_PX,
            height=_TILE_PX,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=from_origin(4 + col - res / 2, 49 - row + res / 2, res, res),
            tiled=True,
            blockxsize=64,
            blockysize=64,
            compress="deflate",
        ) as dst:
            dst.write(data, 1)
        tiles[str(path)] = (row * _TILE_PX, col * _TILE_PX, data)
    return tiles


def _write_constant_geoid(path: Path, undulation: float) -> str:
    """Write a GTX geoid over 3-8 E, 45-50 N and return a compound CRS referring to it."""
    with path.open("wb") as f:
        f.write(struct.pack(">ddddii", 45.0, 3.0, 1.0, 1.0, 6, 6))
        f.write(np.full(36, undulation, dtype=">f4").tobytes())
    return f"+proj=longlat +datum=WGS84 +geoidgrids={path} +vunits=m +type=crs"


def _assert_tiles_in_place(
    output: Path, tiles: dict[str, tuple[int, int, np.ndarray]], *, atol: float = 0.0
) -> None:
    with rasterio.open(output) as src:
        mosaic = src.read(1)
    for row, col, data in tiles.values():
        window = mosaic[row : row + _TILE_PX, col : col + _TILE_PX]
        np.testing.assert_allclose(window, data, rtol=0, atol=atol)


def test_cog_places_tiles_exactly_with_threaded_reads(tmp_path: Path) -> None:
    glo30 = _load_script()
    tiles = _write_tiles(tmp_path)
    output = tmp_path / "dem.tif"

    with rasterio.Env(GDAL_NUM_THREADS="4"):
        glo30.write_mosaic_cog(
            list(tiles), output, nodata=float("nan"), overview_resampling="AVERAGE"
        )

    with rasterio.open(output) as src:
        assert src.driver == "GTiff"
        assert src.crs == "EPSG:4326"
        assert src.overviews(1)
    _assert_tiles_in_place(output, tiles)


def test_cog_converts_the_heights(tmp_path: Path) -> None:
    glo30 = _load_script()
    tiles = _write_tiles(tmp_path)
    geoid = _write_constant_geoid(tmp_path / "geoid.gtx", 10.0)
    output = tmp_path / "dem.tif"

    with rasterio.Env(GDAL_NUM_THREADS="4"):
        glo30.write_mosaic_cog(
            list(tiles),
            output,
            nodata=float("nan"),
            overview_resampling="AVERAGE",
            orthometric_crs=geoid,
        )

    shifted = {path: (row, col, data + 10.0) for path, (row, col, data) in tiles.items()}
    # GDAL adds the shift in double precision before rounding to float32.
    _assert_tiles_in_place(output, shifted, atol=1e-5)


def test_cog_takes_its_origin_from_the_tile_headers(tmp_path: Path) -> None:
    """The mosaic's corner is the western tile's own, not one stepped to from another tile.

    Stepping from the middle tile of this row lands one double-precision step off.
    """
    glo30 = _load_script()
    res, px = 1 / 360, 36
    paths = []
    for col in range(3):
        path = tmp_path / f"tile_{col}.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=px,
            height=px,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=from_origin(4 + col * px * res - res / 2, 47 + res / 2, res, res),
        ) as dst:
            dst.write(np.ones((px, px), dtype="float32"), 1)
        paths.append(str(path))
    output = tmp_path / "dem.tif"

    glo30.write_mosaic_cog(
        [paths[1], paths[0], paths[2]], output, nodata=float("nan"), overview_resampling="AVERAGE"
    )

    with rasterio.open(paths[0]) as west, rasterio.open(output) as src:
        assert src.transform == west.transform
