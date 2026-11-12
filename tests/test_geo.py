"""Tile coverage of a vector inventory on the MGRS grid; rasterisation on plain geoboxes."""

from __future__ import annotations

from collections.abc import Iterable

from affine import Affine
from freezebase.mgrs import MGRSGeoBox
import geopandas as gpd
import numpy as np
from odc.geo.geobox import GeoBox
import pytest
from shapely.geometry import box

from glace_production.geo import MGRSInventory

TILE = "32TMS35"
"""Covers x 430 to 440 km, y 5150 to 5160 km in EPSG:32632."""


def _inventory(*geoms: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=[box(*bounds) for bounds in geoms], crs="EPSG:32632")


def test_adjacent_outlines_are_not_double_counted_once_buffered() -> None:
    # Two 4 x 4 km squares sharing an edge: their 250 m buffers overlap along it.
    inventory = _inventory(
        (431_000, 5_151_000, 435_000, 5_155_000),
        (435_000, 5_151_000, 439_000, 5_155_000),
    )
    fraction = MGRSInventory(inventory, buffer_m=250.0).coverage_fraction(
        MGRSGeoBox.from_mgrs(TILE),
    )
    expected = gpd.GeoSeries(inventory.geometry).buffer(250.0).union_all().area / 1e8
    assert fraction == pytest.approx(expected)


def test_overlapping_buffers_filling_the_tile_stay_at_one() -> None:
    inventory = _inventory(
        (430_000, 5_150_000, 435_100, 5_160_000),
        (434_900, 5_150_000, 440_000, 5_160_000),
    )
    geobox = MGRSGeoBox.from_mgrs(TILE)
    assert MGRSInventory(inventory, buffer_m=250.0).coverage_fraction(geobox) == pytest.approx(1.0)
    assert MGRSInventory(inventory).coverage_fraction(geobox) == pytest.approx(1.0)


def test_outlines_outside_the_tile_cover_nothing() -> None:
    inventory = _inventory((500_000, 5_300_000, 501_000, 5_301_000))
    assert MGRSInventory(inventory).coverage_fraction(MGRSGeoBox.from_mgrs(TILE)) == 0.0


def test_prepare_fills_the_reprojection_and_index_caches() -> None:
    # Accessing private state (SLF001, already ignored repo-wide for tests/) is
    # the only way to prove prepare() actually populated the caches rather than
    # relying on later calls happening to build them on demand.
    inv = MGRSInventory(_inventory((431_000, 5_151_000, 435_000, 5_155_000)), buffer_m=250.0)
    epsg = 32632
    assert epsg not in inv._base
    inv.prepare(epsg)
    assert epsg in inv._base
    assert epsg in inv._buffered
    assert len(inv._base[epsg].sindex) == 1  # the index was built, not just the geometry cached


def _gid_inventory() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"gid": [7, 9]},
        geometry=[box(20, 20, 50, 50), box(60, 10, 80, 30)],
        crs="EPSG:32632",
    )


def test_rasterize_accepts_padded_geobox() -> None:
    gb = GeoBox((10, 10), Affine(10, 0, 0, 0, -10, 100), "EPSG:32632")
    mask = MGRSInventory(_gid_inventory()).rasterize(gb.pad(2))
    assert mask.shape == (14, 14)
    assert mask[2 + 5, 2 + 2] == 1  # inside the first box (row 5 = y 50..40, col 2 = x 20..30)
    assert mask[0, 0] == 0


def test_rasterize_values_burns_attribute_with_fill() -> None:
    gb = GeoBox((10, 10), Affine(10, 0, 0, 0, -10, 100), "EPSG:32632")
    gids = MGRSInventory(_gid_inventory()).rasterize_values(gb, "gid")
    assert gids.dtype == np.int32
    assert gids[5, 2] == 7
    assert gids[7, 6] == 9
    assert gids[0, 0] == -1


def _stacked_inventory(gids: Iterable[int]) -> gpd.GeoDataFrame:
    # Ten nested boxes, all covering pixel row 9, col 0 (x 0..10, y 0..10); the
    # smallest (row position 9) still reaches x,y 0..91. An unsorted STRtree
    # traversal visits these ten leaves out of row order (verified: querying
    # the full extent returns position order [9, 8, ..., 0], the reverse of
    # insertion order), so only a sorted query burns them in row order.
    geoms = [box(0, 0, 100 - i, 100 - i) for i in range(10)]
    return gpd.GeoDataFrame({"gid": list(gids)}, geometry=geoms, crs="EPSG:32632")


def test_rasterize_values_last_row_wins_despite_unsorted_tree_traversal() -> None:
    gb = GeoBox((10, 10), Affine(10, 0, 0, 0, -10, 100), "EPSG:32632")
    gids = MGRSInventory(_stacked_inventory(range(10))).rasterize_values(gb, "gid")
    assert gids[9, 0] == 9

    reversed_gids = MGRSInventory(_stacked_inventory(range(9, -1, -1))).rasterize_values(gb, "gid")
    assert reversed_gids[9, 0] == 0
