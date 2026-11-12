"""Label masks: the year-split rule, the domain, the file contract."""

from __future__ import annotations

from typing import TYPE_CHECKING

from affine import Affine
from freezebase.raster import rasterio_open
import geopandas as gpd
import numpy as np
from odc.geo.geobox import GeoBox
import pandas as pd
import pytest
from shapely.geometry import box

from glace_production.geo import MGRSInventory
from glace_production.labels import (
    IGNORE_VALUE,
    LABEL_SOURCES,
    LabelContractError,
    LabelSource,
    acquisition_years,
    check_label_tags,
    compose_label,
    label_tags,
    split_years,
    write_label_masks,
)

if TYPE_CHECKING:
    from pathlib import Path

# A 10 x 10 px tile of 10 m pixels: x 0..100, y 0..100 in EPSG:32632.
GEOBOX = GeoBox((10, 10), Affine(10, 0, 0, 0, -10, 100), "EPSG:32632")
SOURCE = LabelSource(key="TEST", year_split=True, domain=None)


def _inventory() -> gpd.GeoDataFrame:
    """Three glaciers mapped in three years.

    A (2021) rows 0-3, cols 0-2; B (2022) rows 0-3, cols 4-5; C (2023) rows 6-9, cols 8-9.
    """
    return gpd.GeoDataFrame(
        {"year_acq": [2021, 2022, 2023]},
        geometry=[box(0, 60, 30, 100), box(40, 60, 60, 100), box(80, 0, 100, 40)],
        crs="EPSG:32632",
    )


def _masks(domain: MGRSInventory | None = None) -> dict[int, np.ndarray]:
    return {
        y.year: compose_label(GEOBOX, y.outlines, y.ignore, domain)
        for y in split_years(_inventory(), SOURCE, buffer_m=0.0, ignore_buffer_m=10.0)
    }


def test_split_years_sorted_and_named() -> None:
    years = split_years(_inventory(), SOURCE, buffer_m=0.0, ignore_buffer_m=10.0)
    assert [y.year for y in years] == [2021, 2022, 2023]


def test_own_outlines_are_glacier_and_exact() -> None:
    masks = _masks()
    assert (masks[2022][0:4, 4:6] == 1).all()
    assert (masks[2022] == 1).sum() == 8


def test_earlier_years_ignore_interior_but_label_band_background() -> None:
    m = _masks()[2022]
    assert (m[0:4, 0:3] == IGNORE_VALUE).all()  # A itself is unknown in 2022
    assert m[0, 3] == 0  # the pixel next to A is background: A cannot have grown
    assert m[4, 0] == 0


def test_later_years_ignore_outline_and_buffer() -> None:
    m = _masks()[2022]
    assert (m[6:10, 8:10] == IGNORE_VALUE).all()  # C itself
    assert m[5, 7] == IGNORE_VALUE  # one pixel of buffer around C
    assert m[4, 7] == 0  # beyond the buffer


def test_first_and_last_year() -> None:
    masks = _masks()
    first, last = masks[2021], masks[2023]
    assert (first[0:4, 0:3] == 1).all()
    assert first[0, 3] == IGNORE_VALUE  # B's buffer reaches col 3
    assert first[4, 4] == IGNORE_VALUE
    assert first[5, 4] == 0
    assert (last[6:10, 8:10] == 1).all()
    assert (last[0:4, 0:3] == IGNORE_VALUE).all()
    assert (last[0:4, 4:6] == IGNORE_VALUE).all()
    assert last[0, 3] == 0  # both earlier: no buffer between them


def test_domain_wins_over_everything() -> None:
    domain = MGRSInventory(gpd.GeoDataFrame(geometry=[box(0, 0, 50, 100)], crs="EPSG:32632"))
    masks = _masks(domain)
    assert (masks[2023][:, 5:] == IGNORE_VALUE).all()  # C lies outside
    assert masks[2022][0, 4] == 1  # B's western column is inside
    assert masks[2022][0, 5] == IGNORE_VALUE  # its eastern column is not
    assert masks[2022][5, 0] == 0


def test_single_year_source_has_no_ignore() -> None:
    mask = compose_label(GEOBOX, MGRSInventory(_inventory()), [], None)
    assert set(np.unique(mask)) == {0, 1}


def test_acquisition_years_from_dates() -> None:
    gdf = gpd.GeoDataFrame(
        {"date": pd.to_datetime(["2015-08-01", "2016-09-12"])},
        geometry=[box(0, 0, 1, 1)] * 2,
        crs="EPSG:32632",
    )
    assert acquisition_years(gdf, "date").tolist() == [2015, 2016]
    assert acquisition_years(_inventory(), "year_acq").tolist() == [2021, 2022, 2023]


def test_registered_sources() -> None:
    assert LABEL_SOURCES["SGI2023"].domain == "switzerland"
    assert LABEL_SOURCES["PAULETAL2020"].domain is None
    assert all(s.year_split for s in LABEL_SOURCES.values())


def test_write_label_masks_contract(tmp_path: Path) -> None:
    # A real 10 km tile so the file is a valid COG on the MGRS grid.
    tile_inventory = gpd.GeoDataFrame(
        {"year_acq": [2022, 2023]},
        geometry=[
            box(431_000, 5_151_000, 435_000, 5_155_000),
            box(436_000, 5_156_000, 437_000, 5_157_000),
        ],
        crs="EPSG:32632",
    )
    source = LabelSource(key="TEST", year_split=True, domain=None)
    years = split_years(tile_inventory, source, buffer_m=250.0, ignore_buffer_m=250.0)
    n = write_label_masks(years[0], tmp_path, source, domain=None, ignore_buffer_m=250.0)
    assert n == 1
    path = tmp_path / "LABELS_TEST_2022_32TMS35.tif"
    with rasterio_open(path) as src:
        tags = src.tags()
        data = src.read(1)
        assert src.nodata == IGNORE_VALUE
    assert tags["INVENTORY"] == "TEST"
    assert tags["ACQUISITION_YEAR"] == "2022"
    assert tags["YEAR_RULE"] == "retreat"
    assert tags["IGNORE_BUFFER_M"] == "250.0"
    assert tags["DOMAIN"] == "none"
    assert data.dtype == np.uint8
    assert (data == 1).sum() == 400 * 400
    assert (data == IGNORE_VALUE).sum() > 100 * 100


def test_check_label_tags_accepts_own_tags_and_rejects_legacy_masks() -> None:
    tags = label_tags(SOURCE, year=2022, ignore_buffer_m=250.0)
    check_label_tags(tags, year=2022)
    check_label_tags(tags)
    with pytest.raises(LabelContractError, match="ACQUISITION_YEAR"):
        check_label_tags(tags, year=2023)
    with pytest.raises(LabelContractError, match="regenerate"):
        check_label_tags({"AREA_OR_POINT": "Area"}, year=2022)
    with pytest.raises(LabelContractError, match="DOMAIN missing"):
        check_label_tags({k: v for k, v in tags.items() if k != "DOMAIN"})
