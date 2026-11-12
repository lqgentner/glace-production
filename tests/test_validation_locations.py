"""Locations on a synthetic tile: the scope, distances, censoring and ids."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from glace_production.validation.config import LocationsConfig
from glace_production.validation.locations import (
    LOCATION_COLUMNS,
    LabelMismatchError,
    parse_label_name,
    select_tile_locations,
)

N = 100  # tile is 100 x 100 px = 1 km
PAD = 30  # pad_m 300 -> 30 px

_Scene = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _config(**kw: object) -> LocationsConfig:
    base = {
        "name": "t",
        "labels_dir": "l",
        "inventory": "SGI2023",
        "years": [2023],
        "dem": "d.tif",
        "pad_m": 300.0,
    }
    return LocationsConfig.model_validate(base | kw)


def _scene(
    *,
    glacier: tuple[int, int, int, int] = (40, 60, 40, 60),
    edge: bool = False,
    domain: bool = True,
) -> _Scene:
    """Build a padded glacier mask + gid raster, tile label, debris, elevation."""
    padded = np.zeros((N + 2 * PAD, N + 2 * PAD), dtype=bool)
    r0, r1, c0, c1 = glacier
    padded[PAD + r0 : PAD + r1, PAD + c0 : PAD + c1] = True
    if edge:  # glacier continues through the whole halo beyond the tile's right edge
        padded[PAD + r0 : PAD + r1, PAD + c1 :] = True
    gids = np.where(padded, 42, -1).astype(np.int32)
    label = padded[PAD:-PAD, PAD:-PAD].astype(np.uint8)
    label[0:5, 0:5] = 255  # ignore patch
    if domain:
        label[:, 90:] = 255  # outside the domain
    debris = np.zeros((N, N), dtype=bool)
    debris[45:50, 45:50] = True
    elevation = np.full((N, N), 2500.0, dtype=np.float32)
    return padded, gids, label, debris, elevation


def _select(
    config: LocationsConfig | None = None,
    *,
    edge: bool = False,
    domain: bool = True,
    glacier: tuple[int, int, int, int] = (40, 60, 40, 60),
) -> tuple[pd.DataFrame, dict[str, int | str]]:
    padded, gids, label, debris, elevation = _scene(glacier=glacier, edge=edge, domain=domain)
    return select_tile_locations(
        tile="32TMS55",
        year=2023,
        label=label,
        outlines=padded,
        gids=gids,
        debris=debris,
        elevation=elevation,
        config=config or _config(),
    )


def test_parse_label_name() -> None:
    assert parse_label_name("LABELS_SGI2023_2023_32TMS55.tif") == ("SGI2023", 2023, "32TMS55")
    assert parse_label_name("S1_COH12_LRW_32TMS55_VV_x.tif") is None


def test_scope_is_glacier_plus_ring() -> None:
    frame, report = _select()
    inside = frame[frame.label == 1]
    outside = frame[frame.label == 0]
    assert len(inside) == 20 * 20  # every glacier pixel, interior included
    assert (inside.dist_m < 0).all()
    assert (outside.dist_m > 0).all()
    assert (outside.dist_m <= 250).all()
    assert outside.dist_m.min() == 10  # the first pixel outside
    # a 25 px ring around a 20 x 20 square: four sides plus four rounded corners
    assert len(outside) == report["ring"]
    assert 20 * 4 * 25 < len(outside) < 70 * 70 - 20 * 20
    assert report["glacier"] == 400
    assert report["selected"] == len(frame)
    assert report["ignore"] == 25 + N * 10
    assert report["total"] == N * N


def test_ignore_never_selected() -> None:
    frame, _ = _select()
    assert not ((frame.row < 5) & (frame.col < 5)).any()
    assert (frame.col < 90).all()


def test_determinism() -> None:
    a, _ = _select()
    b, _ = _select()
    pd.testing.assert_frame_equal(a, b)


def test_no_false_boundary_at_tile_edge() -> None:
    frame, _ = _select(edge=True, domain=False, glacier=(40, 60, 70, 100))
    # the glacier continues past the tile edge: the last column is not an outline,
    # and (50, 99) is 10 px from the nearest background, row 60
    inside = frame[frame.label == 1]
    assert inside.loc[(inside.col == 99) & (inside.row == 50), "dist_m"].iloc[0] == -100
    assert (inside.dist_m <= -10).all()


def test_censoring_and_near_gid() -> None:
    frame, _ = _select()
    outside = frame[frame.label == 0]
    assert (outside.near_gid == 42).all()
    inside = frame[frame.label == 1]
    assert (inside.gid == 42).all()
    assert (inside.near_gid == 42).all()
    assert (frame.loc[frame.label == 0, "gid"] == -1).all()


def test_interior_pixel_keeps_own_gid_beyond_pad() -> None:
    # An 80 x 80 glacier leaves 35 px around a 10 x 10 interior block, beyond
    # the 30 px pad: those distances are censored to -pad_m, but the pixels
    # keep their own gid.
    frame, _ = _select(glacier=(10, 90, 10, 90))
    interior = frame[frame.row.between(45, 54) & frame.col.between(45, 54)]
    assert len(interior) == 10 * 10
    assert (interior.label == 1).all()
    assert (interior.dist_m == -300.0).all()
    assert (interior.near_gid == 42).all()
    assert (interior.gid == 42).all()


def test_pad_must_exceed_ring() -> None:
    with pytest.raises(ValueError, match="pad_m"):
        _config(pad_m=200.0)


def test_debris_and_elevation_columns() -> None:
    frame, _ = _select()
    assert frame.loc[frame.debris, "label"].eq(1).all()
    assert frame.debris.any()
    assert (frame.elevation == 2500.0).all()


def test_label_mismatch_raises() -> None:
    padded, gids, label, debris, elevation = _scene()
    label[50, 50] = 0  # inventory says glacier, mask says background
    with pytest.raises(LabelMismatchError):
        select_tile_locations(
            tile="32TMS55",
            year=2023,
            label=label,
            outlines=padded,
            gids=gids,
            debris=debris,
            elevation=elevation,
            config=_config(),
        )


def test_outline_pixels_may_be_ignore_but_not_background() -> None:
    frame, _ = _select(glacier=(40, 60, 80, 95))
    assert (frame.col < 90).all()
    assert frame.label.eq(1).any()


def test_empty_tile_selects_nothing() -> None:
    frame, report = _select(glacier=(0, 0, 0, 0))
    assert frame.empty
    assert report["selected"] == 0


def test_columns_and_dtypes() -> None:
    frame, _ = _select()
    assert list(frame.columns) == LOCATION_COLUMNS
    assert frame.row.dtype == np.uint16
    assert frame.dist_m.dtype == np.float32
    assert frame.gid.dtype == np.int32
    assert frame.year.dtype == np.int16
