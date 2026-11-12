"""Spatial folds: k-means regions over tiles, straddling glaciers, CSV round trip."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from freezebase.mgrs import MGRSGeoBox
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import OptimizeResult
from shapely.geometry import box

from glace_production.validation.folds import (
    Glacier,
    _objective,
    _solve_balanced,
    _tile_frame,
    assign_folds,
    balanced_folds,
    fold_pieces,
    glacier_tiles,
    read_folds,
    straddling_glaciers,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

TILES = [f"32TMS{c}{r}" for c in range(0, 10, 2) for r in range(0, 10, 2)]  # 25 tiles


def test_tile_frame_rejects_empty_and_missing_epsg(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="no tiles"):
        _tile_frame([])

    class _NoCRSBox:
        crs: None = None

    monkeypatch.setattr(
        "glace_production.validation.folds.MGRSGeoBox.from_mgrs", lambda _t: _NoCRSBox()
    )
    with pytest.raises(ValueError, match="no EPSG"):
        _tile_frame(["32TMS00"])


def test_assign_folds_partitions_tiles_deterministically() -> None:
    folds = assign_folds(TILES, k=5, seed=0)
    assert sorted(folds["tile"]) == sorted(TILES)
    assert set(folds["fold"]) == set(range(5))
    assert folds.equals(assign_folds(TILES, k=5, seed=0))
    # renumbered west to east: fold 0 has the smallest mean easting
    x = {t: MGRSGeoBox.from_mgrs(t).extent.geom.centroid.x for t in TILES}
    means = folds.assign(x=folds["tile"].map(x)).groupby("fold")["x"].mean()
    assert list(means.sort_values().index) == [0, 1, 2, 3, 4]


def test_assign_folds_rejects_mixed_zones() -> None:
    with pytest.raises(ValueError, match="UTM"):
        assign_folds(["32TMS55", "33TUN20"], k=2)


def test_straddling_glaciers_and_csv(tmp_path: Path) -> None:
    folds = assign_folds(TILES, k=5, seed=0)
    a, b = "32TMS00", "32TMS80"  # far apart: different folds
    fa, fb = folds.set_index("tile")["fold"][[a, b]]
    assert fa != fb
    ga = MGRSGeoBox.from_mgrs(a).extent.geom
    gb = MGRSGeoBox.from_mgrs(b).extent.geom
    inv = gpd.GeoDataFrame(
        {"gid": [1, 2]},
        geometry=[
            box(ga.bounds[0] + 100, ga.bounds[1] + 100, gb.bounds[0] + 100, gb.bounds[1] + 100),
            box(ga.bounds[0] + 100, ga.bounds[1] + 100, ga.bounds[0] + 500, ga.bounds[1] + 500),
        ],
        crs="EPSG:32632",
    )
    straddling = straddling_glaciers(folds, inv)
    assert straddling["gid"].tolist() == [1]
    path = tmp_path / "folds.csv"
    folds.to_csv(path, index=False)
    series = read_folds(path)
    assert series[a] == fa
    assert series.dtype.kind == "i"


# A 5 x 5 grid of tiles 20 km apart (TILES); each easting column holds the weights
# 0, 80, 100, 120, 200, so a +-5 % partition into five folds of 500 exists.
GRID_WEIGHTS = dict(zip(TILES, [0.0, 80.0, 100.0, 120.0, 200.0] * 5, strict=True))
EMPTY_INVENTORY = gpd.GeoDataFrame({"gid": []}, geometry=[], crs="EPSG:32632")
LINE = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
LINE_NAMES = ("a", "b", "c", "d")


def _bounds(tile: str) -> tuple[float, float, float, float]:
    return MGRSGeoBox.from_mgrs(tile).extent.geom.bounds


def _solve(
    xy: NDArray[np.float64],
    weights: NDArray[np.float64],
    glaciers: Sequence[Glacier],
    *,
    k: int,
    penalty: float,
    tolerance: float = 0.0,
    names: Sequence[str] = LINE_NAMES,
) -> NDArray[np.intp]:
    """``_solve_balanced`` with the shared test defaults (seed=0, max_iter=20, time_limit=60)."""
    return _solve_balanced(
        xy,
        weights,
        glaciers,
        k=k,
        tolerance=tolerance,
        penalty=penalty,
        seed=0,
        max_iter=20,
        time_limit=60.0,
        names=names,
    )


def test_balanced_folds_meets_tolerance_and_is_deterministic() -> None:
    folds = balanced_folds(TILES, GRID_WEIGHTS, EMPTY_INVENTORY, k=5, tolerance=0.05)
    assert folds["tile"].tolist() == sorted(TILES)
    assert set(folds["fold"]) == set(range(5))
    per_fold = folds.assign(w=folds["tile"].map(GRID_WEIGHTS)).groupby("fold")["w"].sum()
    assert ((per_fold >= 475) & (per_fold <= 525)).all()
    shuffled = TILES[::-1]
    assert folds.equals(balanced_folds(shuffled, GRID_WEIGHTS, EMPTY_INVENTORY, k=5))


def test_balanced_folds_renumbered_west_to_east() -> None:
    folds = balanced_folds(TILES, GRID_WEIGHTS, EMPTY_INVENTORY, k=5)
    x = {t: MGRSGeoBox.from_mgrs(t).extent.geom.centroid.x for t in TILES}
    means = folds.assign(x=folds["tile"].map(x)).groupby("fold")["x"].mean()
    assert means.is_monotonic_increasing


@pytest.mark.parametrize(
    ("weights", "kwargs", "match"),
    [
        (dict.fromkeys(TILES[1:], 1.0), {}, "no glacier pixel count"),
        ({**GRID_WEIGHTS, TILES[0]: 10_000.0}, {}, "heavier than a fold"),
        ({**GRID_WEIGHTS, TILES[0]: -1.0}, {}, "finite and non-negative"),
        ({**GRID_WEIGHTS, TILES[0]: float("nan")}, {}, "finite and non-negative"),
        (dict.fromkeys(TILES, 0.0), {}, "no glacier pixels"),
        (GRID_WEIGHTS, {"k": 1}, "at least two folds"),
        (GRID_WEIGHTS, {"k": 30}, "fewer tiles than folds"),
        (GRID_WEIGHTS, {"tolerance": 1.0}, "tolerance"),
        (GRID_WEIGHTS, {"penalty": -1.0}, "penalty"),
        (GRID_WEIGHTS, {"max_iter": 0}, "max_iter"),
    ],
)
def test_balanced_folds_rejects_bad_input(
    weights: dict[str, float], kwargs: dict[str, float], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        # kwargs mixes int and float overrides for compact parametrization; every
        # entry above is valid for its named parameter.
        # pyrefly: ignore[bad-argument-type]
        balanced_folds(TILES, weights, EMPTY_INVENTORY, **kwargs)


def test_solve_balanced_infeasible_partition_raises() -> None:
    with pytest.raises(ValueError, match="no assignment"):
        _solve(
            LINE[:3],
            np.full(3, 100.0),
            [],
            k=2,
            penalty=0.0,
            tolerance=0.05,
            names=LINE_NAMES[:3],
        )


def test_solve_balanced_raises_on_unproven_solution(monkeypatch: pytest.MonkeyPatch) -> None:
    def time_out(*_args: object, **_kwargs: object) -> OptimizeResult:
        # scipy's OptimizeResult stubs do not type its dict-style **kwargs constructor.
        # pyrefly: ignore[unexpected-keyword]
        return OptimizeResult(status=1, message="Time limit reached", x=np.zeros(8))

    monkeypatch.setattr("glace_production.validation.folds.milp", time_out)
    with pytest.raises(RuntimeError, match="not solved to optimality"):
        _solve(LINE, np.ones(4), [], k=2, penalty=0.0)


def test_solve_balanced_penalty_keeps_shared_glacier_together() -> None:
    # Two per fold. Without a penalty the compact split is {a, b} | {c, d}; a large
    # glacier over b and c must pull them into one fold. The tied split {b, c} | {a, d}
    # has coincident centres, so the numbering falls back to the first tile name.
    glacier = [(50.0, (1, 2))]
    free = _solve(LINE, np.ones(4), glacier, k=2, penalty=0.0)
    assert free.tolist() == [0, 0, 1, 1]
    tied = _solve(LINE, np.ones(4), glacier, k=2, penalty=100.0)
    assert tied.tolist() == [0, 1, 1, 0]


def test_objective_counts_a_split_glacier_once_per_extra_fold() -> None:
    centres = np.array([[0.5, 0.0], [2.5, 0.0]])
    folds = np.array([0, 0, 1, 1])
    spread = [(10.0, (0, 1, 2, 3))]  # over all four tiles, split 2 / 2: one extra fold
    distance = _objective(LINE, centres, folds, [], penalty=1.0)
    assert distance == pytest.approx(4 * 0.25)
    assert _objective(LINE, centres, folds, spread, penalty=1.0) == pytest.approx(distance + 10.0)


def test_glacier_tiles_lists_multi_tile_glaciers_with_area() -> None:
    a, b, c = "32TMS00", "32TMS01", "32TMS02"  # one easting column, adjacent northwards
    ba, bc = _bounds(a), _bounds(c)
    inv = gpd.GeoDataFrame(
        {"gid": [1, 2, 3]},
        geometry=[
            box(ba[0] + 100, ba[1] + 100, ba[0] + 1100, bc[1] + 100),  # a, b, c
            box(ba[0] + 2000, ba[3] - 500, ba[0] + 2500, ba[3] + 500),  # a, b
            box(ba[0] + 100, ba[1] + 100, ba[0] + 1100, ba[1] + 1100),  # a only
        ],
        crs="EPSG:32632",
    ).set_index(pd.Index([7, 7, 8]))  # non-unique index must not merge glaciers
    got = sorted(glacier_tiles([c, a, b], inv), key=lambda g: g[1])
    assert [names for _, names in got] == [(a, b), (a, b, c)]
    assert got[1][0] == pytest.approx(inv.geometry.iloc[0].area / 1e6)


def test_fold_pieces_counts_edge_and_corner_neighbours() -> None:
    folds = pd.DataFrame(
        {
            "tile": ["32TMS00", "32TMS11", "32TMS08", "32TMS80", "32TMS81"],
            "fold": [0, 0, 0, 1, 1],
        }
    )  # 00-11 touch at a corner, 08 is apart; 80-81 share an edge
    assert fold_pieces(folds) == {0: 2, 1: 1}


def test_straddling_glaciers_reports_area() -> None:
    folds = pd.DataFrame({"tile": ["32TMS00", "32TMS01"], "fold": [0, 1]})
    b0, b1 = _bounds("32TMS00"), _bounds("32TMS01")
    inv = gpd.GeoDataFrame(
        {"gid": [5]},
        geometry=[box(b0[0] + 100, b0[3] - 500, b0[0] + 1100, b1[1] + 500)],
        crs="EPSG:32632",
    )
    out = straddling_glaciers(folds, inv)
    assert out["area_km2"].tolist() == pytest.approx([1.0])
