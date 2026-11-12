"""Spatial folds over MGRS tiles.

Folds are assigned to tiles, so every year of a tile shares a fold and the
guarantee is unseen *tiles*. `balanced_folds` gives every fold an equal share
of the glacier pixels while keeping folds compact and splitting little glacier
area between them; `assign_folds` is plain k-means over tile centres. A glacier
crossing a fold border is listed by `straddling_glaciers`, not prevented.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from freezebase.mgrs import MGRSGeoBox
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans
from upath import UPath

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

TILE_WIDTH_M = 10_000.0
"""Side of an MGRS tile; distances in the fold objective are in tile widths."""

type Glacier = tuple[float, tuple[int, ...]]
"""Area (km²) of a glacier and the positions of the tiles it intersects."""

_MILP_OPTIMAL = 0
"""``scipy.optimize.milp`` status: optimal solution found."""

_MILP_INFEASIBLE = 2
"""``scipy.optimize.milp`` status: problem is infeasible."""

_MIN_FOLDS = 2
"""Fewest folds a cross-validation split can have."""


def _tile_frame(tiles: Sequence[str]) -> gpd.GeoDataFrame:
    if not tiles:
        msg = "no tiles"
        raise ValueError(msg)
    boxes = [MGRSGeoBox.from_mgrs(t) for t in tiles]
    codes = [b.crs.epsg if b.crs is not None else None for b in boxes]
    if any(code is None for code in codes):
        msg = "some tiles have no EPSG code"
        raise ValueError(msg)
    epsgs: set[int] = {code for code in codes if code is not None}
    if len(epsgs) != 1:
        zones = sorted(epsgs)
        msg = f"tiles span several UTM zones ({zones}); one zone is required"
        raise ValueError(msg)
    return gpd.GeoDataFrame(
        {"tile": list(tiles)}, geometry=[b.extent.geom for b in boxes], crs=f"EPSG:{epsgs.pop()}"
    )


def assign_folds(tiles: Sequence[str], k: int = 5, seed: int = 0) -> pd.DataFrame:
    """Assign ``k`` contiguous folds to tiles by k-means over their centres.

    Folds are renumbered west to east so the numbering is stable across seeds
    that produce the same regions.
    """
    frame = _tile_frame(tiles)
    xy = np.column_stack([frame.geometry.centroid.x, frame.geometry.centroid.y])
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(xy)
    order = np.argsort(km.cluster_centers_[:, 0])
    remap = {int(old): new for new, old in enumerate(order)}
    folds = np.array([remap[int(c)] for c in km.labels_], dtype=np.int64)
    frame = pd.DataFrame({"tile": list(tiles), "fold": folds})
    return frame.sort_values("tile", ignore_index=True)


def _outlines_on(frame: gpd.GeoDataFrame, inventory: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Outlines in the tiles' CRS with a positional id and their area in km²."""
    crs = frame.crs
    if crs is None:  # pragma: no cover - _tile_frame always sets crs
        msg = "tile frame has no CRS"
        raise ValueError(msg)
    outlines = inventory.reset_index(drop=True).to_crs(crs)
    outlines["_pos"] = np.arange(len(outlines))
    outlines["area_km2"] = outlines.geometry.area / 1e6
    return outlines


def glacier_tiles(
    tiles: Sequence[str], inventory: gpd.GeoDataFrame
) -> list[tuple[float, tuple[str, ...]]]:
    """Every glacier intersecting two or more tiles: its area (km²) and sorted tile names."""
    frame = _tile_frame(sorted(tiles))
    outlines = _outlines_on(frame, inventory)
    joined = gpd.sjoin(outlines[["_pos", "area_km2", "geometry"]], frame, predicate="intersects")
    out: list[tuple[float, tuple[str, ...]]] = []
    for _, group in joined.groupby("_pos"):
        names = tuple(sorted(set(group["tile"])))
        if len(names) > 1:
            out.append((float(group["area_km2"].iloc[0]), names))
    return out


def _objective(
    xy: NDArray[np.float64],
    centres: NDArray[np.float64],
    folds: NDArray[np.intp],
    glaciers: Sequence[Glacier],
    penalty: float,
) -> float:
    """Squared tile-to-centre distance plus ``penalty`` x area per extra fold a glacier touches."""
    distance = float(((xy - centres[folds]) ** 2).sum())
    split = sum(area * (len(set(folds[list(members)].tolist())) - 1) for area, members in glaciers)
    return distance + penalty * split


def _assign(
    xy: NDArray[np.float64],
    centres: NDArray[np.float64],
    weights: NDArray[np.float64],
    glaciers: Sequence[Glacier],
    *,
    k: int,
    tolerance: float,
    penalty: float,
    time_limit: float,
) -> NDArray[np.intp]:
    """Optimal fold of every tile for fixed centres.

    Variables are ``x[i, f]`` (tile ``i`` in fold ``f``, binary) followed by
    ``u[g, f]`` (glacier ``g`` touches fold ``f``, continuous in [0, 1];
    ``u[g, f] >= x[i, f]`` for its tiles, and minimisation sets it to their
    maximum). ``sum_f u[g, f]`` is the number of folds the glacier touches.
    """
    n, m = len(weights), len(glaciers)
    n_x = n * k
    target = float(weights.sum()) / k
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    def constrain(entries: Sequence[tuple[int, float]], low: float, high: float) -> None:
        row = len(lower)
        for col, val in entries:
            rows.append(row)
            cols.append(col)
            vals.append(val)
        lower.append(low)
        upper.append(high)

    for i in range(n):  # each tile in exactly one fold
        constrain([(i * k + f, 1.0) for f in range(k)], 1.0, 1.0)
    for f in range(k):  # glacier pixels of every fold within the tolerance
        constrain(
            [(i * k + f, float(weights[i])) for i in range(n)],
            target * (1 - tolerance),
            target * (1 + tolerance),
        )
    for g, (_, members) in enumerate(glaciers):  # u[g, f] >= x[i, f]
        for i in members:
            for f in range(k):
                constrain([(n_x + g * k + f, 1.0), (i * k + f, -1.0)], 0.0, np.inf)
    matrix = sparse.csr_array((vals, (rows, cols)), shape=(len(lower), n_x + m * k))
    distance = ((xy[:, None, :] - centres[None, :, :]) ** 2).sum(axis=-1).ravel()
    split_cost = penalty * np.repeat([area for area, _ in glaciers], k)
    result = milp(
        np.concatenate([distance, split_cost]),
        constraints=LinearConstraint(matrix, lower, upper),
        integrality=np.concatenate([np.ones(n_x, dtype=np.intp), np.zeros(m * k, dtype=np.intp)]),
        bounds=Bounds(0, 1),
        options={"mip_rel_gap": 0.0, "time_limit": time_limit},
    )
    if result.status == _MILP_INFEASIBLE:
        msg = f"no assignment keeps every fold within ±{tolerance:.0%} of the glacier pixels"
        raise ValueError(msg)
    if result.status != _MILP_OPTIMAL or result.x is None:
        msg = f"fold assignment not solved to optimality: {result.message}"
        raise RuntimeError(msg)
    return np.rint(result.x[:n_x]).reshape(n, k).argmax(axis=1)


def _solve_balanced(
    xy: NDArray[np.float64],
    weights: NDArray[np.float64],
    glaciers: Sequence[Glacier],
    *,
    k: int,
    tolerance: float,
    penalty: float,
    seed: int,
    max_iter: int,
    time_limit: float,
    names: Sequence[str],
) -> NDArray[np.intp]:
    """Alternate assignment and centre steps until the objective stops decreasing."""
    upper = float(weights.sum()) / k * (1 + tolerance)
    if (weights > upper).any():
        msg = f"a tile is heavier than a fold's upper bound ({upper:.0f} glacier pixels)"
        raise ValueError(msg)
    centres = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(xy).cluster_centers_
    t0 = time.perf_counter()
    folds = _assign(
        xy,
        centres,
        weights,
        glaciers,
        k=k,
        tolerance=tolerance,
        penalty=penalty,
        time_limit=time_limit,
    )
    logger.info("initial assignment solved in %.1f s", time.perf_counter() - t0)
    best = np.inf
    for iteration in range(max_iter):
        centres = np.array([xy[folds == f].mean(axis=0) for f in range(k)])
        current = _objective(xy, centres, folds, glaciers, penalty)
        logger.info("fold assignment iteration %d: objective %.4f", iteration, current)
        if current >= best - 1e-9:
            break
        best = current
        t0 = time.perf_counter()
        candidate = _assign(
            xy,
            centres,
            weights,
            glaciers,
            k=k,
            tolerance=tolerance,
            penalty=penalty,
            time_limit=time_limit,
        )
        logger.info("assignment solved in %.1f s", time.perf_counter() - t0)
        if _objective(xy, centres, candidate, glaciers, penalty) >= current - 1e-9:
            break
        folds = candidate
    else:
        logger.warning("fold assignment still improving after %d iterations", max_iter)
    keys: list[tuple[float, str]] = [
        (
            float(xy[folds == f, 0].mean()),
            min(n for n, g in zip(names, folds, strict=True) if g == f),
        )
        for f in range(k)
    ]
    order = sorted(range(k), key=lambda f: keys[f])
    remap = np.empty(k, dtype=np.intp)
    remap[order] = np.arange(k)
    return remap[folds]


def balanced_folds(
    tiles: Sequence[str],
    glacier_px: Mapping[str, float],
    inventory: gpd.GeoDataFrame,
    *,
    k: int = 5,
    tolerance: float = 0.05,
    penalty: float = 10.0,
    seed: int = 0,
    max_iter: int = 20,
    time_limit: float = 600.0,
) -> pd.DataFrame:
    """Assign ``k`` compact folds with equal glacier pixels and little glacier area split.

    Parameters
    ----------
    tiles
        MGRS tile names, one UTM zone.
    glacier_px
        Glacier pixels of every tile, summed over its years.
    inventory
        Glacier outlines; a glacier's area is charged once for every fold it
        touches beyond the first.
    k
        Number of folds.
    tolerance
        Allowed relative deviation of a fold's glacier pixels from the mean.
    penalty
        Cost per km² of split glacier, relative to the squared distance (in
        tile widths) of a tile to its fold centre.
    seed
        Seed of the initial k-means centres.
    max_iter
        Maximum number of assignment and centre steps.
    time_limit
        Seconds allowed for one assignment; exceeding it raises.

    Returns
    -------
    pandas.DataFrame
        Columns ``tile, fold``, sorted by tile, folds numbered west to east.
    """
    names = sorted(tiles)
    missing = [t for t in names if t not in glacier_px]
    if missing:
        msg = f"tiles with no glacier pixel count: {missing}"
        raise ValueError(msg)
    weights = np.array([float(glacier_px[t]) for t in names])
    checks = [
        (k >= _MIN_FOLDS, "at least two folds are needed"),
        (len(names) >= k, "fewer tiles than folds"),
        (0.0 <= tolerance < 1.0, "tolerance must lie in [0, 1)"),
        (penalty >= 0.0, "penalty must be non-negative"),
        (max_iter >= 1, "max_iter must be at least 1"),
        (
            bool(np.isfinite(weights).all() and (weights >= 0).all()),
            "glacier pixel counts must be finite and non-negative",
        ),
        (weights.sum() > 0, "the tiles hold no glacier pixels"),
    ]
    for ok, msg in checks:
        if not ok:
            raise ValueError(msg)
    frame = _tile_frame(names)
    xy = np.column_stack([frame.geometry.centroid.x, frame.geometry.centroid.y]) / TILE_WIDTH_M
    index = {t: i for i, t in enumerate(names)}
    glaciers = [
        (area, tuple(index[t] for t in members))
        for area, members in glacier_tiles(names, inventory)
    ]
    folds = _solve_balanced(
        xy,
        weights,
        glaciers,
        k=k,
        tolerance=tolerance,
        penalty=penalty,
        seed=seed,
        max_iter=max_iter,
        time_limit=time_limit,
        names=names,
    )
    return pd.DataFrame({"tile": names, "fold": folds.astype(np.int64)})


def fold_pieces(folds: pd.DataFrame) -> dict[int, int]:
    """Count the connected pieces of every fold; tiles touching at an edge or a corner connect."""
    frame = _tile_frame(list(folds["tile"])).merge(folds, on="tile")
    fold_values = frame["fold"].to_numpy()
    pieces: dict[int, int] = {}
    for fold in np.unique(fold_values):
        geoms = list(frame.loc[fold_values == fold, "geometry"])
        touching = np.array([[a.intersects(b) for b in geoms] for a in geoms], dtype=np.int8)
        pieces[int(fold)] = int(
            connected_components(sparse.csr_array(touching), directed=False)[0]
        )
    return pieces


def straddling_glaciers(
    folds: pd.DataFrame, inventory: gpd.GeoDataFrame, *, id_column: str = "gid"
) -> pd.DataFrame:
    """Glaciers whose outline intersects tiles of more than one fold.

    Includes their area in km² in the tiles' CRS.
    """
    tiles = _tile_frame(list(folds["tile"])).merge(folds, on="tile")
    if tiles.crs is None:  # pragma: no cover - _tile_frame always sets crs
        msg = "tile frame lost its CRS in the merge"
        raise ValueError(msg)
    outlines = _outlines_on(tiles, inventory)
    joined = gpd.sjoin(
        outlines[[id_column, "area_km2", "geometry"]], tiles, predicate="intersects"
    )
    grouped = joined.groupby(id_column).agg(
        # pandas types a NamedAgg callable as taking `...`, leaving the group
        # under aggregation without a type to infer the lambda parameter from.
        # pyrefly: ignore[implicit-any-lambda]
        tiles=("tile", lambda s: sorted(set(s))),
        # pyrefly: ignore[implicit-any-lambda]  # as above
        folds=("fold", lambda s: sorted(set(s))),
        area_km2=("area_km2", "first"),
    )
    out = grouped[grouped["folds"].map(len) > 1].reset_index()
    return out.rename(columns={id_column: "gid"})[["gid", "tiles", "folds", "area_km2"]]


def read_folds(path: Path | UPath) -> pd.Series:
    """Read the committed fold CSV as a ``tile -> fold`` series."""
    with UPath(path).open("r") as handle:
        frame = pd.read_csv(handle, dtype={"tile": str, "fold": int})
    return frame.set_index("tile")["fold"]
