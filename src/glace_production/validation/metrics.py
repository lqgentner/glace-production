"""Scores from stored out-of-fold predictions.

Every row is classified with the threshold of its own fold, then confusion
counts are pooled over the group. AUC and AP are rank-based on the pooled
probabilities and, per fold, on one model's probabilities. Distance bands
hold one class each, so their informative scores are recall inside the
outline and false-positive rate outside.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import multiprocessing
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from glace_production.validation.bands import distance_band
from glace_production.validation.store import KEY_COLUMNS

if TYPE_CHECKING:
    from collections.abc import Mapping

    import geopandas as gpd
    from numpy.typing import NDArray

    from glace_production.validation.config import DatasetConfig, RunConfig
    from glace_production.validation.store import ValidationStore

logger = logging.getLogger(__name__)

_SHARED: dict[str, object] = {}
"""Module state a forked worker process inherits from its parent.

``metrics_table`` fills this right before creating a ``fork``-context
``ProcessPoolExecutor`` and clears it in a ``finally`` once the pool is done.
Forking copies the parent's memory copy-on-write, so every worker sees this
module's globals as they were at fork time without anything being pickled;
only the small ``feature_set``/``model`` strings travel through the task
queue per pair, instead of the multi-gigabyte ``samples`` frame that would
otherwise have to be pickled once per task. ``workers == 1`` never touches
this holder.
"""

ELEVATION_BANDS = np.arange(0, 5001, 500)
N_OBS_BANDS = np.array([0, 5, 10, 15, 20, 25, 1_000])
# pandas-stubs' `cut` overloads only accept `Sequence[float]` bins, not an ndarray;
# pre-converting to plain Python floats also sidesteps a numpy-integer/float mismatch.
_ELEVATION_BINS: list[float] = [float(b) for b in ELEVATION_BANDS]
_N_OBS_BINS: list[float] = [float(b) for b in N_OBS_BANDS]
METRIC_NAMES = ("auc", "ap", "iou", "precision", "recall", "fpr", "f1")
METRICS_COLUMNS = [
    "feature_set",
    "model",
    "group_by",
    "group",
    "metric",
    "value",
    "n",
    "reason",
]


def confusion(y: NDArray[np.integer], pred: NDArray[np.bool_]) -> tuple[int, int, int, int]:
    """(tp, fp, fn, tn) of the glacier class."""
    pos = y == 1
    return (
        int((pred & pos).sum()),
        int((pred & ~pos).sum()),
        int((~pred & pos).sum()),
        int((~pred & ~pos).sum()),
    )


def score(
    y: NDArray[np.integer], proba: NDArray[np.floating], pred: NDArray[np.bool_]
) -> dict[str, float | str]:
    """All metrics of one group; ranking metrics are NaN with a reason for a single class."""
    tp, fp, fn, tn = confusion(y, pred)
    total = tp + fp + fn + tn
    out: dict[str, float | str] = {
        "iou": tp / (tp + fp + fn) if tp + fp + fn else float("nan"),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "fpr": fp / (fp + tn) if fp + tn else float("nan"),
        "n": float(total),
        "prevalence": (tp + fn) / total if total else float("nan"),
        "reason": "",
    }
    # Directly from the counts: a group with glacier pixels and no correct
    # prediction scores 0, not NaN, so failures stay in the summaries.
    out["f1"] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan")
    if len(y) and y.min() != y.max():
        out["auc"] = float(roc_auc_score(y, proba))
        out["ap"] = float(average_precision_score(y, proba))
    else:
        out["auc"] = out["ap"] = float("nan")
        out["reason"] = "single class" if len(y) else "empty"
    return out


def _groupings(
    samples: pd.DataFrame, count_column: str | None
) -> dict[str, NDArray[np.object_] | None]:
    groups: dict[str, NDArray[np.object_] | None] = {
        "all": None,
        "tile": samples["tile"].astype(str).to_numpy(),
        "year": samples["year"].astype(str).to_numpy(),
        "fold": samples["fold"].astype(str).to_numpy(),
        "elevation": pd.cut(samples["elevation"], _ELEVATION_BINS).astype(str).to_numpy(),
        "debris": np.where(samples["debris"], "debris", "clean"),
        "band": distance_band(samples["dist_m"].to_numpy(dtype=np.float32)).astype(object),
    }
    if count_column is not None and count_column in samples:
        groups["n_obs"] = (
            pd.cut(samples[count_column], _N_OBS_BINS, right=False).astype(str).to_numpy()
        )
    return groups


def _rows_for(
    samples: pd.DataFrame,
    *,
    feature_set: str,
    model: str,
    group_by: str,
    key: NDArray[np.object_] | None,
) -> list[dict[str, object]]:
    """Score every group of one grouping.

    Group membership is resolved with a single ``groupby`` over ``key``
    rather than a scan per group, which matters for the ~1300 glaciers.
    """
    y = samples["label"].to_numpy(dtype=np.uint8)
    proba = samples["proba"].to_numpy(dtype=np.float32)
    pred = samples["pred"].to_numpy(dtype=bool)
    if key is None:
        members: list[tuple[object, NDArray[np.intp]]] = [("all", np.arange(len(samples)))]
    else:
        positions = pd.Series(key).groupby(key).indices
        members = [(group, np.asarray(pos)) for group, pos in positions.items()]
    rows: list[dict[str, object]] = []
    for group, sel in members:
        s = score(y[sel], proba[sel], pred[sel])
        rows.extend(
            {
                "feature_set": feature_set,
                "model": model,
                "group_by": group_by,
                "group": str(group),
                "metric": m,
                "value": s[m],
                "n": s["n"],
                "reason": s["reason"],
            }
            for m in (*METRIC_NAMES, "prevalence")
        )
    return rows


def _with_predictions(
    store: ValidationStore, run: RunConfig, feature_set: str, model: str, samples: pd.DataFrame
) -> pd.DataFrame:
    pred = store.read_parquet(store.predictions_path(run.name, feature_set, model))
    thr = store.read_parquet(store.thresholds_path(run.name, feature_set, model))
    frame = samples.merge(pred, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    if len(frame) != len(samples):
        msg = (
            f"{feature_set}/{model}: predictions cover {len(frame)} of {len(samples)} "
            "support rows; the run is incomplete or from another generation"
        )
        raise ValueError(msg)
    if not np.isfinite(frame["proba"].to_numpy()).all():
        msg = f"{feature_set}/{model}: non-finite probabilities"
        raise ValueError(msg)
    folds = set(frame["fold"].unique())
    t = thr.set_index("fold")["threshold"]
    if set(t.index) != folds or t.index.has_duplicates or not np.isfinite(t).all():
        msg = f"{feature_set}/{model}: thresholds do not cover folds {sorted(folds)}"
        raise ValueError(msg)
    frame["pred"] = frame["proba"] >= frame["fold"].map(t).to_numpy()
    return frame


def _support_fraction_rows(
    samples: pd.DataFrame, fold_all: NDArray[np.object_], in_support: NDArray[np.bool_]
) -> list[dict[str, object]]:
    """Support fraction by class and fold, once per run."""
    rows: list[dict[str, object]] = []
    labels = samples["label"].to_numpy()
    for cls in (0, 1):
        for fold in np.unique(fold_all):
            sel = (labels == cls) & (fold_all == fold)
            n = int(sel.sum())
            rows.append(
                {
                    "feature_set": "",
                    "model": "",
                    "group_by": "fold_class",
                    "group": f"{fold}/{cls}",
                    "metric": "support_frac",
                    "value": float(in_support[sel].sum() / n) if n else float("nan"),
                    "n": float(n),
                    "reason": "",
                }
            )
    return rows


def _score_pair(
    store: ValidationStore,
    run: RunConfig,
    feature_set: str,
    model: str,
    samples: pd.DataFrame,
    count_column: str | None,
) -> list[dict[str, object]] | None:
    """Every metric row of one (feature set, model) pair, ``None`` if its outputs are missing."""
    pred_path = store.predictions_path(run.name, feature_set, model)
    thr_path = store.thresholds_path(run.name, feature_set, model)
    if not pred_path.exists() or not thr_path.exists():
        logger.warning("%s/%s: predictions or thresholds missing, skipped", feature_set, model)
        return None
    frame = _with_predictions(store, run, feature_set, model, samples)
    groups = _groupings(frame, count_column)
    rows: list[dict[str, object]] = []
    for group_by, key in groups.items():
        rows.extend(
            _rows_for(frame, feature_set=feature_set, model=model, group_by=group_by, key=key)
        )
    near = frame["near_gid"].to_numpy().astype(str)
    rows.extend(
        _rows_for(frame, feature_set=feature_set, model=model, group_by="glacier", key=near)
    )
    return rows


def _score_pair_shared(feature_set: str, model: str) -> list[dict[str, object]] | None:
    """Score one pair from the state a forked worker inherited from the parent.

    Reads ``_SHARED``, which ``metrics_table`` populates in the parent
    process immediately before the pool is created, so it is already present
    in this process's memory by the time this function runs.
    """
    store = cast("ValidationStore", _SHARED["store"])
    run = cast("RunConfig", _SHARED["run"])
    samples = cast("pd.DataFrame", _SHARED["samples"])
    count_column = cast("str | None", _SHARED["count_column"])
    return _score_pair(store, run, feature_set, model, samples, count_column)


def _score_pairs_sequential(
    store: ValidationStore,
    run: RunConfig,
    samples: pd.DataFrame,
    count_column: str | None,
    pairs: list[tuple[str, str]],
) -> dict[tuple[str, str], list[dict[str, object]] | None]:
    """Score every pair in this process, logging each as it finishes."""
    pair_results: dict[tuple[str, str], list[dict[str, object]] | None] = {}
    for feature_set, model in pairs:
        pair_rows = _score_pair(store, run, feature_set, model, samples, count_column)
        pair_results[feature_set, model] = pair_rows
        if pair_rows is not None:
            logger.info("%s/%s scored", feature_set, model)
    return pair_results


def _score_pairs_parallel(
    store: ValidationStore,
    run: RunConfig,
    samples: pd.DataFrame,
    count_column: str | None,
    pairs: list[tuple[str, str]],
    workers: int,
    mp_context: multiprocessing.context.BaseContext,
) -> dict[tuple[str, str], list[dict[str, object]] | None]:
    """Score every pair in forked worker processes, logging each as a result arrives.

    ``_SHARED`` is filled here, immediately before the pool is created, so
    every worker forked from this process inherits it; it is cleared in the
    ``finally`` regardless of how scoring ends.
    """
    _SHARED["store"] = store
    _SHARED["run"] = run
    _SHARED["samples"] = samples
    _SHARED["count_column"] = count_column
    pair_results: dict[tuple[str, str], list[dict[str, object]] | None] = {}
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp_context) as pool:
            futures = {
                pool.submit(_score_pair_shared, feature_set, model): (feature_set, model)
                for feature_set, model in pairs
            }
            for future in as_completed(futures):
                feature_set, model = futures[future]
                pair_rows = future.result()
                pair_results[feature_set, model] = pair_rows
                if pair_rows is not None:
                    logger.info("%s/%s scored", feature_set, model)
    finally:
        _SHARED.clear()
    return pair_results


def metrics_table(
    run: RunConfig,
    store: ValidationStore,
    *,
    datasets: Mapping[str, DatasetConfig],
    folds: pd.Series,
    workers: int = 1,
    upstream: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Every metric of every (feature set, model) at every grouping.

    Parameters
    ----------
    upstream
        Identity of the run's inputs (:func:`~glace_production.validation.crossval.run_upstream`);
        when given, the stored run must belong to exactly this config and these inputs.
    workers
        Number of (feature set, model) pairs to score at once. With
        ``workers > 1``, pairs are scored in worker processes started with
        ``multiprocessing.get_context("fork")``: forking copies this
        process's memory copy-on-write, so every worker already has the
        loaded, multi-gigabyte ``samples`` frame without it being pickled
        per task — only the small ``feature_set``/``model`` strings are. See
        ``_SHARED`` for how that state reaches the workers. ``workers == 1``
        scores every pair in this process. On a platform without a ``fork``
        start method, scoring falls back to ``workers == 1`` with a warning.
    """
    store.verify_config(store.run_dir(run.name), run, upstream=upstream)
    primary = datasets[run.datasets[0]]
    count_columns = [k for k, v in primary.layers.items() if v.kind == "count"]
    count_column = count_columns[0] if count_columns else None
    samples = store.load_samples(run.datasets, columns=count_columns)
    support = store.read_parquet(store.support_path(run.name))
    in_support = pd.MultiIndex.from_frame(samples[KEY_COLUMNS]).isin(
        pd.MultiIndex.from_frame(support)
    )
    fold_all = samples["tile"].map(folds).astype(str).to_numpy()
    rows = _support_fraction_rows(samples, fold_all, in_support)
    samples = samples[in_support].reset_index(drop=True)

    pairs: list[tuple[str, str]] = [
        (feature_set, model) for feature_set in run.feature_sets for model in run.models
    ]

    mp_context = None
    if workers > 1:
        try:
            mp_context = multiprocessing.get_context("fork")
        except ValueError:
            logger.warning("fork start method unavailable, scoring pairs with workers=1")

    pair_results = (
        _score_pairs_parallel(store, run, samples, count_column, pairs, workers, mp_context)
        if mp_context is not None
        else _score_pairs_sequential(store, run, samples, count_column, pairs)
    )

    scored_any = False
    for feature_set, model in pairs:
        pair_rows = pair_results[feature_set, model]
        if pair_rows is None:
            continue
        scored_any = True
        rows.extend(pair_rows)
    if not scored_any:
        msg = f"no predictions found for run {run.name!r}: run cross-validation first"
        raise FileNotFoundError(msg)
    return pd.DataFrame(rows, columns=METRICS_COLUMNS)


def glacier_table(
    run: RunConfig,
    store: ValidationStore,
    *,
    inventory: gpd.GeoDataFrame,
    debris: gpd.GeoDataFrame,
    straddling: pd.DataFrame,
) -> pd.DataFrame:
    """Per-glacier scores joined with inventory attributes and debris fraction."""
    metrics = store.read_parquet(store.metrics_path(run.name))
    per_glacier = metrics[(metrics.group_by == "glacier") & (metrics.group != "-1")].copy()
    per_glacier["gid"] = per_glacier["group"].astype(int)
    attrs = inventory[["gid", "sgi-id", "name", "area_km2", "masl_med", "year_acq"]].rename(
        columns={"sgi-id": "sgi_id"}
    )
    debris_area = debris.groupby("sgi-id")["area_km2"].sum().rename("debris_area")
    attrs = attrs.merge(debris_area, left_on="sgi_id", right_index=True, how="left")
    attrs["debris_fraction"] = (attrs["debris_area"].fillna(0.0) / attrs["area_km2"]).clip(
        upper=1.0
    )
    attrs["straddles_fold"] = attrs["gid"].isin(set(straddling["gid"]))
    out = per_glacier.merge(attrs.drop(columns="debris_area"), on="gid", how="left")
    return out[
        [
            "feature_set",
            "model",
            "gid",
            "sgi_id",
            "name",
            "area_km2",
            "masl_med",
            "year_acq",
            "debris_fraction",
            "straddles_fold",
            "metric",
            "value",
            "n",
        ]
    ]
