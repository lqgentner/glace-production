"""Validation 2: spatial cross-validation with fixed baseline models.

For every (feature set, model, fold): choose the IoU threshold on an inner
fold, fit on the remaining training folds, predict the held-out fold. All
feature sets of a run share one support (rows where every run column is
finite) so their scores are paired.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import time
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd

from glace_production.validation.config import DatasetConfig, dataset_config_path
from glace_production.validation.folds import read_folds
from glace_production.validation.models import make_model
from glace_production.validation.store import KEY_COLUMNS, file_sha256

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from numpy.typing import NDArray
    from upath import UPath

    from glace_production.validation.config import RunConfig
    from glace_production.validation.models import Estimator
    from glace_production.validation.store import ValidationStore

logger = logging.getLogger(__name__)

THRESHOLD_GRID: NDArray[np.float64] = np.round(np.arange(0.005, 1.0, 0.005), 3)
"""Candidate probability thresholds, 0.005 to 0.995."""

MAX_FOLDS = np.iinfo(np.uint8).max
"""Largest fold count the ``uint8`` fold column of the predictions table can hold."""

type ModelFactory = Callable[[str, int, int], Estimator]
type FitHook = Callable[[int, int, set[str]], None]


@dataclass(frozen=True)
class RunInputs:
    """Everything a run needs in memory: samples on the key columns, folds, dataset configs.

    ``upstream`` identifies the inputs (sample generations, folds file) the
    run's outputs are bound to; the store refuses to resume or score a run
    whose inputs changed.
    """

    samples: pd.DataFrame
    folds: pd.Series
    datasets: Mapping[str, DatasetConfig]
    upstream: Mapping[str, str] = field(default_factory=dict)


def run_upstream(
    store: ValidationStore, run: RunConfig, folds_path: Path | UPath
) -> dict[str, str]:
    """Return a run's input identity: each dataset's sample generation and the folds file hash."""
    upstream = {
        f"samples:{name}": store.generation(store.samples_dir(name)) for name in run.datasets
    }
    upstream["folds_sha256"] = file_sha256(folds_path)
    return upstream


def load_run_inputs(
    run: RunConfig,
    store: ValidationStore,
    *,
    folds_path: Path | UPath,
    dataset_configs: Mapping[str, DatasetConfig],
) -> RunInputs:
    """Load the run's samples (only the columns it uses), folds and dataset configs."""
    run.validate_columns(dataset_configs)
    samples = store.load_samples(run.datasets, columns=run.columns)
    folds = read_folds(folds_path)
    missing = set(samples["tile"].unique()) - set(folds.index)
    if missing:
        msg = f"tiles without a fold: {sorted(missing)}"
        raise ValueError(msg)
    return RunInputs(
        samples=samples,
        folds=folds,
        datasets=dataset_configs,
        upstream=run_upstream(store, run, folds_path),
    )


def load_dataset_configs(run_path: Path, run: RunConfig) -> dict[str, DatasetConfig]:
    """Load the dataset configs a run names, from ``<config root>/datasets/``."""
    return {
        name: DatasetConfig.from_yaml(dataset_config_path(run_path, name)) for name in run.datasets
    }


def compute_support(samples: pd.DataFrame, columns: Sequence[str]) -> NDArray[np.bool_]:
    """Rows where every listed column is finite."""
    return np.isfinite(samples[list(columns)].to_numpy(dtype=np.float32)).all(axis=1)


def training_subsample(
    index: NDArray[np.intp], n: int, seed: int, fold: int, seed_index: int
) -> NDArray[np.intp]:
    """Up to ``n`` positions drawn without replacement, reproducible from the seeds."""
    if len(index) <= n:
        return np.sort(index)
    rng = np.random.default_rng([seed, fold, seed_index])
    # Draw positions into `index`, not values from it: rng.choice on the whole
    # candidate array permutes every element, while choosing among `len(index)`
    # positions and indexing costs only `n`.
    return index[np.sort(rng.choice(len(index), size=n, replace=False))]


def iou_at(y: NDArray[np.integer], proba: NDArray[np.floating], t: float) -> float:
    """IoU of the glacier class at ``predict = proba >= t``."""
    pred = proba >= t
    pos = y == 1
    tp = int((pred & pos).sum())
    fp = int((pred & ~pos).sum())
    fn = int((~pred & pos).sum())
    denom = tp + fp + fn
    return tp / denom if denom else float("nan")


def grid_ious(y: NDArray[np.integer], proba: NDArray[np.floating]) -> NDArray[np.float64]:
    """IoU of the glacier class at every threshold of :data:`THRESHOLD_GRID`.

    One sort of the probabilities and a cumulative count of the positives
    give every grid value; the result equals :func:`iou_at` per threshold.
    """
    order = np.argsort(proba, kind="stable")
    sorted_proba = proba[order].astype(np.float64)
    positives_from = np.cumsum((y[order] == 1)[::-1])[::-1]  # positives with proba >= this row
    n_pos = int(positives_from[0]) if len(positives_from) else 0
    n = len(proba)
    first = np.searchsorted(sorted_proba, THRESHOLD_GRID, side="left")  # rows with proba >= t
    predicted = n - first
    tp = np.where(first < n, positives_from[np.minimum(first, n - 1)], 0)
    denom = n_pos + predicted - tp  # tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, tp / denom, np.nan)


def best_threshold(y: NDArray[np.integer], proba: NDArray[np.floating]) -> tuple[float, float]:
    """Return the grid threshold maximising IoU; ties go to the lowest threshold."""
    if y.min() == y.max():
        msg = "threshold selection needs both classes"
        raise ValueError(msg)
    ious = grid_ious(y, proba)
    best = int(np.argmax(ious))  # argmax returns the first maximum: the lowest threshold
    return float(THRESHOLD_GRID[best]), float(ious[best])


def fit_on_arrays(
    x_all: NDArray[np.float32],
    y_all: NDArray[np.uint8],
    fold_of_row: NDArray[np.floating],
    tiles: NDArray[np.object_],
    fold: int,
    model: str,
    seed: int,
    seed_index: int,
    *,
    train_rows: int,
    factory: ModelFactory,
    exclude: Sequence[int] = (),
    on_fit: FitHook | None = None,
) -> Estimator:
    """Fit one model on every fold except ``fold`` and ``exclude`` (the inner fold).

    Takes the feature matrix, labels, fold assignment and tile names as
    precomputed arrays shared across every fit of a run, so a fit only slices
    and never recomputes them.
    """
    train = np.flatnonzero(~np.isin(fold_of_row, [fold, *exclude]))
    train = training_subsample(train, train_rows, seed, fold, seed_index)
    x = x_all[train]
    y = y_all[train]
    estimator = factory(model, x_all.shape[1], seed + seed_index)
    if on_fit is not None:
        on_fit(fold, exclude[0] if exclude else -1, set(tiles[train]))
    estimator.fit(x, y)
    return estimator


def fit_fold_model(
    samples: pd.DataFrame,
    folds: pd.Series,
    fold: int,
    columns: Sequence[str],
    model: str,
    seed: int,
    seed_index: int,
    *,
    train_rows: int,
    factory: ModelFactory,
    exclude: Sequence[int] = (),
    on_fit: FitHook | None = None,
) -> Estimator:
    """Fit one model on every fold except ``fold`` and ``exclude`` (the inner fold).

    Thin, standalone wrapper over :func:`fit_on_arrays` that builds the
    feature/fold/tile arrays from ``samples`` itself; for a map writer or
    other single-fit caller. ``run_crossval`` and ``benchmark`` precompute
    those arrays once per run and call :func:`fit_on_arrays` directly instead
    of recomputing them on every fit.
    """
    x_all = samples[list(columns)].to_numpy(dtype=np.float32)
    y_all = samples["label"].to_numpy(dtype=np.uint8)
    fold_of_row = samples["tile"].map(folds).to_numpy()
    tiles = samples["tile"].astype(str).to_numpy()
    return fit_on_arrays(
        x_all,
        y_all,
        fold_of_row,
        tiles,
        fold,
        model,
        seed,
        seed_index,
        train_rows=train_rows,
        factory=factory,
        exclude=exclude,
        on_fit=on_fit,
    )


def _predict(estimators: Sequence[Estimator], x: NDArray[np.float32]) -> NDArray[np.float32]:
    return np.mean([e.predict_proba(x)[:, 1] for e in estimators], axis=0).astype(np.float32)


def _fit_predict_feature_set_model(
    run: RunConfig,
    store: ValidationStore,
    samples: pd.DataFrame,
    fold_of_row: NDArray[np.floating],
    tiles: NDArray[np.object_],
    labels: NDArray[np.uint8],
    k: int,
    fs_name: str,
    model: str,
    x_all: NDArray[np.float32],
    *,
    factory: ModelFactory,
    on_fit: FitHook | None,
) -> None:
    """Thresholds and held-out predictions of one (feature set, model) pair, every fold."""
    proba = np.full(len(samples), np.nan, dtype=np.float32)
    fold_col = np.zeros(len(samples), dtype=np.uint8)
    thresholds: list[dict[str, object]] = []
    for fold in range(k):
        t0 = time.perf_counter()
        inner = (fold + 1) % k
        inner_rows = fold_of_row == inner
        inner_models = [
            fit_on_arrays(
                x_all,
                labels,
                fold_of_row,
                tiles,
                fold,
                model,
                run.seed,
                s,
                train_rows=run.train_rows,
                factory=factory,
                exclude=[inner],
                on_fit=on_fit,
            )
            for s in range(run.n_seeds)
        ]
        inner_proba = _predict(inner_models, x_all[inner_rows])
        t, iou = best_threshold(labels[inner_rows], inner_proba)
        thresholds.append(
            {
                "fold": fold,
                "threshold": t,
                "inner_fold": inner,
                "inner_iou": iou,
                "inner_n": int(inner_rows.sum()),
            }
        )
        outer_models = [
            fit_on_arrays(
                x_all,
                labels,
                fold_of_row,
                tiles,
                fold,
                model,
                run.seed,
                s,
                train_rows=run.train_rows,
                factory=factory,
                on_fit=on_fit,
            )
            for s in range(run.n_seeds)
        ]
        test_rows = fold_of_row == fold
        proba[test_rows] = _predict(outer_models, x_all[test_rows])
        fold_col[test_rows] = fold
        if run.save_models:
            path = store.models_path(run.name, fs_name, model, fold)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as handle:
                joblib.dump(outer_models, handle)
        logger.info("%s/%s fold %d: %.0f s", fs_name, model, fold, time.perf_counter() - t0)
    n_unpredicted = int(np.isnan(proba).sum())
    if n_unpredicted:
        msg = f"{fs_name}/{model}: {n_unpredicted} support rows received no prediction"
        raise RuntimeError(msg)
    predictions = samples[KEY_COLUMNS].assign(fold=fold_col, proba=proba)
    store.write_parquet(pd.DataFrame(thresholds), store.thresholds_path(run.name, fs_name, model))
    store.write_parquet(predictions, store.predictions_path(run.name, fs_name, model))


def run_crossval(
    run: RunConfig,
    inputs: RunInputs,
    store: ValidationStore,
    *,
    feature_sets: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    overwrite: bool = False,
    factory: ModelFactory | None = None,
    on_fit: FitHook | None = None,
) -> None:
    """Run every (feature set, model) of the config that is not complete yet."""
    factory = factory or make_model
    run_dir = store.run_dir(run.name)
    store.check_config(run_dir, run, overwrite=overwrite, upstream=inputs.upstream or None)

    support = compute_support(inputs.samples, run.columns)
    samples = inputs.samples[support].reset_index(drop=True)
    store.write_parquet(samples[KEY_COLUMNS], store.support_path(run.name))
    logger.info("support: %d of %d rows", len(samples), len(inputs.samples))
    fold_of_row = samples["tile"].map(inputs.folds).to_numpy()
    k = int(inputs.folds.max()) + 1
    if k > MAX_FOLDS:
        msg = f"{k} folds exceeds the uint8 fold-column capacity ({MAX_FOLDS})"
        raise ValueError(msg)
    labels = samples["label"].to_numpy(dtype=np.uint8)
    tiles = samples["tile"].astype(str).to_numpy()

    for fs_name in feature_sets or list(run.feature_sets):
        columns = run.feature_sets[fs_name]
        x_all = samples[list(columns)].to_numpy(dtype=np.float32)
        for model in models or run.models:
            pred_path = store.predictions_path(run.name, fs_name, model)
            thr_path = store.thresholds_path(run.name, fs_name, model)
            if pred_path.exists() and thr_path.exists() and not overwrite:
                logger.info("%s/%s: complete, skipped", fs_name, model)
                continue
            _fit_predict_feature_set_model(
                run,
                store,
                samples,
                fold_of_row,
                tiles,
                labels,
                k,
                fs_name,
                model,
                x_all,
                factory=factory,
                on_fit=on_fit,
            )


def benchmark(
    run: RunConfig,
    inputs: RunInputs,
    *,
    feature_set: str,
    factory: ModelFactory | None = None,
) -> dict[str, float]:
    """Time one fit and one fold prediction per model on the run's biggest feature set."""
    factory = factory or make_model
    support = compute_support(inputs.samples, run.columns)
    samples = inputs.samples[support].reset_index(drop=True)
    columns = run.feature_sets[feature_set]
    x_all = samples[list(columns)].to_numpy(dtype=np.float32)
    y_all = samples["label"].to_numpy(dtype=np.uint8)
    fold_of_row = samples["tile"].map(inputs.folds).to_numpy()
    tiles = samples["tile"].astype(str).to_numpy()
    timings: dict[str, float] = {}
    for model in run.models:
        t0 = time.perf_counter()
        estimator = fit_on_arrays(
            x_all,
            y_all,
            fold_of_row,
            tiles,
            0,
            model,
            run.seed,
            0,
            train_rows=run.train_rows,
            factory=factory,
        )
        timings[f"{model}_fit_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        x = x_all[fold_of_row == 0]
        estimator.predict_proba(x)
        timings[f"{model}_predict_s"] = time.perf_counter() - t0
    return timings
