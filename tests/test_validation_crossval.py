"""Spatial cross-validation on a toy store: support, leakage, thresholds, resume."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

import numpy as np
import pandas as pd
import pytest

from glace_production.validation.config import DatasetConfig, LayerSpec, RunConfig
from glace_production.validation.crossval import (
    THRESHOLD_GRID,
    RunInputs,
    best_threshold,
    compute_support,
    grid_ious,
    iou_at,
    run_crossval,
    training_subsample,
)
from glace_production.validation.store import KEY_COLUMNS, ConfigMismatchError, ValidationStore

if TYPE_CHECKING:
    from glace_production.validation.models import Estimator

TILES = [f"32TMS{i}{j}" for i in range(5) for j in range(2)]  # 10 tiles


def _samples(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for tile in TILES:
        for year in (2022, 2023):
            n = 300
            label = rng.integers(0, 2, n).astype(np.uint8)
            a = rng.normal(0.3, 0.1, n) + (1 - label) * 0.3  # separable: glacier low
            b = rng.normal(0.0, 1.0, n)
            frames.append(
                pd.DataFrame(
                    {
                        "tile": tile,
                        "year": np.int16(year),
                        "row": np.arange(n, dtype=np.uint16),
                        "col": np.zeros(n, np.uint16),
                        "label": label,
                        "dist_m": np.float32(0),
                        "gid": -1,
                        "near_gid": -1,
                        "debris": False,
                        "elevation": np.float32(2500),
                        "a": a.astype(np.float32),
                        "b": b.astype(np.float32),
                    }
                )
            )
    samples = pd.concat(frames, ignore_index=True)
    samples.loc[:9, "b"] = np.nan  # a NaN patch in feature b only
    samples["tile"] = samples["tile"].astype("category")
    return samples


def _inputs() -> RunInputs:
    folds = pd.Series({t: i % 5 for i, t in enumerate(TILES)}, name="fold")
    dataset = DatasetConfig(
        name="d",
        comp_dir="x",
        layers={
            "a": LayerSpec(pattern="{tile}", kind="coherence"),
            "b": LayerSpec(pattern="{tile}", kind="backscatter"),
        },
    )
    return RunInputs(samples=_samples(), folds=folds, datasets={"d": dataset})


def _run(**kw: object) -> RunConfig:
    base = {
        "name": "toy",
        "datasets": ["d"],
        "folds": "f.csv",
        "feature_sets": {"a": ["a"], "ab": ["a", "b"]},
        "models": ["rf"],
        "train_rows": 500,
        "seed": 0,
    }
    return RunConfig.model_validate(base | kw)


def test_compute_support_is_run_wide() -> None:
    samples = _samples()
    support = compute_support(samples, ["a", "b"])
    assert support.sum() == len(samples) - 10
    assert compute_support(samples, ["a"]).all()


def test_training_subsample_is_deterministic_and_capped() -> None:
    idx = np.arange(1000)
    s1 = training_subsample(idx, 100, seed=0, fold=2, seed_index=0)
    s2 = training_subsample(idx, 100, seed=0, fold=2, seed_index=0)
    s3 = training_subsample(idx, 100, seed=0, fold=3, seed_index=0)
    assert len(s1) == 100
    assert np.array_equal(s1, s2)
    assert not np.array_equal(s1, s3)
    assert len(training_subsample(idx[:50], 100, seed=0, fold=0, seed_index=0)) == 50


def test_best_threshold_from_grid() -> None:
    proba = np.array([0.1, 0.2, 0.6, 0.9])
    y = np.array([0, 0, 1, 1])
    t, iou = best_threshold(y, proba)
    assert t in THRESHOLD_GRID
    assert 0.2 < t <= 0.6
    assert iou == 1.0
    # ties -> lowest threshold: all thresholds in (0.2, 0.6] give IoU 1
    assert t == pytest.approx(0.205)


def test_run_crossval_outputs_and_no_leakage(tmp_path: Path) -> None:
    store = ValidationStore(tmp_path)
    inputs = _inputs()
    run = _run()
    seen: list[tuple[int, int, set[str]]] = []
    run_crossval(
        run,
        inputs,
        store,
        factory=_recording_factory,
        on_fit=lambda f, inner, tiles: seen.append((f, inner, tiles)),
    )
    support = store.read_parquet(store.support_path("toy"))
    assert len(support) == len(inputs.samples) - 10
    pred = store.read_parquet(store.predictions_path("toy", "a", "rf"))
    assert list(pred.columns) == [*KEY_COLUMNS, "fold", "proba"]
    assert len(pred) == len(support)
    assert not pred.duplicated(KEY_COLUMNS).any()
    thr = store.read_parquet(store.thresholds_path("toy", "a", "rf"))
    assert list(thr.columns) == ["fold", "threshold", "inner_fold", "inner_iou", "inner_n"]
    assert len(thr) == 5  # one row per fold
    assert thr["threshold"].between(0.005, 0.995).all()
    assert thr["threshold"].isin(THRESHOLD_GRID).all()
    assert (thr["inner_fold"] == (thr["fold"] + 1) % 5).all()
    assert store.predictions_path("toy", "ab", "rf").exists()
    # leakage: the tiles each fit saw never include the test fold's tiles,
    # and an inner fit (inner >= 0) never saw the inner fold either
    fold_of = inputs.folds
    test_tiles = {f: {t for t in TILES if fold_of[t] == f} for f in range(5)}
    assert len(seen) == 20  # one inner and one outer fit per fold and feature set
    for f, inner, fitted in seen:
        assert fitted
        assert not (fitted & test_tiles[f])
        if inner >= 0:
            assert inner == (f + 1) % 5
            assert not (fitted & test_tiles[inner])


def test_run_crossval_predictions_are_informative(tmp_path: Path) -> None:
    store = ValidationStore(tmp_path)
    inputs = _inputs()
    run_crossval(_run(), inputs, store, factory=_recording_factory)
    pred = store.read_parquet(store.predictions_path("toy", "a", "rf"))
    merged = pred.merge(inputs.samples[[*KEY_COLUMNS, "label"]], on=KEY_COLUMNS)
    assert merged.groupby("label")["proba"].mean()[1] > merged.groupby("label")["proba"].mean()[0]


def test_run_crossval_resume_rule(tmp_path: Path) -> None:
    store = ValidationStore(tmp_path)
    inputs = _inputs()
    run_crossval(_run(), inputs, store, factory=_recording_factory)
    with pytest.raises(ConfigMismatchError):
        run_crossval(_run(train_rows=400), inputs, store, factory=_recording_factory)
    run_crossval(_run(train_rows=400), inputs, store, overwrite=True, factory=_recording_factory)


def test_run_crossval_raises_when_a_tile_has_no_fold(tmp_path: Path) -> None:
    store = ValidationStore(tmp_path)
    samples = _samples()
    # Drop one tile's fold assignment: its support rows never match any test
    # fold, so they should never receive a prediction.
    folds = pd.Series({t: i % 5 for i, t in enumerate(TILES) if t != TILES[0]}, name="fold")
    dataset = DatasetConfig(
        name="d",
        comp_dir="x",
        layers={
            "a": LayerSpec(pattern="{tile}", kind="coherence"),
            "b": LayerSpec(pattern="{tile}", kind="backscatter"),
        },
    )
    inputs = RunInputs(samples=samples, folds=folds, datasets={"d": dataset})
    with pytest.raises(RuntimeError):
        run_crossval(_run(), inputs, store, factory=_recording_factory)


class _RecordingModel:
    """Stands in for a forest: remembers the fit weights, scores feature ``a``."""

    weights: ClassVar[list[np.ndarray | None]] = []

    def fit(
        self,
        x: np.ndarray,  # noqa: ARG002 -- sklearn signature
        y: np.ndarray,  # noqa: ARG002 -- sklearn signature
        sample_weight: np.ndarray | None = None,
    ) -> _RecordingModel:
        _RecordingModel.weights.append(sample_weight)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p = np.clip(1.0 - x[:, 0], 0, 1)
        return np.column_stack([1 - p, p])


def _recording_factory(model: str, n_features: int, seed: int) -> Estimator:  # noqa: ARG001
    # the toy estimator only implements the two methods the run calls
    return cast("Estimator", _RecordingModel())


def test_run_crossval_fits_unweighted(tmp_path: Path) -> None:
    inputs = _inputs()
    _RecordingModel.weights = []
    run_crossval(
        _run(models=["rf"]),
        inputs,
        ValidationStore(tmp_path / "plain"),
        factory=_recording_factory,
    )
    assert _RecordingModel.weights
    assert all(w is None for w in _RecordingModel.weights)


def test_best_threshold_unweighted_iou() -> None:
    y = np.array([1, 1, 0, 0, 1, 0], np.uint8)
    proba = np.array([0.9, 0.6, 0.55, 0.2, 0.4, 0.1], np.float32)
    t, iou = best_threshold(y, proba)
    # any t in (0.2, 0.4] predicts rows 0, 1, 2, 4: 3 tp, 1 fp, 0 fn -> IoU 0.75,
    # and the tie goes to the lowest grid value
    assert t == pytest.approx(0.205)
    assert iou == pytest.approx(0.75)
    assert iou_at(y, proba, 0.6) == pytest.approx(2 / 3)  # 2 tp, 0 fp, 1 fn
    with pytest.raises(ValueError, match="both classes"):
        best_threshold(np.ones(3, np.uint8), proba[:3])


def test_grid_ious_match_the_per_threshold_definition() -> None:
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 5000).astype(np.uint8)
    proba = np.clip(rng.normal(0.35, 0.25, 5000) + 0.3 * y, 0, 1).astype(np.float32)
    proba[:50] = np.float32(0.7)  # ties on a grid value, float32 just below float64 0.7
    expected = np.array([iou_at(y, proba, t) for t in THRESHOLD_GRID])
    np.testing.assert_allclose(grid_ious(y, proba), expected, rtol=0, atol=1e-12)
    t, iou = best_threshold(y, proba)
    assert t == THRESHOLD_GRID[int(np.argmax(expected))]
    assert iou == pytest.approx(expected.max())
