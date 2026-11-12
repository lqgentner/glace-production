"""Scores from stored predictions: confusion, fold thresholds, groupings."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from glace_production.validation.config import DatasetConfig, LayerSpec, RunConfig
from glace_production.validation.metrics import (
    _rows_for,
    confusion,
    glacier_table,
    metrics_table,
    score,
)
from glace_production.validation.store import KEY_COLUMNS, ConfigMismatchError, ValidationStore


def test_confusion_and_score_hand_computed() -> None:
    y = np.array([1, 1, 0, 0, 1], np.uint8)
    pred = np.array([True, False, True, False, True])
    assert confusion(y, pred) == (2, 1, 1, 1)
    s = score(y, np.array([0.9, 0.2, 0.7, 0.1, 0.8]), pred)
    assert s["iou"] == pytest.approx(1 / 2)
    assert s["precision"] == pytest.approx(2 / 3)
    assert s["recall"] == pytest.approx(2 / 3)
    assert s["fpr"] == pytest.approx(1 / 2)
    assert s["prevalence"] == pytest.approx(3 / 5)
    assert s["n"] == 5
    auc = s["auc"]
    assert isinstance(auc, float)
    assert 0 < auc <= 1
    assert s["reason"] == ""


def test_f1_is_zero_not_nan_when_nothing_is_right() -> None:
    y = np.array([1, 1, 0, 0], np.uint8)
    proba = np.array([0.1, 0.2, 0.8, 0.9], np.float32)
    # nothing predicted glacier: precision undefined, recall 0, F1 0
    s = score(y, proba, np.zeros(4, bool))
    assert s["f1"] == 0.0
    assert s["iou"] == 0.0
    # everything wrong: precision 0 and recall 0, F1 0
    s = score(y, proba, np.array([False, False, True, True]))
    assert s["f1"] == 0.0
    # no glacier and nothing predicted: undefined
    s = score(np.zeros(4, np.uint8), proba, np.zeros(4, bool))
    assert np.isnan(s["f1"])
    assert s["fpr"] == 0.0


def test_score_single_class() -> None:
    s = score(np.ones(3, np.uint8), np.array([0.1, 0.5, 0.9]), np.array([True, True, False]))
    assert np.isnan(s["auc"])
    assert s["reason"] == "single class"
    assert s["recall"] == pytest.approx(2 / 3)
    assert np.isnan(s["fpr"])


def test_rows_for_groups_by_index() -> None:
    # Groups A and B give different confusion counts, so an indexing bug
    # (scoring the wrong rows per group) shows up as a wrong value.
    frame = pd.DataFrame(
        {
            "label": np.array([1, 0, 1, 1, 0, 1], np.uint8),
            "proba": np.array([0.9, 0.2, 0.8, 0.4, 0.3, 0.7], np.float32),
            "pred": np.array([True, False, True, False, False, True]),
        }
    )
    key = np.array(["A", "A", "A", "B", "B", "B"], dtype=object)
    rows = _rows_for(frame, feature_set="fs", model="m", group_by="glacier", key=key)
    table = pd.DataFrame(rows).set_index(["group", "metric"])["value"]
    n = pd.DataFrame(rows).set_index(["group", "metric"])["n"]
    # group A: rows 0, 1, 2 -> y=[1, 0, 1], pred=[T, F, T] -> tp=2 fp=0 fn=0 -> iou=1.0
    assert table[("A", "iou")] == pytest.approx(1.0)
    assert n[("A", "iou")] == 3
    # group B: rows 3, 4, 5 -> y=[1, 0, 1], pred=[F, F, T] -> tp=1 fp=0 fn=1 -> iou=0.5
    assert table[("B", "iou")] == pytest.approx(0.5)
    assert n[("B", "iou")] == 3


def _store_with_run(tmp_path: Path) -> tuple[ValidationStore, RunConfig, pd.Series]:
    store = ValidationStore(tmp_path)
    rng = np.random.default_rng(0)
    rows = []
    for tile, _fold in (("32TMS00", 0), ("32TMS10", 1)):
        n = 200
        label = rng.integers(0, 2, n).astype(np.uint8)
        label[:20] = np.arange(20) % 2  # the ambiguous block holds both classes
        rows.append(
            pd.DataFrame(
                {
                    "tile": tile,
                    "year": np.int16(2023),
                    "row": np.arange(n, dtype=np.uint16),
                    "col": np.uint16(0),
                    "label": label,
                    "dist_m": np.where(label == 1, -50.0, 50.0).astype(np.float32),
                    "gid": np.where(label == 1, 7, -1).astype(np.int32),
                    "near_gid": np.int32(7),
                    "debris": np.arange(n) % 5 == 0,
                    "elevation": np.float32(2600.0),
                    "n_obs_vv": np.float32(12.0),
                }
            )
        )
    loc = pd.concat(rows, ignore_index=True)
    # pandas-stubs types a groupby(["tile", "year"]) key as tuple[Hashable, ...],
    # not the concrete str/int columns it actually holds.
    for (tile, year), part in loc.groupby(["tile", "year"]):
        store.write_parquet(
            part.drop(columns="n_obs_vv"),
            # pyrefly: ignore[bad-argument-type]  # year is Hashable, see above
            store.pair_path(store.locations_dir, str(tile), int(year)),
        )
        store.write_parquet(
            part[[*KEY_COLUMNS, "n_obs_vv"]].assign(a=np.float32(0.5)),
            # pyrefly: ignore[bad-argument-type]  # as above
            store.pair_path(store.samples_dir("d"), str(tile), int(year)),
        )
    run = RunConfig(
        name="r", datasets=["d"], folds="f.csv", feature_sets={"a": ["a"]}, models=["rf"]
    )
    store.check_config(store.run_dir("r"), run, overwrite=True)
    store.write_parquet(loc[KEY_COLUMNS], store.support_path("r"))
    proba = np.where(loc.label == 1, 0.8, 0.3).astype(np.float32)
    proba[:20] = 0.5
    proba[200:220] = 0.5  # one ambiguous block per tile: fold thresholds decide
    fold = loc["tile"].map({"32TMS00": 0, "32TMS10": 1}).astype(np.uint8)
    store.write_parquet(
        loc[KEY_COLUMNS].assign(fold=fold, proba=proba), store.predictions_path("r", "a", "rf")
    )
    store.write_parquet(
        pd.DataFrame(
            {
                "fold": [0, 1],
                "threshold": [0.45, 0.55],
                "inner_fold": [1, 0],
                "inner_iou": 0.9,
                "inner_n": 100,
            }
        ),
        store.thresholds_path("r", "a", "rf"),
    )
    folds = pd.Series({"32TMS00": 0, "32TMS10": 1})
    return store, run, folds


def test_metrics_table_groups_and_fold_thresholds(tmp_path: Path) -> None:
    store, run, folds = _store_with_run(tmp_path)
    table = metrics_table(run, store, datasets=_datasets_for_store(), folds=folds)
    assert "scope" not in table.columns
    groupings = {"all", "tile", "year", "fold", "elevation", "debris", "band", "n_obs", "glacier"}
    assert groupings <= set(table["group_by"])
    pooled = table[table.group_by == "all"].set_index("metric")["value"]
    assert pooled["auc"] > 0.9
    # fold 0 (threshold 0.45) classifies its 0.5 block as glacier: full recall, some
    # false positives; fold 1 (0.55) does not: misses the block's glaciers, no false positives
    per_fold = table[table.group_by == "fold"].pivot_table(
        index="group", columns="metric", values="value"
    )

    def cell(group: str, metric: str) -> float:
        # pandas-stubs types a scalar `.loc[row, col]` lookup as a union over every
        # dtype pandas can hold, not the float this pivot table actually stores.
        # pyrefly: ignore[bad-argument-type]
        return float(per_fold.loc[group, metric])

    assert cell("0", "recall") == 1.0
    assert cell("0", "fpr") > 0
    assert cell("1", "recall") < 1.0
    assert cell("1", "fpr") == 0.0
    # bands: every glacier row is inner (dist -50), every background row outer (dist 50)
    band = table[(table.group_by == "band") & (table.metric == "recall")].set_index("group")
    assert set(band.index) == {"inner", "outer"}
    prevalence = table[(table.group_by == "all") & (table.metric == "prevalence")].iloc[0]["value"]
    assert band.loc["inner", "n"] == pytest.approx(prevalence * 400)
    fpr = table[(table.group_by == "band") & (table.metric == "fpr")].set_index("group")["value"]
    assert 0 <= fpr["outer"] <= 1
    assert np.isnan(fpr["inner"])
    support = table[table.metric == "support_frac"]
    assert not support.empty
    assert (support["value"] == 1.0).all()


def test_metrics_table_parallel_matches_sequential(tmp_path: Path) -> None:
    store, run, folds = _store_with_run(tmp_path)
    datasets = {
        "d": DatasetConfig(
            name="d",
            comp_dir="x",
            layers={
                "a": LayerSpec(pattern="{tile}", kind="coherence"),
                "n_obs_vv": LayerSpec(pattern="{tile}", kind="count"),
            },
        )
    }
    sequential = metrics_table(run, store, datasets=datasets, folds=folds, workers=1)
    parallel = metrics_table(run, store, datasets=datasets, folds=folds, workers=2)
    sort_cols = ["feature_set", "model", "group_by", "group", "metric"]
    pd.testing.assert_frame_equal(
        sequential.sort_values(sort_cols).reset_index(drop=True),
        parallel.sort_values(sort_cols).reset_index(drop=True),
    )


def _stamp(store: ValidationStore, run: RunConfig) -> RunConfig:
    """Mark the stored run as produced by ``run`` without touching its outputs."""
    (store.run_dir(run.name) / "config.yaml").write_text(run.to_yaml())
    return run


def _datasets_for_store() -> dict[str, DatasetConfig]:
    return {
        "d": DatasetConfig(
            name="d",
            comp_dir="x",
            layers={
                "a": LayerSpec(pattern="{tile}", kind="coherence"),
                "n_obs_vv": LayerSpec(pattern="{tile}", kind="count"),
            },
        )
    }


def test_metrics_table_skips_missing_pairs_and_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store, run, folds = _store_with_run(tmp_path)
    # feature set "b" was never run: its pair is skipped with a warning.
    run_with_b = _stamp(store, run.model_copy(update={"feature_sets": {"a": ["a"], "b": ["a"]}}))
    with caplog.at_level("WARNING"):
        table = metrics_table(run_with_b, store, datasets=_datasets_for_store(), folds=folds)
    assert any("b/rf" in r.message for r in caplog.records)
    scored_models = set(table.loc[table["model"] != "", "model"])
    assert scored_models == {"rf"}


def test_metrics_table_raises_when_no_pair_is_scored(tmp_path: Path) -> None:
    store, run, folds = _store_with_run(tmp_path)
    # feature set "b" was never run: no pair can be scored.
    run_without_matches = _stamp(store, run.model_copy(update={"feature_sets": {"b": ["a"]}}))
    with pytest.raises(FileNotFoundError, match="no predictions"):
        metrics_table(run_without_matches, store, datasets=_datasets_for_store(), folds=folds)


def test_glacier_table_joins_attributes(tmp_path: Path) -> None:
    store, run, folds = _store_with_run(tmp_path)
    inventory = gpd.GeoDataFrame(
        {
            "gid": [7],
            "sgi-id": ["B36-26"],
            "name": ["Aletsch"],
            "area_km2": [80.0],
            "masl_med": [3000],
            "year_acq": [2023],
        },
        geometry=[box(0, 0, 1, 1)],
        crs="EPSG:2056",
    )
    debris = gpd.GeoDataFrame(
        {"sgi-id": ["B36-26", "B36-26"], "area_km2": [4.0, 4.0]},
        geometry=[box(0, 0, 1, 1)] * 2,
        crs="EPSG:2056",
    )
    straddling = pd.DataFrame({"gid": [7], "tiles": [["32TMS00", "32TMS10"]], "folds": [[0, 1]]})
    datasets = {
        "d": DatasetConfig(
            name="d", comp_dir="x", layers={"a": LayerSpec(pattern="{tile}", kind="coherence")}
        )
    }
    store.write_parquet(
        metrics_table(run, store, datasets=datasets, folds=folds), store.metrics_path("r")
    )
    table = glacier_table(run, store, inventory=inventory, debris=debris, straddling=straddling)
    row = table[table.metric == "iou"].iloc[0]
    assert row["name"] == "Aletsch"
    assert row["debris_fraction"] == pytest.approx(0.1)
    assert bool(row["straddles_fold"]) is True
    assert row["n"] == 400


def test_metrics_table_rejects_incomplete_predictions_and_thresholds(tmp_path: Path) -> None:
    store, run, folds = _store_with_run(tmp_path)
    pred_path = store.predictions_path("r", "a", "rf")
    pred = store.read_parquet(pred_path)
    store.write_parquet(pred.iloc[:-1], pred_path)
    with pytest.raises(ValueError, match="support rows"):
        metrics_table(run, store, datasets=_datasets_for_store(), folds=folds)
    store.write_parquet(pred, pred_path)
    thr_path = store.thresholds_path("r", "a", "rf")
    thr = store.read_parquet(thr_path)
    store.write_parquet(thr[thr.fold == 0], thr_path)
    with pytest.raises(ValueError, match="thresholds do not cover"):
        metrics_table(run, store, datasets=_datasets_for_store(), folds=folds)
    store.write_parquet(thr, thr_path)
    with pytest.raises(ConfigMismatchError):
        metrics_table(
            run,
            store,
            datasets=_datasets_for_store(),
            folds=folds,
            upstream={"folds_sha256": "other"},
        )
