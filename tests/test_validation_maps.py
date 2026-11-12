"""Full-tile probability maps from fold models."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from freezebase.mgrs import MGRSGeoBox
import numpy as np
import pandas as pd
import pytest
import rasterio

from glace_production.validation.config import DatasetConfig, LayerSpec, RunConfig
from glace_production.validation.maps import (
    fold_thresholds,
    predict_tile_proba,
    to_class_map,
    to_uint8_map,
)
from glace_production.validation.store import ValidationStore

if TYPE_CHECKING:
    from pathlib import Path

    from glace_production.validation.models import Estimator

TILE = "32TMS55"


def _write_tile(path: Path, value: float) -> None:
    """Write a constant float32 tile on the MGRS grid with one nodata pixel at (0, 0)."""
    gb = MGRSGeoBox.from_mgrs(TILE)
    data = np.full(gb.shape.yx, value, dtype=np.float32)
    data[0, 0] = np.nan
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=gb.shape.x,
        height=gb.shape.y,
        count=1,
        dtype="float32",
        crs="EPSG:32632",
        transform=gb.affine,
        nodata=np.nan,
    ) as dst:
        dst.write(data, 1)


def test_to_uint8_map_masks_invalid_and_ignore() -> None:
    proba = np.array([[0.0, 0.5], [np.nan, 1.0]], dtype=np.float32)
    label = np.array([[0, 255], [1, 255]], dtype=np.uint8)
    out = to_uint8_map(proba, label=label)
    assert out.dtype == np.uint8
    assert out.tolist() == [[0, 255], [255, 255]]


class _MeanModel:
    """Scores the second feature column (a window mean) as the probability."""

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p = x[:, 1]
        return np.column_stack([1 - p, p])


def test_predict_tile_map_with_derived_columns(tmp_path: Path) -> None:
    gb = MGRSGeoBox.from_mgrs(TILE)
    _write_tile(tmp_path / "comp" / "2023" / f"S1_COH12_LRW_{TILE}_VV_x.tif", 0.4)
    dataset = DatasetConfig(
        name="d",
        comp_dir="comp/{year}",
        layers={
            "coh_vv": LayerSpec(
                pattern="S1_COH12_LRW_{tile}_VV_*.tif", kind="coherence", derive=[{"stat": "mean"}]
            )
        },
    )
    label = np.zeros(gb.shape.yx, np.uint8)
    proba = predict_tile_proba(
        # the toy estimator only implements predict_proba
        [cast("Estimator", _MeanModel())],
        ["coh_vv", "coh_vv_mean5"],
        {"d": dataset},
        TILE,
        2023,
        data_root=tmp_path,
    )
    out = to_uint8_map(proba, label=label)
    assert out[500, 500] == 40
    assert out[0, 500] == 255  # edge ring without a neighbouring tile
    assert out[1, 1] == 255  # next to the nodata corner
    assert out[3, 3] == 40
    cls = to_class_map(proba, 0.35, label=label)
    assert cls[500, 500] == 1
    assert cls[0, 500] == 255
    assert to_class_map(proba, 0.45, label=label)[500, 500] == 0


def test_to_class_map_cuts_in_float64() -> None:
    # float32 0.7 is below float64 0.7, as the metrics compare it
    proba = np.array([[0.7]], dtype=np.float32)
    label = np.zeros((1, 1), np.uint8)
    assert to_class_map(proba, 0.7, label=label).tolist() == [[0]]
    assert to_class_map(proba, 0.69, label=label).tolist() == [[1]]


def test_fold_thresholds_rejects_duplicate_folds(tmp_path: Path) -> None:
    store = ValidationStore(tmp_path)
    run = RunConfig(
        name="r",
        datasets=["d"],
        folds="folds.csv",
        feature_sets={"all": ["coh_vv"]},
        models=["rf"],
    )
    path = store.thresholds_path(run.name, "all", "rf")
    path.parent.mkdir(parents=True, exist_ok=True)
    store.write_parquet(pd.DataFrame({"fold": [0, 0], "threshold": [0.4, 0.6]}), path)
    with pytest.raises(ValueError, match="more than once"):
        fold_thresholds(store, run, "all", "rf")
