"""End to end: the maps reproduce the cross-validation's held-out predictions.

Three synthetic tiles with non-constant layers go through extraction (with
the halo statistics), a three-fold cross-validation and the map writer; the
probability map of the test tile must equal the stored out-of-fold
probabilities at the sampled pixels, and the classified map must be that
probability cut at the fold's stored threshold.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import uuid

from freezebase.mgrs import MGRSGeoBox
from freezebase.raster import rasterio_open
import numpy as np
import pandas as pd
import rasterio

from glace_production.labels import LabelSource, label_tags
from glace_production.validation.config import DatasetConfig, LayerSpec, RunConfig
from glace_production.validation.crossval import load_run_inputs, run_crossval
from glace_production.validation.extract import write_samples
from glace_production.validation.maps import NODATA, fold_thresholds, write_maps
from glace_production.validation.store import KEY_COLUMNS, ValidationStore

if TYPE_CHECKING:
    from pathlib import Path

    from upath import UPath

    from glace_production.validation.maps import MapKind

TILES = {"32TMS55": 0, "32TMS65": 1, "32TMS56": 2}
YEAR = 2023
IGNORE_ROWS = 10  # the first rows of every label mask are 255
IGNORE_COLS = 100  # the westernmost columns of the test tile are 255 (outside the domain)
N_ROWS = 400


def _write(
    path: Path,
    data: np.ndarray,
    gb: MGRSGeoBox,
    dtype: str,
    nodata: float,
    tags: dict[str, str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=gb.shape.x,
        height=gb.shape.y,
        count=1,
        dtype=dtype,
        crs=gb.crs.to_wkt() if gb.crs is not None else None,
        transform=gb.affine,
        nodata=nodata,
    ) as dst:
        dst.write(data.astype(dtype), 1)
        if tags:
            dst.update_tags(**tags)


def _build(tmp_path: Path) -> tuple[ValidationStore, DatasetConfig, Path]:
    rng = np.random.default_rng(0)
    store = ValidationStore(tmp_path / "store")
    labels_dir = tmp_path / "labels"
    for tile in TILES:
        gb = MGRSGeoBox.from_mgrs(tile)
        cols = np.arange(1000)[None, :].repeat(1000, axis=0)
        coh = 0.2 + 0.6 * cols / 1000 + rng.normal(0, 0.05, (1000, 1000))
        rtc = np.exp(rng.normal(-2.5, 0.5, (1000, 1000)))
        coh[3, 3] = np.nan  # a nodata pixel, so a few statistics are NaN inside the tile
        _write(
            tmp_path / "comp" / str(YEAR) / f"S1_COH12_LRW_{tile}_VV_x.tif",
            coh,
            gb,
            "float32",
            np.nan,
        )
        _write(
            tmp_path / "comp" / str(YEAR) / f"S1_RTC_LRW_{tile}_VV_x.tif",
            rtc,
            gb,
            "float32",
            np.nan,
        )
        label = (cols >= 500).astype(np.uint8)
        label[:IGNORE_ROWS] = 255
        if tile == "32TMS55":
            label[:, :IGNORE_COLS] = 255
        _write(
            labels_dir / f"LABELS_SGI2023_{YEAR}_{tile}.tif",
            label,
            gb,
            "uint8",
            255,
            label_tags(LabelSource("SGI2023", year_split=True), year=YEAR, ignore_buffer_m=250.0),
        )
        rows = rng.integers(IGNORE_ROWS, 1000, N_ROWS)
        pix_cols = rng.integers(IGNORE_COLS, 1000, N_ROWS)
        frame = pd.DataFrame(
            {
                "tile": tile,
                "year": np.int16(YEAR),
                "row": rows.astype(np.uint16),
                "col": pix_cols.astype(np.uint16),
                "label": label[rows, pix_cols],
                "dist_m": np.where(label[rows, pix_cols] == 1, np.float32(-20), np.float32(20)),
                "gid": np.int32(-1),
                "near_gid": np.int32(1),
                "debris": False,
                "elevation": np.float32(2500),
            }
        )
        store.write_parquet(frame, store.pair_path(store.locations_dir, tile, YEAR))
    # The generation id a stage run writes beside its outputs.
    (store.locations_dir / "generation.txt").write_text(uuid.uuid4().hex)
    dataset = DatasetConfig(
        name="d",
        comp_dir="comp/{year}",
        layers={
            "coh_vv": LayerSpec(
                pattern="S1_COH12_LRW_{tile}_VV_*.tif",
                kind="coherence",
                derive=[{"stat": "mean", "window": 5}, {"stat": "std", "window": 5}],
            ),
            "rtc_vv": LayerSpec(pattern="S1_RTC_LRW_{tile}_VV_*.tif", kind="backscatter"),
        },
    )
    return store, dataset, labels_dir


def test_maps_reproduce_held_out_predictions(tmp_path: Path) -> None:
    store, dataset, labels_dir = _build(tmp_path)
    write_samples(dataset, store, data_root=tmp_path)
    folds_path = tmp_path / "folds.csv"
    pd.DataFrame({"tile": list(TILES), "fold": list(TILES.values())}).to_csv(
        folds_path, index=False
    )
    run = RunConfig(
        name="e2e",
        datasets=["d"],
        folds=str(folds_path),
        feature_sets={"all": ["coh_vv", "coh_vv_mean5", "coh_vv_std5", "rtc_vv"]},
        models=["rf"],
        train_rows=10_000,
    )
    inputs = load_run_inputs(run, store, folds_path=folds_path, dataset_configs={"d": dataset})
    assert set(inputs.upstream) == {"samples:d", "folds_sha256"}
    run_crossval(run, inputs, store)
    pred = store.read_parquet(store.predictions_path("e2e", "all", "rf"))
    # a stale model file from another generation must not be picked up (save_models is off)
    stale = store.models_path("e2e", "all", "rf", 0)
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"not a model")

    def maps(kind: MapKind) -> list[Path | UPath]:
        return write_maps(
            run,
            inputs,
            store,
            feature_set="all",
            model="rf",
            labels_dir=labels_dir,
            inventory="SGI2023",
            kind=kind,
            data_root=tmp_path,
        )

    assert len(maps("proba")) == len(TILES)
    maps("class")

    tile = "32TMS55"
    maps_dir = store.maps_dir("e2e", "all", "rf")
    with rasterio_open(maps_dir / f"PRED_{tile}_{YEAR}.tif") as src:
        proba_map = src.read(1)
    with rasterio_open(maps_dir / f"CLS_{tile}_{YEAR}.tif") as src:
        class_map = src.read(1)
    threshold = fold_thresholds(store, run, "all", "rf")[0]

    rows = pred[pred["tile"] == tile]
    assert len(rows) > 0
    r = rows["row"].to_numpy()
    c = rows["col"].to_numpy()
    expected_pct = np.round(rows["proba"].to_numpy() * 100).astype(int)
    got = proba_map[r, c].astype(int)
    assert (got != NODATA).all()
    assert np.abs(got - expected_pct).max() <= 1
    expected_class = (rows["proba"].to_numpy(dtype=np.float64) >= threshold).astype(np.uint8)
    got_class = class_map[r, c]
    np.testing.assert_array_equal(got_class, expected_class)
    # masks: ignore rows and columns, and pixels with a missing statistic
    assert (proba_map[:IGNORE_ROWS] == NODATA).all()
    assert (proba_map[IGNORE_ROWS:, :IGNORE_COLS] == NODATA).all()
    assert (class_map[:IGNORE_ROWS] == NODATA).all()
    assert proba_map[IGNORE_ROWS + 1, IGNORE_COLS + 1] != NODATA
    assert set(np.unique(class_map)) <= {0, 1, NODATA}
    # the west edge has no neighbour: the two outermost columns carry no statistic
    assert (proba_map[IGNORE_ROWS:998, 998:] != NODATA).all()  # east edge borrows from 32TMS65
    with rasterio_open(maps_dir / f"PRED_32TMS65_{YEAR}.tif") as src:
        east = src.read(1)
    assert (east[IGNORE_ROWS:998, :2] != NODATA).all()  # its west edge borrows from 32TMS55
    assert (east[IGNORE_ROWS:, 998:] == NODATA).all()  # no tile further east
    # the support the run stored is what the maps used
    support = store.read_parquet(store.support_path("e2e"))
    assert len(support) == len(pred)
    assert list(support.columns) == KEY_COLUMNS
