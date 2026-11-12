"""Full-tile probability maps from the fold models of a run."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from freezebase.mgrs import MGRSGeoBox
from freezebase.raster import COG_PROFILE, write_cog
import joblib
import numpy as np
import pandas as pd

from glace_production.config import settings
from glace_production.labels import IGNORE_VALUE
from glace_production.validation.config import PIXEL_SIZE_M
from glace_production.validation.crossval import compute_support, fit_fold_model
from glace_production.validation.extract import layer_columns
from glace_production.validation.models import make_model
from glace_production.validation.rasters import read_label
from glace_production.validation.store import KEY_COLUMNS

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from numpy.typing import NDArray
    from upath import UPath

    from glace_production.validation.config import DatasetConfig, LayerSpec, RunConfig
    from glace_production.validation.crossval import RunInputs
    from glace_production.validation.models import Estimator
    from glace_production.validation.store import ValidationStore

logger = logging.getLogger(__name__)
NODATA = 255


type MapKind = Literal["proba", "class"]
MAP_PREFIX: dict[str, str] = {"proba": "PRED", "class": "CLS"}
"""File-name prefix of each map kind."""


def to_uint8_map(proba: NDArray[np.floating], *, label: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Probability x 100 as uint8; 255 where a feature is invalid or the label is ignore."""
    valid = np.isfinite(proba) & (label != IGNORE_VALUE)
    out = np.full(proba.shape, NODATA, dtype=np.uint8)
    out[valid] = np.round(np.clip(proba[valid], 0, 1) * 100).astype(np.uint8)
    return out


def to_class_map(
    proba: NDArray[np.floating],
    threshold: float,
    *,
    label: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Glacier (1) where ``proba >= threshold``, else 0; 255 where invalid or ignore."""
    valid = np.isfinite(proba) & (label != IGNORE_VALUE)
    out = np.full(proba.shape, NODATA, dtype=np.uint8)
    # float64 comparison, as in the metrics: a float32 probability compared
    # against a float32 threshold can fall on the other side of the cut.
    out[valid] = (proba[valid].astype(np.float64) >= threshold).astype(np.uint8)
    return out


def _feature_sources(
    datasets: Mapping[str, DatasetConfig], year: int, data_root: Path | UPath | None
) -> dict[str, tuple[Path | UPath, str, LayerSpec]]:
    """Map every feature column to the (composite directory, layer name, spec) it comes from."""
    out: dict[str, tuple[Path | UPath, str, LayerSpec]] = {}
    for config in datasets.values():
        raw = config.comp_dir.format(year=year)
        directory = data_root / raw if data_root is not None else settings.resolve_data_path(raw)
        for column, feature in config.features.items():
            out[column] = (directory, feature.layer, feature.spec)
    return out


def predict_tile_proba(
    estimators: Sequence[Estimator],
    columns: Sequence[str],
    datasets: Mapping[str, DatasetConfig],
    tile: str,
    year: int,
    *,
    data_root: Path | UPath | None = None,
) -> NDArray[np.float32]:
    """Predict the glacier probability of every pixel of a tile; NaN where a feature is missing."""
    geobox = MGRSGeoBox.from_mgrs(tile, resolution=PIXEL_SIZE_M)
    sources = _feature_sources(datasets, year, data_root)
    arrays: dict[str, NDArray[np.float32]] = {}
    for column in columns:
        if column in arrays:
            continue
        directory, layer, spec = sources[column]
        # One read per layer serves the layer and every statistic derived from it.
        arrays.update(layer_columns(directory, spec, tile, layer))
    x = np.column_stack([arrays[c].ravel() for c in columns])
    proba = np.full(x.shape[0], np.nan, dtype=np.float32)
    finite = np.isfinite(x).all(axis=1)
    if finite.any():
        proba[finite] = np.mean([e.predict_proba(x[finite])[:, 1] for e in estimators], axis=0)
    return proba.reshape(geobox.shape.yx)


def fold_thresholds(
    store: ValidationStore, run: RunConfig, feature_set: str, model: str
) -> dict[int, float]:
    """Return the stored threshold of every fold of a (feature set, model)."""
    table = store.read_parquet(store.thresholds_path(run.name, feature_set, model))
    if table["fold"].duplicated().any():
        msg = f"thresholds of {feature_set}/{model}: a fold appears more than once"
        raise ValueError(msg)
    return {int(f): float(t) for f, t in zip(table["fold"], table["threshold"], strict=True)}


def write_maps(
    run: RunConfig,
    inputs: RunInputs,
    store: ValidationStore,
    *,
    feature_set: str,
    model: str,
    labels_dir: Path | UPath,
    inventory: str,
    kind: MapKind = "proba",
    overwrite: bool = False,
    data_root: Path | UPath | None = None,
) -> list[Path | UPath]:
    """Write one COG per test (tile, year) from each fold's model.

    ``kind`` selects the probability map (``PRED_*.tif``, percent) or the
    classified map (``CLS_*.tif``, 1 glacier / 0 other at the fold's stored
    threshold). The stored run must belong to this config and these inputs,
    and its support must equal the support recomputed here. A fold's
    estimators are loaded from ``store.models_path`` when the run saved them
    (``save_models``); otherwise they are refit exactly as the run fitted
    them. ``inventory`` is the label-source key in the mask file names
    (``SGI2023``), used to build ``LABELS_{inventory}_{year}_{tile}.tif``.
    The masks' ignore value already covers everything outside the inventory's
    domain, so the maps carry nodata there.
    """
    store.verify_config(store.run_dir(run.name), run, upstream=inputs.upstream or None)
    columns = run.feature_sets[feature_set]
    prefix = MAP_PREFIX[kind]
    thresholds: dict[int, float] = (
        fold_thresholds(store, run, feature_set, model) if kind == "class" else {}
    )
    support = compute_support(inputs.samples, run.columns)
    samples = inputs.samples[support].reset_index(drop=True)
    stored = store.read_parquet(store.support_path(run.name))
    if len(stored) != len(samples) or not pd.MultiIndex.from_frame(samples[KEY_COLUMNS]).equals(
        pd.MultiIndex.from_frame(stored[KEY_COLUMNS])
    ):
        msg = f"run {run.name}: the stored support differs from the samples loaded now"
        raise ValueError(msg)
    pairs = store.list_pairs(store.locations_dir)
    out_dir = store.maps_dir(run.name, feature_set, model)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path | UPath] = []
    for fold in sorted(inputs.folds.unique()):
        fold_pairs = [(t, y) for t, y in pairs if inputs.folds[t] == fold]
        if (
            all((out_dir / f"{prefix}_{t}_{y}.tif").exists() for t, y in fold_pairs)
            and not overwrite
        ):
            continue
        models_path = store.models_path(run.name, feature_set, model, int(fold))
        if run.save_models and models_path.exists():
            with models_path.open("rb") as handle:
                estimators: list[Estimator] = joblib.load(handle)
        else:
            estimators = [
                fit_fold_model(
                    samples,
                    inputs.folds,
                    int(fold),
                    columns,
                    model,
                    run.seed,
                    s,
                    train_rows=run.train_rows,
                    factory=make_model,
                )
                for s in range(run.n_seeds)
            ]
        for tile, year in fold_pairs:
            out_path = out_dir / f"{prefix}_{tile}_{year}.tif"
            if out_path.exists() and not overwrite:
                continue
            geobox = MGRSGeoBox.from_mgrs(tile, resolution=PIXEL_SIZE_M)
            label = read_label(
                labels_dir / f"LABELS_{inventory}_{year}_{tile}.tif", geobox, year=year
            )
            proba = predict_tile_proba(
                estimators, columns, inputs.datasets, tile, year, data_root=data_root
            )
            tags = {"RUN": run.name, "FEATURE_SET": feature_set, "MODEL": model, "FOLD": str(fold)}
            if kind == "class":
                threshold = thresholds[int(fold)]
                data = to_class_map(proba, threshold, label=label)
                band = "glacier_class"
                tags["THRESHOLD"] = f"{threshold:.3f}"
            else:
                data = to_uint8_map(proba, label=label)
                band = "glacier_probability_pct"
            profile = COG_PROFILE | {
                "dtype": "uint8",
                "count": 1,
                "nodata": NODATA,
                "width": geobox.shape.x,
                "height": geobox.shape.y,
                "crs": geobox.crs,
                "transform": geobox.affine,
            }
            write_cog(data[np.newaxis], out_path, profile, band_names=[band], tags=tags)
            written.append(out_path)
            logger.info("%s written", out_path.name)
    return written
