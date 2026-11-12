"""AUC and JM on synthetic samples: direction, ranges, grouping."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from glace_production.validation.config import DatasetConfig, DerivedSpec, Feature, LayerSpec
from glace_production.validation.crossval import compute_support
from glace_production.validation.separability import (
    LAYER_SUPPORT,
    SEPARABILITY_COLUMNS,
    common_support,
    common_support_columns,
    common_support_name,
    directional_auc,
    jm_distance,
    load_layer_samples,
    separability_table,
    separability_tables,
)
from glace_production.validation.store import ValidationStore

if TYPE_CHECKING:
    from pathlib import Path


def _two_classes(n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    labels = (np.arange(n) % 2).astype(np.uint8)
    values = (rng.normal(0.6, 0.1, n) - 0.4 * labels).astype(np.float32)
    return values, labels


def test_directional_auc_glacier_low() -> None:
    values, labels = _two_classes()
    assert directional_auc(values, labels) == pytest.approx(roc_auc_score(labels, -values))
    assert directional_auc(values, labels, glacier_low=False) == pytest.approx(
        roc_auc_score(labels, values)
    )
    assert directional_auc(values, labels) > 0.9


def test_jm_distance_range_and_clipping() -> None:
    labels = np.array([1] * 50 + [0] * 50, np.uint8)
    values = np.concatenate([np.full(50, 0.1), np.full(50, 0.9)])
    jm, below, above = jm_distance(values, labels, value_range=(0.0, 1.0))
    assert jm == pytest.approx(2.0)
    assert below == 0
    assert above == 0
    jm_same, _, _ = jm_distance(np.full(100, 0.5), labels, value_range=(0.0, 1.0))
    assert jm_same == pytest.approx(0.0)
    _, below, above = jm_distance(
        np.array([-1.0, 2.0, 0.5, 0.5]), np.array([1, 1, 0, 0]), value_range=(0.0, 1.0)
    )
    assert (below, above) == (1, 1)


def _samples(n: int = 400) -> pd.DataFrame:
    values, labels = _two_classes(n)
    return pd.DataFrame(
        {
            "tile": pd.Categorical(np.where(np.arange(n) < n // 2, "32TMS55", "32TMS65")),
            "year": np.where(np.arange(n) % 4 < 2, 2022, 2023).astype(np.int16),
            "label": labels,
            "x": values,
        }
    )


def _feature() -> Feature:
    return Feature("x", LayerSpec(pattern="{tile}", kind="coherence"))


def test_separability_table_groups() -> None:
    table = separability_table(_samples(), {"x": _feature()}, dataset="d")
    assert list(table.columns) == SEPARABILITY_COLUMNS
    assert set(table.group_by) == {"all", "tile", "year"}
    pooled = table[table.group_by == "all"].iloc[0]
    assert pooled["n_glacier"] == 200
    assert pooled["n_other"] == 200
    assert pooled["auc"] > 0.9
    assert 0 < pooled["jm"] <= 2
    assert set(table[table.group_by == "tile"].group) == {"32TMS55", "32TMS65"}
    assert set(table[table.group_by == "year"].group) == {"2022", "2023"}


def test_single_class_group_gives_nan() -> None:
    samples = _samples()
    samples.loc[samples.tile == "32TMS65", "label"] = 1
    table = separability_table(samples, {"x": _feature()}, dataset="d")
    row = table[(table.group_by == "tile") & (table.group == "32TMS65")].iloc[0]
    assert np.isnan(row["auc"])
    assert np.isnan(row["jm"])
    assert row["n_other"] == 0


def test_nan_values_are_counted_not_scored() -> None:
    samples = _samples()
    samples.loc[:9, "x"] = np.nan
    table = separability_table(samples, {"x": _feature()}, dataset="d")
    pooled = table[table.group_by == "all"].iloc[0]
    assert pooled["n_nan"] == 10
    assert pooled["n_glacier"] + pooled["n_other"] == len(samples) - 10
    assert (table.support == LAYER_SUPPORT).all()


def _two_layers() -> pd.DataFrame:
    """Layer ``x`` finite everywhere, layer ``y`` NaN on the glacier rows ``x`` separates worst."""
    samples = _samples()
    x = samples["x"].to_numpy()
    glacier = samples["label"].to_numpy() == 1
    hard = glacier & (x > np.quantile(x[glacier], 0.8))
    samples["y"] = np.where(hard, np.nan, x + 0.01).astype(np.float32)
    return samples


def _pooled(table: pd.DataFrame, feature: str) -> pd.Series:
    return table[(table.group_by == "all") & (table.feature == feature)].iloc[0]


def test_common_support_scores_every_layer_on_the_same_rows() -> None:
    samples = _two_layers()
    features = {"x": _feature(), "y": _feature()}
    support = compute_support(samples, list(features))
    name = "common:a,b"
    table = separability_table(samples, features, dataset="d", support=support, support_name=name)
    assert (table.support == name).all()
    x, y = _pooled(table, "x"), _pooled(table, "y")
    glacier = samples["label"].to_numpy() == 1
    assert x["n_glacier"] == y["n_glacier"] == (support & glacier).sum()
    assert x["n_other"] == y["n_other"] == (support & ~glacier).sum()
    assert x["n_nan"] == y["n_nan"] == (~support).sum()
    # y is x shifted: on the same rows both rank alike, and JM stays defined
    assert x["auc"] == pytest.approx(y["auc"])
    assert 0 < x["jm"] <= 2
    # per layer, x keeps the hard glacier rows y lacks and scores lower
    own = separability_table(samples, features, dataset="d")
    own_x, own_y = _pooled(own, "x"), _pooled(own, "y")
    assert own_x["n_glacier"] > x["n_glacier"]
    assert own_x["auc"] < own_y["auc"]
    per_tile = table[(table.group_by == "tile") & (table.feature == "x")]
    assert per_tile["n_glacier"].sum() == x["n_glacier"]


def test_support_must_match_samples_and_name() -> None:
    samples = _samples()
    features = {"x": _feature()}
    with pytest.raises(ValueError, match="rows"):
        separability_table(
            samples, features, dataset="d", support=np.ones(3, bool), support_name="a"
        )
    with pytest.raises(ValueError, match="support_name"):
        separability_table(samples, features, dataset="d", support=np.ones(len(samples), bool))
    with pytest.raises(ValueError, match="support_name"):
        separability_table(samples, features, dataset="d", support_name="a")


def test_derived_feature_uses_its_own_direction_and_range() -> None:
    values, labels = _two_classes()
    samples = pd.DataFrame(
        {
            "tile": pd.Categorical(["A"] * len(values)),
            "year": np.int16(2023),
            "label": labels,
            "coh_vv_std5": (values * 0.1).astype(np.float32),
        }
    )
    spec = LayerSpec(pattern="{tile}", kind="coherence", derive=[{"stat": "std"}])
    # glacier_low inherited from the layer: the std is lower for glacier here as well
    feature = Feature("coh_vv", spec, spec.derive[0])
    assert feature.value_range == (0.0, 0.5)
    table = separability_table(samples, {"coh_vv_std5": feature}, dataset="d")
    row = table[table.group_by == "all"].iloc[0]
    assert row["feature"] == "coh_vv_std5"
    assert row["kind"] == "coherence"
    assert row["auc"] > 0.9
    flipped = Feature("coh_vv", spec, DerivedSpec(stat="std", glacier_low=False))
    table = separability_table(samples, {"coh_vv_std5": flipped}, dataset="d")
    assert table[table.group_by == "all"].iloc[0]["auc"] < 0.1


def _dataset(name: str, layers: dict[str, dict[str, object]]) -> DatasetConfig:
    return DatasetConfig.model_validate({"name": name, "comp_dir": "c", "layers": layers})


def _configs() -> list[DatasetConfig]:
    """Return ``b`` with one layer and ``a`` with a windowed coherence layer and a count layer."""
    a = _dataset(
        "a",
        {
            "coh": {"pattern": "{tile}", "kind": "coherence", "derive": [{"stat": "mean"}]},
            "n_obs": {"pattern": "{tile}", "kind": "count"},
        },
    )
    b = _dataset("b", {"bs": {"pattern": "{tile}", "kind": "backscatter"}})
    return [b, a]


def _store(tmp_path: Path) -> ValidationStore:
    """Two tiles of 6 rows; ``b`` has no samples of the second tile, ``a`` a NaN ``coh``.

    ``a``'s window mean and count are NaN on rows where ``coh`` and ``bs`` are
    finite: they must not narrow the support.
    """
    store = ValidationStore(tmp_path)
    rng = np.random.default_rng(0)
    for tile, year in (("32TMS55", 2023), ("32TMS56", 2022)):
        n = 6
        keys = pd.DataFrame(
            {
                "tile": tile,
                "year": np.int16(year),
                "row": np.arange(n, dtype=np.uint16),
                "col": np.zeros(n, np.uint16),
            }
        )
        store.write_parquet(
            keys.assign(label=(np.arange(n) % 2).astype(np.uint8)),
            store.pair_path(store.locations_dir, tile, year),
        )
        coh = rng.uniform(0, 1, n).astype(np.float32)
        coh[0] = np.nan
        store.write_parquet(
            keys.assign(
                coh=coh,
                coh_mean5=np.float32(np.nan),
                n_obs=np.float32(np.nan),
            ),
            store.pair_path(store.samples_dir("a"), tile, year),
        )
        if tile == "32TMS55":
            store.write_parquet(
                keys.assign(bs=rng.uniform(-20, 0, n).astype(np.float32)),
                store.pair_path(store.samples_dir("b"), tile, year),
            )
    return store


def test_common_support_columns_are_measurement_layers_only() -> None:
    configs = _configs()
    assert common_support_columns(configs) == ["bs", "coh"]
    assert common_support_name(configs) == "common:a,b"
    assert common_support_name(configs) != LAYER_SUPPORT


def test_common_support_drops_nan_and_rows_missing_in_one_dataset(tmp_path: Path) -> None:
    configs = _configs()
    samples = load_layer_samples(_store(tmp_path), configs)
    assert "coh_mean5" not in samples.columns
    assert "n_obs" not in samples.columns
    support = common_support(samples, configs)
    # 12 rows; the second tile lacks b, and row 0 of the first tile has no coh
    assert len(samples) == 12
    assert support.sum() == 5
    kept = samples[support]
    assert set(kept["tile"].astype(str)) == {"32TMS55"}
    assert 0 not in set(kept["row"])


def test_separability_tables_share_the_common_support(tmp_path: Path) -> None:
    configs = _configs()
    samples = load_layer_samples(_store(tmp_path), configs)
    tables = separability_tables(samples, configs)
    assert set(tables) == {"a", "b"}
    pooled = pd.concat(t[t.group_by == "all"] for t in tables.values())
    assert set(pooled.feature) == {"coh", "bs"}
    assert (pooled.support == "common:a,b").all()
    assert set(pooled.n_glacier + pooled.n_other) == {5}

    own = separability_tables(samples, configs, per_layer_support=True)
    own_pooled = pd.concat(t[t.group_by == "all"] for t in own.values())
    assert (own_pooled.support == LAYER_SUPPORT).all()
    scored = dict(zip(own_pooled.feature, own_pooled.n_glacier + own_pooled.n_other, strict=True))
    assert scored == {"coh": 10, "bs": 6}


def test_separability_tables_refuse_a_single_dataset_on_the_common_support(
    tmp_path: Path,
) -> None:
    configs = _configs()
    samples = load_layer_samples(_store(tmp_path), configs[1:])
    with pytest.raises(ValueError, match="every compared dataset"):
        separability_tables(samples, configs[1:])
    alone = separability_tables(samples, configs[1:], per_layer_support=True)
    assert (alone["a"].support == LAYER_SUPPORT).all()
    with pytest.raises(ValueError, match="twice"):
        separability_tables(samples, [configs[1], configs[1]])
