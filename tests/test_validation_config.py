"""Validation configs: YAML loading, overrides and the constraints they enforce."""

from __future__ import annotations

from pathlib import Path
import re

from pydantic import ValidationError
import pytest

from glace_production.validation.config import (
    DatasetConfig,
    DerivedSpec,
    LayerSpec,
    LocationsConfig,
    RunConfig,
    dataset_config_path,
)

DATASET_YAML = """
name: glace-ch
comp_dir: alps-total/{year}/s1-comp-ch
layers:
  coh_vv: {pattern: "S1_COH12_LRW_{tile}_VV_*.tif", kind: coherence}
  rtc_vv:
    pattern: "S1_RTC_LRW_{tile}_VV_*.tif"
    kind: backscatter
    decibel: false
    derive: [{stat: mean, window: 5}, {stat: std, window: 5}]
  n_obs_vv: {pattern: "S1_COH12_LRW_QA-NUM_{tile}_VV_*.tif", kind: count}
"""

RUN_YAML = """
name: ablation
datasets: [glace-ch]
folds: config/validation/folds-ch.csv
feature_sets:
  coh_vv: [coh_vv]
  sar_vv: [rtc_vv, coh_vv]
models: [rf]
"""


def _dataset(tmp_path: Path) -> DatasetConfig:
    path = tmp_path / "glace-ch.yml"
    path.write_text(DATASET_YAML)
    return DatasetConfig.from_yaml(path)


def test_dataset_layers_and_measurements(tmp_path: Path) -> None:
    config = _dataset(tmp_path)
    assert set(config.layers) == {"coh_vv", "rtc_vv", "n_obs_vv"}
    assert set(config.measurement_layers) == {"coh_vv", "rtc_vv"}
    assert config.layers["coh_vv"].glacier_low is True
    assert list(config.features) == ["coh_vv", "rtc_vv", "rtc_vv_mean5", "rtc_vv_std5"]
    std = config.features["rtc_vv_std5"]
    assert std.layer == "rtc_vv"
    assert std.kind == "backscatter"
    assert std.glacier_low is True  # inherited from the layer
    assert std.value_range == (0.0, 22.5)
    assert config.features["rtc_vv_mean5"].value_range == (-35.0, 10.0)


def test_derived_specs_are_validated() -> None:
    with pytest.raises(ValidationError, match="odd"):
        DerivedSpec(stat="mean", window=4)
    with pytest.raises(ValidationError, match="measurement"):
        LayerSpec(pattern="{tile}", kind="count", derive=[{"stat": "mean"}])
    with pytest.raises(ValidationError, match="duplicate"):
        LayerSpec(pattern="{tile}", kind="coherence", derive=[{"stat": "mean"}, {"stat": "mean"}])
    with pytest.raises(ValidationError, match="clash"):
        DatasetConfig(
            name="d",
            comp_dir="x",
            layers={
                "a": LayerSpec(pattern="{tile}", kind="coherence", derive=[{"stat": "std"}]),
                "a_std5": LayerSpec(pattern="{tile}", kind="coherence"),
            },
        )


def test_layer_pattern_requires_tile_placeholder() -> None:
    with pytest.raises(ValidationError, match=re.escape("{tile}")):
        LayerSpec(pattern="S1_COH12_LRW_VV.tif", kind="coherence")


def test_decibel_only_for_backscatter() -> None:
    with pytest.raises(ValidationError, match="decibel"):
        LayerSpec(pattern="x_{tile}.tif", kind="coherence", decibel=True)


def test_run_columns_and_validation(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    run_path = tmp_path / "runs" / "ablation.yml"
    run_path.parent.mkdir()
    run_path.write_text(RUN_YAML)
    run = RunConfig.from_yaml(run_path)
    assert run.columns == ["coh_vv", "rtc_vv"]
    run.validate_columns({"glace-ch": dataset})
    derived = run.model_copy(update={"feature_sets": {"ctx": ["coh_vv", "rtc_vv_std5"]}})
    derived.validate_columns({"glace-ch": dataset})
    assert dataset_config_path(run_path, "glace-ch") == tmp_path / "datasets" / "glace-ch.yml"


def test_run_rejects_unknown_and_count_columns(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    run = RunConfig(
        name="r",
        datasets=["glace-ch"],
        folds="f.csv",
        feature_sets={"bad": ["n_obs_vv"]},
        models=["rf"],
    )
    with pytest.raises(ValueError, match="n_obs_vv"):
        run.validate_columns({"glace-ch": dataset})
    run = RunConfig(
        name="r",
        datasets=["glace-ch"],
        folds="f.csv",
        feature_sets={"bad": ["nope"]},
        models=["rf"],
    )
    with pytest.raises(ValueError, match="nope"):
        run.validate_columns({"glace-ch": dataset})


def test_run_rejects_dataset_missing_from_mapping(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    run = RunConfig(
        name="r",
        datasets=["missing"],
        folds="f.csv",
        feature_sets={"a": ["coh_vv"]},
        models=["rf"],
    )
    with pytest.raises(ValueError, match=re.escape("dataset 'missing' has no config")):
        run.validate_columns({"glace-ch": dataset})


def test_overrides_and_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "loc.yml"
    path.write_text(
        "name: ch\nlabels_dir: labels/sgi2023\ninventory: SGI2023\n"
        "years: [2021, 2022]\ndem: s3://b/dem.tif\n"
    )
    config = LocationsConfig.from_yaml(path, overrides=["pad_m=1100"])
    assert config.pad_m == 1100.0
    assert config.year_column == "year_acq"
    again = LocationsConfig.model_validate(__import__("yaml").safe_load(config.to_yaml()))
    assert again == config
