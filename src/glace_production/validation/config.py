"""Configuration models of the validation stages.

Three YAML-backed configs drive the pipeline: which scope rows are selected as
sample locations (:class:`LocationsConfig`), which raster layers a dataset variant
provides (:class:`DatasetConfig`) and what a cross-validation run trains
(:class:`RunConfig`). Loading mirrors :class:`glace_production.composites.CompositeConfig`:
OmegaConf reads the file, dot-list overrides are merged, pydantic validates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Literal, Self

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import yaml

from glace_production.validation.bands import RING_M

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

PIXEL_SIZE_M = 10.0
"""Grid spacing of every tile raster, in metres."""

type LayerKind = Literal["coherence", "backscatter", "count"]
type ModelName = Literal["rf"]
type DerivedStat = Literal["mean", "std"]

HIST_RANGES: dict[str, tuple[float, float]] = {
    "coherence": (0.0, 1.0),
    "backscatter": (-35.0, 10.0),
}
"""Value range of the JM histograms per measurement kind (backscatter in dB)."""


def feature_range(kind: str, stat: DerivedStat | None = None) -> tuple[float, float]:
    """Histogram range of a base layer (``stat=None``) or of a neighbourhood statistic.

    A window mean shares the layer's range; a window standard deviation is
    non-negative and bounded by half the layer's range.
    """
    lo, hi = HIST_RANGES[kind]
    if stat == "std":
        return (0.0, (hi - lo) / 2)
    return (lo, hi)


class _YamlConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_yaml(cls, path: Path, overrides: list[str] | None = None) -> Self:
        """Load a config from YAML with optional ``key=value`` dot-list overrides."""
        cfg = OmegaConf.load(path)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        return cls.model_validate(OmegaConf.to_container(cfg, resolve=True))

    def to_yaml(self) -> str:
        """Serialise the resolved config; the stored copy every stage compares against."""
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)


class LocationsConfig(_YamlConfig):
    """Which label masks the validation locations come from and how far the halo reaches."""

    name: str
    labels_dir: str
    inventory: Literal["SGI2016", "SGI2023"]
    year_column: str = "year_acq"
    years: list[int] = Field(min_length=1)
    pad_m: float = Field(1000.0, gt=0)
    """Halo read from the neighbouring tiles when rasterising the outlines, in metres."""
    dem: str

    @model_validator(mode="after")
    def _halo_beyond_ring(self) -> Self:
        if self.pad_m <= RING_M:
            msg = (
                f"pad_m={self.pad_m} must exceed the {RING_M} m ring; "
                "distances are censored at pad_m"
            )
            raise ValueError(msg)
        return self

    @property
    def pad_px(self) -> int:
        """Halo width in pixels; ``pad_m`` must be a whole number of pixels."""
        px = self.pad_m / PIXEL_SIZE_M
        if not math.isclose(px, round(px)):
            msg = f"pad_m={self.pad_m} is not a multiple of {PIXEL_SIZE_M} m"
            raise ValueError(msg)
        return round(px)


class DerivedSpec(_YamlConfig):
    """A neighbourhood statistic of a layer, computed over a square window at extraction."""

    stat: DerivedStat
    window: int = Field(5, ge=3)
    glacier_low: bool | None = None
    """A-priori direction of the statistic; ``None`` inherits the layer's."""

    @field_validator("window")
    @classmethod
    def _odd(cls, value: int) -> int:
        if value % 2 == 0:
            msg = f"window must be odd, got {value}"
            raise ValueError(msg)
        return value

    def column(self, layer: str) -> str:
        """Column name of the statistic: ``<layer>_<stat><window>``."""
        return f"{layer}_{self.stat}{self.window}"


class LayerSpec(_YamlConfig):
    """One raster layer of a dataset variant."""

    pattern: str
    kind: LayerKind
    decibel: bool = False
    glacier_low: bool = True
    derive: list[DerivedSpec] = []

    @field_validator("pattern")
    @classmethod
    def _needs_tile(cls, value: str) -> str:
        if "{tile}" not in value:
            msg = f"layer pattern {value!r} must contain '{{tile}}'"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _decibel_is_backscatter(self) -> Self:
        if self.decibel and self.kind != "backscatter":
            msg = "decibel applies to backscatter layers only"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _derive_measurements_only(self) -> Self:
        if self.derive and not self.is_measurement:
            msg = "derive applies to measurement layers only"
            raise ValueError(msg)
        keys = [(d.stat, d.window) for d in self.derive]
        if len(set(keys)) != len(keys):
            msg = f"duplicate derived statistics: {keys}"
            raise ValueError(msg)
        return self

    @property
    def is_measurement(self) -> bool:
        """Whether the layer takes part in separability and feature sets."""
        return self.kind != "count"


@dataclass(frozen=True)
class Feature:
    """A feature column: a measurement layer itself or one of its neighbourhood statistics."""

    layer: str
    spec: LayerSpec
    derived: DerivedSpec | None = None

    @property
    def kind(self) -> LayerKind:
        """Measurement kind of the underlying layer."""
        return self.spec.kind

    @property
    def glacier_low(self) -> bool:
        """Whether glacier pixels are expected to have the lower value."""
        if self.derived is not None and self.derived.glacier_low is not None:
            return self.derived.glacier_low
        return self.spec.glacier_low

    @property
    def value_range(self) -> tuple[float, float]:
        """Fixed histogram range of the column."""
        return feature_range(self.kind, None if self.derived is None else self.derived.stat)


class DatasetConfig(_YamlConfig):
    """A tiled dataset variant: where its tiles live and which layers it has."""

    name: str
    comp_dir: str
    layers: dict[str, LayerSpec] = Field(min_length=1)

    @property
    def measurement_layers(self) -> dict[str, LayerSpec]:
        """Layers usable as features (coherence and backscatter)."""
        return {k: v for k, v in self.layers.items() if v.is_measurement}

    @property
    def features(self) -> dict[str, Feature]:
        """Every feature column the extraction writes: layers, then their statistics."""
        out: dict[str, Feature] = {}
        for name, spec in self.measurement_layers.items():
            out[name] = Feature(name, spec)
            for derived in spec.derive:
                out[derived.column(name)] = Feature(name, spec, derived)
        return out

    @model_validator(mode="after")
    def _columns_are_distinct(self) -> Self:
        columns = [d.column(k) for k, v in self.layers.items() for d in v.derive]
        clashes = sorted(set(columns) & set(self.layers)) + sorted(
            c for c in set(columns) if columns.count(c) > 1
        )
        if clashes:
            msg = f"derived columns clash with layers: {clashes}"
            raise ValueError(msg)
        return self


class RunConfig(_YamlConfig):
    """A cross-validation run: datasets, feature sets, models, budget."""

    name: str
    datasets: list[str] = Field(min_length=1)
    folds: str
    feature_sets: dict[str, list[str]] = Field(min_length=1)
    models: list[ModelName] = Field(min_length=1)
    train_rows: int = Field(2_000_000, gt=0)
    n_seeds: int = Field(1, ge=1)
    seed: int = 0
    save_models: bool = False

    @model_validator(mode="after")
    def _check_sets(self) -> Self:
        for name, columns in self.feature_sets.items():
            if not columns or len(set(columns)) != len(columns):
                msg = f"feature set {name!r} must list distinct, non-empty columns"
                raise ValueError(msg)
        return self

    @property
    def columns(self) -> list[str]:
        """Every feature column used anywhere in the run, sorted."""
        return sorted({c for cols in self.feature_sets.values() for c in cols})

    def validate_columns(self, datasets: Mapping[str, DatasetConfig]) -> None:
        """Raise if a listed dataset has no config, or a column is not one of its features."""
        known: set[str] = set()
        for name in self.datasets:
            if name not in datasets:
                msg = f"dataset {name!r} has no config"
                raise ValueError(msg)
            known.update(datasets[name].features)
        for column in self.columns:
            if column not in known:
                msg = f"feature column {column!r} is not a feature of {self.datasets}"
                raise ValueError(msg)


def dataset_config_path(run_path: Path, name: str) -> Path:
    """Where a run's dataset config lives: ``<config root>/datasets/<name>.yml``."""
    return run_path.resolve().parent.parent / "datasets" / f"{name}.yml"
