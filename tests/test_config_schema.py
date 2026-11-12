"""The composite configs against the s1bursts 0.8 schema."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError
import pytest
from s1bursts import RTCConfig
import yaml

from glace_production.composites import CompositeConfig

CONFIG_DIR = Path(__file__).parents[1] / "config"
REPOSITORY_CONFIGS = sorted(CONFIG_DIR.glob("*.yml"))


@pytest.mark.parametrize("path", REPOSITORY_CONFIGS, ids=lambda p: p.name)
def test_repository_configs_load(path: Path) -> None:
    config = CompositeConfig.from_yaml(path, ["year=2025"])
    assert isinstance(config.backscatter, RTCConfig)


@pytest.mark.parametrize(
    ("override", "field"),
    [
        ("rtc_decibel=false", "rtc_decibel"),
        ("cqm_decibel=false", "cqm_decibel"),
        ("max_items=1000", "max_items"),
        ("backscatter.flatten_terrain=true", "flatten_terrain"),
        ("coherence.square_pixel=false", "square_pixel"),
    ],
)
def test_old_schema_fields_are_refused(override: str, field: str) -> None:
    with pytest.raises(ValidationError) as excinfo:
        CompositeConfig.from_yaml(
            CONFIG_DIR / "lrw-alps.yml", ["year=2025", override], resolve_window=False
        )
    message = str(excinfo.value)
    assert field in message
    assert "s1bursts 0.8" in message


def test_every_setting_is_explicit() -> None:
    raw = yaml.safe_load((CONFIG_DIR / "lrw-alps.yml").read_text())
    assert {"align_to_standard_grid", "correct_coherence_bias", "remove_local_phase_ramp"} <= set(
        raw["coherence"]
    )
    assert "align_to_standard_grid" in raw["backscatter"]


def test_the_composites_go_to_s1_comp() -> None:
    raw = yaml.safe_load((CONFIG_DIR / "lrw-alps.yml").read_text())
    assert raw["dirs"]["comp_dir"] == "alps-total/{year}/s1-comp"
