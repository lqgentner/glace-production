"""The directory settings: only the data directory is required."""

from __future__ import annotations

from pathlib import Path

import pytest

from glace_production.config import Settings


def test_only_the_data_dir_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GLACIERS_DATA_DIR", "/data")
    for name in ("REPORTS_DIR", "FIGURES_DIR", "MODELS_DIR"):
        monkeypatch.delenv(f"GLACIERS_{name}", raising=False)
    settings = Settings()
    assert settings.data_dir == Path("/data")
    with pytest.raises(ValueError, match="GLACIERS_MODELS_DIR is not set"):
        _ = settings.models_dir


def test_a_set_optional_dir_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GLACIERS_MODELS_DIR", "/models")
    assert Settings().models_dir == Path("/models")


def test_config_paths_resolve_under_the_data_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GLACIERS_DATA_DIR", "/data")
    settings = Settings()
    assert settings.resolve_data_path("dem/glo30.tif") == Path("/data/dem/glo30.tif")
    assert settings.resolve_data_path("/elsewhere/glo30.tif") == Path("/elsewhere/glo30.tif")
