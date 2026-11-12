"""Data paths of the paper notebooks, read from the run configs (``data_paths.py``)."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import TYPE_CHECKING

import pytest

from glace_production import config

if TYPE_CHECKING:
    from collections.abc import Callable

sys.path.insert(0, str(Path(__file__).parents[1] / "notebooks"))

from data_paths import PASS_CONFIG, run_dirs, run_scope


@pytest.fixture
def use_data_dir(monkeypatch: pytest.MonkeyPatch) -> Callable[[str], None]:
    """Point the settings at a data directory, wherever a module bound them by name."""

    def use(data_dir: str) -> None:
        monkeypatch.setenv("GLACIERS_DATA_DIR", data_dir)
        settings = config.Settings()
        monkeypatch.setattr("glace_production.config.settings", settings)
        monkeypatch.setattr("glace_production.composites.settings", settings)

    return use


def test_run_dirs_lrw(tmp_path: Path, use_data_dir: Callable[[str], None]) -> None:
    use_data_dir(str(tmp_path / "data"))
    dirs = run_dirs(2023)

    assert dirs.comp_dir == tmp_path / "data/alps-total/2023/s1-comp"
    assert dirs.prod_dir == tmp_path / "data/alps-total/2023/s1-prod"
    assert dirs.slc_dir == tmp_path / "data/alps-total/2023/s1-slc"


def test_run_dirs_pass(tmp_path: Path, use_data_dir: Callable[[str], None]) -> None:
    use_data_dir(str(tmp_path / "data"))

    assert run_dirs(2023, PASS_CONFIG).comp_dir == tmp_path / "data/alps-total/2023/s1-comp-pass"


def test_run_dirs_create_nothing(tmp_path: Path, use_data_dir: Callable[[str], None]) -> None:
    """A notebook only reads: resolving a run's directories must not make them."""
    use_data_dir(str(tmp_path / "data"))
    run_dirs(2023)

    assert not (tmp_path / "data").exists()


def test_run_dirs_follow_an_object_store_data_dir(
    monkeypatch: pytest.MonkeyPatch, use_data_dir: Callable[[str], None]
) -> None:
    monkeypatch.setenv("GLACIERS_S3_PROFILE", "placeholder")  # never contacted
    use_data_dir("s3://bucket/data")

    assert str(run_dirs(2023).comp_dir) == "s3://bucket/data/alps-total/2023/s1-comp"


def test_run_scope() -> None:
    assert run_scope() == "alps-total"
