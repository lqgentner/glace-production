"""The compositing window is a committed input, not something a run computes.

A window may come from the windows file the config names (the production path:
the config stays year-agnostic and ``year`` is an override) or inline in the config
(a saved run configuration being rerun, or an experiment). When both are
present they must agree: a saved config that no longer matches the committed
table means the window was recomputed since that run, which is a decision to
make rather than something to resolve silently in either direction.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from pydantic import ValidationError
import pytest
from s1bursts import SearchQuery, TimeRange
import yaml

from glace_production.composites import (
    CompositeConfig,
    _BaseConfig,
    default_config_dir,
    resolve_time_range,
)

WINDOW: dict[str, date] = {"start": date(2023, 7, 18), "end": date(2023, 10, 15)}


def _windows_file(tmp_path: Path, window: str = "switzerland") -> Path:
    path = tmp_path / "windows" / f"{window}.yml"
    path.parent.mkdir()
    rows = {2023: {**WINDOW, "window_days": 90, "computed": date(2026, 9, 7)}}
    path.write_text(yaml.safe_dump(rows, sort_keys=False))
    return path


def test_from_the_windows_file_when_the_config_has_none(tmp_path: Path) -> None:
    _windows_file(tmp_path)
    resolved = resolve_time_range(None, window="switzerland", year=2023, config_dir=tmp_path)
    assert resolved == TimeRange(start=WINDOW["start"], end=WINDOW["end"])


def test_inline_alone_is_used(tmp_path: Path) -> None:
    inline = TimeRange(start=date(2023, 7, 1), end=date(2023, 9, 30))
    resolved = resolve_time_range(inline, window="nowhere", year=2023, config_dir=tmp_path)
    assert resolved == inline


def test_a_saved_config_is_checked_against_the_repository_table(tmp_path: Path) -> None:
    """A saved config.yaml lives beside the archive, not under config/.

    The committed table still applies to it.
    """
    committed = yaml.safe_load(
        (default_config_dir() / "windows" / "alps-total.yml").read_text(),
    )[2023]
    agreeing = TimeRange(start=committed["start"], end=committed["end"])
    assert (
        resolve_time_range(agreeing, window="alps-total", year=2023, config_dir=tmp_path)
        == agreeing
    )
    with pytest.raises(ValueError, match="disagrees"):
        resolve_time_range(
            TimeRange(start=date(2023, 7, 1), end=date(2023, 9, 30)),
            window="alps-total",
            year=2023,
            config_dir=tmp_path,
        )


def test_inline_that_agrees_with_the_file_is_fine(tmp_path: Path) -> None:
    _windows_file(tmp_path)
    inline = TimeRange(start=WINDOW["start"], end=WINDOW["end"])
    assert (
        resolve_time_range(inline, window="switzerland", year=2023, config_dir=tmp_path) == inline
    )


def test_inline_that_disagrees_with_the_file_is_refused(tmp_path: Path) -> None:
    _windows_file(tmp_path)
    inline = TimeRange(start=date(2023, 7, 1), end=date(2023, 9, 30))
    with pytest.raises(ValueError, match="disagrees"):
        resolve_time_range(inline, window="switzerland", year=2023, config_dir=tmp_path)


def test_neither_names_the_script(tmp_path: Path) -> None:
    _windows_file(tmp_path)  # has 2023, not 2022
    with pytest.raises(ValueError, match="compute-snow-window") as excinfo:
        resolve_time_range(None, window="switzerland", year=2022, config_dir=tmp_path)
    assert "2022" in str(excinfo.value)


def test_a_missing_windows_file_names_the_script_too(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="compute-snow-window"):
        resolve_time_range(None, window="nowhere", year=2019, config_dir=tmp_path)


def _config_file(tmp_path: Path, **fields: str) -> Path:
    path = tmp_path / "run.yml"
    path.write_text(yaml.safe_dump({"dirs": {"base_dir": "s1"}, **fields}))
    return path


def test_the_window_is_read_from_the_file_the_config_names(tmp_path: Path) -> None:
    """A smaller scope composes over the window of a larger one, not its own."""
    _windows_file(tmp_path, "alps-total")
    path = _config_file(tmp_path, scope="switzerland", window="alps-total")
    config = CompositeConfig.from_yaml(path, overrides=["year=2023"])
    assert config.scope == "switzerland"
    assert config.time_range == TimeRange(start=WINDOW["start"], end=WINDOW["end"])


def test_the_window_defaults_to_the_scope(tmp_path: Path) -> None:
    _windows_file(tmp_path, "alps-total")
    path = _config_file(tmp_path, scope="alps-total")
    config = CompositeConfig.from_yaml(path, overrides=["year=2023"])
    assert config.window == "alps-total"
    assert config.time_range == TimeRange(start=WINDOW["start"], end=WINDOW["end"])


def test_the_repository_configs_share_one_window() -> None:
    """The per-pass run composes over the Alpine window, so its days match the LRW run's."""
    windows = {
        path.name: CompositeConfig.from_yaml(
            path, overrides=["year=2023"], resolve_window=False
        ).window
        for path in default_config_dir().glob("*.yml")
    }
    assert set(windows) == {"lrw-alps.yml", "pass-alps.yml"}
    assert set(windows.values()) == {"alps-total"}


def test_end_before_start_is_rejected() -> None:
    with pytest.raises(ValueError, match="end"):
        TimeRange(start=date(2023, 10, 15), end=date(2023, 7, 18))


def test_a_reversed_window_in_a_config_is_rejected() -> None:
    """Pydantic builds the window from its fields without the constructor.

    The check must still run.
    """
    with pytest.raises(ValidationError, match="before its start"):
        _BaseConfig.model_validate(
            {
                "scope": "alps",
                "year": 2023,
                "dirs": {"base_dir": "s1"},
                "time_range": {"start": "2023-10-15", "end": "2023-07-18"},
            }
        )


def test_the_window_is_the_one_object_the_search_and_the_composite_take() -> None:
    """One window in the config reaches SearchQuery and compose_* as the same days."""
    window = TimeRange(start=WINDOW["start"], end=WINDOW["end"])
    # s1bursts 0.8 keeps the TimeRange itself on the query, inclusive of its last day.
    assert SearchQuery(time_range=window).time_range == window
    assert pd.Timestamp("2023-10-15T17:12:44", tz="UTC") in window
