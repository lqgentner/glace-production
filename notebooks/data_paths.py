"""Data directories of the paper notebooks, read from the run configs.

The notebooks take every composite, product and SLC directory from the ``dirs:`` block
of the run config that wrote it, resolved against ``GLACIERS_DATA_DIR`` like the run
itself does, so a notebook follows the configs and never names a directory of its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import yaml

from glace_production import config
from glace_production.composites import CompositeConfig, default_config_dir

if TYPE_CHECKING:
    from pathlib import Path

    from upath import UPath

LRW_CONFIG = default_config_dir() / "lrw-alps.yml"
PASS_CONFIG = default_config_dir() / "pass-alps.yml"


@dataclass(frozen=True)
class RunDirs:
    """Where one year's run keeps its SLCs, burst products and composites."""

    slc_dir: Path | UPath
    prod_dir: Path | UPath
    comp_dir: Path | UPath


def run_config(year: int, config_path: Path = LRW_CONFIG) -> CompositeConfig:
    """Return the run config of one year, with its compositing window resolved."""
    return CompositeConfig.from_yaml(config_path, [f"year={year}"])


def run_dirs(year: int, config_path: Path = LRW_CONFIG) -> RunDirs:
    """Return the SLC, product and composite directories of one year's run.

    The directories are only resolved, never created. A directory the config leaves
    out sits under ``base_dir`` with the name s1bursts gives it (``s1-slc``,
    ``s1-prod``, ``s1-comp``).
    """
    run = run_config(year, config_path)

    def resolve(template: str) -> Path | UPath:
        return config.settings.resolve_data_path(run.expand_template(template))

    base = resolve(run.dirs.base_dir)
    d = run.dirs
    return RunDirs(
        slc_dir=resolve(d.slc_dir) if d.slc_dir else base / "s1-slc",
        prod_dir=resolve(d.prod_dir) if d.prod_dir else base / "s1-prod",
        comp_dir=resolve(d.comp_dir) if d.comp_dir else base / "s1-comp",
    )


def run_scope(config_path: Path = LRW_CONFIG) -> str:
    """Return the scope a run config composes, e.g. ``alps-total``."""
    return yaml.safe_load(config_path.read_text())["scope"]
