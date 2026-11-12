#!/usr/bin/env python3
r"""Compute one scope-year's snow-free compositing window and commit it as input.

The window is the lowest-snow run of consecutive days in the year over the
scope's AOI, from MODIS Terra/Aqua NDSI. This script writes the row into
``config/windows/{scope}.yml``, and ``process-composites.py`` reads it back for
every config whose ``window`` names that scope, whatever the config's own scope.
The config given here must therefore have ``window`` equal to ``scope``.
An existing row is left alone unless ``--force`` is given.

Examples
--------
::

    pixi run python scripts/s1-composites/compute-snow-window.py \
        config/lrw-alps.yml year=2022
    pixi run python scripts/s1-composites/compute-snow-window.py \
        config/lrw-alps.yml year=2019 --plot
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import logging
from pathlib import Path

import pandas as pd
import shapely
import yaml

from glace_production.composites import CompositeConfig, TimeRange, resolve_aoi, windows_file
from glace_production.snow import find_modis_snowfree_window, year_is_complete

logger = logging.getLogger(Path(__file__).stem)

DEFAULT_WINDOW_DAYS = 90


def compute_window(config: CompositeConfig, *, window_days: int, plot: Path | None) -> TimeRange:
    """Find the lowest-snow ``window_days`` window of ``config.year`` over the scope's AOI."""
    aoi = resolve_aoi(
        config.scope,
        simplify_m=config.simplify_m,
        glacier_buffer_m=config.glacier_buffer_m,
    )
    start = pd.Timestamp(year=config.year, month=1, day=1)
    stop = start + pd.offsets.YearBegin()
    first, last = find_modis_snowfree_window(
        shapely.union_all(aoi),
        start,
        stop,
        window_days=window_days,
        plot_file=plot,
    )
    return TimeRange(start=first.date(), end=last.date())


def write_row(path: Path, year: int, window: TimeRange, *, window_days: int) -> None:
    """Add or replace ``year`` in the windows file, keeping the rows in year order."""
    rows: dict[int, dict] = yaml.safe_load(path.read_text()) or {} if path.exists() else {}
    rows[year] = {
        "start": window.start,
        "end": window.end,
        "window_days": window_days,
        "computed": datetime.now(UTC).date(),
    }
    header = (
        f"# Snow-free compositing windows for scope `{path.stem}`, one row per year.\n"
        "# Written by scripts/s1-composites/compute-snow-window.py from MODIS NDSI over the\n"
        "# scope's AOI (`resolve_aoi`). Review, then commit: a run whose config names this "
        "file\n"
        "# as its `window` reads its window from here and refuses to start without one.\n"
        "# `window_days` is the length that was searched for.\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(dict(sorted(rows.items())), sort_keys=False, default_flow_style=False)
    path.write_text(header + body)


def main() -> None:
    """Compute and record one scope-year's window."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", type=Path, help="the composite config of the scope")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf overrides; `year=YYYY` is required",
    )
    parser.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS)
    parser.add_argument("--force", action="store_true", help="replace an existing row")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="save the NDSI series plot beside the row",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = CompositeConfig.from_yaml(
        args.config,
        overrides=args.overrides or None,
        resolve_window=False,
    )
    if config.window != config.scope:
        msg = (
            f"{args.config} reads the window {config.window!r} but covers the scope "
            f"{config.scope!r}. A window is computed over the AOI of the scope it is named "
            f"after; pass a config whose scope is {config.window!r}."
        )
        raise SystemExit(msg)
    path = windows_file(args.config.resolve().parent, config.window)
    existing = (yaml.safe_load(path.read_text()) or {}).get(config.year) if path.exists() else None
    if existing is not None and not args.force:
        msg = (
            f"{path} already has a window for {config.year}: "
            f"{existing['start']}..{existing['end']}. Pass --force to replace it; "
            "a shifted window renames every composite of the year."
        )
        raise SystemExit(msg)

    if not year_is_complete(config.year):
        logger.warning(
            "MODIS coverage of %d is not final: near-real-time values are revised for a while, "
            "so this window may move by a few days if recomputed later. Commit it knowing that, "
            "or wait until the year has settled.",
            config.year,
        )
    plot = path.with_name(f"{path.stem}-{config.year}-ndsi.png") if args.plot else None
    window = compute_window(config, window_days=args.window_days, plot=plot)
    write_row(path, config.year, window, window_days=args.window_days)
    logger.info("%s %d: %s to %s -> %s", config.scope, config.year, window.start, window.end, path)


if __name__ == "__main__":
    main()
