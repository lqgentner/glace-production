"""Rasterize a glacier inventory onto MGRS 10-km tile grids as label COGs.

Selects the MGRS tiles intersecting the inventory (buffered by ``--buffer-m``,
default 250 m) and writes one uint8 mask COG per tile (1 = glacier,
0 = background, 255 = ignore). Year-split inventories are rasterized once per
acquisition year under the retreat rule, and sources with a domain have
everything outside it set to ignore; see ``glace_production.labels``.

Output naming: ``LABELS_{source}_{tile}.tif``, or
``LABELS_{source}_{year}_{tile}.tif`` for year-split inventories.

Usage:
    pixi run python scripts/inventories/rasterize-labels.py \
        --source SGI2023 --out-dir $GLACIERS_DATA_DIR/labels/sgi2023 [--overwrite]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from glace_production.config import settings
from glace_production.geo import MGRSInventory
from glace_production.labels import (
    DOMAINS,
    INVENTORIES,
    LABEL_SOURCES,
    split_years,
    write_label_masks,
)

logger = logging.getLogger(Path(__file__).stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        choices=sorted(INVENTORIES),
        type=str.upper,
        help="inventory to rasterize",
    )
    parser.add_argument("--out-dir", required=True, help="local path or s3:// URI")
    parser.add_argument(
        "--buffer-m",
        type=float,
        default=250.0,
        help="buffer applied to the outlines (in local UTM) for tile selection",
    )
    parser.add_argument(
        "--ignore-buffer-m",
        type=float,
        default=250.0,
        help="buffer around later years' outlines burned as ignore (year-split sources)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace masks already in --out-dir"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    source = LABEL_SOURCES[args.source]
    out_dir = settings.resolve_path(args.out_dir)
    existing = next(iter(out_dir.glob(f"LABELS_{source.key}*.tif")), None)
    if existing is not None and not args.overwrite:
        # The validation was built from these masks; a rerun must not change them unasked.
        parser.error(f"{out_dir} already holds masks ({existing.name}); pass --overwrite")
    gdf = INVENTORIES[source.key]().get_data(download=True)
    domain = MGRSInventory(DOMAINS[source.domain]()) if source.domain else None
    out_dir.mkdir(parents=True, exist_ok=True)

    if source.year_split:
        selections = split_years(
            gdf, source, buffer_m=args.buffer_m, ignore_buffer_m=args.ignore_buffer_m
        )
    else:
        selections = [MGRSInventory(gdf, buffer_m=args.buffer_m)]
    total = sum(
        write_label_masks(
            sel, out_dir, source, domain=domain, ignore_buffer_m=args.ignore_buffer_m
        )
        for sel in selections
    )
    logger.info("Wrote %d label masks to %s", total, out_dir)


if __name__ == "__main__":
    main()
