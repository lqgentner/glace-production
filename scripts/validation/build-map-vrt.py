r"""Mosaic the prediction maps of one (feature set, model) into a VRT.

The maps are one COG per label (tile, year). A tile mapped in two acquisition
years has two maps whose valid pixels are complementary: each year's map holds
255 (nodata) where the other year's glaciers lie. Sources are therefore added
in ascending year order and every map contributes where it has data, which
composes the SGI-matched mosaic. ``--year`` restricts the mosaic to one year.
Sources are referenced relative to the VRT, so a local copy of the directory
tree reads without S3 credentials.

``--classified`` mosaics the ``CLS_*`` class maps instead of the ``PRED_*``
probability maps.

Usage:
    pixi run python scripts/validation/build-map-vrt.py config/validation/runs/ablation.yml \
        --feature-set all_ctx --model rf [--classified] [--year 2023] [--out URI] [--store URI]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import re

from glace_production.config import settings
from glace_production.validation.config import RunConfig
from glace_production.validation.store import ValidationStore
from glace_production.validation.vrt import build_relative_vrt

logger = logging.getLogger(Path(__file__).stem)

_MAP_RE = re.compile(r"^(?P<prefix>PRED|CLS)_(?P<tile>\d{2}[A-Z]{3}\d{2})_(?P<year>\d{4})\.tif$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--feature-set", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--classified", action="store_true", help="mosaic the CLS_* class maps")
    parser.add_argument("--year", type=int, help="keep only this acquisition year's maps")
    parser.add_argument("--out", help="output VRT (default: beside the maps)")
    parser.add_argument("--store")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run = RunConfig.from_yaml(args.config)
    store = (
        ValidationStore(settings.resolve_path(args.store))
        if args.store
        else ValidationStore.default()
    )
    maps_dir = store.maps_dir(run.name, args.feature_set, args.model)
    prefix = "CLS" if args.classified else "PRED"
    files = []
    for path in maps_dir.glob(f"{prefix}_*.tif"):
        match = _MAP_RE.match(path.name)
        if match and (args.year is None or int(match["year"]) == args.year):
            files.append((int(match["year"]), match["tile"], path))
    if not files:
        msg = f"no prediction maps under {maps_dir}"
        raise FileNotFoundError(msg)
    files.sort()

    suffix = f"_{args.year}" if args.year is not None else ""
    out = (
        settings.resolve_path(args.out)
        if args.out
        else maps_dir / f"{prefix}_{args.feature_set}_{args.model}{suffix}.vrt"
    )
    build_relative_vrt([path for _, _, path in files], out)
    logger.info("%s: %d maps over %d tiles", out, len(files), len({tile for _, tile, _ in files}))


if __name__ == "__main__":
    main()
