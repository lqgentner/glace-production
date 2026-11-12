r"""Mosaic the rasterised inventory masks into VRTs, per year and combined.

One VRT per acquisition year holds that year's masks (1 glacier, 0 background,
255 ignore). The combined VRT adds the years in ascending order; because every
mask marks the other years' glaciers as 255, the later year falls through to
the earlier one there, and the union is the SGI-matched mosaic that pairs with
the prediction-map VRT of ``build-map-vrt.py``. The VRTs are written beside the
masks and reference them by filename, so a local copy of the directory reads
without S3 credentials; the masks themselves are never touched.

Usage:
    pixi run python scripts/validation/build-label-vrt.py config/validation/locations-ch.yml \
        [--out-dir URI]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from glace_production.config import settings
from glace_production.validation.config import LocationsConfig
from glace_production.validation.locations import parse_label_name
from glace_production.validation.vrt import build_relative_vrt

logger = logging.getLogger(Path(__file__).stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--out-dir", help="output directory (default: the label directory)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = LocationsConfig.from_yaml(args.config)
    labels_dir = settings.resolve_data_path(config.labels_dir)
    out_dir = settings.resolve_path(args.out_dir) if args.out_dir else labels_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    masks = []
    for path in labels_dir.glob("LABELS_*.tif"):
        parsed = parse_label_name(path.name)
        if parsed and parsed[0] == config.inventory and parsed[1] in config.years:
            masks.append((parsed[1], parsed[2], path))
    if not masks:
        msg = f"no {config.inventory} masks for {config.years} under {labels_dir}"
        raise FileNotFoundError(msg)
    masks.sort()

    for year in sorted({year for year, _, _ in masks}):
        files = [path for y, _, path in masks if y == year]
        out = out_dir / f"LABELS_{config.inventory}_{year}.vrt"
        build_relative_vrt(files, out)
        logger.info("%s: %d masks", out, len(files))
    out = out_dir / f"LABELS_{config.inventory}.vrt"
    build_relative_vrt([path for _, _, path in masks], out)
    logger.info("%s: %d masks over %d tiles", out, len(masks), len({t for _, t, _ in masks}))


if __name__ == "__main__":
    main()
