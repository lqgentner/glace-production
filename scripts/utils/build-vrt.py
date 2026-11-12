"""Mosaic MGRS tiles matching a glob into a relocatable VRT.

Works for any single-band tile set on the MGRS grid: composites, label masks or
prediction maps. Sources are added in filename order, so where a tile has one
file per acquisition year, the later year falls through to the earlier one
where it is nodata. Same-store sources are referenced relative to the VRT, so
a local copy of the directory tree reads without S3 credentials.

Usage:
    pixi run python scripts/utils/build-vrt.py alps-total/2023/s1-comp \
        --pattern "S1_COH12_LRW_{tile}_VV_*.tif" [--out URI]

``{tile}`` in the pattern stands for one MGRS tile code, so the count layers
(``..._QA-NUM_{tile}_...``) beside the composites are not swept in by a bare
``*``. ``dir`` is relative to ``settings.data_dir`` unless it is an absolute path or
URI. Without ``--out`` the VRT is written beside the tiles, named after the
pattern with its wildcards removed (``S1_COH12_LRW_VV.vrt``).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from glace_production.config import settings
from glace_production.validation.vrt import (
    build_relative_vrt,
    default_vrt_name,
    expand_tile_pattern,
)

logger = logging.getLogger(Path(__file__).stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dir", help="tile directory: local path, s3:// URI or relative to data_dir"
    )
    parser.add_argument(
        "--pattern", required=True, help="glob of the tiles to mosaic; {tile} = one MGRS code"
    )
    parser.add_argument(
        "--out", help="output VRT (default: beside the tiles, named after the pattern)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tile_dir = settings.resolve_data_path(args.dir)
    files = list(tile_dir.glob(expand_tile_pattern(args.pattern)))
    if not files:
        msg = f"no tiles matching {args.pattern!r} under {tile_dir}"
        raise FileNotFoundError(msg)
    out = (
        settings.resolve_path(args.out) if args.out else tile_dir / default_vrt_name(args.pattern)
    )
    n = build_relative_vrt(files, out)
    logger.info("%s: %d tiles", out, n)


if __name__ == "__main__":
    main()
