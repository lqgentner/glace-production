r"""Precompute a scope's per-tile RGI7 glacier fractions, once, against a pinned inventory.

Every tile item publishes ``glace:glacier_fraction`` and
``glace:glacier_fraction_buffered`` — the share of the tile covered by RGI7
outlines, plain and buffered by 250 m. They are what makes a glacier-weighted
sampler possible, so they stay.

They are computed **here, once per scope**, rather than inside the per-year
publish — and the reason is consistency, not speed. The fraction is
year-independent, but the publisher runs once per year, so recomputing it against
a freshly downloaded RGI7 could silently give one tile different values in 2015
and 2025: a published property drifting inside one dataset. Computing it once and
pinning the inventory version removes that entirely.

The table is published as a collection asset of the tiles collection, so the
lookup behind a published property is auditable and travels with the store.

Usage:
    pixi run python scripts/inventories/derive-glacier-fractions.py \
        --scope alps --perimeter-tiles 2210 --out OUT.parquet [--overwrite]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import time
from typing import Final

from freezebase.mgrs import MGRSGeoBox, MGRSGrid
import pandas as pd

from glace_production.config import settings
from glace_production.datasets import RGI7, SGI2023, AlpineConventionPerimeter
from glace_production.geo import MGRSInventory

logger = logging.getLogger(Path(__file__).stem)

BUFFER_M = 250.0
RGI_REGION: Final = "11"
RGI_VERSION = "RGI7-C-11"
"""Inventory edition the published fractions are pinned to.

Recorded in the table so a fraction can always be traced to the outlines it was
measured against, and so a later RGI release is a visible change rather than a
silent drift.
"""


def perimeter_tiles(scope_key: str) -> list[str]:
    """MGRS 10 km tile codes covering a scope's perimeter.

    The fractions follow the perimeter the dataset *claims*, exactly as the
    canonical extent does, so a tile that appears in a later year already has its
    fraction rather than triggering a recomputation.
    """
    if scope_key == "alps":
        frame = AlpineConventionPerimeter().get_data(download=True)
        return sorted(
            str(code) for code in MGRSGrid(frame.to_crs("EPSG:4326").geometry).mgrs_codes
        )
    if scope_key == "ch":
        # The Swiss run composites glacier tiles only, so the claim is the
        # buffered inventory rather than the national boundary.
        inventory = SGI2023().get_data(download=True)
        return sorted(
            str(code) for code in MGRSInventory(inventory, buffer_m=BUFFER_M).intersecting_tiles()
        )
    msg = f"Unknown scope {scope_key!r}"
    raise ValueError(msg)


def compute_fractions(tiles: list[str]) -> pd.DataFrame:
    """Plain and buffered RGI7 coverage of every tile, as the published table."""
    started = time.monotonic()
    rgi = RGI7(product="C", region=RGI_REGION).get_data(download=True)
    logger.info(
        "Loaded %d RGI7 region-%s outlines in %.1f s",
        len(rgi),
        RGI_REGION,
        time.monotonic() - started,
    )

    plain = MGRSInventory(rgi)
    buffered = MGRSInventory(rgi, buffer_m=BUFFER_M)
    rows = []
    started = time.monotonic()
    for index, code in enumerate(tiles, start=1):
        geobox = MGRSGeoBox.from_mgrs(code)
        rows.append(
            {
                "mgrs_code": code,
                "glacier_fraction": round(plain.coverage_fraction(geobox), 5),
                "glacier_fraction_buffered": round(buffered.coverage_fraction(geobox), 5),
                "rgi_version": RGI_VERSION,
                "buffer_m": BUFFER_M,
            },
        )
        if index % 250 == 0:
            logger.info("%d/%d tiles", index, len(tiles))
    logger.info(
        "Computed %d tiles in %.1f s (%.0f ms/tile)",
        len(tiles),
        time.monotonic() - started,
        (time.monotonic() - started) / len(tiles) * 1000,
    )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    """Command-line arguments of the glacier-fraction builder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", required=True, choices=["ch", "alps"])
    parser.add_argument(
        "--perimeter-tiles",
        type=int,
        default=None,
        help=(
            "tile count the scope's committed extent was derived from, for a "
            "completeness check; see glace_catalog.scopes. Passed rather than "
            "imported so this repository does not depend on the catalog's."
        ),
    )
    parser.add_argument("--out", required=True, help="local path or s3:// URI")
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing table at --out"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = settings.resolve_path(args.out)
    if out.exists() and not args.overwrite:
        # The catalog reads this table every year; a rerun must not change it unasked.
        msg = f"{out} already exists; pass --overwrite to replace it"
        raise SystemExit(msg)

    tiles = perimeter_tiles(args.scope)
    if args.perimeter_tiles is not None and len(tiles) != args.perimeter_tiles:
        logger.warning(
            "%s perimeter yielded %d tiles, but --perimeter-tiles says %d. The "
            "catalog's committed extent was derived from that count, so check "
            "which of the two moved before publishing.",
            args.scope,
            len(tiles),
            args.perimeter_tiles,
        )
    logger.info("%s: %d perimeter tiles", args.scope, len(tiles))

    frame = compute_fractions(tiles)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as handle:
        frame.to_parquet(handle, index=False)
    logger.info(
        "Wrote %s: %d tiles, %d with any glacier cover",
        out,
        len(frame),
        int((frame["glacier_fraction"] > 0).sum()),
    )


if __name__ == "__main__":
    main()
