"""Export the glacier inventories as GeoJSON overlays for the glace-viewer repo.

Writes one trimmed, simplified ``.geojson`` per inventory plus an
``inventories.json`` index. Those are committed by the glace-viewer repo.

The inventories need this repository's dataset classes, so they are exported
here; only the page itself lives in glace-viewer.

Usage:
    pixi run python scripts/inventories/export-inventories.py
"""

from __future__ import annotations

import argparse
from functools import partial
import json
import logging
from pathlib import Path

from glace_production.datasets import RGI7, SGI2016, SGI2023, AustrianGI5, PaulEtAl2020
from glace_production.inventories import INVENTORY_OVERLAYS, export_inventory

logger = logging.getLogger(__name__)

INVENTORIES = {
    "agi5": AustrianGI5,
    "pauletal2020": PaulEtAl2020,
    "rgi7": partial(RGI7, region="11-01"),  # Alps
    "sgi2016": SGI2016,
    "sgi2023": SGI2023,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default="../glace-viewer/data",
        help="glace-viewer's data directory, where the overlays are committed",
    )
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        choices=sorted(INVENTORIES),
        help="restrict to this inventory (repeatable; default: all)",
    )
    parser.add_argument(
        "--simplify",
        type=float,
        default=10.0,
        help=(
            "Douglas-Peucker tolerance in metres (default: 10, finer than the "
            "26.5 m the rasters resolve at their maximum zoom)"
        ),
    )
    parser.add_argument("--precision", type=int, default=5, help="coordinate decimals")
    parser.add_argument(
        "--max-zoom",
        type=int,
        default=14,
        help="maximum zoom glace-viewer should tile to; the main lever on size",
    )
    parser.add_argument("--min-zoom", type=int, default=4, help="minimum zoom to tile to")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out_dir = Path(args.out_dir)
    index = []
    for key in args.sources or sorted(INVENTORIES):
        spec = INVENTORY_OVERLAYS[key]
        dataset = INVENTORIES[key]()
        meta = dataset.metadata
        frame = dataset.get_data(download=True)
        out_path = out_dir / f"{key}.geojson"
        count, first_year, last_year = export_inventory(
            frame,
            out_path,
            name_column=spec["name_column"],
            year_column=spec["year_column"],
            extra_columns=spec.get("extra_columns"),
            simplify=args.simplify,
            precision=args.precision,
        )
        logger.info("Wrote %s: %.1f MB", out_path.name, out_path.stat().st_size / 1e6)

        base = spec["title"] or meta.name
        span = f"{first_year}" if first_year == last_year else f"{first_year}\u2013{last_year}"
        index.append(
            {
                "id": key,
                "title": base,
                "span": span,
                "color": spec["color"],
                # What the viewer loads after the glace-viewer build step
                "url": f"{key}.pmtiles",
                "source_layer": key,
                "min_zoom": args.min_zoom,
                "max_zoom": args.max_zoom,
                "features": count,
                "simplify_m": args.simplify,
                "has_names": spec["name_column"] is not None,
                "attribution": meta.attribution,
                "citation": meta.additional_fields.get("citation"),
                "license": meta.license,
                "license_url": meta.license_url,
                "version": meta.version,
                "links": [
                    {
                        "label": "Dataset",
                        "url": meta.additional_fields.get("dataset_url") or meta.source_url,
                    },
                ],
            },
        )
        logger.info("  -> %s (%s)", index[-1]["title"], index[-1]["span"])

    index_path = out_dir / "inventories.json"
    index_path.write_text(
        json.dumps({"inventories": index}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    total = sum((out_dir / f"{entry['id']}.geojson").stat().st_size for entry in index)
    logger.info(
        "Wrote %s with %d overlays (%.1f MB total)",
        index_path,
        len(index),
        total / 1e6,
    )


if __name__ == "__main__":
    main()
