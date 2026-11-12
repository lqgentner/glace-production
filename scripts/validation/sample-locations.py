"""Select the validation locations of a label set.

Usage:
    pixi run python scripts/validation/sample-locations.py config/validation/locations-ch.yml \
        [--overwrite] [--max-workers 4] [--store s3://.../validation] [key=value ...]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from glace_production.config import settings
from glace_production.validation.config import LocationsConfig
from glace_production.validation.locations import write_locations
from glace_production.validation.store import ValidationStore

logger = logging.getLogger(Path(__file__).stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("overrides", nargs="*", help="key=value overrides")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--store", help="store root (default: models_dir/validation)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = LocationsConfig.from_yaml(args.config, args.overrides)
    store = (
        ValidationStore(settings.resolve_path(args.store))
        if args.store
        else ValidationStore.default()
    )
    report = write_locations(config, store, overwrite=args.overwrite, max_workers=args.max_workers)
    logger.info(
        "%d pairs, %d rows (%d glacier)",
        len(report),
        int(report["selected"].sum()),
        int(report["glacier"].sum()),
    )


if __name__ == "__main__":
    main()
