"""Extract a dataset's layers at the validation sample locations.

Usage:
    pixi run python scripts/validation/extract-samples.py config/validation/datasets/lrw-alps.yml \
        [--allow-missing] [--overwrite] [--max-workers 8] [--store URI] [key=value ...]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from glace_production.config import settings
from glace_production.validation.config import DatasetConfig
from glace_production.validation.extract import write_samples
from glace_production.validation.store import ValidationStore

logger = logging.getLogger(Path(__file__).stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--store")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = DatasetConfig.from_yaml(args.config, args.overrides)
    store = (
        ValidationStore(settings.resolve_path(args.store))
        if args.store
        else ValidationStore.default()
    )
    report = write_samples(
        config,
        store,
        allow_missing=args.allow_missing,
        overwrite=args.overwrite,
        max_workers=args.max_workers,
    )
    missing = int(report["missing"].sum()) if "missing" in report else 0
    logger.info("%s: %d layer rows, %d pairs missing", config.name, len(report), missing)


if __name__ == "__main__":
    main()
