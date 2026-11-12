#!/usr/bin/env python3
"""Create Sentinel-1 composites of the Alpine glaciers (LRW or per-pass)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from glace_production.composites import CompositeConfig, run_composite

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the composite pipeline from a YAML config file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python %(prog)s config/lrw-alps.yml year=2023\n"
            "  python %(prog)s config/pass-alps.yml year=2023\n"
        ),
    )
    parser.add_argument("config", type=Path, help="Path to YAML config file")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dot-notation overrides, e.g. year=2023 scope=switzerland",
    )
    args = parser.parse_args()

    config = CompositeConfig.from_yaml(args.config, overrides=args.overrides or None)

    # The aggregation is part of the name so mean and median runs of the same
    # scope and year do not write to the same log.
    parts = ["glacier", config.scope.lower(), str(config.year)]
    if config.method != "lrw":
        parts.append(config.method)
    parts.append(str(config.aggregation))
    log_file = Path("logs") / f"{'-'.join(parts)}.log"
    log_file.parent.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )

    run_composite(config)
