"""Validation 1: separability of the measurement layers of the compared datasets.

Layers only, as pixel values: the window statistics a layer derives are inputs
of the cross-validation's model, not features whose separability is reported.

Every layer of every listed dataset is scored on one common support, the sample
rows where all of these layers are finite, so the datasets compare on the same
pixels (the per-pass composites miss what the merged ones fill in, the global
seasonal product has its own gaps). The common support needs every compared
dataset at once: a single config is refused, since its table would sit on a
different support than the others'. ``--per-layer-support`` scores each layer
on its own finite rows instead, and accepts any number of configs. The table of
each dataset is written to ``separability/{dataset}.parquet``; its ``support``
column records which support it is on. Without configs, the three datasets of
the paper are compared (``DEFAULT_CONFIGS``), so a plain run keeps every table
on the same support.

Usage:
    pixi run python scripts/validation/compute-separability.py [CONFIG ...] [--store URI]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from glace_production.config import settings
from glace_production.validation.config import DatasetConfig
from glace_production.validation.separability import (
    MIN_COMPARED,
    common_support,
    common_support_name,
    load_layer_samples,
    separability_tables,
)
from glace_production.validation.store import ValidationStore

logger = logging.getLogger(Path(__file__).stem)
DATASETS_DIR = Path(__file__).resolve().parents[2] / "config" / "validation" / "datasets"
DEFAULT_CONFIGS = [
    DATASETS_DIR / f"{name}.yml" for name in ("lrw-alps", "pass-alps", "globalseasonal-ch")
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "configs",
        type=Path,
        nargs="*",
        default=DEFAULT_CONFIGS,
        help="dataset configs (default: lrw-alps, pass-alps and globalseasonal-ch)",
    )
    parser.add_argument("--store")
    parser.add_argument(
        "--per-layer-support",
        action="store_true",
        help="score each layer on its own finite rows, not on the common support",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    configs = [DatasetConfig.from_yaml(path) for path in args.configs]
    if not args.per_layer_support and len(configs) < MIN_COMPARED:
        parser.error(
            "the common support needs every compared dataset config (e.g. lrw-alps, "
            "pass-alps and globalseasonal-ch); pass --per-layer-support to score one alone"
        )
    store = (
        ValidationStore(settings.resolve_path(args.store))
        if args.store
        else ValidationStore.default()
    )
    samples = load_layer_samples(store, configs)
    if not args.per_layer_support:
        support = common_support(samples, configs)
        glacier = samples["label"].to_numpy() == 1
        logger.info(
            "%s: %d of %d rows (%d glacier, %d non-glacier)",
            common_support_name(configs),
            support.sum(),
            len(samples),
            (support & glacier).sum(),
            (support & ~glacier).sum(),
        )
    tables = separability_tables(samples, configs, per_layer_support=args.per_layer_support)
    for name, table in tables.items():
        store.write_parquet(table, store.separability_path(name))
        pooled = table[table.group_by == "all"].set_index("feature")["auc"]
        logger.info("%s pooled AUC:\n%s", name, pooled.round(4).to_string())


if __name__ == "__main__":
    main()
