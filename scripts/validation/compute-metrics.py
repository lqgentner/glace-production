"""Score a cross-validation run: metrics and per-glacier tables.

Usage:
    pixi run python scripts/validation/compute-metrics.py config/validation/runs/ablation.yml \
        [--store URI] [--workers N]
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from glace_production.config import settings
from glace_production.datasets import SGI2023
from glace_production.validation.config import RunConfig
from glace_production.validation.crossval import load_dataset_configs, run_upstream
from glace_production.validation.folds import read_folds, straddling_glaciers
from glace_production.validation.metrics import glacier_table, metrics_table
from glace_production.validation.store import ValidationStore

logger = logging.getLogger(Path(__file__).stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--store")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="feature-set/model pairs to score concurrently; each worker holds its own "
        "merged frame beside the shared samples (default: %(default)s)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run = RunConfig.from_yaml(args.config)
    store = (
        ValidationStore(settings.resolve_path(args.store))
        if args.store
        else ValidationStore.default()
    )
    datasets = load_dataset_configs(args.config, run)
    folds = read_folds(Path(run.folds))
    metrics = metrics_table(
        run,
        store,
        datasets=datasets,
        folds=folds,
        workers=args.workers,
        upstream=run_upstream(store, run, Path(run.folds)),
    )
    store.write_parquet(metrics, store.metrics_path(run.name))
    pooled = metrics[(metrics.group_by == "all") & metrics.metric.isin(["auc", "iou"])]
    logger.info(
        "pooled:\n%s",
        pooled.pivot_table(index=["feature_set", "model"], columns="metric", values="value")
        .round(4)
        .to_string(),
    )

    inventory = SGI2023().get_data(download=True)
    debris = SGI2023(product="D").get_data(download=True)
    straddling = straddling_glaciers(folds.reset_index(), inventory)
    glaciers = glacier_table(run, store, inventory=inventory, debris=debris, straddling=straddling)
    store.write_parquet(glaciers, store.glaciers_path(run.name))
    logger.info("%d per-glacier rows", len(glaciers))


if __name__ == "__main__":
    main()
