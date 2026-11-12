"""Write full-tile maps for one (feature set, model) of a run.

By default the probability map (``PRED_*.tif``, percent); with ``--classify``
the classified map (``CLS_*.tif``, 1 glacier / 0 other, cut at the fold's
stored IoU threshold).

Usage:
    pixi run python scripts/validation/predict-maps.py config/validation/runs/ablation.yml \
        --feature-set all_ctx --model rf --locations-config config/validation/locations-ch.yml \
        [--classify] [--overwrite] [--store URI]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from glace_production.config import settings
from glace_production.validation.config import LocationsConfig, RunConfig
from glace_production.validation.crossval import load_dataset_configs, load_run_inputs
from glace_production.validation.maps import write_maps
from glace_production.validation.store import ValidationStore

logger = logging.getLogger(Path(__file__).stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--feature-set", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--locations-config", type=Path, required=True)
    parser.add_argument("--classify", action="store_true", help="write CLS_* class maps")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--store")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run = RunConfig.from_yaml(args.config)
    locations = LocationsConfig.from_yaml(args.locations_config)
    store = (
        ValidationStore(settings.resolve_path(args.store))
        if args.store
        else ValidationStore.default()
    )
    datasets = load_dataset_configs(args.config, run)
    inputs = load_run_inputs(run, store, folds_path=Path(run.folds), dataset_configs=datasets)
    written = write_maps(
        run,
        inputs,
        store,
        feature_set=args.feature_set,
        model=args.model,
        labels_dir=settings.resolve_data_path(locations.labels_dir),
        inventory=locations.inventory,
        kind="class" if args.classify else "proba",
        overwrite=args.overwrite,
    )
    logger.info("%d maps written", len(written))


if __name__ == "__main__":
    main()
