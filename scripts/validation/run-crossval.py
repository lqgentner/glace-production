"""Validation 2: spatial cross-validation of a run config.

Usage:
    pixi run python scripts/validation/run-crossval.py config/validation/runs/ablation.yml \
        [--feature-set NAME ...] [--model NAME ...] [--overwrite] [--benchmark] \
        [--learning-curve] [--store URI]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import resource

from sklearn.metrics import roc_auc_score

from glace_production.config import settings
from glace_production.validation.config import RunConfig
from glace_production.validation.crossval import (
    RunInputs,
    benchmark,
    compute_support,
    fit_fold_model,
    load_dataset_configs,
    load_run_inputs,
    run_crossval,
)
from glace_production.validation.models import make_model
from glace_production.validation.store import ValidationStore

logger = logging.getLogger(Path(__file__).stem)

LEARNING_CURVE_ROWS = (100_000, 300_000, 1_000_000, 2_000_000)


def _learning_curve(run: RunConfig, inputs: RunInputs, feature_set: str | None = None) -> None:
    """Fit ``rf`` on one feature set at increasing caps, log the inner-fold AUC.

    The feature set defaults to the run's largest. The fit excludes fold 0 and
    the inner fold 1 and is scored on every scope row of fold 1, so the budget
    is chosen on training regions only.
    """
    if feature_set is None:
        feature_set = max(run.feature_sets, key=lambda k: len(run.feature_sets[k]))
    columns = run.feature_sets[feature_set]
    logger.info("learning curve on %s (%d columns)", feature_set, len(columns))
    support = compute_support(inputs.samples, run.columns)
    samples = inputs.samples[support].reset_index(drop=True)
    fold_of_row = samples["tile"].map(inputs.folds).to_numpy()
    test_rows = fold_of_row == 1
    x = samples[list(columns)].to_numpy()[test_rows]
    y = samples["label"].to_numpy()[test_rows]
    for train_rows in LEARNING_CURVE_ROWS:
        cap = min(train_rows, len(samples))
        estimator = fit_fold_model(
            samples,
            inputs.folds,
            0,
            columns,
            "rf",
            run.seed,
            0,
            train_rows=cap,
            factory=make_model,
            exclude=[1],
        )
        proba = estimator.predict_proba(x)[:, 1]
        auc = roc_auc_score(y, proba)
        logger.info("train_rows=%d (capped at %d): AUC=%.4f", train_rows, cap, auc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--feature-set", action="append", dest="feature_sets")
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--benchmark", action="store_true", help="time one fit per model and exit")
    parser.add_argument(
        "--learning-curve",
        action="store_true",
        help="fit rf at increasing train_rows caps and exit; on the first --feature-set, "
        "else the largest",
    )
    parser.add_argument("--store")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run = RunConfig.from_yaml(args.config)
    store = (
        ValidationStore(settings.resolve_path(args.store))
        if args.store
        else ValidationStore.default()
    )
    datasets = load_dataset_configs(args.config, run)
    inputs = load_run_inputs(run, store, folds_path=Path(run.folds), dataset_configs=datasets)
    if args.benchmark:
        biggest = max(run.feature_sets, key=lambda k: len(run.feature_sets[k]))
        timings = benchmark(run, inputs, feature_set=biggest)
        for key, value in timings.items():
            logger.info("%s: %.1f s", key, value)
        logger.info("peak RSS: %.1f GB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
        return
    if args.learning_curve:
        _learning_curve(run, inputs, args.feature_sets[0] if args.feature_sets else None)
        return
    run_crossval(
        run,
        inputs,
        store,
        feature_sets=args.feature_sets,
        models=args.models,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
