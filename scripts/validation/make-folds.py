"""Assign spatial folds to the label tiles and write the committed CSV.

Balanced folds (default) need the locations of the same config, so run
`sample-locations.py` first.

Usage:
    pixi run python scripts/validation/make-folds.py \
        --locations-config config/validation/locations-ch.yml --out config/validation/folds-ch.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from glace_production.config import settings
from glace_production.datasets import SGI2016, SGI2023
from glace_production.validation.config import LocationsConfig
from glace_production.validation.folds import (
    assign_folds,
    balanced_folds,
    fold_pieces,
    straddling_glaciers,
)
from glace_production.validation.locations import parse_label_name
from glace_production.validation.store import ValidationStore

logger = logging.getLogger(Path(__file__).stem)
INVENTORIES = {"SGI2016": SGI2016, "SGI2023": SGI2023}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locations-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", choices=["balanced", "kmeans"], default="balanced")
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--penalty", type=float, default=10.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = LocationsConfig.from_yaml(args.locations_config)
    labels_dir = settings.resolve_data_path(config.labels_dir)
    pairs = sorted(
        {
            (parsed[2], parsed[1])
            for path in labels_dir.glob("LABELS_*.tif")
            if (parsed := parse_label_name(path.name))
            and parsed[0] == config.inventory
            and parsed[1] in config.years
        }
    )
    tiles = sorted({tile for tile, _ in pairs})
    inventory = INVENTORIES[config.inventory]().get_data(download=True)
    store = ValidationStore.default()
    report = store.read_report(store.locations_dir)

    if args.method == "balanced":
        store.verify_config(store.locations_dir, config)
        have = set(zip(report["tile"], report["year"].astype(int), strict=True))
        if have != set(pairs):
            msg = (
                f"locations report does not match the label tile-years: missing "
                f"{sorted(set(pairs) - have)[:5]}, extra {sorted(have - set(pairs))[:5]}"
            )
            raise SystemExit(msg)
        glacier_sums = report.groupby("tile")["glacier"].sum()
        glacier_px: dict[str, float] = {
            str(tile): float(count) for tile, count in glacier_sums.items()
        }
        folds = balanced_folds(
            tiles,
            glacier_px,
            inventory,
            k=args.k,
            tolerance=args.tolerance,
            penalty=args.penalty,
            seed=args.seed,
        )
    else:
        folds = assign_folds(tiles, k=args.k, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(args.out, index=False)
    logger.info(
        "%d tiles into %d folds: %s",
        len(tiles),
        args.k,
        folds["fold"].value_counts().sort_index().to_dict(),
    )

    straddling = straddling_glaciers(folds, inventory)
    names = inventory.set_index("gid")
    for row in straddling.itertuples():
        info = names.loc[row.gid]
        logger.info(
            "straddles folds %s: %s (%.2f km2) tiles %s",
            row.folds,
            info["name"],
            row.area_km2,
            row.tiles,
        )
    logger.info(
        "%d glaciers (%.1f km2) straddle a fold border; largest %.1f km2",
        len(straddling),
        straddling["area_km2"].sum(),
        straddling["area_km2"].max() if len(straddling) else 0.0,
    )
    logger.info("pieces per fold: %s", fold_pieces(folds))
    if not report.empty:
        per_fold = (
            report.merge(folds, on="tile").groupby("fold")[["selected", "glacier", "ring"]].sum()
        )
        per_fold["tiles"] = folds["fold"].value_counts().sort_index()
        per_fold["glacier_share"] = per_fold["glacier"] / per_fold["selected"]
        cv = per_fold["glacier"].std() / per_fold["glacier"].mean()
        logger.info("rows per fold (CV of glacier pixels %.3f):\n%s", cv, per_fold.to_string())


if __name__ == "__main__":
    main()
