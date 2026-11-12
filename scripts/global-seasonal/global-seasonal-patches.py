"""Resample the global seasonal coherence and backscatter mosaics onto the 10 m label tiles.

Reads the WGS84 mosaics written by ``scripts/global-seasonal/download-global-seasonal.py``
(run it first with ``--bbox 5 45 11 48`` for Switzerland, adding
``--products coherence backscatter`` for ``--backscatter``) and writes one COG per
label tile and layer so the product can be treated as any other validation dataset.
Backscatter stays linear gamma0 power, resampled in linear space.

Usage:
    pixi run python scripts/global-seasonal/global-seasonal-patches.py --baseline 12 \
        --locations-config config/validation/locations-ch.yml [--season summer] \
        [--backscatter] [--overwrite]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from freezebase.mgrs import MGRSGeoBox
from freezebase.raster import COG_PROFILE, write_cog
import numpy as np
from rasterio.enums import Resampling

from glace_production.config import settings
from glace_production.validation.config import PIXEL_SIZE_M, LocationsConfig
from glace_production.validation.locations import parse_label_name
from glace_production.validation.rasters import warp_to_geobox

if TYPE_CHECKING:
    from upath import UPath

logger = logging.getLogger(Path(__file__).stem)
GLOBAL_DIR = settings.data_dir / "global-seasonal"


@dataclass(frozen=True)
class Layer:
    """A global mosaic and the per-tile COGs it is resampled into."""

    source: Path | UPath
    name: str
    """Output filename with a ``{tile}`` placeholder."""
    band: str
    tags: dict[str, str]


def label_tiles(config: LocationsConfig) -> list[str]:
    """Tiles that have a label mask of the configured inventory and years."""
    labels_dir = settings.resolve_data_path(config.labels_dir)
    return sorted(
        {
            parsed[2]
            for path in labels_dir.glob("LABELS_*.tif")
            if (parsed := parse_label_name(path.name))
            and parsed[0] == config.inventory
            and parsed[1] in config.years
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locations-config", type=Path, required=True)
    parser.add_argument("--baseline", type=int, default=12, choices=[6, 12, 18, 24, 36, 48])
    parser.add_argument("--season", default="summer")
    parser.add_argument(
        "--backscatter", action="store_true", help="also tile the VV and VH gamma0 mosaics"
    )
    parser.add_argument("--out-dir", default="global-seasonal/tiles-ch")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    season = args.season.upper()
    common_tags = {
        "SOURCE": "Kellndorfer et al. (2022), https://doi.org/10.1038/s41597-022-01189-6",
        "SEASON": args.season,
        "RESAMPLING": "bilinear from 3 arcsec",
    }
    layers = [
        Layer(
            source=GLOBAL_DIR / f"global_seasonal_{args.season}_vv_COH{args.baseline:02d}.tif",
            name=f"GLOBALSEASONAL_COH{args.baseline:02d}_{{tile}}_VV_{season}.tif",
            band="COH",
            tags=common_tags | {"TEMPORAL_BASELINE_DAYS": str(args.baseline)},
        )
    ]
    if args.backscatter:
        layers += [
            Layer(
                source=GLOBAL_DIR / f"global_seasonal_{args.season}_{pol}_GAMMA0.tif",
                name=f"GLOBALSEASONAL_GAMMA0_{{tile}}_{pol.upper()}_{season}.tif",
                band="GAMMA0",
                tags=common_tags | {"POLARIZATION": pol.upper(), "SCALE": "linear power"},
            )
            for pol in ("vv", "vh")
        ]
    for layer in layers:
        if not layer.source.exists():
            msg = (
                f"{layer.source} missing; "
                "run scripts/global-seasonal/download-global-seasonal.py first"
            )
            raise FileNotFoundError(msg)
    out_dir = settings.resolve_data_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles = label_tiles(LocationsConfig.from_yaml(args.locations_config))
    for tile in tiles:
        geobox = MGRSGeoBox.from_mgrs(tile, resolution=PIXEL_SIZE_M)
        for layer in layers:
            out_path = out_dir / layer.name.format(tile=tile)
            if out_path.exists() and not args.overwrite:
                continue
            data = warp_to_geobox(layer.source, geobox, resampling=Resampling.bilinear)
            profile = COG_PROFILE | {
                "dtype": "float32",
                "count": 1,
                "nodata": np.nan,
                "width": geobox.shape.x,
                "height": geobox.shape.y,
                "crs": geobox.crs,
                "transform": geobox.affine,
            }
            write_cog(
                data[np.newaxis], out_path, profile, band_names=[layer.band], tags=layer.tags
            )
            logger.info("%s: %.1f%% valid", out_path.name, 100 * np.isfinite(data).mean())


if __name__ == "__main__":
    main()
