#!/usr/bin/env python3
"""Download selected tiles of the global seasonal Sentinel-1 coherence and backscatter dataset.

Coherence is stored as a float in [0, 1]; backscatter (the ``AMP`` layers) is
converted from the product's amplitude DN to linear gamma0 power, with
``dB = 20 * log10(DN) - 83`` and ``gamma0 = 10 ** (dB / 10)``. Each product is
mosaicked into one COG per season and polarization under ``global-seasonal/``.

Run with:
    pixi run python scripts/global-seasonal/download-global-seasonal.py [--help]

Kellndorfer, J., Cartus, O., Lavalle, M., Magnard, C., Milillo, P., Oveisgharan, S., Osmanoglu, B.,
Rosen, P. A., & Wegmüller, U. (2022). Global seasonal Sentinel-1 interferometric coherence and
backscatter data set. Scientific Data, 9(1). https://doi.org/10.1038/s41597-022-01189-6
"""

from __future__ import annotations

import argparse
import tempfile
from typing import TYPE_CHECKING

from freezebase import vrt
from freezebase.download import HTTPDownloader
from freezebase.raster import (
    COG_PROFILE,
    RASTERIO_PROFILE_DEFAULTS,
    AnyPath,
    rasterio_open,
    write_cog,
)
from freezebase.utils import get_credentials_from_env
import numpy as np

from glace_production.config import settings

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

BASE_URL = "https://sentinel1.asf.alaska.edu/global_coherence/s1/tiles"
BASE_NAME = "global_seasonal"
SAVE_DIR = settings.data_dir / "global-seasonal"

DEFAULT_SEASONS = ["summer"]
DEFAULT_BASELINES = [12]
DEFAULT_BBOX = (7, 46, 9, 47)
DEFAULT_POLARIZATION = "VV"
DEFAULT_PRODUCTS = ["coherence"]

VALID_SEASONS = ["spring", "summer", "fall", "winter"]
VALID_BASELINES = [6, 12, 18, 24, 36, 48]
VALID_POLARIZATIONS = ["VV", "HH"]
VALID_PRODUCTS = ["coherence", "backscatter"]
CROSS_POLARIZATION = {"VV": "VH", "HH": "HV"}
AMP_DB_OFFSET = 83.0
"""Calibration offset of the amplitude DN: ``dB = 20 * log10(DN) - 83``."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download tiles of the global seasonal coherence and backscatter dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --seasons summer winter --baselines 12 24 --bbox 7 46 9 47
  %(prog)s --polarization HH --seasons fall
  %(prog)s --products coherence backscatter --bbox 5 45 11 48
        """,
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        choices=VALID_SEASONS,
        default=DEFAULT_SEASONS,
        help=f"Season(s) to download (default: {DEFAULT_SEASONS})",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        type=int,
        choices=VALID_BASELINES,
        default=DEFAULT_BASELINES,
        help=f"Temporal baseline(s) in days (default: {DEFAULT_BASELINES})",
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=int,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        default=list(DEFAULT_BBOX),
        help=f"Bounding box as min_lon min_lat max_lon max_lat (default: {DEFAULT_BBOX})",
    )
    parser.add_argument(
        "--polarization",
        choices=VALID_POLARIZATIONS,
        default=DEFAULT_POLARIZATION,
        help=(
            f"Co-polarization of the coherence (default: {DEFAULT_POLARIZATION}); "
            "backscatter is downloaded in it and its cross-polarization"
        ),
    )
    parser.add_argument(
        "--products",
        nargs="+",
        choices=VALID_PRODUCTS,
        default=DEFAULT_PRODUCTS,
        help=f"Product(s) to download (default: {DEFAULT_PRODUCTS})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    auth = get_credentials_from_env("EOSDIS_USERNAME", "EOSDIS_PASSWORD")
    downloader = HTTPDownloader(auth=auth, trusted_hosts=["urs.earthdata.nasa.gov"])
    tiles = get_tiles(tuple(args.bbox))
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    if "coherence" in args.products:
        download_coherences(tiles, args.seasons, args.polarization, args.baselines, downloader)
    if "backscatter" in args.products:
        polarizations = [args.polarization, CROSS_POLARIZATION[args.polarization]]
        for season in args.seasons:
            for polarization in polarizations:
                download_backscatter(tiles, season, polarization, downloader)


def get_tiles(bbox: tuple[int, int, int, int]) -> list[str]:
    lonrange = range(bbox[0], bbox[2])
    latrange = range(bbox[1], bbox[3])
    # Tile name constructed from the top-left corner
    return [f"N{lat + 1:02d}E{lon:03d}" for lat in latrange for lon in lonrange]


def _rescale(path: AnyPath, scale: Callable[[NDArray[np.float32]], NDArray[np.float32]]) -> None:
    """Rewrite the DN of ``path`` in place as float32, with DN 0 as NaN nodata."""
    with rasterio_open(path) as src:
        dn = src.read(1).astype(np.float32)
        src_profile = src.profile
    data = np.where(dn != 0, scale(dn), np.nan).astype(np.float32)
    dst_profile = RASTERIO_PROFILE_DEFAULTS | src_profile | {"dtype": np.float32, "nodata": np.nan}
    with rasterio_open(path, "w", **dst_profile) as dst:
        dst.write(data, 1)


def dn_to_coherence(dn: NDArray[np.float32]) -> NDArray[np.float32]:
    """Coherence in [0, 1] from DN in percent."""
    return (dn / 100).astype(np.float32)


def scale_coherence(path: AnyPath) -> None:
    _rescale(path, dn_to_coherence)


def amplitude_to_gamma0(dn: NDArray[np.float32]) -> NDArray[np.float32]:
    """Linear gamma0 power from amplitude DN, ``DN = 10 ** ((dB + 83) / 20)``."""
    # 10 ** (dB / 10) with dB = 20 * log10(DN) - 83 is DN**2 / 10**8.3
    return (dn.astype(np.float64) ** 2 / 10 ** (AMP_DB_OFFSET / 10)).astype(np.float32)


def scale_backscatter(path: AnyPath) -> None:
    _rescale(path, amplitude_to_gamma0)


def download_mosaic(
    urls: list[str],
    scale: Callable[[AnyPath], None],
    name: str,
    band_name: str,
    downloader: HTTPDownloader,
) -> None:
    """Download tiles, rescale each in place, and mosaic them into ``SAVE_DIR / name``."""
    with tempfile.TemporaryDirectory() as tempdir:
        files = []
        for url in urls:
            file = downloader(url, save_dir=tempdir)
            scale(file)
            files.append(file)
        write_cog(vrt.mosaic(files), SAVE_DIR / name, COG_PROFILE, band_names=[band_name])


def download_coherence(
    tiles: list[str],
    season: str,
    baseline: int,
    polarization: str,
    downloader: HTTPDownloader,
) -> None:
    suffix = f"{season}_{polarization.lower()}_COH{baseline:02d}"
    download_mosaic(
        [f"{BASE_URL}/{tile}/{tile}_{suffix}.tif" for tile in tiles],
        scale_coherence,
        f"{BASE_NAME}_{suffix}.tif",
        "COH",
        downloader,
    )


def download_backscatter(
    tiles: list[str],
    season: str,
    polarization: str,
    downloader: HTTPDownloader,
) -> None:
    download_mosaic(
        [f"{BASE_URL}/{tile}/{tile}_{season}_{polarization.lower()}_AMP.tif" for tile in tiles],
        scale_backscatter,
        f"{BASE_NAME}_{season}_{polarization.lower()}_GAMMA0.tif",
        "GAMMA0",
        downloader,
    )


def download_coherences(
    tiles: list[str],
    seasons: list[str],
    polarization: str,
    baselines: list[int],
    downloader: HTTPDownloader,
) -> None:
    for season in seasons:
        for baseline in baselines:
            download_coherence(tiles, season, baseline, polarization, downloader)


if __name__ == "__main__":
    main()
