#!/usr/bin/env python3
"""Download Copernicus GLO-30 DEM tiles for a bounding box and merge them into one COG.

The 1x1 degree EPSG:4326 tiles are read straight from the public AWS Open Data
mirror (https://copernicus-dem-30m.s3.amazonaws.com) and mosaicked on their own
grid, without reprojection. Their EGM2008 (EPSG:3855) orthometric heights are
converted to WGS84 ellipsoid heights, which the SAR processing expects, unless
--no-vertical-transform is given.

--edm-output also mosaics the EDM (Editing Mask) auxiliary layer into a QA COG,
marking pixels ESA filled, interpolated or otherwise edited. Its values, per the
Copernicus DEM Product Handbook:
    0  Void (no data)              7  Flattened pixels
    1  Not edited (original data)  8  Ocean pixels
    2  Infill of external data     9  Lake pixels
    3  Interpolated pixels         10 River pixels
    4  Smoothed pixels             11 Shoreline pixels
    5  Airport editing             12 Morphed pixels
    6  Raised negative elevation   13 Shifted pixels

Usage
-----
    pixi run python download-glo30.py --config config/dem/glo30-alps.yml [output.tif]
        [--overwrite]
    pixi run python download-glo30.py <output.tif> [--bbox MIN_LON MIN_LAT MAX_LON MAX_LAT]
        [--edm-output <edm.tif>] [--no-vertical-transform] [--overwrite]

``--config`` reads the bounding box, the output (relative to the data directory)
and the vertical transform from a committed YAML file; an output given on the
command line takes precedence. An existing output is replaced only with
``--overwrite``. A failed run leaves no partial output.

The EGM2008 grid is fetched by PROJ from cdn.proj.org, which the pixi environment
allows (``PROJ_NETWORK=ON``).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

from freezebase import vrt
from freezebase.download import retry_request
from freezebase.progress import track_progress
from freezebase.raster import COG_PROFILE, write_cog
import rasterio
import requests

from glace_production.dem import Glo30DownloadConfig, ellipsoidal_heights, resolve_download_output

if TYPE_CHECKING:
    from collections.abc import Callable

    from upath import UPath

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_URL = "https://copernicus-dem-30m.s3.amazonaws.com"
TILE_PATTERN = "Copernicus_DSM_COG_10_{tile_id}_DEM"
EDM_PATTERN = "Copernicus_DSM_COG_10_{tile_id}_EDM"
DEFAULT_BBOX = (7.0, 46.0, 9.0, 47.0)
# GLO-30's native vertical datum, as a compound CRS on top of the tiles' EPSG:4326.
_ORTHOMETRIC_CRS = "EPSG:4326+3855"
# Above this latitude, GLO-30 tiles use coarser longitude pixel spacing.
HIGH_LAT_WARNING_THRESHOLD = 50.0
# EDM is a Byte mask whose valid class codes are 0-13; use a value outside that range
# to mark gaps left by missing tiles, distinct from the in-band "0 = Void" class.
EDM_NODATA = 255
_HEAD_TIMEOUT = 10.0
_EPS = 1e-9


def parse_nodata(value: str) -> float:
    if value.lower() == "nan":
        return float("nan")
    try:
        return float(value)
    except ValueError as err:
        msg = f"Invalid nodata value '{value}'. Use a float or 'nan'."
        raise argparse.ArgumentTypeError(msg) from err


def tile_id(lat: int, lon: int) -> str:
    lat_hem = "N" if lat >= 0 else "S"
    lon_hem = "E" if lon >= 0 else "W"
    return f"{lat_hem}{abs(lat):02d}_00_{lon_hem}{abs(lon):03d}_00"


def tile_url(tile: str) -> str:
    name = TILE_PATTERN.format(tile_id=tile)
    return f"{BASE_URL}/{name}/{name}.tif"


def edm_tile_url(tile: str) -> str:
    dem_dir = TILE_PATTERN.format(tile_id=tile)
    edm_name = EDM_PATTERN.format(tile_id=tile)
    return f"{BASE_URL}/{dem_dir}/AUXFILES/{edm_name}.tif"


def _tile_range(lo: float, hi: float) -> range:
    """SW-corner integer tile indices whose [n, n+1) cell intersects [lo, hi)."""
    return range(math.floor(lo), math.floor(hi - _EPS) + 1)


def get_tiles(bbox: tuple[float, float, float, float]) -> list[str]:
    min_lon, min_lat, max_lon, max_lat = bbox
    if min_lon >= max_lon or min_lat >= max_lat:
        msg = f"Invalid bbox {bbox}: expected min_lon < max_lon and min_lat < max_lat."
        raise ValueError(msg)
    lats = _tile_range(min_lat, max_lat)
    lons = _tile_range(min_lon, max_lon)
    return [tile_id(lat, lon) for lat in lats for lon in lons]


def has_mixed_resolution(bbox: tuple[float, float, float, float]) -> bool:
    """Whether the bbox reaches the latitudes where tiles coarsen their longitude spacing."""
    _, min_lat, _, max_lat = bbox
    if max(abs(min_lat), abs(max_lat)) < HIGH_LAT_WARNING_THRESHOLD:
        return False
    log.warning(
        "bbox extends to |lat| >= %.0f deg: Copernicus DEM tiles there use coarser "
        "longitude pixel spacing than at low latitudes. Mixed-resolution tiles in the "
        "mosaic will be nearest-neighbour aligned onto the finest grid present.",
        HIGH_LAT_WARNING_THRESHOLD,
    )
    return True


@retry_request(logger=log)
def check_tile_exists(tile: str, session: requests.Session, url_fn: Callable[[str], str]) -> bool:
    """Check whether a tile exists; a 404 is "no such tile", other errors are retried."""
    url = url_fn(tile)
    resp = session.head(url, timeout=_HEAD_TIMEOUT, allow_redirects=True)
    if resp.status_code == requests.codes.not_found:
        return False
    resp.raise_for_status()
    return True


def filter_existing_tiles(
    tiles: list[str],
    url_fn: Callable[[str], str],
    *,
    max_workers: int,
    description: str = "Checking tile availability...",
) -> list[str]:
    session = requests.Session()
    check = partial(check_tile_exists, session=session, url_fn=url_fn)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        found = list(
            track_progress(
                executor.map(check, tiles),
                description=description,
                total=len(tiles),
            ),
        )
    existing = [t for t, ok in zip(tiles, found, strict=True) if ok]
    missing = [t for t, ok in zip(tiles, found, strict=True) if not ok]
    if missing:
        log.warning("Skipping %d missing tile(s): %s", len(missing), ", ".join(missing))
    log.info("%d/%d candidate tile(s) available", len(existing), len(tiles))
    return existing


def write_mosaic_cog(
    sources: list[str],
    output: Path | UPath,
    *,
    nodata: float,
    overview_resampling: str,
    orthometric_crs: str | None = None,
    mixed_resolution: bool = False,
) -> None:
    """Mosaic tiles on their common grid and write the result as a COG.

    Parameters
    ----------
    sources : list[str]
        Tile paths or URLs.
    output : Path or UPath
        Output COG, local or in an object store.
    nodata : float
        NODATA of the gaps between tiles.
    overview_resampling : str
        Resampling of the overviews, which are built from the full-resolution mosaic.
    orthometric_crs : str or None, optional
        Compound CRS of the tiles' heights, converted to ellipsoidal heights when given.
    mixed_resolution : bool, optional
        Resample tiles of different pixel sizes onto the finest one. Otherwise every
        tile must lie on one grid, whose corner is taken from the tile headers.
    """
    log.info("Mosaicking %d tile(s) -> %s", len(sources), output)
    mosaic = vrt.mosaic(sources, resolution="finest" if mixed_resolution else None, nodata=nodata)
    if orthometric_crs is not None:
        mosaic = ellipsoidal_heights(mosaic, orthometric_crs=orthometric_crs)
    write_cog(mosaic, output, COG_PROFILE | {"overview_resampling": overview_resampling})


def _resolve_config(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> tuple[Glo30DownloadConfig, Path | UPath]:
    """Merge ``--config`` with the command line and return the config and the output."""
    if args.config is not None:
        config = Glo30DownloadConfig.from_yaml(args.config)
    elif args.output is not None:
        config = Glo30DownloadConfig(output=str(args.output), bbox=DEFAULT_BBOX)
    else:
        parser.error("give an output path or --config")
    if args.bbox is not None:
        config = config.model_copy(update={"bbox": tuple(args.bbox)})
    if args.vertical_transform is not None:
        config = config.model_copy(update={"vertical_transform": args.vertical_transform})
    try:
        target = resolve_download_output(config, args.output, overwrite=args.overwrite)
    except FileExistsError as exc:
        parser.error(str(exc))
    return config, target


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download Copernicus GLO-30 DEM tiles for a bounding box and merge them "
            "into a single Cloud Optimised GeoTIFF (COG), with no reprojection or "
            "resampling of the source data."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --config config/dem/glo30-alps.yml
  %(prog)s dem.tif --bbox 7 46 9 47
  %(prog)s dem.tif --bbox 7 46 9 47 --edm-output dem-edm.tif
  %(prog)s dem.tif --bbox 7 46 9 47 --no-vertical-transform
        """,
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=None,
        help="Output COG file path (default: the config's output).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="PATH",
        help="Download config (YAML) with the output, bbox and vertical transform.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output.",
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        default=None,
        help=f"Bounding box as min_lon min_lat max_lon max_lat (default: {DEFAULT_BBOX})",
    )
    parser.add_argument(
        "--check-workers",
        type=int,
        default=16,
        metavar="N",
        help="Parallel workers for tile-existence HEAD checks (default: 16).",
    )
    parser.add_argument(
        "--dst-nodata",
        type=parse_nodata,
        default=float("nan"),
        metavar="VALUE",
        help="Nodata value of the gaps left by missing tiles (default: nan).",
    )
    parser.add_argument(
        "--no-vertical-transform",
        dest="vertical_transform",
        action="store_false",
        default=None,
        help="Keep the native EGM2008 orthometric heights.",
    )
    parser.add_argument(
        "--edm-output",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Also mosaic the EDM (Editing Mask) layer into a QA COG at this path "
            "(uint8; 255 marks gaps from missing tiles)."
        ),
    )
    args = parser.parse_args()
    config, target = _resolve_config(parser, args)
    bbox = config.bbox

    candidate_tiles = get_tiles(bbox)
    log.info("Enumerated %d candidate tile(s) for bbox %s", len(candidate_tiles), bbox)
    mixed_resolution = has_mixed_resolution(bbox)

    tiles = filter_existing_tiles(candidate_tiles, tile_url, max_workers=args.check_workers)
    if not tiles:
        log.error(
            "None of the %d candidate tile(s) exist on the Copernicus DEM bucket "
            "for bbox %s; nothing to download.",
            len(candidate_tiles),
            bbox,
        )
        raise SystemExit(1)

    with rasterio.Env(GDAL_NUM_THREADS="ALL_CPUS"):
        write_mosaic_cog(
            [tile_url(t) for t in tiles],
            target,
            nodata=args.dst_nodata,
            overview_resampling="AVERAGE",
            orthometric_crs=_ORTHOMETRIC_CRS if config.vertical_transform else None,
            mixed_resolution=mixed_resolution,
        )
    log.info("Done: %s (%d/%d candidate tile(s) used)", target, len(tiles), len(candidate_tiles))

    if args.edm_output:
        edm_tiles = filter_existing_tiles(
            tiles,
            edm_tile_url,
            max_workers=args.check_workers,
            description="Checking EDM tile availability...",
        )
        if not edm_tiles:
            log.error("No EDM tiles available for the selected tile(s); skipping QA COG.")
            return
        with rasterio.Env(GDAL_NUM_THREADS="ALL_CPUS"):
            write_mosaic_cog(
                [edm_tile_url(t) for t in edm_tiles],
                args.edm_output,
                nodata=EDM_NODATA,
                overview_resampling="MODE",
                mixed_resolution=mixed_resolution,
            )
        log.info("Done: %s (EDM, %d/%d tile(s) used)", args.edm_output, len(edm_tiles), len(tiles))


if __name__ == "__main__":
    main()
