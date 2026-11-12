"""Label masks rasterized from glacier inventories onto the MGRS 10 km grid.

One uint8 COG per tile and inventory, ``LABELS_{source}_{tile}.tif``, or per
tile and acquisition year, ``LABELS_{source}_{year}_{tile}.tif``, holding
1 = glacier, 0 = background and :data:`IGNORE_VALUE` where the inventory makes
no claim. Masks are burned from the true (unbuffered) outlines.

Inventories flagged ``year_split`` (the Swiss inventories and Paul et al.
2020, whose outlines carry an acquisition year or date) are rasterized once
per year, so every glacier is labelled exactly once, in its acquisition year.
The mask of year *Y* treats the other years under the assumption that Alpine
glaciers only retreat:

- outlines mapped **before** *Y* are burned as ignore without a buffer: the
  glacier may have retreated inside them, but cannot have grown, so the
  terrain around them is background in *Y*;
- outlines mapped **after** *Y* are buffered by ``ignore_buffer_m`` and burned
  as ignore: in *Y* the glacier may have reached into that band.

A source with a ``domain`` (the Swiss national territory for the SGI) has
everything outside it burned as ignore last, whatever the outlines say: the
inventory is not authoritative there. The rule, buffer and domain are recorded
as GeoTIFF tags, so a mask describes itself to whatever reads it.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING

from freezebase.mgrs import MGRSGeoBox
from freezebase.raster import COG_PROFILE, write_cog
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from glace_production.datasets import SGI2016, SGI2023, PaulEtAl2020, SwisstopoBoundaries
from glace_production.geo import MGRSInventory

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from freezebase.vectordata import GeoVectorData
    from numpy.typing import NDArray
    from odc.geo.geobox import GeoBox
    from upath import UPath

logger = logging.getLogger(__name__)

IGNORE_VALUE = 255
"""Burned where the inventory makes no claim: other years' glaciers, outside the domain.

Excluded from training and evaluation rather than treated as background. Also
the masks' nodata value. Part of the on-disk contract with whatever reads them.
"""

YEAR_RULE = "retreat"
"""Name of the year-split rule written to the ``YEAR_RULE`` tag."""


class LabelContractError(ValueError):
    """A mask does not carry the tags this module writes, so its ignore value cannot be trusted."""


def check_label_tags(tags: Mapping[str, str], *, year: int | None = None) -> None:
    """Raise :class:`LabelContractError` unless ``tags`` describe a mask written by this module.

    Readers rely on the mask to carry the domain and the year rule; a mask
    from an earlier rasterization lacks the tags and has to be regenerated.
    """
    expected = {"IGNORE_VALUE": str(IGNORE_VALUE)}
    if year is not None:
        expected |= {"ACQUISITION_YEAR": str(year), "YEAR_RULE": YEAR_RULE}
    problems = [
        f"{k}={tags.get(k)!r}, expected {v!r}" for k, v in expected.items() if tags.get(k) != v
    ]
    problems += [f"{k} missing" for k in ("INVENTORY", "DOMAIN") if k not in tags]
    if problems:
        msg = (
            "label mask does not carry the current contract ("
            + "; ".join(problems)
            + "); regenerate it with scripts/inventories/rasterize-labels.py"
        )
        raise LabelContractError(msg)


@dataclass(frozen=True)
class LabelSource:
    """How one glacier inventory is rasterized.

    Parameters
    ----------
    key
        Filename token, e.g. ``SGI2016``.
    year_split
        Rasterize once per acquisition year under the retreat rule.
    year_column
        Outline attribute the acquisition year is read from. May hold a year
        or a full date.
    domain
        Key of :data:`DOMAINS` outside which every pixel is ignore, or
        ``None`` for an inventory that is authoritative everywhere it covers.
    """

    key: str
    year_split: bool = False
    year_column: str = "year_acq"
    domain: str | None = None


LABEL_SOURCES: dict[str, LabelSource] = {
    "PAULETAL2020": LabelSource(key="PAULETAL2020", year_split=True, year_column="date"),
    "SGI2016": LabelSource(key="SGI2016", year_split=True, domain="switzerland"),
    "SGI2023": LabelSource(key="SGI2023", year_split=True, domain="switzerland"),
}
"""Glacier inventories available as label masks, keyed by filename token."""

INVENTORIES: dict[str, type[GeoVectorData]] = {
    "PAULETAL2020": PaulEtAl2020,
    "SGI2016": SGI2016,
    "SGI2023": SGI2023,
}
"""Dataset class of every label source."""


def swiss_domain() -> gpd.GeoDataFrame:
    """Return the Swiss national territory as a 2-D polygon in EPSG:2056."""
    bounds = SwisstopoBoundaries().get_data(download=True)
    country = bounds[(bounds["objektart"] == "Land") & (bounds["name"] == "Schweiz")]
    if len(country) != 1:
        msg = f"expected one Swiss national polygon, found {len(country)}"
        raise ValueError(msg)
    geom = shapely.force_2d(country.geometry.iloc[0])
    return gpd.GeoDataFrame({"name": ["Schweiz"]}, geometry=[geom], crs="EPSG:2056")


DOMAINS: dict[str, Callable[[], gpd.GeoDataFrame]] = {"switzerland": swiss_domain}
"""Loaders of the areas a label source is authoritative in."""


def acquisition_years(gdf: gpd.GeoDataFrame, column: str) -> pd.Series:
    """Per-outline acquisition year, from either a year or a full-date column."""
    values = gdf[column]
    if pd.api.types.is_datetime64_any_dtype(values):
        return values.dt.year
    return values


@dataclass(frozen=True)
class YearLabels:
    """The outlines of one acquisition year and what its masks ignore.

    Parameters
    ----------
    year
        Acquisition year.
    outlines
        The year's glaciers, buffered for tile selection only.
    ignore
        Inventories burned as :data:`IGNORE_VALUE` before the outlines, each
        with its own buffer: earlier years unbuffered, later years buffered.
    """

    year: int
    outlines: MGRSInventory
    ignore: list[MGRSInventory]


def split_years(
    gdf: gpd.GeoDataFrame,
    source: LabelSource,
    *,
    buffer_m: float,
    ignore_buffer_m: float,
) -> list[YearLabels]:
    """Split a year-split inventory into one :class:`YearLabels` per acquisition year."""
    years = acquisition_years(gdf, source.year_column)
    result = []
    for year in sorted(int(y) for y in years.unique()):
        selected = MGRSInventory(gpd.GeoDataFrame(gdf[years == year]), buffer_m=buffer_m)
        ignore = []
        if (years < year).any():
            ignore.append(MGRSInventory(gpd.GeoDataFrame(gdf[years < year])))
        if (years > year).any():
            ignore.append(
                MGRSInventory(gpd.GeoDataFrame(gdf[years > year]), buffer_m=ignore_buffer_m)
            )
        result.append(YearLabels(year=year, outlines=selected, ignore=ignore))
    return result


def compose_label(
    geobox: GeoBox,
    outlines: MGRSInventory,
    ignore: Sequence[MGRSInventory],
    domain: MGRSInventory | None,
) -> NDArray[np.uint8]:
    """Burn one label mask: ignore first, then the outlines, then outside the domain."""
    mask = outlines.rasterize(geobox, ignore=ignore, ignore_value=IGNORE_VALUE)
    if domain is not None:
        mask[domain.rasterize(geobox) == 0] = IGNORE_VALUE
    return mask


def label_tags(source: LabelSource, *, year: int | None, ignore_buffer_m: float) -> dict[str, str]:
    """GeoTIFF tags describing how a mask was made."""
    tags = {
        "INVENTORY": source.key,
        "IGNORE_VALUE": str(IGNORE_VALUE),
        "DOMAIN": source.domain or "none",
    }
    if year is not None:
        tags |= {
            "ACQUISITION_YEAR": str(year),
            "YEAR_RULE": YEAR_RULE,
            "IGNORE_BUFFER_M": str(ignore_buffer_m),
        }
    return tags


def write_label_masks(
    labels: YearLabels | MGRSInventory,
    out_dir: Path | UPath,
    source: LabelSource,
    *,
    domain: MGRSInventory | None,
    ignore_buffer_m: float,
) -> int:
    """Rasterize one inventory selection into per-tile label COGs; return the tile count.

    Tiles are those intersecting the (buffered) outlines, so a tile selected
    only through the buffer holds no glacier pixel, only background and ignore.
    """
    if isinstance(labels, YearLabels):
        year, outlines, ignore = labels.year, labels.outlines, labels.ignore
    else:
        year, outlines, ignore = None, labels, list[MGRSInventory]()
    tiles = outlines.intersecting_tiles()
    logger.info(
        "Rasterizing %s%s for %d tiles", source.key, f" {year}" if year else "", len(tiles)
    )
    suffix = "" if year is None else f"_{year}"
    tags = label_tags(source, year=year, ignore_buffer_m=ignore_buffer_m)
    for tile in tiles:
        geobox = MGRSGeoBox.from_mgrs(tile)
        mask = compose_label(geobox, outlines, ignore, domain)
        out_path = out_dir / f"LABELS_{source.key}{suffix}_{tile}.tif"
        profile = COG_PROFILE | {
            "dtype": "uint8",
            "count": 1,
            "nodata": IGNORE_VALUE,
            "width": geobox.shape.x,
            "height": geobox.shape.y,
            "crs": geobox.crs,
            "transform": geobox.affine,
        }
        write_cog(mask[np.newaxis], out_path, profile, band_names=["glacier_mask"], tags=tags)
        logger.info(
            "%s: %.2f%% glacier, %.2f%% ignore",
            out_path.name,
            np.count_nonzero(mask == 1) / mask.size * 100,
            np.count_nonzero(mask == IGNORE_VALUE) / mask.size * 100,
        )
    return len(tiles)
