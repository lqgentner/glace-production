"""Copernicus datasets."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, override

from freezebase.vectools import drop_z_if_zero, simplify_multipolygons, write_parquet
from freezebase.vectordata import DatasetMetadata, GeoVectorData
import geopandas as gpd
from shapely import GeometryCollection, MultiPolygon, Polygon

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class Sentinel2Tiles(GeoVectorData):
    """
    Sentinel-2 Tiling Grid.

    Sentinel-2 Level-1C and Level-2A products are delivered in granules, also called tiles.
    Each tile is a 110 km x 110 km ortho-image projected onto the Universal Transversal Mercator
    (UTM) spatial reference system. These tiles do not change position over time;
    the tiling grid remains constant throughout the mission. Every tile has a unique ID, allowing
    the location of all imagery for a given tile.
    """

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        """
        Initialize the Sentinel-2 Tiling grid.

        Parameters
        ----------
        cache_dir : str or Path or None, default: None
            The root directory where the dataset is stored. Defaults to the 'FREEZEBASE_CACHE'
            environment variable or the system cache directory (in that order).

        """
        super().__init__(cache_dir)

    @property
    @override
    def metadata(self) -> DatasetMetadata:
        """Metadata container for dataset attribution and licensing."""
        return DatasetMetadata(
            name="Sentinel-2 Tiling Grid",
            source_url="https://sentiwiki.copernicus.eu/__attachments/1692737/S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.zip",
            attribution="ESA/Copernicus",
        )

    @property
    @override
    def raw_path(self) -> Path:
        """Path to the raw downloaded data."""
        return self.cache_dir / "raw/optical/s2-tiling-grid.zip"

    @property
    @override
    def processed_path(self) -> Path:
        """Path to the processed data."""
        return self.cache_dir / "processed/optical/s2-tiling-grid.parquet"

    @override
    def _prepare(self) -> None:
        """Extract the ZIP, select most important columns and save as GeoParquet."""
        logger.info(
            "Processing data and saving '%s' to '%s'.",
            self.processed_path.name,
            self.processed_path.parent,
        )

        # The Geometries are of type GeometryCollection.
        # They hold one point (the Polygon center) and one or two polygons
        # We remove the point and convert the geometries to MultiPolygons
        def extract_multipolygon(geom: GeometryCollection) -> MultiPolygon:
            """Extract polygons from a GeometryCollection and returns a MultiPolygon."""
            polys = [g for g in geom.geoms if isinstance(g, Polygon)]
            return MultiPolygon(polys)

        # Extract UTM zone from "Description" column
        def extract_epsg(html_string: str) -> int | None:
            """Extract the EPSG string from the HTML metadata."""
            match = re.search(r"<b>EPSG</b>.*?<font[^>]*>([^<]+)</font>", html_string)
            return int(match.group(1)) if match else None

        cols_to_keep = ["tile_id", "utm_zone", "geometry"]

        tiles = (
            gpd.read_file(self.raw_path, layer="Features")
            .assign(geometry=lambda df: df["geometry"].map(extract_multipolygon))
            .pipe(drop_z_if_zero)
            .pipe(simplify_multipolygons)
            .assign(utm_zone=lambda df: df["description"].map(extract_epsg))
            .rename(columns={"Name": "tile_id"})
            .filter(items=cols_to_keep)
        )
        write_parquet(tiles, self.processed_path)
