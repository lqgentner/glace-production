"""Swisstopo swissBOUNDARIES3D dataset."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast, override

from freezebase.pandas_utils import clean_names
from freezebase.vectools import write_parquet
from freezebase.vectordata import DatasetMetadata, GeoVectorData
import geopandas as gpd
import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

SWISS_BOUNDARIES_VERSION = "2025-04"
SWISS_BOUNDARIES_DATA_NAME = f"swissboundaries3d_{SWISS_BOUNDARIES_VERSION}"
SWISS_BOUNDARIES_URL_BASE = "https://data.geo.admin.ch/ch.swisstopo.swissboundaries3d/"


class SwisstopoBoundaries(GeoVectorData):
    """
    Swisstopo swissBOUNDARIES3D dataset.

    swissBOUNDARIES3D contains all administrative units and national boundaries of Switzerland
    and the Principality of Liechtenstein in vector form.
    """

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        """
        Initialize the SwissTopo Boundaries 3D dataset.

        Parameters
        ----------
        cache_dir : str or Path or None, default: None
            The root directory where the dataset is stored.
        """
        super().__init__(cache_dir)
        self.version = SWISS_BOUNDARIES_VERSION

    @property
    @override
    def metadata(self) -> DatasetMetadata:
        """Metadata container for dataset attribution and licensing."""
        # The provided URL contains both the version and the name
        source_zip_name = (
            f"{SWISS_BOUNDARIES_DATA_NAME}/"
            f"swissboundaries3d_{SWISS_BOUNDARIES_VERSION}_2056_5728.shp.zip"
        )
        return DatasetMetadata(
            name=f"Swiss Boundaries 3D ({self.version})",
            source_url=SWISS_BOUNDARIES_URL_BASE + source_zip_name,
            attribution="swisstopo",
            license="Free Geodata",
            license_url="https://www.swisstopo.admin.ch/en/terms-of-use-free-geodata-and-geoservices",
            additional_fields={"version": self.version},
        )

    @property
    @override
    def raw_path(self) -> Path:
        """Path to the raw downloaded data (the ZIP file)."""
        return self.cache_dir / f"raw/swisstopo_boundaries/{SWISS_BOUNDARIES_DATA_NAME}.shp.zip"

    @property
    @override
    def processed_path(self) -> Path:
        """Path to the processed data (GeoParquet)."""
        return (
            self.cache_dir / f"processed/swisstopo_boundaries/{SWISS_BOUNDARIES_DATA_NAME}.parquet"
        )

    @override
    def _prepare(self) -> None:
        """Extract the ZIP, clean columns, combine shapefiles, and save as GeoParquet."""
        logger.info(
            "Processing data and saving '%s' to '%s'.",
            self.processed_path.name,
            self.processed_path.parent,
        )

        # List of internal shapefile paths to load and combine
        shapefile_names = [
            "swissBOUNDARIES3D_1_5_TLM_LANDESGEBIET.shp",
            "swissBOUNDARIES3D_1_5_TLM_KANTONSGEBIET.shp",
        ]

        gdfs_to_combine = []
        zip_path_prefix = f"zip://{self.raw_path}/"

        for shp_name in shapefile_names:
            internal_shp_path = zip_path_prefix + shp_name
            try:
                # geopandas can read directly from a zipped shapefile
                logger.info("Reading shapefile: %s", shp_name)
                gdf = gpd.read_file(internal_shp_path)
                gdfs_to_combine.append(gdf)
            except Exception as e:
                msg = f"Failed to read data from the downloaded ZIP for: {shp_name}"
                raise OSError(msg) from e

        if not gdfs_to_combine:
            msg = "No GeoDataFrames were loaded from the ZIP file."
            raise ValueError(msg)

        logger.info("Combining %d GeoDataFrames...", len(gdfs_to_combine))

        cols_to_keep = ["objektart", "name", "geometry"]

        boundaries = (
            gpd.GeoDataFrame(pd.concat(gdfs_to_combine, ignore_index=True))
            .pipe(clean_names)
            .filter(cols_to_keep)
            .assign(objektart=lambda df: df["objektart"].astype("category"))
        )
        # assign is inherited from pandas, which types it as returning a DataFrame;
        # geopandas returns the GeoDataFrame it was called on.
        write_parquet(cast("gpd.GeoDataFrame", boundaries), self.processed_path)
