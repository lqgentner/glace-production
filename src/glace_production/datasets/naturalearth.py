"""Natural Earth dataset."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, override

from freezebase.pandas_utils import clean_names
from freezebase.utils import format_valid_options
from freezebase.vectools import write_parquet
from freezebase.vectordata import DatasetMetadata, GeoVectorData
import geopandas as gpd
import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

# logger = get_logger()
logger = logging.getLogger(__name__)

type NEResolution = Literal["10m", "50m", "110m"]
type NELakesSupplement = Literal["europe", "north_america"]
NE_RESOLUTION_DESC = {
    "10m": "1:10m",
    "50m": "1:50m",
    "110m": "1:110m",
}


class NaturalEarthCountries(GeoVectorData):
    """
    Natural Earth: Admin 0 - Countries.

    Natural Earth is a public domain map dataset available at
    1:10m, 1:50m, and 1:110 million scales.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        *,
        resolution: NEResolution = "50m",
    ) -> None:
        """
        Initialize the Natural Earth: Admin 0 - Countries dataset.

        Parameters
        ----------
        cache_dir : str or Path or None, default: None
            The root directory where the dataset is stored. Defaults to the 'DEEPFREEZER_CACHE'
            environment variable or the system cache directory (in that order).
        resolution : {"10m", "50m", 110m"}, default: "50m"
            The spatial resolution.

        """
        super().__init__(cache_dir)
        # Validate resolution
        try:
            NE_RESOLUTION_DESC[resolution]
        except KeyError as e:
            valid_values_bulleted = format_valid_options(NE_RESOLUTION_DESC)
            msg = f"Invalid product: '{resolution}'. Use one of:\n{valid_values_bulleted}"
            raise ValueError(msg) from e
        self.resolution = resolution

    @property
    @override
    def metadata(self) -> DatasetMetadata:
        """Metadata container for dataset attribution and licensing."""
        return DatasetMetadata(
            name="Natural Earth: Admin 0 - Countries",
            source_url=f"https://naciscdn.org/naturalearth/{self.resolution}/cultural/ne_{self.resolution}_admin_0_countries.zip",
            attribution="Natural Earth",
            license="CC0 1.0 Public Domain",
            license_url="https://www.naturalearthdata.com/about/terms-of-use/",
            doi="https://doi.org/10.14714/CP64.148",
            additional_fields={"resolution": NE_RESOLUTION_DESC[self.resolution]},
        )

    @property
    @override
    def raw_path(self) -> Path:
        """Path to the raw downloaded data."""
        return self.cache_dir / f"raw/countries/ne_{self.resolution}_admin_0_countries.zip"

    @property
    @override
    def processed_path(self) -> Path:
        """Path to the processed data."""
        return (
            self.cache_dir / f"processed/countries/ne_{self.resolution}_admin_0_countries.parquet"
        )

    @override
    def _prepare(self) -> None:
        """Extract the ZIP, select most important columns and save as GeoParquet."""
        logger.info(
            "Processing data and saving '%s' to '%s'.",
            self.processed_path.name,
            self.processed_path.parent,
        )
        ne = gpd.read_file(self.raw_path)
        # Lower-case column names
        ne = clean_names(ne)
        # Only retain important columns
        cols_to_keep = [
            "name",
            "name_long",
            "name_sort",
            "abbrev",
            "iso_a2",
            "iso_a3",
            "type",
            "geometry",
        ]
        ne = (
            ne.filter(items=cols_to_keep)
            # Replace missing string value with NA
            .replace("-99", pd.NA)
            # Convert str columns to categorical to save space
            .assign(type=lambda df: df["type"].astype("category"))
        )
        write_parquet(ne, self.processed_path)


class NaturalEarthLakes(GeoVectorData):
    """
    Natural Earth: Lakes and reservoirs.

    The physical layer of the major lakes, available at 1:10m, 1:50m and 1:110 million
    scales. At 1:10m, the regional supplements (``"europe"``, ``"north_america"``) add the
    smaller lakes of that continent. Natural Earth is public domain and needs no attribution.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        *,
        resolution: NEResolution = "10m",
        supplement: NELakesSupplement | None = None,
    ) -> None:
        """
        Initialize the Natural Earth: Lakes dataset.

        Parameters
        ----------
        cache_dir : str or Path or None, default: None
            The root directory where the dataset is stored. Defaults to the 'DEEPFREEZER_CACHE'
            environment variable or the system cache directory (in that order).
        resolution : {"10m", "50m", 110m"}, default: "10m"
            The spatial resolution.
        supplement : {"europe", "north_america"} or None, default: None
            A regional supplement of smaller lakes instead of the global layer. Only
            published at 1:10m.

        """
        super().__init__(cache_dir)
        if resolution not in NE_RESOLUTION_DESC:
            valid_values_bulleted = format_valid_options(NE_RESOLUTION_DESC)
            msg = f"Invalid resolution: '{resolution}'. Use one of:\n{valid_values_bulleted}"
            raise ValueError(msg)
        if supplement is not None and resolution != "10m":
            msg = f"The '{supplement}' lakes supplement is only published at 1:10m."
            raise ValueError(msg)
        self.resolution = resolution
        self.supplement = supplement

    @property
    def _layer(self) -> str:
        return f"ne_{self.resolution}_lakes" + (f"_{self.supplement}" if self.supplement else "")

    @property
    @override
    def metadata(self) -> DatasetMetadata:
        """Metadata container for dataset attribution and licensing."""
        return DatasetMetadata(
            name="Natural Earth: Lakes",
            source_url=f"https://naciscdn.org/naturalearth/{self.resolution}/physical/{self._layer}.zip",
            attribution="Natural Earth",
            license="CC0 1.0 Public Domain",
            license_url="https://www.naturalearthdata.com/about/terms-of-use/",
            doi="https://doi.org/10.14714/CP64.148",
            additional_fields={"resolution": NE_RESOLUTION_DESC[self.resolution]},
        )

    @property
    @override
    def raw_path(self) -> Path:
        """Path to the raw downloaded data."""
        return self.cache_dir / f"raw/lakes/{self._layer}.zip"

    @property
    @override
    def processed_path(self) -> Path:
        """Path to the processed data."""
        return self.cache_dir / f"processed/lakes/{self._layer}.parquet"

    @override
    def _prepare(self) -> None:
        """Extract the ZIP, select the name columns and save as GeoParquet."""
        logger.info(
            "Processing data and saving '%s' to '%s'.",
            self.processed_path.name,
            self.processed_path.parent,
        )
        lakes = clean_names(gpd.read_file(self.raw_path))
        lakes = lakes.filter(items=["name", "name_alt", "featurecla", "geometry"])
        write_parquet(lakes, self.processed_path)
