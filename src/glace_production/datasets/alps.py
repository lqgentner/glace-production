"""Frank Paul's Glacier inventory of the Alps dataset."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast, override

from freezebase.pandas_utils import clean_names
from freezebase.vectools import write_parquet
from freezebase.vectordata import DatasetMetadata, GeoVectorData
import geopandas as gpd

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class PaulEtAl2020(GeoVectorData):
    """
    Frank Paul's Glacier inventory of the Alps.

    Alpine-wide glacier inventory with a spatial resolution of 10 m.
    Created using Sentinel-2 scenes from 2015 to 2017 and topographic information from
    ALOS AW3D30 DEM.

    Paper:
    > Paul, F., Rastner, P., Azzoni, R. S., Diolaiuti, G., Fugazza, D., Le Bris, R., Nemec, J.,
    Rabatel, A., Ramusovic, M., Schwaizer, G., & Smiraglia, C. (2020).
    Glacier shrinkage in the Alps continues unabated as revealed by a new glacier inventory from
    Sentinel-2. Earth System Science Data, 12(3), 1805-1821.
    https://doi.org/10.5194/essd-12-1805-2020

    Dataset:
    > Paul, F., Rastner, P., Azzoni, R. S., Diolaiuti, G., Fugazza, D., Le Bris, R., Nemec, J.,
    Rabatel, A., Ramusovic, M., Schwaizer, G., & Smiraglia, C. (2019).
    Glacier inventory of the Alps from Sentinel-2, shape files [Dataset]. PANGAEA.
    https://doi.org/10.1594/PANGAEA.909133
    """

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        """
        Initialize the Frank Paul's Glacier inventory of the Alps dataset.

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
            name="Glacier inventory of the Alps from Sentinel-2",
            source_url="https://store.pangaea.de/Publications/PaulF-etal_2019/c3s_gi_rgi11_s2_2015_v2.zip",
            attribution="Frank Paul et al.",
            license="CC BY 4.0",
            license_url="https://creativecommons.org/licenses/by/4.0/",
            version="2.0",
            doi="https://doi.org/10.5194/essd-12-1805-2020",
            additional_fields={
                "citation": (
                    "Paul, F., Rastner, P., Azzoni, R. S., Diolaiuti, G., Fugazza, D., "
                    "Le Bris, R., Nemec, J., Rabatel, A., Ramusovic, M., Schwaizer, G., "
                    "& Smiraglia, C. (2020). Glacier shrinkage in the Alps continues "
                    "unabated as revealed by a new glacier inventory from Sentinel-2. "
                    "Earth System Science Data, 12(3), 1805\u20131821. "
                    "https://doi.org/10.5194/essd-12-1805-2020"
                ),
                "dataset_url": "https://doi.org/10.1594/PANGAEA.909133",
            },
        )

    @property
    @override
    def raw_path(self) -> Path:
        """Path to the raw downloaded data."""
        return self.cache_dir / "raw/glaciers/alps-2015-paul.zip"

    @property
    @override
    def processed_path(self) -> Path:
        """Path to the processed data."""
        return self.cache_dir / "processed/glaciers/alps-2015-paul.parquet"

    @override
    def _prepare(self) -> None:
        """Extract the ZIP, select most important columns and save as GeoParquet."""
        logger.info(
            "Processing data and saving '%s' to '%s'.",
            self.processed_path.name,
            self.processed_path.parent,
        )
        paul = gpd.read_file(f"{self.raw_path}!c3s_gi_rgi11_s2_2015_v2").pipe(clean_names)
        # read_file is typed as returning a plain DataFrame for its ignore_geometry
        # variant; this call keeps the geometry column.
        write_parquet(cast("gpd.GeoDataFrame", paul), self.processed_path)
