"""Swiss Glacier Inventory (SGI) datasets."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, override

from freezebase.utils import format_valid_options
from freezebase.vectools import write_parquet
from freezebase.vectordata import DatasetMetadata, GeoVectorData
import geopandas as gpd

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

type SGIProduct = Literal["G", "D", "I", "L"]
SGI_PRODUCT_DESC = {
    # Descriptions (values) are used for locating the shape files
    "G": "glaciers",
    "D": "debriscover",
    "I": "icedivides",
    "L": "centerlines",
    "P": "glacier_locations",
}


class SGIBase(GeoVectorData):
    """Swiss Glacier Inventory base class."""

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        *,
        product: SGIProduct = "G",
    ) -> None:
        """
        Initialize the Swiss Glacier Inventory.

        Parameters
        ----------
        cache_dir : str or Path or None, default: None
            The root directory where the dataset is stored. Defaults to the 'DEEPFREEZER_CACHE'
            environment variable or the system cache directory (in that order).
        product : {"G", "D", "I", "L"}, default: "G"
            The SGI product type

        """
        super().__init__(cache_dir)
        # Validate product
        try:
            SGI_PRODUCT_DESC[product]
        except KeyError as e:
            valid_values_bulleted = format_valid_options(SGI_PRODUCT_DESC)
            msg = f"Invalid product: '{product}'. Use one of:\n{valid_values_bulleted}"
            raise ValueError(msg) from e
        self.product = product


class SGI2016(SGIBase):
    """
    Swiss Glacier Inventory 2016.

    The Swiss Glacier Inventory 2016 provides glacier outlines (areas), debris cover, ice divides
    and location points of all glaciers in Switzerland referring to the years 2013-2018. The data
    is based on aerial imagery and stereophotogramatic analysis of the Federal office of
    Topography, Switzerland (swisstopo) within the framework of the Topographic landscape Model of
    Switzerland (swissTLM3D). Furthermore the dataset has been revised by Glaciologists of
    VAW-ETH Zurich, University of Zurich and University of Fribourg (Switzerland).

    Paper:
    > Linsbauer, A., Huss, M., Hodel, E., Bauder, A., Fischer, M., Weidmann, Y., Bärtschi, H.,
    & Schmassmann, E. (2021). The New Swiss Glacier Inventory SGI2016:
    From a Topographical to a Glaciological Dataset. Frontiers in Earth Science, 9.
    https://doi.org/10.3389/feart.2021.704189
    """

    @property
    @override
    def metadata(self) -> DatasetMetadata:
        """Metadata container for dataset attribution and licensing."""
        return DatasetMetadata(
            name="Swiss Glacier Inventory 2016",
            source_url="https://doi.glamos.ch/data/inventory/inventory_sgi2016_r2020.zip",
            attribution="Glacier Monitoring Switzerland (GLAMOS)",
            license="CC BY 4.0",
            license_url="https://creativecommons.org/licenses/by/4.0/",
            version="R2020",
            doi="https://doi.org/10.18750/inventory.sgi2016.r2020",
            additional_fields={
                "product": SGI_PRODUCT_DESC[self.product],
                "citation": (
                    "Linsbauer, A., Huss, M., Hodel, E., Bauder, A., Fischer, M., "
                    "Weidmann, Y., Bärtschi, H., & Schmassmann, E. (2021). The New Swiss "
                    "Glacier Inventory SGI2016: From a Topographical to a Glaciological Dataset. "
                    "Frontiers in Earth Science, 9. https://doi.org/10.3389/feart.2021.704189"
                ),
            },
        )

    @property
    @override
    def raw_path(self) -> Path:
        """Path to the raw downloaded data."""
        return self.cache_dir / "raw/glaciers/SGI2016-R2020.zip"

    @property
    @override
    def processed_path(self) -> Path:
        """Path to the processed data."""
        return (
            self.cache_dir
            / f"processed/glaciers/SGI2016-R2020-{SGI_PRODUCT_DESC[self.product]}.parquet"
        )

    @override
    def _prepare(self) -> None:
        """Extract the ZIP, select most important columns and save as GeoParquet."""
        logger.info(
            "Processing data and saving '%s' to '%s'.",
            self.processed_path.name,
            self.processed_path.parent,
        )
        cols_to_drop = ["pk_glacier", "rl_0", "rl_1", "rl_2", "rl_3", "i_code"]

        sgi = gpd.read_file(
            f"{self.raw_path}",
            layer=f"SGI_2016_{SGI_PRODUCT_DESC[self.product]}",
        ).drop(columns=cols_to_drop, errors="ignore")

        write_parquet(sgi, self.processed_path)


class SGI2023(SGIBase):
    """
    Swiss Glacier Inventory 2023.

    The preliminary release R2026, subject to change before the final release.
    """

    @property
    @override
    def metadata(self) -> DatasetMetadata:
        """Metadata container for dataset attribution and licensing."""
        return DatasetMetadata(
            name="Swiss Glacier Inventory 2023",
            source_url="https://doi.glamos.ch/data/inventory/inventory_sgi2023_r2026.zip",
            attribution="Glacier Monitoring Switzerland (GLAMOS)",
            license="CC BY 4.0",
            license_url="https://creativecommons.org/licenses/by/4.0/",
            version="R2026",
            doi="https://doi.org/10.18750/inventory.sgi2023.r2026",
            additional_fields={
                "product": SGI_PRODUCT_DESC[self.product],
                "citation": (
                    "GLAMOS-Glacier Monitoring Switzerland. (2026). Swiss Glacier Inventory 2023 "
                    "[Dataset]. GLAMOS - Glacier Monitoring Switzerland; Laboratory of "
                    "Hydraulics, Hydrology and Glaciology (VAW), Swiss Federal Institute of "
                    "Technology Zurich (ETH Zurich); Department of Geosciences, University of "
                    "Fribourg, Switzerland; Department of Geography, University of Zürich, "
                    "Switzerland. https://doi.org/10.18750/INVENTORY.SGI2023.R2026"
                ),
            },
        )

    @property
    @override
    def raw_path(self) -> Path:
        """Path to the raw downloaded data."""
        return self.cache_dir / "raw/glaciers/SGI_2023.zip"

    @property
    @override
    def processed_path(self) -> Path:
        """Path to the processed data."""
        return (
            self.cache_dir / f"processed/glaciers/SGI2023-{SGI_PRODUCT_DESC[self.product]}.parquet"
        )

    @override
    def _prepare(self) -> None:
        """Extract the ZIP, select most important columns and save as GeoParquet."""
        logger.info(
            "Processing data and saving '%s' to '%s'.",
            self.processed_path.name,
            self.processed_path.parent,
        )
        cols_to_drop = ["pk_glacier", "rl_0", "rl_1", "rl_2", "rl_3", "i_code"]

        # The GeoPackage, not the shapefiles beside it: the debris-cover shapefile of R2026
        # gives each multipart feature the area of a single part.
        sgi = gpd.read_file(
            f"zip://{self.raw_path}!SGI_2023.gpkg",
            layer=f"sgi_2023_{SGI_PRODUCT_DESC[self.product]}",
        ).drop(columns=cols_to_drop, errors="ignore")

        write_parquet(sgi, self.processed_path)
