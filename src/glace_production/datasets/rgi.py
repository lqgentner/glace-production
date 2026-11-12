"""Randolph Glacier Inventory dataset."""

from __future__ import annotations

from io import BytesIO
import logging
import re
from typing import TYPE_CHECKING, Literal, get_args, override
from zipfile import ZipFile

from freezebase.download import HTTPDownloader
from freezebase.utils import format_valid_options, get_credentials_from_env
from freezebase.vectools import drop_z_if_zero, write_parquet
from freezebase.vectordata import DatasetMetadata, GeoVectorData
import geopandas as gpd
import pandas as pd
from pandas.api.types import CategoricalDtype

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

type RGIProduct = Literal["G", "C", "I", "L"]
RGI_PRODUCT_DESC = {
    "G": "Glacier",
    "C": "Complex",
    "I": "Intersects",
    "L": "Centerlines",
}
# fmt: off
type RGIRegion = Literal[
    "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
    "11", "12", "13", "14", "15", "16", "17", "18", "19",
]
"""First-order RGI region codes."""
type RGISubregion = Literal[
    "01-01", "01-02", "01-03", "01-04", "01-05", "01-06", "02-01", "02-02", "02-03", "02-04",
    "02-05", "03-01", "03-02", "03-03", "03-04", "03-05", "03-06", "03-07", "04-01", "04-02",
    "04-03", "04-04", "04-05", "04-06", "04-07", "04-08", "04-09", "05-01", "06-01", "07-01",
    "07-02", "08-01", "08-02", "08-03", "09-01", "09-02", "09-03", "10-01", "10-02", "10-03",
    "10-04", "10-05", "10-06", "10-07", "11-01", "11-02", "12-01", "12-02", "13-01", "13-02",
    "13-03", "13-04", "13-05", "13-06", "13-07", "13-08", "13-09", "14-01", "14-02", "14-03",
    "15-01", "15-02", "15-03", "16-01", "16-02", "16-03", "16-04", "17-01", "17-02", "18-01",
    "19-01", "19-02", "19-03", "19-04", "19-05", "19-11", "19-12", "19-13", "19-14", "19-15",
    "19-16", "19-17", "19-18", "19-19", "19-20", "19-21", "19-22", "19-24",
]
"""Second-order RGI region codes, as ``{region}-{subregion}``."""
# fmt: on
RGI_REGIONS: frozenset[str] = frozenset(get_args(RGIRegion.__value__))
RGI_SUBREGIONS: frozenset[str] = frozenset(get_args(RGISubregion.__value__))
ROW_GROUP_SIZE = 10_000
# C1 control characters: never in a real name, left by bytes decoded with the wrong code page
_C1_CONTROLS = re.compile("[\x80-\x9f]")


def repair_name(name: str) -> str:
    r"""
    Undo the two encoding errors in RGI 7.0 glacier names.

    The shapefiles declare ISO-8859-1, but some GLIMS submissions stored UTF-8
    ('AbbÃ© Glacier' for 'Abbé Glacier') and one stored DOS code page 437
    ('Mont V\x82lan' for 'Mont Vélan'). Correct names pass through unchanged.
    """
    try:
        return name.encode("latin-1").decode("utf-8")
    except UnicodeError:
        pass
    if _C1_CONTROLS.search(name):
        return name.encode("latin-1").decode("cp437")
    return name


class RGI7(GeoVectorData):
    """
    Randolph Glacier Inventory Version 7.0.

    A Dataset of Global Glacier Outlines.

    Dataset:
    > RGI Consortium. (2023). Randolph Glacier Inventory - A Dataset of Global Glacier
    Outlines, Version 7 [Dataset]. NASA National Snow and Ice Data Center Distributed Active
    Archive Center. https://doi.org/10.5067/F6JMOVY5NAVZ

    User guide:
    > Maussion, F., Hock, R., Paul, F., Raup, B., Rastner, P., Zemp, M., Andreassen, L.,
    Barr, I., Bolch, T., Kochtitzky, W., McNabb, R., & Tielidze, L. (2023).
    The Randolph Glacier Inventory version 7.0 User guide. Zenodo.
    https://doi.org/10.5281/ZENODO.8362857
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        *,
        product: RGIProduct = "G",
        region: RGIRegion | RGISubregion | None = None,
    ) -> None:
        """
        Initialize the Randolph Glacier Inventory Version 7.0.

        Parameters
        ----------
        cache_dir : str or Path or None, default: None
            The root directory where the dataset is stored. Defaults to the 'DEEPFREEZER_CACHE'
            environment variable or the system cache directory (in that order).
        product : {"G", "C", "I", "L"}, default: "G"
            The RGI product type
        region : str or None, default: None
            Read only one RGI region (``"11"``, Central Europe) or subregion (``"11-01"``,
            Alps). ``None`` reads the whole world. The cache stays global either way.

        """
        super().__init__(cache_dir)
        # Validate product
        try:
            RGI_PRODUCT_DESC[product]
        except KeyError as e:
            valid_values_bulleted = format_valid_options(RGI_PRODUCT_DESC)
            msg = f"Invalid product: '{product}'. Use one of:\n{valid_values_bulleted}"
            raise ValueError(msg) from e
        self.product = product
        if region is not None and region not in RGI_REGIONS | RGI_SUBREGIONS:
            msg = f"Invalid RGI region: '{region}'. Use a region ('11') or subregion ('11-01')."
            raise ValueError(msg)
        self.region = region

    @property
    @override
    def metadata(self) -> DatasetMetadata:
        """Metadata container for dataset attribution and licensing."""
        return DatasetMetadata(
            name="Randolph Glacier Inventory",
            source_url=f"https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0770_rgi_v7/global_files/RGI2000-v7.0-{self.product}-global.zip",
            attribution="RGI 7.0 Consortium, 2023",
            license="CC BY 4.0",
            license_url="https://creativecommons.org/licenses/by/4.0/",
            version="7.0",
            doi="https://doi.org/10.5067/f6jmovy5navz",
            additional_fields={
                "product": RGI_PRODUCT_DESC[self.product],
                "citation": (
                    "RGI 7.0 Consortium. (2023). Randolph Glacier Inventory - A Dataset of "
                    "Global Glacier Outlines, Version 7.0 [Dataset]. NASA National Snow and "
                    "Ice Data Center Distributed Active Archive Center. "
                    "https://doi.org/10.5067/F6JMOVY5NAVZ"
                ),
                # The source URL needs an Earthdata login
                "dataset_url": "https://doi.org/10.5067/f6jmovy5navz",
            },
        )

    @override
    def _load_data(self) -> gpd.GeoDataFrame:
        """Load the processed data, reading only the row groups of the selected region."""
        if self.region is None:
            return gpd.read_parquet(self.processed_path)
        column = "o1region" if self.region in RGI_REGIONS else "o2region"
        return gpd.read_parquet(self.processed_path, filters=[(column, "==", self.region)])

    @property
    @override
    def raw_path(self) -> Path:
        """Path to the raw downloaded data."""
        return self.cache_dir / f"raw/glaciers/RGI2000-v7.0-{self.product}-global.zip"

    @property
    @override
    def processed_path(self) -> Path:
        """Path to the processed data."""
        return self.cache_dir / f"processed/glaciers/RGI2000-v7.0-{self.product}-global.parquet"

    @override
    def _download(self) -> None:
        """Download the dataset. Requires NASA Earthdata (EOSDIS) credentials."""
        url = self.metadata.source_url
        save_dir = self.raw_path.parent
        filename = self.raw_path.name

        # Get credentials
        try:
            auth = get_credentials_from_env("EOSDIS_USERNAME", "EOSDIS_PASSWORD")
        except KeyError as e:
            logger.warning(
                "%s Requests will attempt to get credentials from the '.netrc' file.",
                str(e),
            )
            auth = None

        downloader = HTTPDownloader(auth=auth, trusted_hosts=["urs.earthdata.nasa.gov"])
        downloader(url=url, save_dir=save_dir, filename=filename)

    @override
    def _prepare(self) -> None:
        """Extract the ZIP and merge the RGI regions into one GeoParquet."""
        logger.info(
            "Processing data and saving '%s' to '%s'.",
            self.processed_path.name,
            self.processed_path.parent,
        )
        # The downloaded file is a nested ZIP:
        # Every RGI region is saved as separate zipped shapefile
        with ZipFile(self.raw_path) as outer_zip:
            # Open nested ZIPs (One ZIP per RGI region)
            region_zips = sorted(f for f in outer_zip.namelist() if f.endswith(".zip"))
            # Combine all regions in one DataFrame (cast to GeoDataFrame for the type checker)
            rgi_world = gpd.GeoDataFrame(
                pd.concat(gpd.read_file(BytesIO(outer_zip.read(f))) for f in region_zips)
                # Make RGI ID the index
                .set_index("rgi_id", drop=True)
                .sort_index(),
            )
            if any(rgi_world.has_z):
                # Drop empty Z axis
                rgi_world = drop_z_if_zero(rgi_world)
            if "glac_name" in rgi_world.columns:
                rgi_world["glac_name"] = rgi_world["glac_name"].map(
                    repair_name, na_action="ignore"
                )

            # Convert column dtypes
            ordered_cat = CategoricalDtype(ordered=True)
            dtype_map: dict[str, str | CategoricalDtype] = {
                "o1region": ordered_cat,
                "o2region": ordered_cat,
                "anlys_id": "uint32",
                "subm_id": "uint16",
                "src_date": "datetime64[us]",
                "utm_zone": "uint8",
                "is_rgi6": "bool",
                "primeclass": "uint8",
                "conn_lvl": "uint8",
                "surge_type": "uint8",
                "term_type": "uint8",
                "lmax_m": "uint32",
                "dem_source": "category",
                # Centerlines dtypes
                "segment_id": "uint8",
                "is_main": "bool",
                "outflow_id": "int8",
                "strahler_n": "uint8",
                "length_m": "uint32",
            }
            dtype_map = {k: v for k, v in dtype_map.items() if k in rgi_world.columns}
            float_cols = rgi_world.select_dtypes(include="float64").columns
            dtype_map.update(dict.fromkeys(float_cols, "float32"))
            rgi_world = rgi_world.astype(dtype_map)
            # Save as GeoParquet in small row groups: sorting by RGI ID keeps each region
            # contiguous, so a region read skips the other groups by their statistics.
            write_parquet(rgi_world, self.processed_path, row_group_size=ROW_GROUP_SIZE)
