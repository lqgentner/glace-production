"""Austrian Glacier Inventory datasets."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, override
from zipfile import ZipFile

from freezebase.vectools import write_parquet
from freezebase.vectordata import DatasetMetadata, GeoVectorData
import geopandas as gpd
import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

GI5_DTYPES = {
    "id": "uint32",
    "year": "uint16",
    "name": "str",
    "data_type": "str",
    "src_date": "str",
    "source": "str",
    "image_ID1": "str",
    "flightstr": "str",
    "state": "category",
    "region": "category",
    "url": "str",
    "analyst": "str",
    "img_qf": "uint8",
    "outline_qf": "uint8",
    "crevs": "uint8",
    "debris": "uint8",
    "comment": "str",
    "data_type2": "str",
    "ROGI": "uint8",
    "van_fragm": "uint8",
    "nr_frag": "uint8",
    "median_elev": "float32",
    "max_elev": "float32",
    "min_elev": "float32",
    "area": "uint32",
}
"""Column types of the merged GI5 outlines, in the order of the source files."""


class AustrianGI5(GeoVectorData):
    """
    5th Austrian Glacier Inventory (GI5).

    This dataset contains the glacier outlines of the 5th Austrian Glacier Inventory (GI5) for the
    years 2021-2023 as an updated version of the 3rd Austrian Glacier Inventory (GI3, Fischer et
    al. 2015). The outlines as Vector files in GeoJSON format are published for 20 subregions. The
    outlines were mapped using high resolution, multi-temporal orthophotos and satellite images,
    hillshades and elevation change data derived from digital elevation models depending on
    regional data availability.

    The subregions are merged into a single GeoParquet in the original CRS (EPSG:31287), indexed
    by glacier ID. The `state` column is harmonised to the German state names. Glaciers spanning
    several states keep a comma-separated list, including "Italien" for a transboundary glacier.
    `src_date` stays a string, since the source mixes full dates (`YYYYMMDD`), dates with an
    unknown day (`YYYYMMXX`) and bare years (`YYYY`).

    Paper:
    > Hartl, L., Abermann, J., Akgün, A., Bertolotti, G., Bolch, T., Conzelmann, S.,
    Diaconu, C.-A., Hansche, I., Hartig, A., Haut, A., Helfricht, K., Hynek, B.,
    Kaucher, M. S., Kellerer-Pirklbauer, A., Kogel, A. C., Krippes, J., Lauria, M. V.,
    Mayer, C., Otto, J.-C., … Zandler, H. (2026). On thin glacial ice: New Austrian Glacier
    Inventory shows accelerating glacier shrinkage and 31 % area loss within two decades.
    The Cryosphere, 20(9), 5157-5180. https://doi.org/10.5194/tc-20-5157-2026
    """

    @property
    @override
    def metadata(self) -> DatasetMetadata:
        """Metadata container for dataset attribution and licensing."""
        return DatasetMetadata(
            name="Austrian Glacier Inventory 5",
            source_url="https://download.pangaea.de/dataset/991106/files/GI5_vector_files.zip",
            attribution="Hartig et al., 2026",
            license="CC BY 4.0",
            license_url="https://creativecommons.org/licenses/by/4.0/",
            doi="https://doi.org/10.1594/PANGAEA.991106",
            additional_fields={
                "dataset_url": "https://doi.org/10.1594/PANGAEA.991106",
                "citation": (
                    "Hartig, A., Akguen, A. M., Bertolotti, G., Bolch, T., Conzelmann, S., "
                    "Hartl, L., Haut, A., Helfricht, K., Hynek, B., Kaucher, M. S., "
                    "Kellerer-Pirklbauer, A., Kogel, A. C., Krippes, J., Lauria, M. V., "
                    "Mayer, C., Otto, J.-C., Prinz, R., Proelß, S., Seiser, B., … Zandler, H. "
                    "(2026). The Austrian Glacier Inventory GI 5, 2021-2023, as vector files "
                    "[Dataset]. PANGAEA. https://doi.org/10.1594/PANGAEA.991106"
                ),
            },
        )

    @property
    @override
    def raw_path(self) -> Path:
        """Path to the raw downloaded data."""
        return self.cache_dir / "raw/glaciers/GI5_vector_files.zip"

    @property
    @override
    def processed_path(self) -> Path:
        """Path to the processed data."""
        return self.cache_dir / "processed/glaciers/Austrian-GI5.parquet"

    @override
    def _prepare(self) -> None:
        """Merge the regional GeoJSONs of the ZIP into one GeoParquet."""
        logger.info(
            "Processing data and saving '%s' to '%s'.",
            self.processed_path.name,
            self.processed_path.parent,
        )
        # The ZIP holds one GeoJSON per subregion, all in EPSG:31287
        with ZipFile(self.raw_path) as zf:
            region_files = sorted(f for f in zf.namelist() if f.endswith(".geojson"))
        gi5 = gpd.GeoDataFrame(
            pd.concat(gpd.read_file(f"/vsizip/{self.raw_path}/{f}") for f in region_files)
            # Empty in every region
            .drop(columns=["source_2", "src_date_2"])
            # Treat blank text as missing, and strip stray whitespace and newlines
            .replace(r"^\s*$", None, regex=True)
            .replace(r"^\s+|\s+$", "", regex=True)
            .assign(
                # The regions spell the states in German or English: use the German names
                state=lambda df: df["state"].replace(
                    {r"Carin?thia": "Kärnten", r"(East )?Tyrol": "Tirol"},
                    regex=True,
                ),
                # Number of vanishing fragments, left empty where there are none
                van_fragm=lambda df: pd.to_numeric(df["van_fragm"]).fillna(0),
            )
            .astype(GI5_DTYPES)
            .set_index("id")
            .sort_index(),
        )

        write_parquet(gi5, self.processed_path)
