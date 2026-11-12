"""Perimeter of the Alpine Convention."""

from __future__ import annotations

import logging
import shutil
from typing import TYPE_CHECKING, override

from freezebase.download import HTTPDownloader
from freezebase.utils import file_sha256
from freezebase.vectools import write_parquet
from freezebase.vectordata import DatasetMetadata, GeoVectorData
import geopandas as gpd
import requests

from glace_production.config import settings

if TYPE_CHECKING:
    from pathlib import Path

    from upath import UPath

logger = logging.getLogger(__name__)

MIRROR_SHA256 = "1d1aaaf141298c5f8c3440f11c4e7da0ccfd0ce5ca818db379d665cfc17c8285"
"""Digest of the atlas export of 2026-09-18, kept beside the project data.

Only the mirror copy is pinned: the atlas generates its SHAPE-ZIP per request,
so the bytes it serves are never the same twice.
"""


class AlpineConventionPerimeter(GeoVectorData):
    """
    Perimeter of the Alpine Convention, version 2025.

    The perimeter defines the area of application of the Alpine Convention. It was prepared from
    the specification in the Annex to the Framework Convention, which remains the only legally
    binding definition of that territory. Updated in 2025 from the previous version (2018).

    More info in the
    [Alpine Convention atlas](https://www.atlas.alpconv.org/catalogue/uuid/098dfdfc-4ab4-11f0-abce-0050568ea38e).

    Dataset:
    > Eurac Research & Permanent Secretariat of the Alpine Convention. (2025). Alpine Convention
    perimeter (Version 2025) [Data set]. Permanent Secretariat of the Alpine Convention.
    https://doi.org/10.48784/VK93-8P54

    The atlas is tried first. When it cannot be reached, the archive is copied from a mirror
    beside the project data, taken from the atlas on 2026-09-18.
    """

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        """
        Initialize the Perimeter of the Alpine Convention.

        Parameters
        ----------
        cache_dir : str or Path or None, default: None
            The root directory where the dataset is stored. Defaults to the 'DEEPFREEZER_CACHE'
            environment variable or the system cache directory (in that order).

        """
        super().__init__(cache_dir)

    @property
    @override
    def metadata(self) -> DatasetMetadata:
        """Metadata container for dataset attribution and licensing."""
        return DatasetMetadata(
            name="Alpine Convention perimeter",
            source_url="https://www.atlas.alpconv.org/geoserver/ows?service=WFS&version=1.0.0&request=GetFeature&typename=geonode%3AAlpine_Convention_Perimeter_2025&outputFormat=SHAPE-ZIP&srs=EPSG%3A3035&format_options=charset%3AUTF-8",
            attribution="Eurac Research & Permanent Secretariat of the Alpine Convention",
            license="CC BY 4.0",
            license_url="https://creativecommons.org/licenses/by/4.0/",
            version="2025",
            doi="https://doi.org/10.48784/VK93-8P54",
            additional_fields={
                "citation": (
                    "Eurac Research & Permanent Secretariat of the Alpine Convention. (2025). "
                    "Alpine Convention perimeter (Version 2025) [Data set]. Permanent Secretariat "
                    "of the Alpine Convention. https://doi.org/10.48784/VK93-8P54"
                ),
            },
        )

    @property
    @override
    def raw_path(self) -> Path:
        """Path to the raw downloaded data."""
        return self.cache_dir / "raw/regions/alpineconvention-2025.zip"

    @property
    @override
    def processed_path(self) -> Path:
        """Path to the processed data."""
        return self.cache_dir / "processed/regions/alpineconvention-2025.parquet"

    @override
    def _download(self) -> None:
        """Download from the atlas, or copy the mirror when the atlas fails."""
        try:
            with HTTPDownloader() as downloader:
                downloader(
                    url=self.metadata.source_url,
                    save_dir=self.raw_path.parent,
                    filename=self.raw_path.name,
                )
        except requests.RequestException as error:
            mirror = settings.data_dir / "regions" / self.raw_path.name
            logger.warning(
                "Origin of '%s' unreachable (%s); copying the mirror from '%s'.",
                self.raw_path.name,
                error,
                mirror.parent,
            )
            self._copy_mirror(mirror)

    def _copy_mirror(self, mirror: Path | UPath) -> None:
        """Copy the mirrored archive into place, refusing a copy that does not match the pin."""
        self.raw_path.parent.mkdir(parents=True, exist_ok=True)
        staging = self.raw_path.with_suffix(".part")
        with mirror.open("rb") as source, staging.open("wb") as target:
            shutil.copyfileobj(source, target)
        actual = file_sha256(staging)
        if actual.lower() != MIRROR_SHA256:
            staging.unlink()
            msg = (
                f"The mirror copy of '{self.raw_path.name}' at '{mirror}' does not match its "
                f"pinned digest: expected {MIRROR_SHA256}, got {actual}."
            )
            raise ValueError(msg)
        staging.replace(self.raw_path)

    @override
    def _prepare(self) -> None:
        """Extract the ZIP and save as GeoParquet."""
        logger.info(
            "Processing data and saving '%s' to '%s'.",
            self.processed_path.name,
            self.processed_path.parent,
        )
        perimeter = gpd.read_file(self.raw_path)
        write_parquet(perimeter, self.processed_path)
