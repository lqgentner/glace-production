"""DeepFreezer datasets."""

from __future__ import annotations

from glace_production.datasets.alpineconvention import AlpineConventionPerimeter
from glace_production.datasets.alps import PaulEtAl2020
from glace_production.datasets.austrian_gi import AustrianGI5
from glace_production.datasets.copernicus import Sentinel2Tiles
from glace_production.datasets.naturalearth import NaturalEarthCountries, NaturalEarthLakes
from glace_production.datasets.rgi import RGI7
from glace_production.datasets.sgi import SGI2016, SGI2023
from glace_production.datasets.swisstopo_boundaries import SwisstopoBoundaries

__all__ = [
    "RGI7",
    "SGI2016",
    "SGI2023",
    "AlpineConventionPerimeter",
    "AustrianGI5",
    "NaturalEarthCountries",
    "NaturalEarthLakes",
    "PaulEtAl2020",
    "Sentinel2Tiles",
    "SwisstopoBoundaries",
]

DATASETS = __all__
"""List of all available dataset classes."""
