"""Windows of the example figures, shared by the paper-plot notebooks.

`mapping-examples.qmd` renders the mapping examples and `lrw-effect.qmd` the local
resolution weighting window; `dataset-overview.qmd` marks all of them on the Alps-wide
map. Keeping them here is what keeps the figures in agreement.
"""

from __future__ import annotations

from typing import TypedDict

REGION_EPSG = 32632
"""UTM 32N: the windows are on the 10 m grid of the zone's composites."""

WIDTH_M = 10_000
HEIGHT_M = 10_000


class Region(TypedDict):
    """A mapping example, selected by the lower-left corner of its window."""

    name: str
    year: int
    easting: float
    northing: float


REGIONS: list[Region] = [
    {"name": "Grosser Aletschgletscher", "year": 2023, "easting": 420_500, "northing": 5_139_000},
    {"name": "Gornergletscher", "year": 2023, "easting": 404_000, "northing": 5_087_000},
    {"name": "Unteraargletscher", "year": 2024, "easting": 434_000, "northing": 5_153_000},
]

LRW_REGION: Region = {
    "name": "Gurgler Ferner",
    "year": 2023,
    "easting": 647_000,
    "northing": 5_179_000,
}
"""The local resolution weighting window, outside the Swiss composites on purpose."""


def region_bounds(region: Region) -> tuple[float, float, float, float]:
    """Window of ``region`` as (left, bottom, right, top) in UTM 32N metres."""
    x0, y0 = region["easting"], region["northing"]
    return (x0, y0, x0 + WIDTH_M, y0 + HEIGHT_M)
