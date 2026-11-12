"""Glacier inventories as vector overlays for the GLACE Viewer."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from freezebase.utils import depends_on_optional

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    import geopandas as gpd

logger = logging.getLogger(__name__)

INVENTORY_OVERLAYS: dict[str, dict[str, Any]] = {
    "agi5": {
        "title": None,
        "color": "#e41a1c",
        "name_column": "name",
        "year_column": "year",
    },
    "pauletal2020": {
        "title": "Alpine Glacier Inventory",
        "color": "#377eb8",
        "name_column": None,
        "year_column": "date",
        # The inventory ships no names, so its own identifier is what a reader
        # can look a feature up by.
        "extra_columns": {"glacier_nr": "glacier_nr"},
    },
    "rgi7": {
        "title": "Randolph Glacier Inventory 7.0",
        "color": "#4daf4a",
        "name_column": "glac_name",
        "year_column": "src_date",
        "extra_columns": {"rgi_id": "rgi_id"},
    },
    "sgi2016": {
        "title": None,  # taken from the dataset's own metadata
        "color": "#984ea3",
        "name_column": "name",
        "year_column": "year_acq",
    },
    "sgi2023": {
        "title": None,
        "color": "#ff7f00",
        "name_column": "name",
        "year_column": "year_acq",
    },
}
"""How each glacier inventory is drawn as a vector overlay, keyed by dataset."""


@depends_on_optional("pandas", install_hint="pip install pandas")
def export_inventory(
    frame: gpd.GeoDataFrame,
    out_path: Path,
    *,
    name_column: str | None,
    year_column: str,
    extra_columns: Mapping[str, str] | None = None,
    simplify: float = 10.0,
    precision: int = 5,
) -> tuple[int, int, int]:
    """Write a glacier inventory as a trimmed, simplified GeoJSON overlay.

    Each feature keeps ``name`` and ``year`` plus whatever ``extra_columns``
    names; everything else is dropped, since the overlay exists only to be
    looked at. Geometries
    are simplified in a metric CRS and coordinates rounded. The default 10 m
    tolerance is finer than the 26.5 m the raster layers resolve at their
    maximum zoom, so the outlines stay crisper than the imagery underneath.

    Parameters
    ----------
    frame
        Inventory as a :class:`geopandas.GeoDataFrame`.
    out_path
        Destination ``.geojson`` file.
    name_column
        Column holding the glacier name, or ``None`` for an inventory that
        ships none, in which case ``name`` is written as null.
    year_column
        Column holding the acquisition year or date; dates are reduced to
        their year.
    extra_columns
        Further attributes to carry through, as ``{property: column}``. Missing
        columns are written as null rather than raising, so an inventory that
        lacks one still exports.
    simplify
        Douglas-Peucker tolerance in metres, applied in EPSG:3035. Zero keeps
        the full geometry.
    precision
        Decimal places kept in the output coordinates; 5 is about a metre.

    Returns
    -------
    tuple[int, int, int]
        Feature count, and the first and last acquisition year present.
    """
    # Lazy: this module otherwise imports no pandas at all
    from pandas.api.types import is_datetime64_any_dtype  # noqa: PLC0415

    export = frame.copy()
    if export.index.name is not None and export.index.name not in export:
        # An identifier kept as the index, like RGI's rgi_id, can be carried as a column
        export = export.reset_index()
    years = export[year_column]
    export["year"] = (years.dt.year if is_datetime64_any_dtype(years) else years).astype("int32")
    if name_column is None:
        export["name"] = None
    else:
        export["name"] = export[name_column].astype("object").where(export[name_column].notna())

    kept = ["name", "year"]
    for prop, column in (extra_columns or {}).items():
        export[prop] = export[column].where(export[column].notna()) if column in export else None
        kept.append(prop)
    export = export[[*kept, "geometry"]]
    if simplify > 0:
        export["geometry"] = (
            export.to_crs("EPSG:3035").geometry.simplify(simplify).to_crs("EPSG:4326")
        )
    else:
        export = export.to_crs("EPSG:4326")

    payload = json.loads(export.to_json(drop_id=True))
    for feature in payload["features"]:
        feature["geometry"]["coordinates"] = _round_coords(
            feature["geometry"]["coordinates"],
            precision,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # One feature per line, so a git diff shows which outlines changed
    features = ",\n".join(
        json.dumps(f, separators=(",", ":"), ensure_ascii=False) for f in payload["features"]
    )
    out_path.write_text(
        f'{{"type":"FeatureCollection","features":[\n{features}\n]}}\n',
        encoding="utf-8",
    )
    logger.debug(
        "Wrote %s: %d features, %.1f MB",
        out_path.name,
        len(export),
        out_path.stat().st_size / 1e6,
    )
    return len(export), int(export["year"].min()), int(export["year"].max())


def _round_coords(coords: Any, precision: int) -> Any:
    """Round a GeoJSON coordinate tree to ``precision`` decimals."""
    if coords and isinstance(coords[0], (int, float)):
        return [round(coords[0], precision), round(coords[1], precision)]
    return [_round_coords(part, precision) for part in coords]
