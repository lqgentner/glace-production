"""Geometry helpers for mapping vector inventories onto the MGRS grid.

Home of :class:`MGRSInventory`, developed here for the GLACE catalog and label
tooling; intended to migrate into ``freezebase`` once mature.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from freezebase.mgrs import WGS84, MGRSGrid, UTMZones
import numpy as np
from rasterio import features
from shapely import union_all

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import SupportsInt

    import geopandas as gpd
    from geopandas import GeoSeries
    from numpy.typing import NDArray
    from odc.geo.geobox import GeoBox
    from shapely.geometry.base import BaseGeometry


def _epsg_of(geobox: GeoBox) -> int:
    """EPSG code of a geobox's CRS."""
    epsg = geobox.crs.epsg if geobox.crs is not None else None
    if epsg is None:
        msg = f"GeoBox {geobox} has no EPSG-identifiable CRS."
        raise TypeError(msg)
    return epsg


class MGRSInventory:
    """A vector inventory mapped onto the MGRS 10 km grid.

    The inventory is reprojected to each UTM zone's CRS on demand and cached, so
    an inventory spanning several zones needs only one reprojection per zone.
    ``buffer_m`` expands the outlines (in each zone's metric CRS) for tile
    selection and coverage; rasterized masks always burn the true, unbuffered
    outlines.

    Parameters
    ----------
    inventory
        Vector geometries in any CRS.
    buffer_m
        Buffer applied to the outlines for :meth:`intersecting_tiles` and
        :meth:`coverage_fraction`, in metres. Default 0 (no buffer).
    """

    def __init__(self, inventory: gpd.GeoDataFrame, buffer_m: float = 0.0) -> None:
        self._inventory = inventory
        self._buffer_m = buffer_m
        self._base: dict[int, GeoSeries] = {}
        self._buffered: dict[int, GeoSeries] = {}

    def _base_geoms(self, epsg: int) -> GeoSeries:
        """Inventory geometry reprojected to ``epsg`` (cached, unbuffered)."""
        if epsg not in self._base:
            self._base[epsg] = self._inventory.to_crs(epsg=epsg).geometry
        return self._base[epsg]

    def buffered_geoms(self, epsg: int) -> GeoSeries:
        """Inventory geometry reprojected to ``epsg`` and buffered by ``buffer_m``."""
        if not self._buffer_m:
            return self._base_geoms(epsg)
        if epsg not in self._buffered:
            self._buffered[epsg] = self._base_geoms(epsg).buffer(self._buffer_m)
        return self._buffered[epsg]

    def prepare(self, epsg: int) -> None:
        """Populate the reprojection and spatial-index caches for ``epsg``.

        ``_base_geoms``/``buffered_geoms`` do an unsynchronised check-then-populate
        of ``self._base``/``self._buffered``, and a ``GeoSeries``'s first
        ``.sindex`` access lazily builds its spatial index. Both are safe to share
        read-only across threads afterwards, but not while either is still being
        built, so call this once per EPSG code from a single thread before an
        instance is handed to a thread pool.
        """
        base = self._base_geoms(epsg)
        base.sindex  # noqa: B018 - accessing the cached_property builds the index
        if self._buffer_m:
            buffered = self.buffered_geoms(epsg)
            buffered.sindex  # noqa: B018 - ditto

    def _candidates(self, geoms: GeoSeries, tile_poly: BaseGeometry) -> GeoSeries:
        """Geometries whose bounds intersect a tile, in inventory row order."""
        return cast("GeoSeries", geoms.iloc[geoms.sindex.query(tile_poly, sort=True)])

    def intersecting_tiles(self) -> list[str]:
        """Return the MGRS 10 km tile codes intersecting the buffered inventory.

        Each UTM zone the inventory touches is buffered in that zone's CRS; the
        resulting tiles are filtered to that zone, so multi-zone inventories are
        handled without collapsing to a single projection and boundary tiles are
        kept.
        """
        geoms_wgs = self._inventory.to_crs(WGS84).geometry
        tiles: set[str] = set()
        for zone in UTMZones().find_intersecting(geoms_wgs).itertuples():
            epsg = int(cast("SupportsInt", zone.epsg))
            buffered = self.buffered_geoms(epsg).to_crs(WGS84)
            grid = MGRSGrid(buffered).to_geodataframe()
            tiles.update(grid.loc[grid.epsg == epsg, "mgrs_code"])
        return sorted(tiles)

    def coverage_fraction(self, geobox: GeoBox) -> float:
        """Fraction of a tile's area covered by the buffered inventory, in [0, 1]."""
        tile_poly = geobox.extent.geom
        candidates = self._candidates(self.buffered_geoms(_epsg_of(geobox)), tile_poly)
        if candidates.empty:
            return 0.0
        # Union first: buffered outlines of neighbouring glaciers overlap, and
        # summing per-outline areas would count the overlap once per outline.
        area = union_all(candidates.values).intersection(tile_poly).area
        return float(area / tile_poly.area)

    def rasterize(
        self,
        geobox: GeoBox,
        *,
        ignore: MGRSInventory | Sequence[MGRSInventory] | None = None,
        ignore_value: int = 255,
    ) -> NDArray[np.uint8]:
        """Burn the true (unbuffered) outlines intersecting a tile into a uint8 mask.

        Parameters
        ----------
        geobox
            Target grid, a tile or a padded tile.
        ignore
            Inventories whose *buffered* outlines are burned as ``ignore_value``
            before the true outlines, marking areas to exclude (e.g. glaciers
            mapped in a different acquisition year), each with its own buffer.
            Where they overlap this inventory's outlines, these outlines win.
        ignore_value
            Value burned for ``ignore``. Default 255.
        """
        epsg = _epsg_of(geobox)
        tile_poly = geobox.extent.geom
        mask = np.zeros(geobox.shape.yx, dtype=np.uint8)
        if isinstance(ignore, MGRSInventory):
            ignore = [ignore]
        burns: list[tuple[GeoSeries, int]] = [
            (self._candidates(inventory.buffered_geoms(epsg), tile_poly), ignore_value)
            for inventory in ignore or ()
        ]
        burns.append((self._candidates(self._base_geoms(epsg), tile_poly), 1))
        for candidates, value in burns:
            if candidates.empty:
                continue
            features.rasterize(
                ((geom, value) for geom in candidates),
                out=mask,
                transform=geobox.affine,
            )
        return mask

    def rasterize_values(
        self, geobox: GeoBox, column: str, *, fill: int = -1
    ) -> NDArray[np.int32]:
        """Burn an integer attribute of the outlines intersecting a grid.

        Parameters
        ----------
        geobox
            Target grid.
        column
            Inventory column holding the value to burn per outline. Later rows
            win where outlines overlap.
        fill
            Value outside every outline. Default -1.
        """
        epsg = _epsg_of(geobox)
        tile_poly = geobox.extent.geom
        geoms = self._base_geoms(epsg)
        positions = geoms.sindex.query(tile_poly, sort=True)
        out = np.full(geobox.shape.yx, fill, dtype=np.int32)
        if len(positions) == 0:
            return out
        values = self._inventory[column].iloc[positions].astype(int).to_numpy()
        features.rasterize(
            zip(geoms.iloc[positions], values.tolist(), strict=True),
            out=out,
            transform=geobox.affine,
        )
        return out
