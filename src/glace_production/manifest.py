"""The run manifest: one row per composite a run left behind.

``glace-catalog`` publishes from this file instead of opening every composite of a
year. The schema is an **on-disk contract** with it, validated on read; the two
repositories share no import.

- **A cache, not the source of truth.** Every value here is also in the raster it
  describes, or is derived from its bytes: ``sha256`` comes from the writer when
  this run produced the file and from hashing the file otherwise. Nothing is
  ever fabricated.
- **A resumed run describes the whole year.** The writer skips files that exist,
  so their digests are not in memory. Rows from the previous manifest are carried
  wherever the filename and size still match, and re-read otherwise.
- **Flushed after every tile**, so a crash loses at most one tile's digests.
"""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
import logging
from operator import itemgetter
import re
from typing import TYPE_CHECKING, Any

from freezebase.raster import rasterio_open
from freezebase.s3 import list_object_sizes
from freezebase.utils import file_sha256
import numpy as np
import pandas as pd
from upath import UPath

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable
    from datetime import date
    from pathlib import Path

    from s1bursts import FileManager

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.parquet"
MANIFEST_VERSION = 2
"""Bumped when a column changes meaning, not when one is added.

2: ``composite_start`` / ``composite_end`` are the compositing window, no longer the
first and last contributing acquisition (those are the ``FIRST_ACQUISITION_DATE`` /
``LAST_ACQUISITION_DATE`` tag columns).
"""

COLUMNS: tuple[str, ...] = (
    "filename",
    "tile",
    "asset_key",
    "product",
    "qa",
    "polarization",
    "composite_start",
    "composite_end",
    "size",
    "sha256",
    "valid_percent",
    "band_name",
    "band_unit",
    "data_type",
    "nodata",
    "spatial_resolution",
    "bits_per_sample",
    "byte_order",
    "stat_minimum",
    "stat_maximum",
    "stat_mean",
    "stat_stddev",
    "stat_valid_percent",
)
"""Fixed columns, in order. Every dataset tag follows as its own column.

Fixed columns are lowercase and tag columns SCREAMING_CASE, so the two cannot
collide.
"""

_STATISTICS = {
    "STATISTICS_MINIMUM": "stat_minimum",
    "STATISTICS_MAXIMUM": "stat_maximum",
    "STATISTICS_MEAN": "stat_mean",
    "STATISTICS_STDDEV": "stat_stddev",
    "STATISTICS_VALID_PERCENT": "stat_valid_percent",
}

_COMPOSITE_RE = re.compile(
    r"^S1_(?P<product>COH\d{2}|RTC)"
    r"_(?:LRW|COMP_(?P<orbit>[AD]))"
    r"(?:_QA-(?P<qa>NUM|CQM))?"
    r"_(?P<tile>\d{2}[A-Z]{3}\d{2})"
    r"_(?P<pol>VV|VH|HH|HV)"
    r"_(?P<start>\d{8})_(?P<end>\d{8})\.tif$",
)


def asset_key(
    product: str,
    polarization: str,
    *,
    qa: str | None = None,
    orbit: str | None = None,
) -> str:
    """Return the published stem of a composite, e.g. ``coh12_vv`` or ``rtc_vh_comp_a_qa_num``.

    Parameters
    ----------
    product
        ``RTC`` or ``COH<baseline>``, as in the filename.
    polarization
        ``VV``, ``VH``, ...
    qa
        ``NUM`` or ``CQM`` for a QA layer, ``None`` for the measurement.
    orbit
        ``A`` or ``D`` for a per-pass composite, ``None`` for LRW.
    """
    key = f"{product.lower()}_{polarization.lower()}"
    if orbit:
        key += f"_comp_{orbit.lower()}"
    if qa:
        key += f"_qa_{qa.lower()}"
    return key


@dataclass(frozen=True)
class CompositeName:
    """What a composite's filename says about it."""

    product: str
    """``COH12`` or ``RTC``."""
    qa: str | None
    """``None`` for a measurement, ``NUM`` or ``CQM`` for its QA layer."""
    tile: str
    polarization: str
    orbit: str | None
    """``None`` for an LRW composite, ``A`` or ``D`` for a pass composite."""
    start: datetime
    end: datetime

    @property
    def asset_key(self) -> str:
        """The published stem, e.g. ``coh12_vv`` or ``rtc_vh_comp_a_qa_num``."""
        return asset_key(self.product, self.polarization, qa=self.qa, orbit=self.orbit)


def parse_composite_name(name: str) -> CompositeName | None:
    """Parse a composite filename; ``None`` for anything that is not one."""
    match = _COMPOSITE_RE.match(name)
    if match is None:
        return None
    return CompositeName(
        product=match["product"],
        qa=match["qa"],
        tile=match["tile"],
        polarization=match["pol"],
        orbit=match["orbit"],
        start=datetime.strptime(match["start"], "%Y%m%d").replace(tzinfo=UTC),
        end=datetime.strptime(match["end"], "%Y%m%d").replace(tzinfo=UTC),
    )


def describe_raster(path: Path | UPath) -> dict[str, Any]:
    """Read one composite's header: band layout, dataset tags and statistics.

    One range request on object storage. Returns the fixed columns except
    ``filename``, ``size``, ``sha256`` and ``valid_percent``, plus one entry per
    dataset tag.
    """
    with rasterio_open(path) as src:
        dataset_tags = dict(src.tags())
        band_tags = dict(src.tags(1))
        info: dict[str, Any] = {
            "band_name": src.descriptions[0] or "",
            "band_unit": src.units[0] or "",
            "data_type": src.dtypes[0],
            "nodata": src.nodatavals[0],
            "spatial_resolution": abs(src.res[0]),
            "bits_per_sample": np.dtype(src.dtypes[0]).itemsize * 8,
        }
    with path.open("rb") as handle:
        marker = handle.read(2)
    if marker not in (b"II", b"MM"):
        msg = f"{path} does not start with a TIFF byte-order marker, got {marker!r}."
        raise ValueError(msg)
    info["byte_order"] = "little-endian" if marker == b"II" else "big-endian"
    for tag, column in _STATISTICS.items():
        info[column] = float(band_tags[tag]) if tag in band_tags else None
    info.update(dataset_tags)
    return info


class RunManifest:
    """The rows of one year's manifest, carried across resumed runs.

    Parameters
    ----------
    comp_dir
        The directory the composites are written to; the manifest sits beside them.
    checksums
        Multihash-encoded sha256 per output path, as ``ResolutionWeighter.checksums``
        records them for the files *this* run wrote.
    rows
        Rows carried from a previous manifest, keyed by filename.
    """

    def __init__(
        self,
        comp_dir: Path | UPath,
        checksums: Mapping[str, str],
        rows: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.comp_dir = comp_dir
        self.path = comp_dir / MANIFEST_NAME
        self._checksums = checksums
        self._rows: dict[str, dict[str, Any]] = rows or {}
        self._unsaved = False
        """Whether rows changed since the last successful write."""
        self.max_workers = 12
        """Concurrent header reads; one tile's twelve layers at once."""

    @classmethod
    def load(cls, comp_dir: Path | UPath, checksums: Mapping[str, str]) -> RunManifest:
        """Start from the previous manifest beside the composites, if there is one."""
        path = comp_dir / MANIFEST_NAME
        rows: dict[str, dict[str, Any]] = {}
        if path.exists():
            with UPath(path).open("rb") as handle:
                previous = pd.read_parquet(handle)
            rows = {
                str(record["filename"]): _clean(record) for record in previous.to_dict("records")
            }
            logger.info("Carrying %d rows from %s", len(rows), path)
        return cls(comp_dir, checksums, rows)

    def record(
        self,
        paths: Iterable[Path | UPath],
        sizes: Mapping[str, int] | None = None,
    ) -> None:
        """Describe the given composites, re-reading only what changed.

        A row is carried from the previous manifest when its filename and size
        still match, this run did not write the file, and the row has a digest.
        Otherwise the header is read, and the digest comes from this run's writer
        or from hashing the file, so every row ends up with a digest.

        Parameters
        ----------
        paths
            The composites to describe. Anything not named like one is ignored.
        sizes
            Size per filename, where a listing already provided it; a file not
            in the mapping is stat'ed.
        """
        sizes = sizes or {}

        def describe(path: Path | UPath) -> tuple[str, dict[str, Any]] | None:
            parsed = parse_composite_name(path.name)
            if parsed is None:
                return None
            size = sizes.get(path.name) or path.stat().st_size
            previous = self._rows.get(path.name)
            written_now = str(path) in self._checksums
            if previous is not None and previous["size"] == size and not written_now:
                if previous["sha256"] is not None:
                    return None
                # Same bytes as before, digest never recorded: hash them once so
                # the manifest converges instead of carrying the gap forever.
                return path.name, {**previous, "sha256": multihash(path)}
            return path.name, {
                "filename": path.name,
                "tile": parsed.tile,
                "asset_key": parsed.asset_key,
                "product": parsed.product,
                "qa": parsed.qa,
                "polarization": parsed.polarization,
                "composite_start": parsed.start.date().isoformat(),
                "composite_end": parsed.end.date().isoformat(),
                "size": size,
                # The writer's digest where this run produced the file; otherwise
                # the file predates the run (a crash before its flush, or an
                # archive without a manifest) and is read once and hashed.
                "sha256": self._checksums.get(str(path)) or multihash(path),
                **describe_raster(path),
            }

        # Each header is one range request; on object storage the latency, not
        # the bytes, is the cost, so the reads go out together.
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for described in pool.map(describe, list(paths)):
                if described is not None:
                    self._rows[described[0]] = described[1]
                    self._unsaved = True

    def complete_tiles(self, assets: Collection[str], start: date, end: date) -> set[str]:
        """Return the tiles a resumed run can skip without looking at them again.

        A tile is complete when every asset key in ``assets`` has a row with a
        digest and the window ``start`` to ``end``, and a listing of the directory
        shows the file at the size the row recorded. Anything short of that, or
        a listing that fails, leaves the tile to the normal path, which fills in
        what is missing.

        Parameters
        ----------
        assets
            The asset keys of one complete tile; empty means no tile is complete.
        start, end
            The compositing window the run names its composites with.

        Returns
        -------
        set[str]
            The MGRS codes of the complete tiles.
        """
        if not assets:
            return set()
        try:
            sizes = _listing(self.comp_dir)
        except Exception:
            logger.warning(
                "Could not list %s; no tile is skipped early",
                self.comp_dir,
                exc_info=True,
            )
            return set()
        window = (start.isoformat(), end.isoformat())
        found: dict[str, set[str]] = {}
        for name, row in self._rows.items():
            if row["sha256"] is None or sizes.get(name) != row["size"]:
                continue
            if (row["composite_start"], row["composite_end"]) != window:
                continue
            found.setdefault(row["tile"], set()).add(row["asset_key"])
        return {tile for tile, keys in found.items() if keys.issuperset(assets)}

    def reconcile(self) -> None:
        """Bring the rows in line with a listing: drop the vanished, add the unrecorded.

        One listing, with sizes, rather than one ``stat`` per object.
        """
        sizes = _listing(self.comp_dir)
        for name in list(self._rows):
            if name not in sizes:
                logger.warning("%s is in the manifest but not on disk; dropping", name)
                del self._rows[name]
        self.record((self.comp_dir / name for name in sorted(sizes)), sizes)

    @property
    def frame(self) -> pd.DataFrame:
        """The rows as a frame in ``(tile, asset_key)`` order, valid percentages filled."""
        rows = sorted(self._rows.values(), key=itemgetter("tile", "asset_key"))
        tag_columns = sorted({key for row in rows for key in row} - set(COLUMNS))
        frame = pd.DataFrame(rows, columns=[*COLUMNS, *tag_columns])
        frame["valid_percent"] = _valid_percents(frame)
        frame.attrs["manifest_version"] = MANIFEST_VERSION
        return frame

    def write(self) -> None:
        """Write the manifest beside the composites."""
        frame = self.frame
        with UPath(self.path).open("wb") as handle:
            frame.to_parquet(handle, index=False)
        self._unsaved = False
        logger.info("Wrote %s with %d rows", self.path, len(frame))

    def checkpoint(self, paths: Iterable[Path | UPath]) -> None:
        """Record ``paths`` and write the manifest, but never fail the run over it.

        A failure to describe or write is logged, not raised: the end-of-run
        :meth:`reconcile` describes whatever a checkpoint missed. The manifest is
        written only when a row changed since the last successful write.

        Parameters
        ----------
        paths
            The composites this tile wrote.
        """
        try:
            self.record(paths)
        except Exception:
            logger.warning("Could not describe this tile's files; continuing", exc_info=True)
        if self._unsaved:
            self.flush()

    def flush(self) -> None:
        """Write the manifest, but never fail the run over it.

        Between tiles the manifest is a checkpoint; the end-of-run write is the
        one that must succeed. An unreachable endpoint at a checkpoint costs the
        rows since the last one, which the next run recovers by hashing.
        """
        try:
            self.write()
        except Exception:
            logger.warning("Could not checkpoint %s; continuing", self.path, exc_info=True)


def multihash(path: Path | UPath) -> str:
    """Return the multihash-encoded sha256 of a file's bytes, as ``file:checksum`` wants it."""
    return "1220" + file_sha256(path)


def _listing(directory: Path | UPath) -> dict[str, int]:
    """Size per filename of the files directly in ``directory``."""
    return list_object_sizes(directory)


def _valid_percents(frame: pd.DataFrame) -> pd.Series:
    """Each measurement's valid percentage, from its QA-NUM band's own statistics.

    The count band's ``STATISTICS_VALID_PERCENT`` is the share of pixels with at
    least one observation, which is the archive's definition of valid. The data
    band's own figure is not the same number: a decibel conversion turns every
    non-positive aggregate into NaN while the count still stands.
    """
    # Keyed by (tile, asset_key): the asset key alone repeats once per tile.
    counts = {
        (row["tile"], row["asset_key"]): row["stat_valid_percent"]
        for row in frame.loc[frame["qa"] == "NUM"].to_dict("records")
    }
    return pd.Series(
        [
            counts.get((row["tile"], f"{row['asset_key']}_qa_num"))
            if row["qa"] is None or pd.isna(row["qa"])
            else None
            for row in frame.to_dict("records")
        ],
        index=frame.index,
        dtype=float,
    )


def _clean(record: Mapping[Any, Any]) -> dict[str, Any]:
    """Convert a parquet row back to plain values: NaN from the store becomes ``None``."""
    return {
        str(key): None
        if isinstance(value, float) and np.isnan(value) and key != "nodata"
        else value
        for key, value in record.items()
    }


def written_paths(fm: FileManager, results: Iterable[Any]) -> list[Path | UPath]:
    """Return the output paths behind what the compose calls returned.

    A compose call hands back a composite product or a per-pass mapping of them;
    this flattens both to one file per layer the product carries, each once.
    """
    paths: list[Path | UPath] = []

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            for value in node.values():
                walk(value)
        elif isinstance(node, (tuple, list)):
            for value in node:
                walk(value)
        elif node is not None:
            paths.extend(fm.get_product_path(node, layer) for layer in sorted(node.layers))

    walk(list(results))
    return list(dict.fromkeys(paths))
