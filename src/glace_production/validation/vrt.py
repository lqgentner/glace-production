"""Relocatable VRTs: mosaic MGRS tiles and reference the sources relative to the VRT.

Any set of single-band tiles on the MGRS grid mosaics the same way, whether
composites, label masks or prediction maps: sources are added in filename
order, so the years of a tile (``LABELS_{inv}_{year}_{tile}``,
``PRED_{tile}_{year}``) ascend and a later year's data falls through to the
earlier year where it is nodata.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from freezebase.vrt import build_vrt_mosaic
from upath import UPath

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

TILE_GLOB = "[0-9][0-9][A-Z][A-Z][A-Z][0-9][0-9]"
"""Glob character classes matching one MGRS 10 km tile code, e.g. ``32TMS55``."""


def build_relative_vrt(files: Sequence[str | Path | UPath], out: str | Path | UPath) -> int:
    """Mosaic tiles into a VRT whose same-store sources are relative to it.

    Sources on another store than the VRT keep their absolute path.

    Parameters
    ----------
    files
        Source tiles, mosaicked in sorted filename order.
    out
        Output VRT.

    Returns
    -------
    int
        Number of sources written.
    """
    if not files:
        msg = "no tiles to mosaic"
        raise ValueError(msg)
    ordered = sorted(files, key=lambda f: UPath(f).name)
    build_vrt_mosaic(ordered, out)
    return len(ordered)


def expand_tile_pattern(pattern: str) -> str:
    """Replace a ``{tile}`` placeholder by :data:`TILE_GLOB`.

    ``S1_COH12_LRW_{tile}_VV_*.tif`` then matches the measurement tiles but not
    ``S1_COH12_LRW_QA-NUM_{tile}_VV_*.tif``, which a bare ``*`` would.
    """
    return pattern.replace("{tile}", TILE_GLOB)


def default_vrt_name(pattern: str) -> str:
    """Derive a VRT filename from a tile glob: the pattern with its wildcards removed.

    ``S1_COH12_LRW_{tile}_VV_*.tif`` becomes ``S1_COH12_LRW_VV.vrt``.
    """
    stem = pattern.rsplit(".", 1)[0] if "." in pattern else pattern
    stem = re.sub(r"(\{tile\}|[*?])+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    if not stem:
        msg = f"cannot derive a VRT name from pattern {pattern!r}"
        raise ValueError(msg)
    return f"{stem}.vrt"
