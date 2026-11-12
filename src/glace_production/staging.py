"""Materialise remote files locally for tools that cannot read object storage.

SNAP's ``externalDEMFile`` and similar parameters take a filesystem path, so a
DEM stored on S3 has to be copied to disk before a graph can reference it.
:func:`stage_file` performs that copy once per file into the local cache.
"""

from __future__ import annotations

import logging
from pathlib import Path
from secrets import token_hex
import shutil
import time
from typing import TYPE_CHECKING

from s1bursts import prepare_dem_for_snap

from glace_production.config import settings

if TYPE_CHECKING:
    from upath import UPath

logger = logging.getLogger(__name__)

_COPY_BUFFER_SIZE = 32 * 1024 * 1024


def stage_file(src: Path | UPath, *, subdir: str = "staged") -> Path:
    """Return a local path to ``src``, copying it into the cache if it is remote.

    A previously staged copy of the same name and size is reused, so repeated
    runs on the same machine download the file only once.

    Parameters
    ----------
    src : Path or UPath
        Source file. A :class:`~pathlib.Path` (including a local ``UPath``) is
        already local and is returned unchanged.
    subdir : str, default "staged"
        Sub-directory of the cache directory to stage into.

    Returns
    -------
    Path
        Path to the local file.

    Raises
    ------
    FileNotFoundError
        If ``src`` does not exist.
    """
    # `UPath("/local/path")` is a Path subclass, `UPath("s3://...")` is not,
    # so isinstance() cleanly separates already-local sources from remote ones.
    if isinstance(src, Path):
        if not src.exists():
            msg = f"File not found: '{src}'"
            raise FileNotFoundError(msg)
        return src

    if not src.exists():
        msg = f"File not found: '{src}'"
        raise FileNotFoundError(msg)

    size = src.stat().st_size
    dest_dir = settings.cache_dir / subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name

    if dest.exists() and dest.stat().st_size == size:
        logger.info("Reusing staged file %s (%.1f MB)", dest, size / 1e6)
        return dest

    logger.info("Staging %s to %s (%.1f MB)", src, dest, size / 1e6)
    start = time.perf_counter()
    # A unique suffix keeps concurrent writers to the same target from
    # clobbering one another's partial file.
    partial = dest.with_name(f"{dest.name}.{token_hex(8)}.partial")
    try:
        with src.open("rb") as fsrc, partial.open("wb") as fdst:
            shutil.copyfileobj(fsrc, fdst, length=_COPY_BUFFER_SIZE)
        partial.replace(dest)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    logger.info("Staged %s in %.1f s", dest.name, time.perf_counter() - start)
    return dest


def stage_dem_for_snap(src: Path | UPath, *, subdir: str = "staged-snap") -> Path:
    """Return a local DEM copy that SNAP's ``externalDEMFile`` reads correctly.

    The DEM is staged with :func:`stage_file` and passed to
    :func:`s1bursts.prepare_dem_for_snap`, which cancels the half-pixel offset SNAP
    applies to an external DEM. The heights are left alone: the project DEMs are
    already ellipsoidal. GDAL and rasterio see the copy half a pixel off, so it serves
    ``externalDEMFile`` only.

    Parameters
    ----------
    src : Path or UPath
        Source DEM, local or remote.
    subdir : str, default "staged-snap"
        Sub-directory of the cache directory to write the compensated copy to.

    Returns
    -------
    Path
        Path to the compensated local DEM.

    Raises
    ------
    FileNotFoundError
        If ``src`` does not exist.
    """
    return prepare_dem_for_snap(stage_file(src), cache_dir=settings.cache_dir / subdir)
