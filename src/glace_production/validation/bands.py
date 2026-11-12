"""Signed-distance bands of the validation scope.

The scope is every glacier pixel plus the ring of ``RING_M`` outside the
outlines. Bands split it by signed distance to the outline (negative inside).
A band holds one class: inside bands are glacier, outside bands background,
so a band is scored by recall inside and false-positive rate outside.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

RING_M = 250.0
"""Width of the ring outside the outlines that belongs to the scope, in metres."""

EDGE_M = 30.0
"""Width of the outside band attributed to outline uncertainty, in metres."""

BAND_NAMES: tuple[str, ...] = ("interior", "inner", "edge", "outer")
"""From the glacier interior outwards: beyond ``RING_M`` inside, within it,
within ``EDGE_M`` outside, the rest of the ring."""


def distance_band(dist_m: NDArray[np.floating]) -> NDArray[np.str_]:
    """Band name of every signed distance: ``-RING_M`` and ``0`` are inner, ``EDGE_M`` is edge."""
    d = np.asarray(dist_m)
    names = np.asarray(BAND_NAMES, dtype=np.str_)
    return np.select([d < -RING_M, d <= 0.0, d <= EDGE_M], names[:3], default=names[3])
