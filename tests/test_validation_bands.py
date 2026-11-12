"""Signed-distance bands of the validation scope."""

from __future__ import annotations

import numpy as np

from glace_production.validation.bands import BAND_NAMES, EDGE_M, RING_M, distance_band


def test_constants() -> None:
    assert RING_M == 250.0
    assert EDGE_M == 30.0
    assert BAND_NAMES == ("interior", "inner", "edge", "outer")


def test_distance_band_edges() -> None:
    dist = np.array([-1000.0, -250.1, -250.0, -10.0, 0.0, 10.0, 30.0, 30.1, 250.0], np.float32)
    bands = distance_band(dist)
    assert bands.tolist() == [
        "interior",
        "interior",
        "inner",
        "inner",
        "inner",
        "edge",
        "edge",
        "outer",
        "outer",
    ]
    assert bands.dtype.kind == "U"
