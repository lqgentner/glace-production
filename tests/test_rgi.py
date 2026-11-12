"""RGI 7.0 glacier names are repaired where a submission used another encoding.

The examples are names as the RGI shapefiles deliver them.
"""

from __future__ import annotations

import pytest

from glace_production.datasets.rgi import repair_name


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        # UTF-8 read as ISO-8859-1 (Scandinavia, North Asia, Caucasus, Arctic Canada)
        ("B\u00c3\u00b8verbrean", "Bøverbrean"),
        ("Ã\x85rnesbreen", "Årnesbreen"),
        ("AbbÃ© Glacier", "Abbé Glacier"),
        ("16\u00d0\u00b0", "16\u0430"),  # Cyrillic a
        # Code page 437 read as ISO-8859-1 (Alps)
        ("Mont V\x82lan-W", "Mont Vélan-W"),
        ("MONT MIN\x90 GLACIER DU", "MONT MINÉ GLACIER DU"),
        ("\x9assers Barrhorn", "Üssers Barrhorn"),
        ("BR\x8eNDJI GLETSCHER", "BRÄNDJI GLETSCHER"),
    ],
)
def test_repairs_misdecoded_names(stored: str, expected: str) -> None:
    assert repair_name(stored) == expected


@pytest.mark.parametrize(
    "name",
    ["TRIENT GLACIER DU", "Glacier de Rhêmes - Calabre", "Längentaler Ferner", ""],
)
def test_keeps_correct_names(name: str) -> None:
    assert repair_name(name) == name
