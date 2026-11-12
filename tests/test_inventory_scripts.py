"""The inventory scripts never replace published outputs without ``--overwrite``.

The glacier fractions and the label masks live in an unversioned object store, where a
rerun would silently overwrite what the catalog and the validation were built from.
The expensive steps are replaced by a tripwire, so a refusal must come first.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Never

import pytest

if TYPE_CHECKING:
    from types import ModuleType

_SCRIPTS = Path(__file__).parents[1] / "scripts" / "inventories"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), _SCRIPTS / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tripwire(*_: object, **__: object) -> Never:
    msg = "the script computed before checking its output"
    raise AssertionError(msg)


def test_glacier_fractions_refuse_an_existing_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _load("derive-glacier-fractions")
    out = tmp_path / "fractions.parquet"
    out.write_bytes(b"published")
    monkeypatch.setattr(script, "perimeter_tiles", _tripwire)
    monkeypatch.setattr(script, "compute_fractions", _tripwire)
    monkeypatch.setattr(
        sys, "argv", ["derive-glacier-fractions", "--scope", "alps", "--out", str(out)]
    )

    with pytest.raises(SystemExit):
        script.main()
    assert out.read_bytes() == b"published"


def test_label_masks_refuse_a_directory_with_masks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _load("rasterize-labels")
    mask = tmp_path / "LABELS_SGI2023_2023_32TLR48.tif"
    mask.write_bytes(b"published")
    monkeypatch.setattr(script, "INVENTORIES", dict.fromkeys(script.INVENTORIES, _tripwire))
    monkeypatch.setattr(
        sys, "argv", ["rasterize-labels", "--source", "SGI2023", "--out-dir", str(tmp_path)]
    )

    with pytest.raises(SystemExit):
        script.main()
    assert mask.read_bytes() == b"published"
