"""The Alpine Convention perimeter: the atlas first, the project store as fallback.

The atlas's geoserver stopped answering on 2026-09-21, and a fresh container
has no cache, so the whole fleet failed to resolve its AOI at start. The atlas
is still tried first; when it fails, the raw archive is copied from a mirror
beside the project data and checked against the checksum of the copy taken on
2026-09-18.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Self

import pytest
import requests

from glace_production.datasets import alpineconvention

if TYPE_CHECKING:
    from collections.abc import Callable

ARCHIVE = b"not really a zip"


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a project store holding the mirrored archive, with the pin matching it."""
    store = tmp_path / "store"
    (store / "regions").mkdir(parents=True)
    (store / "regions" / "alpineconvention-2025.zip").write_bytes(ARCHIVE)
    monkeypatch.setattr(alpineconvention, "settings", SimpleNamespace(data_dir=store))
    monkeypatch.setattr(alpineconvention, "MIRROR_SHA256", hashlib.sha256(ARCHIVE).hexdigest())
    return store


def _origin(monkeypatch: pytest.MonkeyPatch, behaviour: Callable[[Path], None]) -> list[str]:
    """Stand in for the HTTP downloader; returns the list of URLs it was asked for."""
    urls: list[str] = []

    class Downloader:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def __call__(self, url: str, save_dir: Path, filename: str, **_: Any) -> Path:
            urls.append(url)
            behaviour(Path(save_dir) / filename)
            return Path(save_dir) / filename

    monkeypatch.setattr(alpineconvention, "HTTPDownloader", Downloader)
    return urls


def _serve(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fresh from the origin")


def _gone(path: Path) -> None:  # noqa: ARG001
    msg = "Read timed out. (read timeout=30)"
    raise requests.ReadTimeout(msg)


@pytest.mark.usefixtures("store")
def test_a_reachable_origin_is_used_and_the_mirror_is_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    urls = _origin(monkeypatch, _serve)
    dataset = alpineconvention.AlpineConventionPerimeter(cache_dir=tmp_path / "cache")

    dataset._download()

    assert urls == [dataset.metadata.source_url]
    assert dataset.raw_path.read_bytes() == b"fresh from the origin"


def test_a_dead_origin_falls_back_to_the_mirror(
    store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _origin(monkeypatch, _gone)
    dataset = alpineconvention.AlpineConventionPerimeter(cache_dir=tmp_path / "cache")

    dataset._download()

    assert dataset.raw_path.read_bytes() == ARCHIVE
    assert "Read timed out" in caplog.text
    assert str(store / "regions") in caplog.text


def test_a_corrupt_mirror_is_refused(
    store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _origin(monkeypatch, _gone)
    (store / "regions" / "alpineconvention-2025.zip").write_bytes(b"tampered")
    dataset = alpineconvention.AlpineConventionPerimeter(cache_dir=tmp_path / "cache")

    with pytest.raises(ValueError, match="mirror"):
        dataset._download()
    assert not dataset.raw_path.exists()


def test_the_origin_checksum_is_not_pinned() -> None:
    """A WFS export is generated per request, so its bytes are never stable."""
    assert alpineconvention.AlpineConventionPerimeter().metadata.sha256 is None
