"""The run manifest: the on-disk contract the GLACE catalog publisher reads.

These tests write real COGs through the same ``freezebase.write_cog`` the
composite writer uses, so the bytes, tags and digests are the ones a run leaves
behind. Nothing is mocked.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
import hashlib
import logging
from pathlib import Path
from typing import Any

from freezebase.raster import write_cog
import numpy as np
import pandas as pd
import pytest
from rasterio.transform import from_origin
from s1bursts import FileManager
from s1bursts.product import RTCLRWComposite, RTCPassComposite
from s1bursts.product_tags import band_statistics
from upath import UPath

from glace_production.manifest import (
    MANIFEST_NAME,
    MANIFEST_VERSION,
    RunManifest,
    asset_key,
    parse_composite_name,
    written_paths,
)

# The columns glace-catalog's ``read_manifest`` refuses a manifest without.
# Kept verbatim rather than imported: the two repositories share this file
# format and nothing else, so a change on either side has to show up here.
PUBLISHER_COLUMNS = (
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
)

TILE = "32TPS05"
WINDOW = "20240619_20240912"


def _multihash(path: Path) -> str:
    return "1220" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write(
    path: Path,
    data: np.ndarray,
    *,
    nodata: float,
    tags: dict[str, str],
    band_name: str,
    unit: str = "",
) -> str | None:
    profile = {
        "driver": "GTiff",
        "dtype": data.dtype,
        "count": 1,
        "height": data.shape[0],
        "width": data.shape[1],
        "crs": "EPSG:32632",
        "transform": from_origin(500_000, 5_200_000, 10, 10),
        "nodata": nodata,
        "compress": "zstd",
    }
    return write_cog(
        data,
        path,
        profile,
        band_names=[band_name],
        tags=tags,
        band_tags=[band_statistics(data, nodata)],
        units=[unit] if unit else None,
        checksum=True,
    )


@pytest.fixture
def archive(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """One tile's coherence pair: a data band and its QA-NUM count band."""
    rng = np.random.default_rng(0)
    coh = rng.random((8, 8), dtype=np.float32)
    coh[0, :] = np.nan  # one row of nodata -> 87.5 % valid
    num = np.full((8, 8), 3, dtype=np.uint16)
    num[0, :] = 0
    num[1, :2] = 0  # 10 of 64 unobserved -> 84.375 % valid
    checksums: dict[str, str] = {}
    tags = {"PROCESSING_SOFTWARE": "s1bursts 0.4.0", "PASS_DIRECTION": "Ascending"}
    for name, data, nodata, band in (
        (f"S1_COH12_LRW_{TILE}_VV_{WINDOW}.tif", coh, np.nan, "coherence"),
        (f"S1_COH12_LRW_QA-NUM_{TILE}_VV_{WINDOW}.tif", num, 0, "observations"),
    ):
        digest = _write(tmp_path / name, data, nodata=nodata, tags=tags, band_name=band)
        assert digest is not None
        checksums[str(tmp_path / name)] = digest
    return tmp_path, checksums


def test_parse_lrw_composite_name_yields_the_published_asset_key() -> None:
    parsed = parse_composite_name(f"S1_COH12_LRW_{TILE}_VV_{WINDOW}.tif")
    assert parsed is not None
    assert parsed.asset_key == "coh12_vv"
    assert (parsed.product, parsed.qa, parsed.tile, parsed.polarization) == (
        "COH12",
        None,
        TILE,
        "VV",
    )
    assert parsed.start == datetime(2024, 6, 19, tzinfo=UTC)
    assert parsed.end == datetime(2024, 9, 12, tzinfo=UTC)


def test_parse_qa_and_pass_names() -> None:
    qa = parse_composite_name(f"S1_RTC_LRW_QA-CQM_{TILE}_VH_{WINDOW}.tif")
    assert qa is not None
    assert (qa.asset_key, qa.qa) == ("rtc_vh_qa_cqm", "CQM")
    by_pass = parse_composite_name(f"S1_RTC_COMP_D_QA-NUM_{TILE}_VV_{WINDOW}.tif")
    assert by_pass is not None
    assert (by_pass.asset_key, by_pass.orbit) == ("rtc_vv_comp_d_qa_num", "D")


def test_parse_rejects_anything_else() -> None:
    assert parse_composite_name("config.yaml") is None
    assert parse_composite_name(MANIFEST_NAME) is None
    assert parse_composite_name(f"LABELS_SGI2023_2023_{TILE}.tif") is None


def test_record_describes_a_written_file(archive: tuple[Path, dict[str, str]]) -> None:
    comp_dir, checksums = archive
    manifest = RunManifest.load(comp_dir, checksums)
    manifest.record(sorted(comp_dir.glob("*.tif")))
    frame = manifest.frame

    assert set(PUBLISHER_COLUMNS) <= set(frame.columns)
    row = frame.set_index("asset_key").loc["coh12_vv"]
    assert row["filename"] == f"S1_COH12_LRW_{TILE}_VV_{WINDOW}.tif"
    assert row["tile"] == TILE
    assert row["size"] == (comp_dir / row["filename"]).stat().st_size
    assert row["sha256"] == checksums[str(comp_dir / row["filename"])]
    assert row["sha256"].startswith("1220")
    assert row["composite_start"] == "2024-06-19"
    assert row["band_name"] == "coherence"
    assert row["data_type"] == "float32"
    assert np.isnan(row["nodata"])
    assert row["spatial_resolution"] == 10.0
    assert row["bits_per_sample"] == 32
    assert row["byte_order"] == "little-endian"
    # Every dataset tag becomes a column, SCREAMING_CASE, uncurated.
    assert row["PROCESSING_SOFTWARE"] == "s1bursts 0.4.0"
    assert row["PASS_DIRECTION"] == "Ascending"
    # The band statistics the writer embedded come along too.
    assert row["stat_valid_percent"] == pytest.approx(87.5)
    assert 0.0 <= row["stat_minimum"] <= row["stat_maximum"] <= 1.0


def test_valid_percent_comes_from_the_qa_num_band(archive: tuple[Path, dict[str, str]]) -> None:
    comp_dir, checksums = archive
    manifest = RunManifest.load(comp_dir, checksums)
    manifest.record(sorted(comp_dir.glob("*.tif")))
    by_key = manifest.frame.set_index("asset_key")
    # Keyed by the data asset it counts for, and taken from the count band's
    # own statistics -- not from the data band's, which counts the composite.
    assert by_key.loc["coh12_vv", "valid_percent"] == pytest.approx(84.375)
    assert pd.isna(by_key.loc["coh12_vv_qa_num", "valid_percent"])


def test_write_then_load_round_trips(archive: tuple[Path, dict[str, str]]) -> None:
    comp_dir, checksums = archive
    manifest = RunManifest.load(comp_dir, checksums)
    manifest.record(sorted(comp_dir.glob("*.tif")))
    manifest.write()

    written = pd.read_parquet(comp_dir / MANIFEST_NAME)
    assert len(written) == 2
    assert written.attrs["manifest_version"] == MANIFEST_VERSION
    assert list(written["asset_key"]) == ["coh12_vv", "coh12_vv_qa_num"]


def test_resumed_run_carries_digests_of_files_it_did_not_write(
    archive: tuple[Path, dict[str, str]],
) -> None:
    comp_dir, checksums = archive
    first = RunManifest.load(comp_dir, checksums)
    first.record(sorted(comp_dir.glob("*.tif")))
    first.write()

    # A resumed run: _save skips both files, so nothing is in memory.
    resumed = RunManifest.load(comp_dir, {})
    resumed.record(sorted(comp_dir.glob("*.tif")))
    by_name = resumed.frame.set_index("filename")
    for path, digest in checksums.items():
        assert by_name.loc[Path(path).name, "sha256"] == digest


def test_rewritten_file_is_re_read_and_gets_the_digest_of_its_new_bytes(
    archive: tuple[Path, dict[str, str]],
) -> None:
    comp_dir, checksums = archive
    first = RunManifest.load(comp_dir, checksums)
    first.record(sorted(comp_dir.glob("*.tif")))
    first.write()

    name = f"S1_COH12_LRW_{TILE}_VV_{WINDOW}.tif"
    bigger = np.zeros((16, 16), dtype=np.float32)
    _write(
        comp_dir / name,
        bigger,
        nodata=np.nan,
        tags={"REWRITTEN": "yes"},
        band_name="coherence",
    )

    resumed = RunManifest.load(comp_dir, {})
    resumed.record([comp_dir / name])
    row = resumed.frame.set_index("filename").loc[name]
    assert row["size"] == (comp_dir / name).stat().st_size
    assert row["sha256"] == _multihash(comp_dir / name)
    assert row["REWRITTEN"] == "yes"


def test_reconcile_matches_the_listing(archive: tuple[Path, dict[str, str]]) -> None:
    comp_dir, checksums = archive
    manifest = RunManifest.load(comp_dir, checksums)
    manifest.record(sorted(comp_dir.glob("*.tif")))
    manifest.write()

    # One file vanished, one appeared that no run recorded.
    (comp_dir / f"S1_COH12_LRW_QA-NUM_{TILE}_VV_{WINDOW}.tif").unlink()
    stray = comp_dir / f"S1_RTC_LRW_{TILE}_VH_{WINDOW}.tif"
    _write(stray, np.ones((8, 8), dtype=np.float32), nodata=np.nan, tags={}, band_name="rtc")
    (comp_dir / "config.yaml").write_text("year: 2024\n")

    resumed = RunManifest.load(comp_dir, {})
    resumed.reconcile()
    assert list(resumed.frame["asset_key"]) == ["coh12_vv", "rtc_vh"]
    assert (
        resumed.frame.set_index("asset_key").loc["coh12_vv", "sha256"]
        == checksums[str(comp_dir / f"S1_COH12_LRW_{TILE}_VV_{WINDOW}.tif")]
    )
    assert resumed.frame.set_index("asset_key").loc["rtc_vh", "sha256"] == _multihash(stray)


def test_written_paths_names_every_layer(tmp_path: Path) -> None:
    fm = FileManager(tmp_path)
    start, end = pd.Timestamp("2024-06-19"), pd.Timestamp("2024-09-12")
    with_qa = RTCLRWComposite(TILE, start, end, "VV", layers=RTCLRWComposite.supported_layers)
    bare = RTCLRWComposite(TILE, start, end, "VH")
    by_pass = {"Ascending": RTCPassComposite(TILE, start, end, "VV", "Ascending")}

    assert len(written_paths(fm, [with_qa])) == 3
    assert len(written_paths(fm, [bare])) == 1
    # A per-pass mapping, and None for a cell that wrote nothing, are flattened too.
    names = {path.name for path in written_paths(fm, [with_qa, bare, by_pass, None])}
    assert names == {
        *(fm.get_product_path(with_qa, layer).name for layer in with_qa.layers),
        fm.get_product_path(bare).name,
        fm.get_product_path(by_pass["Ascending"]).name,
    }


def test_reconcile_over_a_upath_uses_the_filesystem_listing(
    archive: tuple[Path, dict[str, str]],
) -> None:
    comp_dir, checksums = archive
    manifest = RunManifest.load(UPath(comp_dir), checksums)
    manifest.reconcile()
    frame = manifest.frame
    assert list(frame["asset_key"]) == ["coh12_vv", "coh12_vv_qa_num"]
    assert list(frame["size"]) == [(comp_dir / name).stat().st_size for name in frame["filename"]]
    # The digests still key on the path string the writer used.
    assert list(frame["sha256"]) == [checksums[str(comp_dir / n)] for n in frame["filename"]]


def test_valid_percent_is_keyed_per_tile(archive: tuple[Path, dict[str, str]]) -> None:
    """Two tiles carry the same asset keys; the lookup must not collide across them."""
    comp_dir, checksums = archive
    other = "32TLR49"
    num = np.full((8, 8), 3, dtype=np.uint16)
    num[:4, :] = 0  # 50 % observed on the second tile
    coh = np.full((8, 8), 0.5, dtype=np.float32)
    for name, data, nodata, band in (
        (f"S1_COH12_LRW_{other}_VV_{WINDOW}.tif", coh, np.nan, "coherence"),
        (f"S1_COH12_LRW_QA-NUM_{other}_VV_{WINDOW}.tif", num, 0, "observations"),
    ):
        _write(comp_dir / name, data, nodata=nodata, tags={}, band_name=band)

    manifest = RunManifest.load(comp_dir, checksums)
    manifest.record(sorted(comp_dir.glob("*.tif")))
    by_tile = manifest.frame.set_index(["tile", "asset_key"])["valid_percent"]
    assert by_tile[(TILE, "coh12_vv")] == pytest.approx(84.375)
    assert by_tile[(other, "coh12_vv")] == pytest.approx(50.0)
    assert pd.isna(by_tile[(other, "coh12_vv_qa_num")])


def test_a_file_this_run_wrote_is_always_re_read(archive: tuple[Path, dict[str, str]]) -> None:
    """An overwrite of the same size must not keep the carried row and its stale digest."""
    comp_dir, checksums = archive
    stale = RunManifest.load(comp_dir, {})
    stale.record(sorted(comp_dir.glob("*.tif")))
    stale.write()  # every sha256 None, sizes matching

    fresh = RunManifest.load(comp_dir, checksums)
    fresh.record(sorted(comp_dir.glob("*.tif")))
    by_name = fresh.frame.set_index("filename")
    for path, digest in checksums.items():
        assert by_name.loc[Path(path).name, "sha256"] == digest


def test_orphaned_files_are_hashed_rather_than_left_without_a_digest(
    archive: tuple[Path, dict[str, str]],
) -> None:
    """A file a crashed run wrote before its flush: no row, no in-memory digest."""
    comp_dir, checksums = archive
    resumed = RunManifest.load(comp_dir, {})
    resumed.record(sorted(comp_dir.glob("*.tif")))
    by_name = resumed.frame.set_index("filename")
    # Hashing the bytes gives the same multihash the writer recorded.
    for path, digest in checksums.items():
        assert by_name.loc[Path(path).name, "sha256"] == digest


def test_a_carried_row_without_a_digest_is_hashed_on_the_next_run(
    archive: tuple[Path, dict[str, str]],
) -> None:
    comp_dir, checksums = archive
    frame = RunManifest.load(comp_dir, checksums)
    frame.record(sorted(comp_dir.glob("*.tif")))
    stale = frame.frame
    stale["sha256"] = None  # as a retrofitted manifest would have it
    stale.to_parquet(comp_dir / MANIFEST_NAME, index=False)

    resumed = RunManifest.load(comp_dir, {})
    resumed.record(sorted(comp_dir.glob("*.tif")))
    assert resumed.frame["sha256"].notna().all()


def test_flush_survives_a_failed_write(
    archive: tuple[Path, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A per-tile flush that fails must not end the run; the final write redoes it."""
    comp_dir, checksums = archive
    manifest = RunManifest.load(comp_dir, checksums)
    manifest.record(sorted(comp_dir.glob("*.tif")))

    def boom(self: RunManifest) -> None:  # noqa: ARG001 -- monkeypatched onto the class, so it takes self
        msg = "endpoint unreachable"
        raise OSError(msg)

    monkeypatch.setattr(RunManifest, "write", boom)
    with caplog.at_level(logging.WARNING):
        manifest.flush()
    assert "endpoint unreachable" in caplog.text
    assert not (comp_dir / MANIFEST_NAME).exists()


def test_checkpoint_survives_a_failed_describe(
    archive: tuple[Path, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Describing a tile's files at a checkpoint may fail without ending the run.

    During an object-store outage the per-tile compose is caught and skipped,
    but reading the header of what the other polarization had written raised
    from outside that handler and took every VM of the fleet down. The
    end-of-run reconcile re-describes whatever a checkpoint could not.
    """
    comp_dir, checksums = archive
    manifest = RunManifest.load(comp_dir, checksums)

    def boom(path: Path) -> dict[str, Any]:  # noqa: ARG001
        msg = "Forbidden"
        raise PermissionError(msg)

    monkeypatch.setattr("glace_production.manifest.describe_raster", boom)
    with caplog.at_level(logging.WARNING):
        manifest.checkpoint(sorted(comp_dir.glob("*.tif")))
    assert "Forbidden" in caplog.text
    assert len(manifest.frame) == 0

    monkeypatch.undo()
    manifest.reconcile()
    assert len(manifest.frame) == 2
    assert manifest.frame["sha256"].notna().all()


def test_asset_key_is_what_parsing_a_name_yields() -> None:
    for name, key in (
        (f"S1_RTC_LRW_{TILE}_VV_{WINDOW}.tif", asset_key("RTC", "VV")),
        (f"S1_COH12_LRW_QA-NUM_{TILE}_VH_{WINDOW}.tif", asset_key("COH12", "VH", qa="NUM")),
        (
            f"S1_RTC_COMP_A_QA-NUM_{TILE}_VV_{WINDOW}.tif",
            asset_key("RTC", "VV", qa="NUM", orbit="A"),
        ),
    ):
        parsed = parse_composite_name(name)
        assert parsed is not None
        assert parsed.asset_key == key
    assert asset_key("COH12", "VH", qa="CQM") == "coh12_vh_qa_cqm"


# ---------------------------------------------------------------------------
# Complete tiles: what a resumed run may skip before searching for bursts.
# ---------------------------------------------------------------------------

START, END = date(2024, 6, 19), date(2024, 9, 12)  # WINDOW
ASSETS = frozenset({"rtc_vv", "coh12_vv"})
RTC_NAME = f"S1_RTC_LRW_{TILE}_VV_{WINDOW}.tif"
COH_NAME = f"S1_COH12_LRW_{TILE}_VV_{WINDOW}.tif"


def _done_tile(comp_dir: Path, *, digestless: str | None = None) -> RunManifest:
    """Return a manifest whose rows say TILE has ASSETS, each on disk at its recorded size.

    The row of ``digestless``, if given, has no digest.
    """
    rows: dict[str, dict[str, Any]] = {}
    for name in (RTC_NAME, COH_NAME):
        (comp_dir / name).write_bytes(b"II*\x00" + name.encode())
        parsed = parse_composite_name(name)
        assert parsed is not None
        rows[name] = {
            "filename": name,
            "tile": parsed.tile,
            "asset_key": parsed.asset_key,
            "product": parsed.product,
            "qa": parsed.qa,
            "polarization": parsed.polarization,
            "composite_start": START.isoformat(),
            "composite_end": END.isoformat(),
            "size": (comp_dir / name).stat().st_size,
            "sha256": None if name == digestless else "1220" + "0" * 64,
        }
    return RunManifest(comp_dir, {}, rows)


def test_a_tile_with_every_asset_on_disk_is_complete(tmp_path: Path) -> None:
    assert _done_tile(tmp_path).complete_tiles(ASSETS, START, END) == {TILE}


def test_a_missing_asset_leaves_the_tile_incomplete(tmp_path: Path) -> None:
    assert _done_tile(tmp_path).complete_tiles(ASSETS | {"rtc_vh"}, START, END) == set()


def test_a_row_without_a_digest_is_not_complete(tmp_path: Path) -> None:
    """The normal path hashes it, so the manifest converges."""
    manifest = _done_tile(tmp_path, digestless=RTC_NAME)
    assert manifest.complete_tiles(ASSETS, START, END) == set()


def test_a_file_gone_from_disk_is_not_complete(tmp_path: Path) -> None:
    """Otherwise the end-of-run reconcile would drop its row and the gap would be silent."""
    manifest = _done_tile(tmp_path)
    (tmp_path / COH_NAME).unlink()
    assert manifest.complete_tiles(ASSETS, START, END) == set()


def test_a_file_of_another_size_is_not_complete(tmp_path: Path) -> None:
    manifest = _done_tile(tmp_path)
    (tmp_path / RTC_NAME).write_bytes(b"II*\x00rewritten")
    assert manifest.complete_tiles(ASSETS, START, END) == set()


def test_rows_of_another_window_are_not_complete(tmp_path: Path) -> None:
    manifest = _done_tile(tmp_path)
    assert manifest.complete_tiles(ASSETS, START, date(2024, 9, 13)) == set()


def test_no_expected_assets_means_no_tile_is_complete(tmp_path: Path) -> None:
    """The per-pass method expects nothing and must never skip early."""
    assert _done_tile(tmp_path).complete_tiles(frozenset(), START, END) == set()


def test_complete_tiles_survives_a_parquet_round_trip(tmp_path: Path) -> None:
    """On a real resume the rows come back from ``manifest.parquet``, not from memory."""
    _done_tile(tmp_path).write()
    resumed = RunManifest.load(tmp_path, {})
    assert resumed.complete_tiles(ASSETS, START, END) == {TILE}


def test_complete_tiles_survives_a_failed_listing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unreachable store means no tile is known complete, not a run that never starts."""
    manifest = _done_tile(tmp_path)

    def boom(directory: Path) -> dict[str, int]:  # noqa: ARG001
        msg = "Forbidden"
        raise PermissionError(msg)

    monkeypatch.setattr("glace_production.manifest._listing", boom)
    with caplog.at_level(logging.WARNING):
        assert manifest.complete_tiles(ASSETS, START, END) == set()
    assert "Forbidden" in caplog.text


def test_a_checkpoint_without_paths_does_not_rewrite_the_manifest(tmp_path: Path) -> None:
    """A tile that returned nothing changed no row, so re-uploading the manifest is waste."""
    RunManifest(tmp_path, {}).checkpoint([])
    assert not (tmp_path / MANIFEST_NAME).exists()


def test_a_digestless_row_is_not_complete_after_a_parquet_round_trip(tmp_path: Path) -> None:
    """A null digest must come back as ``None``, not as a NaN that passes the check."""
    _done_tile(tmp_path, digestless=RTC_NAME).write()
    resumed = RunManifest.load(tmp_path, {})
    assert resumed.complete_tiles(ASSETS, START, END) == set()


def test_a_checkpoint_of_carried_rows_does_not_rewrite_the_manifest(
    archive: tuple[Path, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resumed tile whose files all match their rows changed nothing to upload."""
    comp_dir, checksums = archive
    manifest = RunManifest.load(comp_dir, checksums)
    manifest.record(sorted(comp_dir.glob("*.tif")))
    manifest.write()

    writes: list[RunManifest] = []

    # A function, not ``writes.append``: a builtin on the class does not bind
    # ``self``, so the call would raise inside ``flush`` and be swallowed.
    def write(self: RunManifest) -> None:
        writes.append(self)

    monkeypatch.setattr(RunManifest, "write", write)
    RunManifest.load(comp_dir, {}).checkpoint(sorted(comp_dir.glob("*.tif")))
    assert writes == []


def test_a_checkpoint_retries_a_write_that_failed_before(
    archive: tuple[Path, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rows a failed checkpoint could not persist go out with the next one, even an empty one."""
    comp_dir, checksums = archive
    manifest = RunManifest.load(comp_dir, checksums)
    real_write = RunManifest.write

    def boom(self: RunManifest) -> None:  # noqa: ARG001 -- monkeypatched onto the class, so it takes self
        msg = "endpoint unreachable"
        raise OSError(msg)

    monkeypatch.setattr(RunManifest, "write", boom)
    manifest.checkpoint(sorted(comp_dir.glob("*.tif")))
    assert not (comp_dir / MANIFEST_NAME).exists()

    monkeypatch.setattr(RunManifest, "write", real_write)
    manifest.checkpoint([])
    assert len(RunManifest.load(comp_dir, {}).frame) == 2
