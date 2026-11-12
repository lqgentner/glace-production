"""The composite driver's completeness guarantee.

A composite may only be written from the full set of burst products its search
returned. Otherwise an outage — CDSE, the object store — yields a composite
built from whatever survived, and the rerun that produces the missing bursts
skips the tile because a composite already exists. The incomplete one survives.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pandas import Timestamp
import pytest
from s1bursts import FileManager, ResolutionWeighter, TimeRange
from s1bursts.burst import Burst, BurstCollection
from s1bursts.errors import (
    BurstMapUnavailableError,
    CompositeInputError,
    EmptyCompositeError,
    InconsistentInputsError,
    ProductSettingsMismatchError,
)
from s1bursts.product import CohLRWComposite, RTCLRWComposite
from s1bursts.summary import ErrorResult, OperationSummary
from shapely import box

from glace_production import composites
from glace_production.composites import (
    CompositeConfig,
    DemConfig,
    DirectoryConfig,
    IncompleteCellError,
    ensure_burst_map,
    expected_assets,
    find_pairs,
    require_complete,
)
from glace_production.config import Settings, settings
from glace_production.manifest import RunManifest, parse_composite_name


def _summary(n_total: int, n_failed: int) -> OperationSummary[str]:
    ok = n_total - n_failed
    return OperationSummary(
        n_total=n_total,
        n_processed=ok,
        n_skipped=0,
        n_failed=n_failed,
        products=[f"product-{i}" for i in range(ok)],
        failed_items=[
            ErrorResult(source=f"burst-{i}", error=TimeoutError()) for i in range(n_failed)
        ],
    )


def test_a_complete_step_passes() -> None:
    require_complete("backscatter", _summary(100, 0), geobox="32TLR48", pol="VV")


def test_any_failure_refuses_the_cell() -> None:
    with pytest.raises(IncompleteCellError, match="backscatter: 95 of 100 failed") as excinfo:
        require_complete("backscatter", _summary(100, 95), geobox="32TLR48", pol="VV")
    assert "32TLR48" in str(excinfo.value)
    assert "VV" in str(excinfo.value)


def test_one_failure_is_enough() -> None:
    with pytest.raises(IncompleteCellError):
        require_complete("coherence", _summary(72, 1), geobox="32TLR48", pol="VH")


# ---------------------------------------------------------------------------
# Pair selection: which acquisitions may form a 12-day coherence pair.
# ---------------------------------------------------------------------------
#
# s1bursts' ``find_pairs`` accepts any spacing within a day of the baseline by
# default, so a 13-day pair from two satellites a day apart in phase (the
# June 2026 constellation reshuffle) passed as a 12-day one and the composite
# refused the mixed inputs. A genuine repeat pass differs from the baseline by
# seconds, but one CDSE burst timestamp on orbit 088 in 2019 reads two hours off
# its product's sensing time, so the tolerance must keep hours and drop a day.


def _bursts(dates: list[str]) -> BurstCollection:
    return BurstCollection.from_bursts(
        [
            Burst(
                burst_id=30346,
                sub_swath="IW1",
                polarization="VV",
                date=Timestamp(d),
                _geometry=box(7.0, 46.0, 8.0, 47.0),
            )
            for d in dates
        ],
    )


def _pairs(dates: list[str]) -> set[tuple[str, str]]:
    config = CompositeConfig.model_construct(baseline_days=12)
    return {
        (pair.primary.date.strftime("%Y-%m-%d"), pair.secondary.date.strftime("%Y-%m-%d"))
        for pair in find_pairs(_bursts(dates), config)
    }


def test_a_pair_one_day_off_the_baseline_is_not_a_12_day_pair() -> None:
    assert _pairs(["2026-06-19 05:51:57", "2026-07-02 05:51:03"]) == set()


def test_a_repeat_pass_seconds_off_the_baseline_is_a_pair() -> None:
    assert _pairs(["2026-06-07 05:51:57", "2026-06-19 05:51:57"]) == {("2026-06-07", "2026-06-19")}


def test_a_timestamp_two_hours_off_still_pairs() -> None:
    assert _pairs(["2019-06-17 17:22:08", "2019-06-29 15:22:08"]) == {("2019-06-17", "2019-06-29")}


def test_the_tolerance_comes_from_the_config() -> None:
    config = CompositeConfig.model_construct(baseline_days=12, pair_tolerance_hours=1)
    pairs = find_pairs(_bursts(["2019-06-17 17:22:08", "2019-06-29 15:22:08"]), config)
    assert len(pairs) == 0


# ---------------------------------------------------------------------------
# Resuming: a tile whose composites are all there is skipped before the search.
# ---------------------------------------------------------------------------

WINDOW = TimeRange("2023-07-16", "2023-10-13")


def _lrw_config(**fields: Any) -> CompositeConfig:
    values: dict[str, Any] = {
        "method": "lrw",
        "window": "alps-total",
        "write_qa": True,
        "polarizations": ["VV", "VH"],
        "baseline_days": 12,
    }
    return CompositeConfig.model_construct(**(values | fields))


def test_an_lrw_run_with_qa_expects_twelve_assets_per_tile() -> None:
    assets = expected_assets(_lrw_config())
    assert len(assets) == 12
    assert "coh12_vh_qa_cqm" in assets


def test_without_qa_only_the_measurements_are_expected() -> None:
    config = _lrw_config(write_qa=False, polarizations=["VV"], baseline_days=6)
    assert expected_assets(config) == {"rtc_vv", "coh06_vv"}


def test_the_pass_method_expects_nothing_so_it_never_skips_early() -> None:
    assert expected_assets(_lrw_config(method="pass")) == frozenset()


def test_expected_assets_are_the_names_s1bursts_writes(tmp_path: Path) -> None:
    """The skip compares asset keys; this pins them to the writer's filenames."""
    fm = FileManager(tmp_path)
    # The writer names every composite with the window, whatever the inputs span.
    start, end = ResolutionWeighter._composite_dates(
        (Timestamp("2023-07-22"), Timestamp("2023-10-02")),
        WINDOW,
    )
    assert (start.date(), end.date()) == (WINDOW.start, WINDOW.end)
    products = [
        product
        for pol in ("VV", "VH")
        for product in (
            RTCLRWComposite("33UTP83", start, end, pol, layers=RTCLRWComposite.supported_layers),
            CohLRWComposite(
                "33UTP83", start, end, pol, 12, layers=CohLRWComposite.supported_layers
            ),
        )
    ]
    keys: set[str] = set()
    for product in products:
        for layer in product.layers:
            parsed = parse_composite_name(fm.get_product_path(product, layer).name)
            assert parsed is not None
            assert (parsed.start.date(), parsed.end.date()) == (WINDOW.start, WINDOW.end)
            keys.add(parsed.asset_key)
    assert keys == expected_assets(_lrw_config())


@dataclass
class _GeoBox:
    mgrs_code: str


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    overwrite: bool,
    raises: dict[str, Exception] | None = None,
) -> list[tuple[str, str]]:
    """Run the driver over tiles DONE and TODO with everything but the skip stubbed out.

    DONE has all twelve composites in the manifest and on disk; TODO has none.
    Returns the (tile, polarization) cells that reached burst processing.
    ``raises`` maps a tile to the error its RTC compose call raises; such a tile's
    cells return one placeholder RTC product so that the compose call is reached.
    """
    config = _lrw_config(overwrite=overwrite, skip_cdse=False, scope="test")
    object.__setattr__(config, "time_range", WINDOW)
    rows: dict[str, dict[str, Any]] = {}
    for key in expected_assets(config):
        name = f"{key}.tif"
        (tmp_path / name).write_bytes(key.encode())
        rows[name] = {
            "filename": name,
            "tile": "DONE",
            "asset_key": key,
            "qa": None,
            "composite_start": WINDOW.start.isoformat(),
            "composite_end": WINDOW.end.isoformat(),
            "size": len(key),
            "sha256": "1220" + "0" * 64,
        }
    RunManifest(tmp_path, {}, rows).write()

    @dataclass
    class _Weighter:
        checksums: dict[str, str]

    reached: list[tuple[str, str]] = []

    raises = raises or {}

    def process_cell(*, geobox: _GeoBox, pol: str, **_: Any) -> tuple[list[Any], list[Any]]:
        reached.append((geobox.mgrs_code, pol))
        return (["rtc"] if geobox.mgrs_code in raises else []), []

    def compose_rtc(_weighter: Any, _prods: Any, geobox: _GeoBox, *_a: Any, **_k: Any) -> None:
        raise raises[geobox.mgrs_code]

    fm = FileManager(tmp_path, comp_dir=tmp_path)
    monkeypatch.setattr(CompositeConfig, "to_file_manager", lambda _self: fm)
    monkeypatch.setattr(CompositeConfig, "to_snap_configs", lambda _self: (None, None))
    monkeypatch.setattr(CompositeConfig, "to_snap_runtime_config", lambda _self: None)
    monkeypatch.setattr(CompositeConfig, "save_yaml", lambda _self, _path: None)
    monkeypatch.setattr(composites, "resolve_aoi", lambda *_a, **_k: None)
    monkeypatch.setattr(
        composites,
        "MGRSGrid",
        lambda *_a, **_k: [_GeoBox("DONE"), _GeoBox("TODO")],
    )
    monkeypatch.setattr(composites, "BurstCatalog", lambda *_a, **_k: None)
    monkeypatch.setattr(composites, "BurstProcessor", lambda *_a, **_k: None)
    monkeypatch.setattr(composites, "ResolutionWeighter", lambda *_a, **_k: _Weighter({}))
    monkeypatch.setattr(composites, "_process_cell", process_cell)
    monkeypatch.setattr(composites, "_compose_rtc", compose_rtc)
    composites.run_composite(config)
    return reached


def test_a_resumed_run_searches_only_the_incomplete_tiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _run(tmp_path, monkeypatch, overwrite=False) == [("TODO", "VV"), ("TODO", "VH")]


def test_overwrite_searches_every_tile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reached = _run(tmp_path, monkeypatch, overwrite=True)
    assert reached == [("DONE", "VV"), ("DONE", "VH"), ("TODO", "VV"), ("TODO", "VH")]


# ---------------------------------------------------------------------------
# Error policy: a composite error loses its tile, a mismatch stops the run.
# ---------------------------------------------------------------------------

ALL_CELLS = [("DONE", "VV"), ("DONE", "VH"), ("TODO", "VV"), ("TODO", "VH")]


def test_an_empty_cell_is_logged_and_the_run_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    reached = _run(
        tmp_path,
        monkeypatch,
        overwrite=True,
        raises={"DONE": EmptyCompositeError("no valid pixel")},
    )
    assert reached == ALL_CELLS
    assert "no valid pixel" in caplog.text


def test_a_composite_error_skips_only_its_tile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    reached = _run(
        tmp_path,
        monkeypatch,
        overwrite=True,
        raises={"DONE": CompositeInputError("unreadable input")},
    )
    assert reached == ALL_CELLS
    assert "unreadable input" in caplog.text


def test_inconsistent_inputs_stop_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(InconsistentInputsError):
        _run(
            tmp_path,
            monkeypatch,
            overwrite=True,
            raises={"DONE": InconsistentInputsError("two SNAP versions")},
        )


def test_settings_mismatch_stops_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ProductSettingsMismatchError):
        _run(
            tmp_path,
            monkeypatch,
            overwrite=True,
            raises={"DONE": ProductSettingsMismatchError("window differs")},
        )


# ---------------------------------------------------------------------------
# The burst-ID map: a fresh VM has none cached, and the offline index needs it.
# ---------------------------------------------------------------------------


def test_a_missing_burst_map_is_downloaded_once(monkeypatch: pytest.MonkeyPatch) -> None:
    downloads: list[str] = []
    loads: list[str] = []

    def load(*, mode: str) -> None:
        loads.append(mode)
        if not downloads:
            msg = "The Sentinel-1 IW burst-ID map ... has not been downloaded."
            raise BurstMapUnavailableError(msg)

    monkeypatch.setattr(composites, "create_s1_geometry_lut", load)
    monkeypatch.setattr(composites, "download_burst_map", lambda *, mode: downloads.append(mode))
    ensure_burst_map("IW")
    assert downloads == ["IW"]
    assert loads == ["IW", "IW"]


def test_a_cached_burst_map_is_not_downloaded(monkeypatch: pytest.MonkeyPatch) -> None:
    downloads: list[str] = []
    monkeypatch.setattr(composites, "create_s1_geometry_lut", lambda **_: None)
    monkeypatch.setattr(composites, "download_burst_map", lambda *, mode: downloads.append(mode))
    ensure_burst_map("IW")
    assert downloads == []


# ---------------------------------------------------------------------------
# The DEM source: relative paths sit under the data directory.
# ---------------------------------------------------------------------------


def _dem_config(source: str) -> CompositeConfig:
    return CompositeConfig.model_construct(
        scope="alps", year=2023, dem=DemConfig(source=source, name="GLO-30")
    )


def test_a_relative_dem_source_resolves_under_the_data_dir() -> None:
    resolved = _dem_config("dem/{scope}.tif").resolve_dem_source()
    assert resolved == settings.data_dir / "dem" / "alps.tif"


def test_an_absolute_dem_source_is_kept() -> None:
    assert _dem_config("/elsewhere/dem.tif").resolve_dem_source() == Path("/elsewhere/dem.tif")


def test_a_built_in_dem_name_is_passed_through() -> None:
    config = CompositeConfig.model_construct(
        scope="alps", year=2023, dem=DemConfig(source="Copernicus 30m Global DEM")
    )
    assert config.resolve_dem_source() == "Copernicus 30m Global DEM"


# ---------------------------------------------------------------------------
# Run directories: relative under the data directory, absolute and URIs as given.
# ---------------------------------------------------------------------------


def test_run_dirs_resolve_like_every_config_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GLACIERS_DATA_DIR", "/data")
    # A profile lets the s3:// URI open; building the path reaches no store.
    monkeypatch.setenv("GLACIERS_S3_PROFILE", "test")
    monkeypatch.setenv("GLACIERS_S3_ENDPOINT_URL", "https://s3.example.org")
    monkeypatch.setattr(composites, "settings", Settings())
    # FileManager creates its directories; capture them instead.
    dirs: dict[str, Any] = {}
    monkeypatch.setattr(composites, "FileManager", lambda **kwargs: dirs.update(kwargs))
    config = CompositeConfig.model_construct(
        scope="alps",
        year=2023,
        dirs=DirectoryConfig(
            base_dir="{scope}/{year}",
            slc_dir="s3://bucket/{scope}/{year}/s1-slc",
            comp_dir="/elsewhere/{year}/s1-comp",
        ),
    )
    config.to_file_manager()
    assert dirs["base_dir"] == Path("/data/alps/2023")
    assert str(dirs["slc_dir"]) == "s3://bucket/alps/2023/s1-slc"
    assert dirs["slc_dir"].storage_options["profile"] == "test"
    assert dirs["prod_dir"] is None
    assert dirs["comp_dir"] == Path("/elsewhere/2023/s1-comp")
