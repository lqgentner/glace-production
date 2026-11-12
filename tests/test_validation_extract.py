"""Feature extraction at stored locations from a synthetic tiled dataset."""

from __future__ import annotations

from pathlib import Path
import uuid

from freezebase.mgrs import MGRSGeoBox
import numpy as np
import pandas as pd
import pytest
import rasterio

from glace_production.validation.config import DatasetConfig, LayerSpec
from glace_production.validation.extract import (
    layer_tiles,
    read_layer_padded,
    resolve_layer_path,
    window_stats,
    write_samples,
)
from glace_production.validation.store import KEY_COLUMNS, ConfigMismatchError, ValidationStore

TILE = "32TMS55"


def _write_tile(path: Path, value: float, *, tile: str = TILE, nodata: bool = True) -> Path:
    """Write a constant tile; with ``nodata`` its (0, 0) pixel is NaN."""
    gb = MGRSGeoBox.from_mgrs(tile)
    data = np.full(gb.shape.yx, value, dtype=np.float32)
    if nodata:
        data[0, 0] = np.nan
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=1000,
        height=1000,
        count=1,
        dtype="float32",
        crs=gb.crs.to_wkt() if gb.crs is not None else None,
        transform=gb.affine,
        nodata=np.nan,
    ) as dst:
        dst.write(data, 1)
    return path


EAST = "32TMS65"  # the tile's eastern neighbour, same UTM zone
NORTH = "32TMS56"
NORTH_EAST = "32TMS66"


def _dataset(root: Path) -> DatasetConfig:
    for year in (2022, 2023):
        d = root / "comp" / str(year)
        _write_tile(d / f"S1_COH12_LRW_{TILE}_VV_x.tif", 0.4)
        _write_tile(d / f"S1_COH12_LRW_{EAST}_VV_x.tif", 0.8, tile=EAST, nodata=False)
        _write_tile(d / f"S1_COH12_LRW_{NORTH}_VV_x.tif", 0.6, tile=NORTH, nodata=False)
        _write_tile(d / f"S1_COH12_LRW_{NORTH_EAST}_VV_x.tif", 1.0, tile=NORTH_EAST, nodata=False)
        _write_tile(d / f"S1_RTC_LRW_{TILE}_VV_x.tif", 0.01)  # linear power -> -20 dB
    return DatasetConfig(
        name="d",
        comp_dir="comp/{year}",
        layers={
            "coh_vv": LayerSpec(
                pattern="S1_COH12_LRW_{tile}_VV_*.tif",
                kind="coherence",
                derive=[{"stat": "mean", "window": 5}, {"stat": "std", "window": 5}],
            ),
            "rtc_vv": LayerSpec(pattern="S1_RTC_LRW_{tile}_VV_*.tif", kind="backscatter"),
        },
    )


def _locations(store: ValidationStore, year: int) -> None:
    frame = pd.DataFrame(
        {
            "tile": TILE,
            "year": np.int16(year),
            "row": np.array([0, 5, 500, 999, 0], np.uint16),
            "col": np.array([0, 5, 999, 0, 999], np.uint16),
            "label": np.uint8(1),
        }
    )
    store.write_parquet(frame, store.pair_path(store.locations_dir, TILE, year))
    # The generation id a stage run writes beside its outputs, once per directory.
    generation = store.locations_dir / "generation.txt"
    if not generation.exists():
        generation.write_text(uuid.uuid4().hex)


def test_resolve_layer_path_requires_exactly_one_match(tmp_path: Path) -> None:
    spec = LayerSpec(pattern="S1_COH12_LRW_{tile}_VV_*.tif", kind="coherence")
    with pytest.raises(FileNotFoundError):
        resolve_layer_path(tmp_path, spec, TILE)
    _write_tile(tmp_path / f"S1_COH12_LRW_{TILE}_VV_a.tif", 1.0)
    assert resolve_layer_path(tmp_path, spec, TILE).name.endswith("_a.tif")
    _write_tile(tmp_path / f"S1_COH12_LRW_{TILE}_VV_b.tif", 1.0)
    with pytest.raises(ValueError, match="2 matches"):
        resolve_layer_path(tmp_path, spec, TILE)


def test_write_samples_extracts_and_converts(tmp_path: Path) -> None:
    store = ValidationStore(tmp_path / "store")
    _locations(store, 2022)
    _locations(store, 2023)
    config = _dataset(tmp_path)
    report = write_samples(config, store, data_root=tmp_path)
    assert store.list_pairs(store.samples_dir("d")) == [(TILE, 2022), (TILE, 2023)]
    samples = store.read_parquet(store.pair_path(store.samples_dir("d"), TILE, 2023))
    assert list(samples.columns) == [
        *KEY_COLUMNS,
        "coh_vv",
        "coh_vv_mean5",
        "coh_vv_std5",
        "rtc_vv",
    ]
    assert pd.isna(samples.loc[0, "coh_vv"])  # the (0, 0) nodata pixel
    assert samples.loc[1, "coh_vv"] == pytest.approx(0.4)
    assert samples.loc[1, "rtc_vv"] == pytest.approx(-20.0)
    # window statistics of a constant tile: the mean is the value, the std 0;
    # NaN at the nodata corner and at the western edge (999, 0), which has no
    # neighbour; the eastern edge (500, 999) borrows its halo from the 0.8 tile
    # and the north-eastern corner (0, 999) from three neighbours
    assert samples.loc[1, "coh_vv_mean5"] == pytest.approx(0.4)
    assert samples.loc[1, "coh_vv_std5"] == pytest.approx(0.0)
    assert samples["coh_vv_mean5"].isna().tolist() == [True, False, False, True, False]
    assert samples.loc[2, "coh_vv_mean5"] == pytest.approx(0.4 * 15 / 25 + 0.8 * 10 / 25)
    assert samples["coh_vv_std5"].to_numpy()[2] > 0.1
    corner = (9 * 0.4 + 6 * 0.6 + 6 * 0.8 + 4 * 1.0) / 25
    assert samples.loc[4, "coh_vv_mean5"] == pytest.approx(corner)
    rows = report.set_index(["tile", "year", "layer"])
    assert rows.loc[(TILE, 2023, "coh_vv"), "n_nan"] == 1
    assert rows.loc[(TILE, 2023, "coh_vv_std5"), "n_nan"] == 2
    assert (store.samples_dir("d") / "config.yaml").exists()


def test_write_samples_missing_layer(tmp_path: Path) -> None:
    store = ValidationStore(tmp_path / "store")
    _locations(store, 2022)
    _locations(store, 2024)  # no composites for 2024
    config = _dataset(tmp_path)
    with pytest.raises(FileNotFoundError, match="2024"):
        write_samples(config, store, data_root=tmp_path)
    report = write_samples(config, store, data_root=tmp_path, allow_missing=True, overwrite=True)
    assert store.list_pairs(store.samples_dir("d")) == [(TILE, 2022)]
    assert report[report.year == 2024]["missing"].all()


def test_write_samples_checkpoints_completed_pairs_on_failure(tmp_path: Path) -> None:
    store = ValidationStore(tmp_path / "store")
    _locations(store, 2022)
    _locations(store, 2024)  # no composites for 2024
    config = _dataset(tmp_path)
    # A single worker makes the two pairs finish in submission (sorted) order,
    # so the 2022 pair is guaranteed to have checkpointed before 2024 raises.
    with pytest.raises(FileNotFoundError, match="2024"):
        write_samples(config, store, data_root=tmp_path, max_workers=1)
    report = store.read_report(store.samples_dir("d"))
    rows = report.set_index(["tile", "year", "layer"])
    assert rows.loc[(TILE, 2022, "coh_vv"), "n_nan"] == 1
    assert rows.loc[(TILE, 2022, "rtc_vv"), "n_nan"] == 1
    assert 2024 not in report["year"].to_numpy()


def test_write_samples_skips_existing_and_respects_resume_rule(tmp_path: Path) -> None:
    store = ValidationStore(tmp_path / "store")
    _locations(store, 2022)
    config = _dataset(tmp_path)
    write_samples(config, store, data_root=tmp_path)
    changed = config.model_copy(update={"name": "d", "comp_dir": "other/{year}"})
    with pytest.raises(RuntimeError, match="comp_dir"):
        write_samples(changed, store, data_root=tmp_path)


def test_window_stats_are_nan_aware_and_edge_censored() -> None:
    data = np.arange(49, dtype=np.float32).reshape(7, 7)
    data[1, 1] = np.nan
    mean, std = window_stats(data, 3)
    assert mean.shape == data.shape
    # the 1-pixel edge ring has no full window
    assert np.isnan(mean[0]).all()
    assert np.isnan(mean[:, -1]).all()
    # windows touching the NaN are NaN, the ones beyond it are exact
    assert np.isnan(mean[2, 2])
    assert np.isnan(std[1, 2])
    block = data[2:5, 2:5]
    assert mean[3, 3] == pytest.approx(block.mean())
    assert std[3, 3] == pytest.approx(block.std())
    assert std[4, 4] >= 0


def test_halo_ignores_other_zones_and_keeps_own_nodata(tmp_path: Path) -> None:
    # 32TKR78 sits in the zone-32/31 overlap: the 31TGL squares cover its west
    # (interior included) and its western ring but are ignored; 32TKR88 is its
    # eastern neighbour.
    own, east = "32TKR78", "32TKR88"
    cross = ["31TGL39", "31TGL38"]
    spec = LayerSpec(pattern="S1_COH12_LRW_{tile}_VV_*.tif", kind="coherence")
    _write_tile(tmp_path / f"S1_COH12_LRW_{own}_VV_x.tif", 0.4, tile=own)
    _write_tile(tmp_path / f"S1_COH12_LRW_{east}_VV_x.tif", 0.8, tile=east)
    for code in cross:
        _write_tile(tmp_path / f"S1_COH12_LRW_{code}_VV_x.tif", 0.1, tile=code, nodata=False)
    padded = read_layer_padded(tmp_path, spec, own, halo=2)
    assert padded.shape == (1004, 1004)
    # the tile itself is untouched, its nodata corner included
    assert np.isnan(padded[2, 2])
    assert (padded[2:-2, 2:-2][np.isfinite(padded[2:-2, 2:-2])] == np.float32(0.4)).all()
    # eastern ring: the same-zone neighbour, its own nodata pixel (its (0, 0),
    # at padded (2, 1002)) included; the cross-zone tiles never fill anything
    assert (padded[3:-2, -2:] == np.float32(0.8)).all()
    assert padded[2, -1] == np.float32(0.8)
    assert np.isnan(padded[2, -2])
    # western ring: only cross-zone tiles cover it, so it stays NaN
    assert np.isnan(padded[2:-2, :2]).all()


def test_layer_tiles_rejects_duplicate_files(tmp_path: Path) -> None:
    spec = LayerSpec(pattern="S1_COH12_LRW_{tile}_VV_*.tif", kind="coherence")
    _write_tile(tmp_path / f"S1_COH12_LRW_{TILE}_VV_a.tif", 0.4)
    assert list(layer_tiles(tmp_path, spec)) == [TILE]
    dup = tmp_path / "dup"
    _write_tile(dup / f"S1_COH12_LRW_{TILE}_VV_a.tif", 0.4)
    _write_tile(dup / f"S1_COH12_LRW_{TILE}_VV_b.tif", 0.4)
    with pytest.raises(ValueError, match="several files"):
        layer_tiles(dup, spec)


def test_window_stats_full_window_test_is_exact() -> None:
    # one missing pixel in a large window must still invalidate it
    data = np.ones((80, 80), dtype=np.float32)
    data[40, 40] = np.nan
    mean, _ = window_stats(data, 41)
    assert np.isnan(mean[40, 40])
    assert np.isnan(mean[20, 20])  # window reaches the NaN
    assert np.isnan(mean[20, 19])  # ... and the array edge


def test_overwrite_with_missing_source_removes_the_stale_pair(tmp_path: Path) -> None:
    store = ValidationStore(tmp_path / "store")
    _locations(store, 2022)
    config = _dataset(tmp_path)
    write_samples(config, store, data_root=tmp_path)
    assert store.pair_path(store.samples_dir("d"), TILE, 2022).exists()
    for path in (tmp_path / "comp" / "2022").glob("S1_RTC_*.tif"):
        path.unlink()
    report = write_samples(config, store, data_root=tmp_path, allow_missing=True, overwrite=True)
    assert not store.pair_path(store.samples_dir("d"), TILE, 2022).exists()
    assert report["missing"].all()


def test_extraction_is_bound_to_the_locations_generation(tmp_path: Path) -> None:
    store = ValidationStore(tmp_path / "store")
    _locations(store, 2022)
    config = _dataset(tmp_path)
    write_samples(config, store, data_root=tmp_path)
    # the locations are regenerated: the samples belong to the old generation
    (store.locations_dir / "generation.txt").write_text("0" * 32)
    with pytest.raises(ConfigMismatchError, match=r"upstream\.yaml"):
        write_samples(config, store, data_root=tmp_path)
    write_samples(config, store, data_root=tmp_path, overwrite=True)
