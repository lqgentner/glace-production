"""Output layout, resume rule and sample joins of the validation store."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading

import numpy as np
import pandas as pd
import pytest

from glace_production.validation.config import LocationsConfig
from glace_production.validation.store import KEY_COLUMNS, ConfigMismatchError, ValidationStore


def _config(**overrides: object) -> LocationsConfig:
    base = {
        "name": "t",
        "labels_dir": "labels",
        "inventory": "SGI2023",
        "years": [2023],
        "dem": "dem.tif",
    }
    return LocationsConfig.model_validate(base | overrides)


def _locations(tile: str, year: int, n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tile": tile,
            "year": np.int16(year),
            "row": np.arange(n, dtype=np.uint16),
            "col": np.arange(n, dtype=np.uint16),
            "label": np.uint8(1),
        }
    )


def test_check_config_writes_then_refuses_changes(tmp_path: Path) -> None:
    store = ValidationStore(tmp_path)
    store.check_config(store.locations_dir, _config(), overwrite=False)
    assert (store.locations_dir / "config.yaml").exists()
    store.check_config(store.locations_dir, _config(), overwrite=False)  # identical: fine
    with pytest.raises(ConfigMismatchError, match="pad_m"):
        store.check_config(store.locations_dir, _config(pad_m=1100.0), overwrite=False)
    store.check_config(store.locations_dir, _config(pad_m=1100.0), overwrite=True)
    assert "1100.0" in (store.locations_dir / "config.yaml").read_text()


def test_pairs_and_load_samples_left_join(tmp_path: Path) -> None:
    store = ValidationStore(tmp_path)
    store.write_parquet(
        _locations("32TMS55", 2023, 4), store.pair_path(store.locations_dir, "32TMS55", 2023)
    )
    store.write_parquet(
        _locations("32TMS56", 2022, 2), store.pair_path(store.locations_dir, "32TMS56", 2022)
    )
    assert store.list_pairs(store.locations_dir) == [("32TMS55", 2023), ("32TMS56", 2022)]

    feats = _locations("32TMS55", 2023, 4)[KEY_COLUMNS].assign(coh_vv=np.float32(0.5))
    store.write_parquet(feats, store.pair_path(store.samples_dir("d"), "32TMS55", 2023))

    joined = store.load_samples(["d"], columns=["coh_vv"])
    assert len(joined) == 6
    assert joined["coh_vv"].notna().sum() == 4  # the second pair has no features: NaN
    assert list(joined.columns[:4]) == KEY_COLUMNS
    assert joined["tile"].dtype == "category"


def test_load_samples_rejects_duplicate_keys(tmp_path: Path) -> None:
    store = ValidationStore(tmp_path)
    store.write_parquet(
        _locations("32TMS55", 2023, 3), store.pair_path(store.locations_dir, "32TMS55", 2023)
    )
    dup = pd.concat([_locations("32TMS55", 2023, 3)] * 2)[KEY_COLUMNS].assign(x=1.0)
    store.write_parquet(dup, store.pair_path(store.samples_dir("d"), "32TMS55", 2023))
    with pytest.raises(ValueError, match="duplicate"):
        store.load_samples(["d"], columns=["x"])


def test_load_samples_joins_multiple_datasets(tmp_path: Path) -> None:
    store = ValidationStore(tmp_path)
    store.write_parquet(
        _locations("32TMS55", 2023, 4), store.pair_path(store.locations_dir, "32TMS55", 2023)
    )
    store.write_parquet(
        _locations("32TMS56", 2022, 2), store.pair_path(store.locations_dir, "32TMS56", 2022)
    )

    feats1 = _locations("32TMS55", 2023, 4)[KEY_COLUMNS].assign(coh_vv=np.float32(0.5))
    store.write_parquet(feats1, store.pair_path(store.samples_dir("d1"), "32TMS55", 2023))
    feats2 = _locations("32TMS55", 2023, 4)[KEY_COLUMNS].assign(bs_vv=np.float32(-10.0))
    store.write_parquet(feats2, store.pair_path(store.samples_dir("d2"), "32TMS55", 2023))

    joined = store.load_samples(["d1", "d2"], columns=None)
    assert len(joined) == 6
    assert joined["coh_vv"].notna().sum() == 4
    assert joined["bs_vv"].notna().sum() == 4


def test_load_samples_projects_columns_per_dataset(tmp_path: Path) -> None:
    store = ValidationStore(tmp_path)
    store.write_parquet(
        _locations("32TMS55", 2023, 4), store.pair_path(store.locations_dir, "32TMS55", 2023)
    )
    store.write_parquet(
        _locations("32TMS56", 2022, 2), store.pair_path(store.locations_dir, "32TMS56", 2022)
    )

    # d1 and d2 hold disjoint feature layers: neither carries the other's column.
    feats1 = _locations("32TMS55", 2023, 4)[KEY_COLUMNS].assign(a=np.float32(1.0))
    store.write_parquet(feats1, store.pair_path(store.samples_dir("d1"), "32TMS55", 2023))
    feats2 = _locations("32TMS56", 2022, 2)[KEY_COLUMNS].assign(b=np.float32(2.0))
    store.write_parquet(feats2, store.pair_path(store.samples_dir("d2"), "32TMS56", 2022))

    joined = store.load_samples(["d1", "d2"], columns=["a", "b"])
    assert len(joined) == 6
    assert joined["a"].notna().sum() == 4  # only d1's pair has "a"
    assert joined["b"].notna().sum() == 2  # only d2's pair has "b"


def test_load_samples_rejects_unknown_column(tmp_path: Path) -> None:
    store = ValidationStore(tmp_path)
    store.write_parquet(
        _locations("32TMS55", 2023, 4), store.pair_path(store.locations_dir, "32TMS55", 2023)
    )
    feats1 = _locations("32TMS55", 2023, 4)[KEY_COLUMNS].assign(a=np.float32(1.0))
    store.write_parquet(feats1, store.pair_path(store.samples_dir("d1"), "32TMS55", 2023))
    feats2 = _locations("32TMS55", 2023, 4)[KEY_COLUMNS].assign(b=np.float32(2.0))
    store.write_parquet(feats2, store.pair_path(store.samples_dir("d2"), "32TMS55", 2023))

    with pytest.raises(ValueError, match="nope"):
        store.load_samples(["d1", "d2"], columns=["nope"])


def test_load_samples_rejects_colliding_columns(tmp_path: Path) -> None:
    store = ValidationStore(tmp_path)
    store.write_parquet(
        _locations("32TMS55", 2023, 4), store.pair_path(store.locations_dir, "32TMS55", 2023)
    )

    feats1 = _locations("32TMS55", 2023, 4)[KEY_COLUMNS].assign(coh_vv=np.float32(0.5))
    store.write_parquet(feats1, store.pair_path(store.samples_dir("d1"), "32TMS55", 2023))
    feats2 = _locations("32TMS55", 2023, 4)[KEY_COLUMNS].assign(coh_vv=np.float32(0.6))
    store.write_parquet(feats2, store.pair_path(store.samples_dir("d2"), "32TMS55", 2023))

    with pytest.raises(ValueError, match="coh_vv"):
        store.load_samples(["d1", "d2"], columns=None)


def test_report_update_replaces_rows(tmp_path: Path) -> None:
    store = ValidationStore(tmp_path)
    store.update_report(store.locations_dir, [{"tile": "a", "year": 2023, "n": 1}])
    store.update_report(
        store.locations_dir,
        [{"tile": "a", "year": 2023, "n": 2}, {"tile": "b", "year": 2023, "n": 3}],
    )
    report = store.read_report(store.locations_dir)
    assert report.set_index(["tile", "year"])["n"].to_dict() == {("a", 2023): 2, ("b", 2023): 3}


def test_check_config_starts_a_new_generation_and_wipes_outputs(tmp_path: Path) -> None:
    store = ValidationStore(tmp_path)
    directory = tmp_path / "stage"
    assert store.check_config(directory, _config(), overwrite=False) is True
    generation = store.generation(directory)
    (directory / "old.parquet").write_text("x")
    (directory / "sub").mkdir()
    (directory / "sub" / "deep.parquet").write_text("y")
    # unchanged config: resume, outputs kept, generation kept
    assert store.check_config(directory, _config(), overwrite=False) is False
    assert (directory / "old.parquet").exists()
    assert store.generation(directory) == generation
    # changed config without overwrite: refused
    with pytest.raises(ConfigMismatchError, match="different generation"):
        store.check_config(directory, _config(pad_m=1100.0), overwrite=False)
    assert (directory / "old.parquet").exists()
    # changed config with overwrite: every output gone, new generation
    assert store.check_config(directory, _config(pad_m=1100.0), overwrite=True) is True
    assert not (directory / "old.parquet").exists()
    assert not (directory / "sub" / "deep.parquet").exists()
    assert store.generation(directory) != generation
    # overwrite of an unchanged config keeps the generation and the outputs
    (directory / "new.parquet").write_text("z")
    assert store.check_config(directory, _config(pad_m=1100.0), overwrite=True) is False
    assert (directory / "new.parquet").exists()


def test_upstream_stamp_binds_a_stage_to_its_inputs(tmp_path: Path) -> None:
    store = ValidationStore(tmp_path)
    directory = tmp_path / "stage"
    store.check_config(directory, _config(), overwrite=False, upstream={"locations": "a"})
    (directory / "out.parquet").write_text("x")
    store.verify_config(directory, _config(), upstream={"locations": "a"})
    with pytest.raises(ConfigMismatchError, match=r"upstream\.yaml"):
        store.check_config(directory, _config(), overwrite=False, upstream={"locations": "b"})
    with pytest.raises(ConfigMismatchError, match="different generation"):
        store.verify_config(directory, _config(), upstream={"locations": "b"})
    with pytest.raises(ConfigMismatchError, match="different generation"):
        store.verify_config(directory, _config(pad_m=1100.0), upstream={"locations": "a"})
    with pytest.raises(ConfigMismatchError, match="has not been run"):
        store.verify_config(tmp_path / "never", _config())
    store.check_config(directory, _config(), overwrite=True, upstream={"locations": "b"})
    assert not (directory / "out.parquet").exists()


def test_generation_needs_a_checked_stage(tmp_path: Path) -> None:
    store = ValidationStore(tmp_path)
    directory = tmp_path / "stage"
    with pytest.raises(FileNotFoundError):
        store.generation(directory)
    store.check_config(directory, _config(), overwrite=False)
    generation = store.generation(directory)
    assert len(generation) == 32
    assert store.generation(directory) == generation


def test_collect_reports_keeps_finished_pairs_on_failure(tmp_path: Path) -> None:
    store = ValidationStore(tmp_path)
    directory = tmp_path / "stage"
    directory.mkdir()
    started = threading.Event()  # the slow pair is running before the failure fires
    release = threading.Event()  # ... and finishes only after it did

    def ok(tile: str) -> list[dict[str, object]]:
        return [{"tile": tile, "year": 2023, "n": 1}]

    def slow() -> list[dict[str, object]]:
        started.set()
        release.wait(5)
        return [{"tile": "slow", "year": 2023, "n": 1}]

    def bad() -> list[dict[str, object]]:
        started.wait(5)
        release.set()
        msg = "boom"
        raise RuntimeError(msg)

    def pending() -> list[dict[str, object]]:
        return [{"tile": "pending", "year": 2023, "n": 1}]

    with ThreadPoolExecutor(max_workers=2) as pool:
        # two workers: `slow` and `bad` occupy them while `pending` waits in
        # the queue; whether the freed worker starts it before the cancel
        # lands is a race, so only the running pair's fate is asserted
        futures = [pool.submit(slow), pool.submit(bad), pool.submit(pending)]
        with pytest.raises(RuntimeError, match="boom"):
            store.collect_reports(directory, futures, checkpoint_every=1)
    report = store.read_report(directory)
    assert "slow" in set(report["tile"])
    assert set(report["tile"]) <= {"slow", "pending"}
    assert futures[2].cancelled() or futures[2].done()
