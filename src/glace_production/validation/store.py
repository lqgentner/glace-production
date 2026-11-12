"""Where validation outputs live and how stages read and write them.

Everything sits under one root (``settings.models_dir / "validation"`` by
default). Each stage stores its resolved config before writing outputs and
refuses to resume when the config changed.
"""

from __future__ import annotations

from concurrent.futures import Future, as_completed, wait
import hashlib
import logging
import re
from typing import TYPE_CHECKING
import uuid

import pandas as pd
import pyarrow.parquet as pq
from upath import UPath
import yaml

from glace_production.config import settings

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path

    from glace_production.validation.config import _YamlConfig

logger = logging.getLogger(__name__)

KEY_COLUMNS = ["tile", "year", "row", "col"]
"""Columns identifying one sampled pixel; every table is joined on them."""

_PAIR_RE = re.compile(r"^(?P<tile>\d{2}[A-Z]{3}\d{2})_(?P<year>\d{4})\.parquet$")


type Upstream = Mapping[str, str]
"""Identity of the inputs a stage was produced from, stored beside its config."""


def file_sha256(path: Path | UPath) -> str:
    """Return the hex SHA-256 of a file's bytes."""
    return hashlib.sha256(UPath(path).read_bytes()).hexdigest()


class ConfigMismatchError(RuntimeError):
    """A stage was asked to resume with a config different from the stored one."""


class ValidationStore:
    """Layout of, and access to, the validation outputs under ``root``."""

    def __init__(self, root: Path | UPath) -> None:
        self.root = root

    @classmethod
    def default(cls) -> ValidationStore:
        """Build the store under ``settings.models_dir / "validation"``."""
        return cls(settings.models_dir / "validation")

    # -- layout -------------------------------------------------------------

    @property
    def locations_dir(self) -> Path | UPath:
        """Where sampled locations, per (tile, year), live."""
        return self.root / "locations"

    def samples_dir(self, dataset: str) -> Path | UPath:
        """Where a dataset's extracted feature samples, per (tile, year), live."""
        return self.root / "samples" / dataset

    def separability_path(self, dataset: str) -> Path | UPath:
        """Where a dataset's separability table lives."""
        return self.root / "separability" / f"{dataset}.parquet"

    def run_dir(self, run: str) -> Path | UPath:
        """Root directory of one cross-validation run."""
        return self.root / "runs" / run

    def support_path(self, run: str) -> Path | UPath:
        """Where a run's fold/support assignment lives."""
        return self.run_dir(run) / "support.parquet"

    def predictions_path(self, run: str, feature_set: str, model: str) -> Path | UPath:
        """Where one feature set/model's held-out predictions live."""
        return self.run_dir(run) / "predictions" / feature_set / f"{model}.parquet"

    def thresholds_path(self, run: str, feature_set: str, model: str) -> Path | UPath:
        """Where one feature set/model's fitted thresholds live."""
        return self.run_dir(run) / "thresholds" / feature_set / f"{model}.parquet"

    def metrics_path(self, run: str) -> Path | UPath:
        """Where a run's aggregated metrics live."""
        return self.run_dir(run) / "metrics.parquet"

    def glaciers_path(self, run: str) -> Path | UPath:
        """Where a run's per-glacier metrics live."""
        return self.run_dir(run) / "glaciers.parquet"

    def models_path(self, run: str, feature_set: str, model: str, fold: int) -> Path | UPath:
        """Where one feature set/model/fold's saved estimator lives."""
        return self.run_dir(run) / "models" / feature_set / model / f"fold{fold}.joblib"

    def maps_dir(self, run: str, feature_set: str, model: str) -> Path | UPath:
        """Where one feature set/model's prediction maps live."""
        return self.run_dir(run) / "maps" / feature_set / model

    @staticmethod
    def pair_path(directory: Path | UPath, tile: str, year: int) -> Path | UPath:
        """Return the per-(tile, year) parquet file inside a stage directory."""
        return directory / f"{tile}_{year}.parquet"

    @staticmethod
    def list_pairs(directory: Path | UPath) -> list[tuple[str, int]]:
        """List the (tile, year) pairs with a file in ``directory``, sorted."""
        if not directory.exists():
            return []
        pairs = []
        for path in directory.glob("*.parquet"):
            match = _PAIR_RE.match(path.name)
            if match:
                pairs.append((match["tile"], int(match["year"])))
        return sorted(pairs)

    # -- resume rule ----------------------------------------------------------

    @staticmethod
    def _stamps(
        directory: Path | UPath, config: _YamlConfig, upstream: Upstream | None
    ) -> list[tuple[Path | UPath, str]]:
        stamps = [(directory / "config.yaml", config.to_yaml())]
        if upstream is not None:
            stamps.append((directory / "upstream.yaml", yaml.safe_dump(dict(upstream))))
        return stamps

    @staticmethod
    def _mismatch(path: Path | UPath, text: str) -> str | None:
        """Describe how the stored stamp at ``path`` differs from ``text``; ``None`` if same."""
        if not path.exists():
            return None
        stored = path.read_text()
        return None if stored == text else _changed_lines(stored, text)

    def check_config(
        self,
        directory: Path | UPath,
        config: _YamlConfig,
        *,
        overwrite: bool,
        upstream: Upstream | None = None,
    ) -> bool:
        """Stamp a stage directory with its config and inputs; refuse to resume on a change.

        A stage resumes only when the stored ``config.yaml`` and, if given,
        ``upstream.yaml`` (the identity of the inputs it reads) equal the
        current ones. On a difference, ``overwrite`` starts a new generation:
        every file of the directory is removed first, so outputs of two
        generations can never be mixed; without it ``ConfigMismatchError`` is
        raised. Returns whether a new generation was started.
        """
        stamps = self._stamps(directory, config, upstream)
        problems = [(p.name, why) for p, text in stamps if (why := self._mismatch(p, text))]
        if problems and not overwrite:
            detail = "; ".join(f"{name}: {why}" for name, why in problems)
            msg = (
                f"{directory} holds a different generation ({detail}); "
                "pass --overwrite to discard its outputs and start a new one"
            )
            raise ConfigMismatchError(msg)
        fresh = bool(problems) or not (directory / "generation.txt").exists()
        if problems:
            logger.warning("%s: config or inputs changed, discarding its outputs", directory)
            self.clear_outputs(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for path, text in stamps:
            path.write_text(text)
        if fresh:
            (directory / "generation.txt").write_text(uuid.uuid4().hex)
        return fresh

    def verify_config(
        self, directory: Path | UPath, config: _YamlConfig, *, upstream: Upstream | None = None
    ) -> None:
        """Raise unless ``directory`` holds outputs of exactly this config and these inputs."""
        for path, text in self._stamps(directory, config, upstream):
            if not path.exists():
                msg = f"{path} does not exist: the stage has not been run"
                raise ConfigMismatchError(msg)
            why = self._mismatch(path, text)
            if why:
                msg = f"{path} belongs to a different generation ({why}); rerun the stage"
                raise ConfigMismatchError(msg)

    @staticmethod
    def generation(directory: Path | UPath) -> str:
        """Return the generation id of a stage directory."""
        path = directory / "generation.txt"
        if not path.exists():
            msg = f"{directory} holds no stage outputs"
            raise FileNotFoundError(msg)
        return path.read_text().strip()

    @staticmethod
    def clear_outputs(directory: Path | UPath) -> None:
        """Delete every file below a stage directory."""
        if not directory.exists():
            return
        for path in sorted(directory.rglob("*"), key=str, reverse=True):
            if path.is_file():
                path.unlink()

    # -- report collection ----------------------------------------------------

    def collect_reports(
        self,
        directory: Path | UPath,
        futures: Sequence[Future[list[dict[str, object]]]],
        *,
        checkpoint_every: int = 25,
    ) -> list[dict[str, object]]:
        """Gather the report rows of per-pair futures, checkpointing the stage report.

        Rows are merged into the report every ``checkpoint_every`` results.
        When a future raises, the pending ones are cancelled, the running
        ones are waited for and their rows kept, the report is written, and
        the error is re-raised, so no completed pair is left unreported.
        """
        rows: list[dict[str, object]] = []
        done: set[Future[list[dict[str, object]]]] = set()
        since_checkpoint = 0
        try:
            for future in as_completed(futures):
                rows.extend(future.result())
                done.add(future)
                since_checkpoint += 1
                if since_checkpoint >= checkpoint_every:
                    self.update_report(directory, rows)
                    since_checkpoint = 0
        except BaseException:
            for future in futures:
                future.cancel()
            raise
        finally:
            wait([f for f in futures if not f.cancelled()])
            for future in futures:
                finished = future not in done and future.done() and not future.cancelled()
                if finished and future.exception() is None:
                    rows.extend(future.result())
            self.update_report(directory, rows)
        return rows

    # -- parquet I/O ----------------------------------------------------------

    @staticmethod
    def read_parquet(path: Path | UPath, columns: Sequence[str] | None = None) -> pd.DataFrame:
        """Read a parquet file, optionally restricted to ``columns``."""
        with UPath(path).open("rb") as handle:
            return pd.read_parquet(handle, columns=list(columns) if columns else None)

    @staticmethod
    def write_parquet(frame: pd.DataFrame, path: Path | UPath) -> None:
        """Write ``frame`` to ``path``, creating parent directories as needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with UPath(path).open("wb") as handle:
            frame.to_parquet(handle, index=False)

    def _load_dir(self, directory: Path | UPath, columns: Sequence[str] | None) -> pd.DataFrame:
        pairs = self.list_pairs(directory)
        if not pairs:
            msg = f"no per-pair parquet files under {directory}"
            raise FileNotFoundError(msg)
        wanted = None if columns is None else [*KEY_COLUMNS, *columns]
        frames = [self.read_parquet(self.pair_path(directory, t, y), wanted) for t, y in pairs]
        frame = pd.concat(frames, ignore_index=True)
        frame["tile"] = frame["tile"].astype("category")
        return frame

    def _dataset_columns(self, dataset: str) -> set[str]:
        """Column names available in a dataset's files, from the first pair's schema."""
        directory = self.samples_dir(dataset)
        pairs = self.list_pairs(directory)
        if not pairs:
            msg = f"no per-pair parquet files under {directory}"
            raise FileNotFoundError(msg)
        tile, year = pairs[0]
        with UPath(self.pair_path(directory, tile, year)).open("rb") as handle:
            schema = pq.read_schema(handle)
        return set(schema.names)

    def load_locations(self, columns: Sequence[str] | None = None) -> pd.DataFrame:
        """All location rows, key columns first, ``tile`` categorical."""
        return self._load_dir(self.locations_dir, columns)

    def load_samples(
        self, datasets: Sequence[str], columns: Sequence[str] | None = None
    ) -> pd.DataFrame:
        """Locations left-joined with every dataset's features on the key columns.

        The join is validated to be one-to-one: a dataset may lack rows (NaN) but
        never duplicate them. ``columns`` restricts the feature columns read; a
        dataset only has the requested columns it actually carries read from it,
        since datasets may hold different feature layers. A requested column
        absent from every listed dataset raises ``ValueError``. The location
        attributes are always loaded in full. A dataset's feature column
        colliding with an already-present column (a location attribute or an
        earlier dataset's) is a configuration error and raises ``ValueError``.
        """
        frame = self.load_locations()
        available = {dataset: self._dataset_columns(dataset) for dataset in datasets}
        if columns is not None:
            missing = [c for c in columns if not any(c in cols for cols in available.values())]
            if missing:
                msg = f"requested columns not found in any dataset: {missing}"
                raise ValueError(msg)
        for dataset in datasets:
            wanted = None if columns is None else [c for c in columns if c in available[dataset]]
            feats = self._load_dir(self.samples_dir(dataset), wanted)
            collisions = [c for c in feats.columns if c not in KEY_COLUMNS and c in frame.columns]
            if collisions:
                msg = f"samples of {dataset!r} collide with existing columns: {collisions}"
                raise ValueError(msg)
            if feats.duplicated(KEY_COLUMNS).any():
                msg = f"duplicate keys in samples of {dataset!r}"
                raise ValueError(msg)
            # merge on a categorical "tile" key against another categorical key
            # silently produces mismatched categories in pandas; go through str.
            left = frame.copy()
            left["tile"] = left["tile"].astype(str)
            feats = feats.copy()
            feats["tile"] = feats["tile"].astype(str)
            before = len(left)
            frame = left.merge(feats, on=KEY_COLUMNS, how="left", validate="one_to_one")
            if len(frame) != before:  # pragma: no cover - guarded by validate=
                msg = f"join with {dataset!r} changed the row count"
                raise ValueError(msg)
        frame["tile"] = frame["tile"].astype("category")
        return frame

    # -- reports --------------------------------------------------------------

    def read_report(self, directory: Path | UPath) -> pd.DataFrame:
        """Read the stage report of ``directory``, or an empty frame if none exists yet."""
        path = directory / "report.parquet"
        return self.read_parquet(path) if path.exists() else pd.DataFrame()

    def update_report(self, directory: Path | UPath, rows: Iterable[dict[str, object]]) -> None:
        """Merge ``rows`` into the stage report, replacing rows with the same (tile, year)."""
        new = pd.DataFrame(list(rows))
        if new.empty:
            return
        old = self.read_report(directory)
        if not old.empty:
            keys = pd.MultiIndex.from_frame(new[["tile", "year"]])
            old = old[~pd.MultiIndex.from_frame(old[["tile", "year"]]).isin(keys)]
        report = pd.concat([old, new], ignore_index=True).sort_values(["tile", "year"])
        self.write_parquet(report.reset_index(drop=True), directory / "report.parquet")


def _changed_lines(stored: str, current: str) -> str:
    """Names of the top-level keys whose YAML line differs, for the error message."""
    old = set(stored.splitlines())
    keys = sorted({line.split(":")[0] for line in current.splitlines() if line not in old})
    return ", ".join(keys) or "unknown keys"
