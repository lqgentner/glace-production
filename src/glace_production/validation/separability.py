"""Validation 1: how well each feature separates glacier from non-glacier pixels.

Directional AUC (exact, rank-based) is the primary metric; the
Jeffries-Matusita distance from histograms is the secondary one. Both are
computed over every scope row, per tile and per year. The datasets that are
compared with each other are scored on one common support, the rows where every
measurement layer of every one of them is finite (:func:`common_support`), so
their scores describe the same pixels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from glace_production.validation.crossval import compute_support

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

    from glace_production.validation.config import DatasetConfig, Feature
    from glace_production.validation.store import ValidationStore

LAYER_SUPPORT = "layer"
"""``support`` of a table whose every feature is scored on its own finite rows."""

SEPARABILITY_COLUMNS = [
    "dataset",
    "feature",
    "kind",
    "support",
    "group_by",
    "group",
    "auc",
    "jm",
    "n_glacier",
    "n_other",
    "n_nan",
    "n_below",
    "n_above",
]


def directional_auc(
    values: NDArray[np.floating], labels: NDArray[np.integer], *, glacier_low: bool = True
) -> float:
    """ROC AUC of ``values`` for the glacier class, with an a-priori direction.

    Glacier pixels are expected to have the lower value when ``glacier_low``,
    so an informative feature scores above 0.5 and an inverted one below.
    """
    score = -values if glacier_low else values
    return float(roc_auc_score(labels, score))


def jm_distance(
    values: NDArray[np.floating],
    labels: NDArray[np.integer],
    *,
    value_range: tuple[float, float],
    bins: int = 100,
) -> tuple[float, int, int]:
    """Jeffries-Matusita distance on [0, 2] from class histograms.

    ``JM = 2 (1 - sum_i sqrt(p_i q_i))``. Values outside ``value_range`` are
    counted into the edge bins; how many lie below and above is returned.
    """
    lo, hi = value_range
    n_below = int((values < lo).sum())
    n_above = int((values > hi).sum())
    clipped = np.clip(values, lo, hi)
    glacier = labels == 1
    p, _ = np.histogram(clipped[glacier], bins=bins, range=value_range)
    q, _ = np.histogram(clipped[~glacier], bins=bins, range=value_range)
    p = p / p.sum()
    q = q / q.sum()
    bc = float(np.clip(np.sum(np.sqrt(p * q)), 0.0, 1.0))
    return 2.0 * (1.0 - bc), n_below, n_above


MIN_COMPARED = 2
"""Fewest datasets the common support is computed over."""

COMMON_SUPPORT_PREFIX = "common:"
"""Prefix of ``support`` for tables scored on the common support of several datasets."""


def layer_features(config: DatasetConfig) -> dict[str, Feature]:
    """Return the measurement layers of a dataset as pixel values, without window statistics."""
    return {c: f for c, f in config.features.items() if f.derived is None}


def common_support_columns(configs: Sequence[DatasetConfig]) -> list[str]:
    """Every column the common support of ``configs`` requires to be finite."""
    return [c for config in configs for c in layer_features(config)]


def common_support_name(configs: Sequence[DatasetConfig]) -> str:
    """``support`` of tables scored on the common support of ``configs``."""
    return COMMON_SUPPORT_PREFIX + ",".join(sorted(config.name for config in configs))


def load_layer_samples(store: ValidationStore, configs: Sequence[DatasetConfig]) -> pd.DataFrame:
    """Sample rows joined with the measurement layers of every dataset in ``configs``."""
    return store.load_samples(
        [config.name for config in configs], columns=common_support_columns(configs)
    )


def common_support(samples: pd.DataFrame, configs: Sequence[DatasetConfig]) -> NDArray[np.bool_]:
    """Rows where every measurement layer of every dataset in ``configs`` is finite.

    A row a dataset has no sample for is NaN in its layers after the join of
    :meth:`~glace_production.validation.store.ValidationStore.load_samples`, so it
    drops out as well.
    """
    return compute_support(samples, common_support_columns(configs))


def separability_tables(
    samples: pd.DataFrame,
    configs: Sequence[DatasetConfig],
    *,
    per_layer_support: bool = False,
) -> dict[str, pd.DataFrame]:
    """Score each dataset's layers, on the common support of all of them.

    Parameters
    ----------
    samples
        Output of :func:`load_layer_samples` for ``configs``.
    configs
        The datasets compared with each other.
    per_layer_support
        Score each layer on its own finite rows instead of the common support.

    Returns
    -------
    dict of str to pandas.DataFrame
        One :func:`separability_table` per dataset name.

    Raises
    ------
    ValueError
        If a dataset is listed twice, or the common support is asked of a
        single dataset: its table would then sit on a different support than
        the tables of the datasets it is compared with.
    """
    names = [config.name for config in configs]
    if len(set(names)) != len(names):
        msg = f"a dataset is listed twice: {names}"
        raise ValueError(msg)
    if not per_layer_support and len(configs) < MIN_COMPARED:
        msg = (
            f"the common support needs every compared dataset, got only {names}; "
            "list all of them, or score each layer on its own rows (per_layer_support)"
        )
        raise ValueError(msg)
    support = None if per_layer_support else common_support(samples, configs)
    support_name = LAYER_SUPPORT if per_layer_support else common_support_name(configs)
    return {
        config.name: separability_table(
            samples,
            layer_features(config),
            dataset=config.name,
            support=support,
            support_name=support_name,
        )
        for config in configs
    }


def _one(
    values: NDArray[np.float32], labels: NDArray[np.uint8], feature: Feature
) -> dict[str, object]:
    finite = np.isfinite(values)
    v, y = values[finite], labels[finite]
    glacier = y == 1
    n_glacier = int(glacier.sum())
    n_other = int((~glacier).sum())
    row: dict[str, object] = {
        "n_glacier": n_glacier,
        "n_other": n_other,
        "n_nan": int((~finite).sum()),
        "auc": np.nan,
        "jm": np.nan,
        "n_below": 0,
        "n_above": 0,
    }
    if n_glacier and n_other:
        row["auc"] = directional_auc(v, y, glacier_low=feature.glacier_low)
        row["jm"], row["n_below"], row["n_above"] = jm_distance(
            v, y, value_range=feature.value_range
        )
    return row


def separability_table(
    samples: pd.DataFrame,
    features: Mapping[str, Feature],
    *,
    dataset: str,
    support: NDArray[np.bool_] | None = None,
    support_name: str = LAYER_SUPPORT,
) -> pd.DataFrame:
    """AUC and JM per feature column and group (all, tile, year).

    Parameters
    ----------
    samples
        Sample rows with ``label``, ``tile``, ``year`` and the feature columns.
    features
        Feature columns to score.
    dataset
        Dataset name written into every row.
    support
        Rows every feature is scored on; rows outside it count as ``n_nan``.
        ``None`` scores each feature on its own finite rows.
    support_name
        What defines ``support``, written into the ``support`` column.

    Returns
    -------
    pandas.DataFrame
        One row per feature and group, with :data:`SEPARABILITY_COLUMNS`.
    """
    if (support is None) != (support_name == LAYER_SUPPORT):
        msg = f"support_name {support_name!r}: {LAYER_SUPPORT!r} if and only if support is None"
        raise ValueError(msg)
    if support is not None and len(support) != len(samples):
        msg = f"support has {len(support)} rows, samples have {len(samples)}"
        raise ValueError(msg)
    labels = samples["label"].to_numpy(dtype=np.uint8)
    groups: dict[str, NDArray[np.object_] | None] = {
        "all": None,
        "tile": samples["tile"].astype(str).to_numpy(),
        "year": samples["year"].astype(str).to_numpy(),
    }
    rows: list[dict[str, object]] = []
    for column, feature in features.items():
        values = samples[column].to_numpy(dtype=np.float32)
        if support is not None:
            values = np.where(support, values, np.float32(np.nan))
        for group_by, key in groups.items():
            members = (
                [("all", np.ones(len(values), bool))]
                if key is None
                else [(str(g), key == g) for g in np.unique(key)]
            )
            for group, sel in members:
                rows.append(
                    {
                        "dataset": dataset,
                        "feature": column,
                        "kind": feature.kind,
                        "support": support_name,
                        "group_by": group_by,
                        "group": group,
                        **_one(values[sel], labels[sel], feature),
                    }
                )
    return pd.DataFrame(rows, columns=SEPARABILITY_COLUMNS)
