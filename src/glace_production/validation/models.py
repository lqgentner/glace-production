"""The baseline model: a random forest with fixed hyperparameters.

``max_features`` is the only setting that depends on the feature count; the
rest are constants of this module (``RF_PARAMS``) and are stated in the paper
as such.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

# typing.TypedDict does not accept ``closed`` at runtime on 3.13 (PEP 728).
from typing_extensions import TypedDict


class _RFParams(TypedDict, closed=True):
    """Fixed `RandomForestClassifier` hyperparameters, keyed by constructor argument."""

    n_estimators: int
    min_samples_leaf: int
    n_jobs: int


RF_PARAMS: _RFParams = {"n_estimators": 300, "min_samples_leaf": 50, "n_jobs": -1}

type Estimator = RandomForestClassifier


def make_model(
    name: str,
    n_features: int,  # noqa: ARG001 -- part of the ModelFactory signature
    seed: int,
) -> Estimator:
    """Build the baseline estimator.

    Parameters
    ----------
    name
        Model name; only ``"rf"`` exists.
    n_features
        Number of feature columns the model will be fit on.
    seed
        Random seed for the estimator.

    Returns
    -------
    Estimator
        The constructed, unfitted estimator.
    """
    if name == "rf":
        return RandomForestClassifier(max_features="sqrt", random_state=seed, **RF_PARAMS)
    msg = f"unknown model {name!r}"
    raise ValueError(msg)
