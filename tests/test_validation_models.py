"""The fixed-hyperparameter baseline model."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from glace_production.validation.models import make_model


def _toy(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    x = rng.normal(0, 1, (n, 3)) + y[:, None] * np.array([-2.0, 0.0, 1.0])
    return x.astype(np.float32), y


def test_rf_has_fixed_params() -> None:
    rf = make_model("rf", 4, seed=0)
    assert isinstance(rf, RandomForestClassifier)
    assert rf.get_params()["max_features"] == "sqrt"
    assert rf.get_params()["n_estimators"] == 300
    assert rf.get_params()["min_samples_leaf"] == 50


def test_forest_fits_and_predicts() -> None:
    x, y = _toy()
    model = make_model("rf", 3, seed=0)
    model.fit(x, y)
    proba = model.predict_proba(x)[:, 1]
    assert proba.shape == (len(y),)
    assert np.mean((proba > 0.5) == y) > 0.5


@pytest.mark.parametrize("name", ["svm", "xgb", "thresh"])
def test_unknown_or_retired_model(name: str) -> None:
    with pytest.raises(ValueError, match="unknown model"):
        make_model(name, 2, seed=0)
