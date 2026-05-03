"""Tests for the scikit-learn cross-version compatibility helpers."""

import numpy as np
import numpy.testing as npt
import pytest
from sklearn.base import BaseEstimator

from skordinal.utils import _sklearn_compat
from skordinal.utils._sklearn_compat import validate_data


class _DummyEstimator(BaseEstimator):
    """Minimal estimator used to exercise the compat shim's modern path."""


class _LegacyEstimator(BaseEstimator):
    """Estimator exposing the pre-1.6 ``_validate_data`` and recording calls."""

    def _validate_data(self, X, y=None, *, reset=True, **check_params):
        self._last_call = {
            "X": np.asarray(X),
            "y": None if y is None else np.asarray(y),
            "reset": reset,
            "check_params": check_params,
        }
        if y is None:
            return self._last_call["X"]
        return self._last_call["X"], self._last_call["y"]


@pytest.fixture
def estimator():
    """Return a fresh dummy estimator without fitted attributes."""
    return _DummyEstimator()


@pytest.fixture
def X():
    """Two-feature, four-sample float matrix."""
    return np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])


@pytest.fixture
def y():
    """Three-class ordinal target matching ``X``."""
    return np.array([0, 1, 2, 1])


def test_validate_data_returns_X_alone_when_y_is_none(estimator, X):
    """When ``y`` is omitted, only the validated ``X`` is returned."""
    out = validate_data(estimator, X)
    assert isinstance(out, np.ndarray)
    npt.assert_array_equal(out, X)


def test_validate_data_returns_X_y_tuple(estimator, X, y):
    """When ``y`` is given, a ``(X, y)`` tuple is returned."""
    X_out, y_out = validate_data(estimator, X, y)
    npt.assert_array_equal(X_out, X)
    npt.assert_array_equal(y_out, y)


def test_validate_data_reset_records_and_checks_n_features_in(estimator, X):
    """``reset=True`` records ``n_features_in_``; ``reset=False`` enforces it."""
    validate_data(estimator, X, reset=True)
    assert estimator.n_features_in_ == X.shape[1]

    with pytest.raises(ValueError):
        validate_data(estimator, X[:, :1], reset=False)


def test_validate_data_forwards_check_params(estimator):
    """Extra ``check_params`` reach the underlying validator (dtype applied)."""
    out = validate_data(estimator, np.array([[1, 2], [3, 4]]), dtype=np.float64)
    assert out.dtype == np.float64


def test_validate_data_falls_back_to_legacy_method_with_y(monkeypatch, X, y):
    """The pre-1.6 path delegates to ``estimator._validate_data(X, y, ...)``."""
    monkeypatch.setattr(_sklearn_compat, "_HAS_VALIDATE_DATA", False)
    legacy = _LegacyEstimator()

    X_out, y_out = validate_data(legacy, X, y, dtype=np.float64)

    npt.assert_array_equal(X_out, X)
    npt.assert_array_equal(y_out, y)
    assert legacy._last_call["reset"] is True
    assert legacy._last_call["check_params"] == {"dtype": np.float64}


def test_validate_data_falls_back_to_legacy_method_without_y(monkeypatch, X):
    """The pre-1.6 path delegates to ``estimator._validate_data(X, ...)`` only."""
    monkeypatch.setattr(_sklearn_compat, "_HAS_VALIDATE_DATA", False)
    legacy = _LegacyEstimator()

    out = validate_data(legacy, X, reset=False)

    npt.assert_array_equal(out, X)
    assert legacy._last_call["y"] is None
    assert legacy._last_call["reset"] is False
