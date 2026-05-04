"""Tests for the SVOREX classifier."""

from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from skordinal.classifiers import SVOREX
from skordinal.utils._testing import make_balance_scale_split

PREDICTIONS_DIR = Path(__file__).parent / "data" / "SVOREX"


@pytest.fixture
def X():
    """Create sample feature patterns for testing."""
    return np.array([[1, 2], [2, 1], [2, 2], [1, 1], [2, 3]])


@pytest.fixture
def y():
    """Create sample target variables for testing."""
    return np.array([1, 2, 2, 1, 2])


@pytest.mark.parametrize(
    "kernel",
    [
        "gaussian",
        "linear",
        "poly",
    ],
)
def test_svorex_predict_matches_expected(kernel):
    """Test that predictions match expected values."""
    X_train, X_test, y_train, _ = make_balance_scale_split()

    classifier = SVOREX(C=0.5, kernel=kernel, degree=4, tol=0.002, kappa=0.1)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_expected = np.loadtxt(PREDICTIONS_DIR / f"predictions_{kernel}.csv", dtype=int)

    npt.assert_equal(
        y_pred, y_expected, "The prediction doesnt match with the desired values"
    )


@pytest.mark.parametrize(
    "param_name, invalid_value",
    [
        ("C", 0),
        ("C", -1),
        ("degree", -1),
        ("tol", 0),
        ("tol", -1e-5),
        ("kernel", "unknown"),
        ("kappa", -1),
    ],
)
def test_svorex_hyperparameter_value_validation(X, y, param_name, invalid_value):
    """Test that SVOREX raises ValueError for invalid of hyperparameters."""
    classifier = SVOREX(**{param_name: invalid_value})

    with pytest.raises(ValueError, match=rf"The '{param_name}' parameter.*"):
        classifier.fit(X, y)


@pytest.mark.parametrize(
    "param_name, invalid_value",
    [
        ("C", "high"),
        ("kernel", 5),
        ("degree", 2.5),
        ("tol", "tight"),
        ("kappa", "low"),
    ],
)
def test_svorex_hyperparameter_type_validation(X, y, param_name, invalid_value):
    """Test that SVOREX raises ValueError for invalid types of hyperparameters."""
    classifier = SVOREX(**{param_name: invalid_value})

    with pytest.raises(ValueError, match=rf"The '{param_name}' parameter.*"):
        classifier.fit(X, y)


def test_svorex_fit_input_validation(X, y):
    """Test that input data is validated."""
    X_invalid = X[:-1, :-1]
    y_invalid = y[:-1]

    classifier = SVOREX()
    with pytest.raises(ValueError):
        model = classifier.fit(X, y_invalid)
        assert model is None, "The SVOREX fit method doesnt return Null on error"

    with pytest.raises(ValueError):
        model = classifier.fit([], y)
        assert model is None, "The SVOREX fit method doesnt return Null on error"

    with pytest.raises(ValueError):
        model = classifier.fit(X, [])
        assert model is None, "The SVOREX fit method doesnt return Null on error"

    with pytest.raises(ValueError):
        model = classifier.fit(X_invalid, y)
        assert model is None, "The SVOREX fit method doesnt return Null on error"


def test_svorex_validates_internal_model_format(X, y):
    """Test that internal model format is validated."""
    classifier = SVOREX()
    classifier.fit(X, y)

    with pytest.raises(TypeError, match="Model should be a dictionary!"):
        classifier.model_ = 1
        classifier.predict(X)


def test_svorex_predict_invalid_input_raises_error(X, y):
    """Test that invalid input raises an error."""
    classifier = SVOREX()
    classifier.fit(X, y)

    with pytest.raises(ValueError):
        classifier.predict([])
