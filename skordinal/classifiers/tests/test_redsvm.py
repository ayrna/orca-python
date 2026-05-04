"""Tests for the REDSVM classifier."""

from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from skordinal.classifiers import REDSVM
from skordinal.utils._testing import make_balance_scale_split

PREDICTIONS_DIR = Path(__file__).parent / "data" / "REDSVM"


@pytest.fixture
def X():
    """Create sample feature patterns for testing."""
    return np.array([[0, 1], [1, 0], [1, 1], [0, 0], [1, 2]])


@pytest.fixture
def y():
    """Create sample target variables for testing."""
    return np.array([0, 1, 1, 0, 1])


@pytest.mark.parametrize(
    "kernel",
    [
        "linear",
        "poly",
        "rbf",
        "sigmoid",
        "stump",
        "perceptron",
        "laplacian",
        "exponential",
    ],
)
def test_redsvm_predict_matches_expected(kernel):
    """Test that predictions match expected values."""
    X_train, X_test, y_train, _ = make_balance_scale_split()

    classifier = REDSVM(
        C=0.1,
        kernel=kernel,
        degree=2,
        gamma=0.1,
        coef0=0.5,
        shrinking=False,
        tol=0.005,
        cache_size=150,
    )

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
        ("gamma", -0.5),
        ("shrinking", 2),
        ("tol", -1e-5),
        ("cache_size", 0),
        ("kernel", "unknown"),
        ("gamma", "invalid_string"),
    ],
)
def test_redsvm_hyperparameter_value_validation(X, y, param_name, invalid_value):
    """Test that REDSVM raises ValueError for invalid of hyperparameters."""
    classifier = REDSVM(**{param_name: invalid_value})

    with pytest.raises(ValueError, match=rf"The '{param_name}' parameter.*"):
        classifier.fit(X, y)


@pytest.mark.parametrize(
    "param_name, invalid_value",
    [
        ("C", "high"),
        ("kernel", 5),
        ("degree", 2.5),
        ("coef0", "bias"),
        ("shrinking", "yes"),
        ("tol", "tight"),
        ("cache_size", "big"),
    ],
)
def test_redsvm_hyperparameter_type_validation(X, y, param_name, invalid_value):
    """Test that REDSVM raises ValueError for invalid types of hyperparameters."""
    classifier = REDSVM(**{param_name: invalid_value})

    with pytest.raises(ValueError, match=rf"The '{param_name}' parameter.*"):
        classifier.fit(X, y)


def test_redsvm_fit_input_validation(X, y):
    """Test that input data is validated."""
    X_invalid = X[:-1, :-1]
    y_invalid = y[:-1]

    classifier = REDSVM()
    with pytest.raises(ValueError):
        model = classifier.fit(X, y_invalid)
        assert model is None, "The REDSVM fit method doesnt return Null on error"

    with pytest.raises(ValueError):
        model = classifier.fit([], y)
        assert model is None, "The REDSVM fit method doesnt return Null on error"

    with pytest.raises(ValueError):
        model = classifier.fit(X, [])
        assert model is None, "The REDSVM fit method doesnt return Null on error"

    with pytest.raises(ValueError):
        model = classifier.fit(X_invalid, y)
        assert model is None, "The REDSVM fit method doesnt return Null on error"


def test_redsvm_validates_internal_model_format(X, y):
    """Test that internal model format is validated."""
    classifier = REDSVM()
    classifier.fit(X, y)

    with pytest.raises(TypeError, match="Model should be a dictionary!"):
        classifier.model_ = 1
        classifier.predict(X)


def test_redsvm_predict_invalid_input_raises_error(X, y):
    """Test that invalid input raises an error."""
    classifier = REDSVM()
    classifier.fit(X, y)

    with pytest.raises(ValueError):
        classifier.predict([])
