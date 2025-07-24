"""Tests for the NNOP classifier."""

import numpy as np
import pytest

from orca_python.classifiers.NNOP import NNOP


@pytest.fixture
def X():
    """Create sample feature patterns for testing."""
    return np.array([[0, 1], [1, 0], [1, 1], [0, 0], [1, 2]])


@pytest.fixture
def y():
    """Create sample target variables for testing."""
    return np.array([0, 1, 1, 0, 1])


@pytest.mark.parametrize(
    "param_name, invalid_value",
    [
        ("epsilon_init", 0),
        ("epsilon_init", -1),
        ("n_hidden", -1),
        ("max_iter", -1),
        ("lambda_value", -1e-5),
    ],
)
def test_nnop_hyperparameter_value_validation(X, y, param_name, invalid_value):
    """Test that NNOP raises ValueError for invalid of hyperparameters."""
    classifier = NNOP(**{param_name: invalid_value})

    with pytest.raises(ValueError, match=rf"The '{param_name}' parameter.*"):
        classifier.fit(X, y)


@pytest.mark.parametrize(
    "param_name, invalid_value",
    [
        ("epsilon_init", "high"),
        ("n_hidden", 5.5),
        ("max_iter", 2.5),
        ("lambda_value", "tight"),
    ],
)
def test_nnop_hyperparameter_type_validation(X, y, param_name, invalid_value):
    """Test that NNOP raises ValueError for invalid types of hyperparameters."""
    classifier = NNOP(**{param_name: invalid_value})

    with pytest.raises(ValueError, match=rf"The '{param_name}' parameter.*"):
        classifier.fit(X, y)


def test_nnop_fit_input_validation(X, y):
    """Test that input data is validated."""
    X_invalid = X[:-1, :-1]
    y_invalid = y[:-1]

    classifier = NNOP()
    with pytest.raises(ValueError):
        model = classifier.fit(X, y_invalid)
        assert model is None, "The NNOP fit method doesnt return Null on error"

    with pytest.raises(ValueError):
        model = classifier.fit([], y)
        assert model is None, "The NNOP fit method doesnt return Null on error"

    with pytest.raises(ValueError):
        model = classifier.fit(X, [])
        assert model is None, "The NNOP fit method doesnt return Null on error"

    with pytest.raises(ValueError):
        model = classifier.fit(X_invalid, y)
        assert model is None, "The NNOP fit method doesnt return Null on error"


def test_nnop_predict_invalid_input_raises_error(X, y):
    """Test that invalid input raises an error."""
    classifier = NNOP()
    classifier.fit(X, y)

    with pytest.raises(ValueError):
        classifier.predict([])
