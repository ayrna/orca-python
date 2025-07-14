"""Tests for the NNPOM classifier."""

import numpy as np
import pytest

from orca_python.classifiers.NNPOM import NNPOM


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
        ("n_hidden", -1),
        ("max_iter", -1),
    ],
)
def test_nnpom_fit_hyperparameters_validation(X, y, param_name, invalid_value):
    """Test that hyperparameters are validated."""
    classifier = NNPOM(**{param_name: invalid_value})
    model = classifier.fit(X, y)

    assert model is None, "The NNPOM fit method doesnt return Null on error"


def test_nnpom_fit_input_validation(X, y):
    """Test that input data is validated."""
    X_invalid = X[:-1, :-1]
    y_invalid = y[:-1]

    classifier = NNPOM()
    with pytest.raises(ValueError):
        model = classifier.fit(X, y_invalid)
        assert model is None, "The NNPOM fit method doesnt return Null on error"

    with pytest.raises(ValueError):
        model = classifier.fit([], y)
        assert model is None, "The NNPOM fit method doesnt return Null on error"

    with pytest.raises(ValueError):
        model = classifier.fit(X, [])
        assert model is None, "The NNPOM fit method doesnt return Null on error"

    with pytest.raises(ValueError):
        model = classifier.fit(X_invalid, y)
        assert model is None, "The NNPOM fit method doesnt return Null on error"


def test_nnpom_predict_invalid_input_raises_error(X, y):
    """Test that invalid input raises an error."""
    classifier = NNPOM()
    classifier.fit(X, y)

    with pytest.raises(ValueError):
        classifier.predict([])
