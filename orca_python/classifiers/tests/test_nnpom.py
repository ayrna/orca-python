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


def test_nnpom_fit_not_valid_parameter(X, y):
    # Test preparation
    classifiers = [
        NNPOM(epsilon_init=0.5, n_hidden=-1, max_iter=1000, lambda_value=0.01),
        NNPOM(epsilon_init=0.5, n_hidden=10, max_iter=-1, lambda_value=0.01),
    ]

    # Test execution and verification
    for classifier in classifiers:
        model = classifier.fit(X, y)
        assert model is None, "The NNPOM fit method doesnt return Null on error"


def test_nnpom_fit_not_valid_data(X, y):
    # Test preparation
    X_invalid = X[:-1, :-1]
    y_invalid = y[:-1]

    # Test execution and verification
    classifier = NNPOM(epsilon_init=0.5, n_hidden=10, max_iter=1000, lambda_value=0.01)
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


def test_nnpom_predict_not_valid_data(X, y):
    # Test preparation
    classifier = NNPOM(epsilon_init=0.5, n_hidden=10, max_iter=500, lambda_value=0.01)
    classifier.fit(X, y)

    # Test execution and verification
    with pytest.raises(ValueError):
        classifier.predict([])
