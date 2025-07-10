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


def test_nnop_fit_hyperparameters_validation(X, y):
    # Test preparation
    classifiers = [
        NNOP(epsilon_init=0.5, n_hidden=-1, max_iter=1000, lambda_value=0.01),
        NNOP(epsilon_init=0.5, n_hidden=10, max_iter=-1, lambda_value=0.01),
    ]

    # Test execution and verification
    for classifier in classifiers:
        model = classifier.fit(X, y)
        assert model is None, "The NNOP fit method doesnt return Null on error"


def test_nnop_fit_input_validation(X, y):
    # Test preparation
    X_invalid = X[:-1, :-1]
    y_invalid = y[:-1]

    # Test execution and verification
    classifier = NNOP(epsilon_init=0.5, n_hidden=10, max_iter=1000, lambda_value=0.01)
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
    # Test preparation
    classifier = NNOP(epsilon_init=0.5, n_hidden=10, max_iter=500, lambda_value=0.01)
    classifier.fit(X, y)

    # Test execution and verification
    with pytest.raises(ValueError):
        classifier.predict([])
