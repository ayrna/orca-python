"""Tests for the SVOREX classifier."""

import numpy as np
import numpy.testing as npt
import pytest

from orca_python.classifiers.SVOREX import SVOREX
from orca_python.datasets import load_dataset
from orca_python.testing import TEST_DATASETS_DIR, TEST_PREDICTIONS_DIR


@pytest.fixture
def X():
    """Create sample feature patterns for testing."""
    return np.array([[1, 2], [2, 1], [2, 2], [1, 1], [2, 3]])


@pytest.fixture
def y():
    """Create sample target variables for testing."""
    return np.array([1, 2, 2, 1, 2])


def test_svorex_predict_matches_expected():
    # Test preparation
    X_train, y_train, X_test, _ = load_dataset(
        dataset_name="balance-scale", data_path=TEST_DATASETS_DIR
    )

    expected_files = [
        TEST_PREDICTIONS_DIR / "SVOREX" / "expectedPredictions.0",
        TEST_PREDICTIONS_DIR / "SVOREX" / "expectedPredictions.1",
        TEST_PREDICTIONS_DIR / "SVOREX" / "expectedPredictions.2",
    ]

    classifiers = [
        SVOREX(kernel=0, tol=0.002, C=0.5, kappa=0.1),
        SVOREX(kernel=1, tol=0.002, C=0.5, kappa=0.1),
        SVOREX(kernel=2, degree=4, tol=0.002, C=0.5, kappa=0.1),
    ]

    # Test execution and verification
    for expected_file, classifier in zip(expected_files, classifiers):
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        y_expected = np.loadtxt(expected_file)
        npt.assert_equal(
            y_pred,
            y_expected,
            "The prediction doesnt match with the desired values",
        )


def test_svorex_fit_hyperparameters_validation(X, y):
    # Test preparation
    classifiers = [
        SVOREX(tol=0),
        SVOREX(C=0),
        SVOREX(kappa=0),
        SVOREX(kernel=2, degree=0),
        SVOREX(kappa=-1),
    ]

    error_msgs = [
        "- T is invalid",
        "- C is invalid",
        "- K is invalid",
        "- P is invalid",
        "-1 is invalid",
    ]

    # Test execution and verification
    for classifier, error_msg in zip(classifiers, error_msgs):
        with pytest.raises(ValueError, match=error_msg):
            model = classifier.fit(X, y)
            assert model is None, "The SVOREX fit method doesnt return Null on error"


def test_svorex_fit_input_validation(X, y):
    # Test preparation
    X_invalid = X[:-1, :-1]
    y_invalid = y[:-1]

    # Test execution and verification
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
    # Test preparation
    classifier = SVOREX()
    classifier.fit(X, y)

    # Test execution and verification
    with pytest.raises(TypeError, match="Model should be a dictionary!"):
        classifier.model_ = 1
        classifier.predict(X)


def test_svorex_predict_invalid_input_raises_error(X, y):
    # Test preparation
    classifier = SVOREX()
    classifier.fit(X, y)

    # Test execution and verification
    with pytest.raises(ValueError):
        classifier.predict([])
