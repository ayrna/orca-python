"""Tests for the REDSVM classifier."""

import numpy as np
import numpy.testing as npt
import pytest

from orca_python.classifiers.REDSVM import REDSVM
from orca_python.datasets import load_dataset
from orca_python.testing import TEST_DATASETS_DIR, TEST_PREDICTIONS_DIR


@pytest.fixture
def X():
    """Create sample feature patterns for testing."""
    return np.array([[0, 1], [1, 0], [1, 1], [0, 0], [1, 2]])


@pytest.fixture
def y():
    """Create sample target variables for testing."""
    return np.array([0, 1, 1, 0, 1])


def test_redsvm_fit_correct():
    # Test preparation
    X_train, y_train, X_test, _ = load_dataset(
        dataset_name="balance-scale", data_path=TEST_DATASETS_DIR
    )

    expected_predictions = [
        TEST_PREDICTIONS_DIR / "REDSVM" / "expectedPredictions.0",
        TEST_PREDICTIONS_DIR / "REDSVM" / "expectedPredictions.1",
        TEST_PREDICTIONS_DIR / "REDSVM" / "expectedPredictions.2",
        TEST_PREDICTIONS_DIR / "REDSVM" / "expectedPredictions.3",
        TEST_PREDICTIONS_DIR / "REDSVM" / "expectedPredictions.4",
        TEST_PREDICTIONS_DIR / "REDSVM" / "expectedPredictions.5",
        TEST_PREDICTIONS_DIR / "REDSVM" / "expectedPredictions.6",
        TEST_PREDICTIONS_DIR / "REDSVM" / "expectedPredictions.7",
    ]

    classifiers = [
        REDSVM(
            kernel=0,
            degree=2,
            gamma=0.1,
            coef0=0.5,
            C=0.1,
            cache_size=150,
            tol=0.005,
            shrinking=0,
        ),
        REDSVM(
            kernel=1,
            degree=2,
            gamma=0.1,
            coef0=0.5,
            C=0.1,
            cache_size=150,
            tol=0.005,
            shrinking=0,
        ),
        REDSVM(
            kernel=2,
            degree=2,
            gamma=0.1,
            coef0=0.5,
            C=0.1,
            cache_size=150,
            tol=0.005,
            shrinking=0,
        ),
        REDSVM(
            kernel=3,
            degree=2,
            gamma=0.1,
            coef0=0.5,
            C=0.1,
            cache_size=150,
            tol=0.005,
            shrinking=0,
        ),
        REDSVM(
            kernel=4,
            degree=2,
            gamma=0.1,
            coef0=0.5,
            C=0.1,
            cache_size=150,
            tol=0.005,
            shrinking=0,
        ),
        REDSVM(
            kernel=5,
            degree=2,
            gamma=0.1,
            coef0=0.5,
            C=0.1,
            cache_size=150,
            tol=0.005,
            shrinking=1,
        ),
        REDSVM(
            kernel=6,
            degree=2,
            gamma=0.1,
            coef0=0.5,
            C=0.1,
            cache_size=150,
            tol=0.005,
            shrinking=1,
        ),
        REDSVM(
            kernel=7,
            degree=2,
            gamma=0.1,
            coef0=0.5,
            C=0.1,
            cache_size=150,
            tol=0.005,
            shrinking=1,
        ),
    ]

    # Test execution and verification
    for expected_prediction, classifier in zip(expected_predictions, classifiers):
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        expected_prediction = np.loadtxt(expected_prediction)
        npt.assert_equal(
            predictions,
            expected_prediction,
            "The prediction doesnt match with the desired values",
        )


def test_redsvm_fit_not_valid_parameter(X, y):
    # Test preparation
    classifiers = [
        REDSVM(gamma=0.1, C=1, kernel=-1),
        REDSVM(gamma=0.1, C=1, cache_size=-1),
        REDSVM(gamma=0.1, C=1, tol=-1),
        REDSVM(gamma=0.1, C=1, shrinking=2),
    ]

    error_msgs = [
        "unknown kernel type",
        "cache_size <= 0",
        "eps <= 0",
        "shrinking != 0 and shrinking != 1",
    ]

    # Test execution and verification
    for classifier, error_msg in zip(classifiers, error_msgs):
        with pytest.raises(ValueError, match=error_msg):
            model = classifier.fit(X, y)
            assert model is None, "The REDSVM fit method doesnt return Null on error"


def test_redsvm_fit_not_valid_data(X, y):
    # Test preparation
    X_invalid = X[:-1, :-1]
    y_invalid = y[:-1]

    # Test execution and verification
    classifier = REDSVM(gamma=0.1, C=1, kernel=8)
    with pytest.raises(
        ValueError, match="Wrong input format: sample_serial_number out of range"
    ):
        model = classifier.fit(X, y)
        assert model is None, "The REDSVM fit method doesnt return Null on error"

    classifier = REDSVM(gamma=0.1, C=1)
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


def test_redsvm_model_is_not_a_dict(X, y):
    # Test preparation
    classifier = REDSVM(gamma=0.1, C=1)
    classifier.fit(X, y)

    # Test execution and verification
    with pytest.raises(TypeError, match="Model should be a dictionary!"):
        classifier.model_ = 1
        classifier.predict(X)


def test_redsvm_predict_not_valid_data(X, y):
    # Test preparation
    classifier = REDSVM(gamma=0.1, C=1)
    classifier.fit(X, y)

    # Test execution and verification
    with pytest.raises(ValueError):
        classifier.predict([])
