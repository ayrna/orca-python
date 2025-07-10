"""Tests for the SVOREX classifier."""

from pathlib import Path

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


@pytest.fixture
def dataset_path():
    return Path(TEST_DATASETS_DIR) / "balance-scale"


@pytest.fixture
def predictions_path():
    return Path(TEST_PREDICTIONS_DIR) / "SVOREX"


@pytest.fixture
def train_file(dataset_path):
    return np.loadtxt(dataset_path / "train_balance-scale.csv", delimiter=",")


@pytest.fixture
def test_file(dataset_path):
    return np.loadtxt(dataset_path / "test_balance-scale.csv", delimiter=",")


def test_svorex_fit_correct(predictions_path):
    # Test preparation
    X_train, y_train, X_test, _ = load_dataset(
        dataset_name="balance-scale", data_path=TEST_DATASETS_DIR
    )

    expected_predictions = [
        predictions_path / "expectedPredictions.0",
        predictions_path / "expectedPredictions.1",
        predictions_path / "expectedPredictions.2",
    ]

    classifiers = [
        SVOREX(kernel=0, tol=0.002, C=0.5, kappa=0.1),
        SVOREX(kernel=1, tol=0.002, C=0.5, kappa=0.1),
        SVOREX(kernel=2, degree=4, tol=0.002, C=0.5, kappa=0.1),
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


def test_svorex_fit_not_valid_parameter(X, y):
    # Test preparation
    classifiers = [
        SVOREX(C=0.1, kappa=1, tol=0),
        SVOREX(C=0, kappa=1),
        SVOREX(C=0.1, kappa=0),
        SVOREX(kernel=2, degree=0, C=0.1, kappa=1),
        SVOREX(kernel=0, C=0.1, kappa=-1),
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


def test_svorex_fit_not_valid_data(X, y):
    # Test preparation
    X_invalid = X[:-1, :-1]
    y_invalid = y[:-1]

    # Test execution and verification
    classifier = SVOREX(kappa=0.1, C=1)
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


def test_svorex_model_is_not_a_dict(X, y):
    # Test preparation
    classifier = SVOREX(kappa=0.1, C=1)
    classifier.fit(X, y)

    # Test execution and verification
    with pytest.raises(TypeError, match="Model should be a dictionary!"):
        classifier.model_ = 1
        classifier.predict(X)


def test_svorex_predict_not_valid_data(X, y):
    # Test preparation
    classifier = SVOREX(kappa=0.1, C=1)
    classifier.fit(X, y)

    # Test execution and verification
    with pytest.raises(ValueError):
        classifier.predict([])
