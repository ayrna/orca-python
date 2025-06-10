"""Tests for the REDSVM classifier."""

from pathlib import Path

import pytest
import numpy as np
import numpy.testing as npt

from orca_python.classifiers.REDSVM import REDSVM
from orca_python.testing import TEST_DATASETS_DIR
from orca_python.testing import TEST_PREDICTIONS_DIR


@pytest.fixture
def dataset_path():
    return Path(TEST_DATASETS_DIR) / "balance-scale"


@pytest.fixture
def predictions_path():
    return Path(TEST_PREDICTIONS_DIR) / "REDSVM"


@pytest.fixture
def train_file(dataset_path):
    return np.loadtxt(
        dataset_path / "train_balance-scale.csv", delimiter=","
    )


@pytest.fixture
def test_file(dataset_path):
    return np.loadtxt(
        dataset_path / "test_balance-scale.csv", delimiter=","
    )


def test_redsvm_fit_correct(dataset_path, train_file, test_file, predictions_path):
    # Check if this algorithm can correctly classify a toy problem.

    # Test preparation
    X_train = train_file[:, 0:(-1)]
    y_train = train_file[:, (-1)]

    X_test = test_file[:, 0:(-1)]

    expected_predictions = [
        predictions_path / "expectedPredictions.0",
        predictions_path / "expectedPredictions.1",
        predictions_path / "expectedPredictions.2",
        predictions_path / "expectedPredictions.3",
        predictions_path / "expectedPredictions.4",
        predictions_path / "expectedPredictions.5",
        predictions_path / "expectedPredictions.6",
        predictions_path / "expectedPredictions.7",
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


def test_redsvm_fit_not_valid_parameter(train_file):

    # Test preparation
    X_train = train_file[:, 0:(-1)]
    y_train = train_file[:, (-1)]

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
            model = classifier.fit(X_train, y_train)
            assert model is None, "The REDSVM fit method doesnt return Null on error"


def test_redsvm_fit_not_valid_data(train_file):
    # Test preparation
    X_train = train_file[:, 0:(-1)]
    y_train = train_file[:, (-1)]
    X_train_broken = train_file[:(-1), 0:(-1)]
    y_train_broken = train_file[0:(-1), (-1)]

    # Test execution and verification
    classifier = REDSVM(gamma=0.1, C=1, kernel=8)
    with pytest.raises(
        ValueError, match="Wrong input format: sample_serial_number out of range"
    ):
        model = classifier.fit(X_train, y_train)
        assert model is None, "The REDSVM fit method doesnt return Null on error"

    classifier = REDSVM(gamma=0.1, C=1)
    with pytest.raises(ValueError):
        model = classifier.fit(X_train, y_train_broken)
        assert model is None, "The REDSVM fit method doesnt return Null on error"

    with pytest.raises(ValueError):
        model = classifier.fit([], y_train)
        assert model is None, "The REDSVM fit method doesnt return Null on error"

    with pytest.raises(ValueError):
        model = classifier.fit(X_train, [])
        assert model is None, "The REDSVM fit method doesnt return Null on error"

    with pytest.raises(ValueError):
        model = classifier.fit(X_train_broken, y_train)
        assert model is None, "The REDSVM fit method doesnt return Null on error"


def test_redsvm_model_is_not_a_dict(train_file, test_file):
    # Test preparation
    X_train = train_file[:, 0:(-1)]
    y_train = train_file[:, (-1)]

    X_test = test_file[:, 0:(-1)]

    classifier = REDSVM(gamma=0.1, C=1)
    classifier.fit(X_train, y_train)

    # Test execution and verification
    with pytest.raises(TypeError, match="Model should be a dictionary!"):
        classifier.model_ = 1
        classifier.predict(X_test)


def test_redsvm_predict_not_valid_data(train_file):
    # Test preparation
    X_train = train_file[:, 0:(-1)]
    y_train = train_file[:, (-1)]

    classifier = REDSVM(gamma=0.1, C=1)
    classifier.fit(X_train, y_train)

    # Test execution and verification
    with pytest.raises(ValueError):
        classifier.predict([])
