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


def test_redsvm_predict_matches_expected():
    """Test that predictions match expected values."""
    X_train, y_train, X_test, _ = load_dataset(
        dataset_name="balance-scale", data_path=TEST_DATASETS_DIR
    )

    expected_files = [
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

    for expected_file, classifier in zip(expected_files, classifiers):
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        y_expected = np.loadtxt(expected_file)
        npt.assert_equal(
            y_pred,
            y_expected,
            "The prediction doesnt match with the desired values",
        )


def test_redsvm_fit_hyperparameters_validation(X, y):
    """Test that hyperparameters are validated."""
    classifiers = [
        REDSVM(kernel=-1),
        REDSVM(cache_size=-1),
        REDSVM(tol=-1),
        REDSVM(shrinking=2),
        REDSVM(kernel=8),
    ]

    error_msgs = [
        "unknown kernel type",
        "cache_size <= 0",
        "eps <= 0",
        "shrinking != 0 and shrinking != 1",
        "Wrong input format: sample_serial_number out of range",
    ]

    for classifier, error_msg in zip(classifiers, error_msgs):
        with pytest.raises(ValueError, match=error_msg):
            model = classifier.fit(X, y)
            assert model is None, "The REDSVM fit method doesnt return Null on error"


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
