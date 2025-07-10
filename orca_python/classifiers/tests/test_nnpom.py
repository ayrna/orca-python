"""Tests for the NNPOM classifier."""

from pathlib import Path

import numpy as np
import pytest

from orca_python.classifiers.NNPOM import NNPOM
from orca_python.testing import TEST_DATASETS_DIR


@pytest.fixture
def X():
    """Create sample feature patterns for testing."""
    return np.array([[0, 1], [1, 0], [1, 1], [0, 0], [1, 2]])


@pytest.fixture
def y():
    """Create sample target variables for testing."""
    return np.array([0, 1, 1, 0, 1])


@pytest.fixture
def dataset_path():
    return Path(TEST_DATASETS_DIR) / "balance-scale"


@pytest.fixture
def train_file(dataset_path):
    return np.loadtxt(dataset_path / "train_balance-scale.csv", delimiter=",")


@pytest.fixture
def test_file(dataset_path):
    return np.loadtxt(dataset_path / "test_balance-scale.csv", delimiter=",")


# 	-----	NOT APPLIED	-----
# It doesn't apply to the because can't set seed to randomize model weights.
# def test_nnpom_fit_correct(self):
# 	#Check if this algorithm can correctly classify a toy problem.

# 	#Test preparation
# 	X_train = self.train_file[:,0:(-1)]
# 	y_train = self.train_file[:,(-1)]

# 	X_test = self.test_file[:,0:(-1)]

# 	expected_predictions = [self.dataset_path / "expectedPredictions.0"]
# 							# self.dataset_path / "expectedPredictions.1",
# 							# self.dataset_path / "expectedPredictions.2",
# 							# self.dataset_path / "expectedPredictions.3")]

# 	classifiers = [NNPOM(epsilon_init = 0.5, n_hidden = 10, max_iter = 500, lambda_value = 0.01)]

# 	#			   NNPOM(epsilon_init = 0.5, n_hidden = 20, max_iter = 500, lambda_value = 0.01),
# 	#			   NNPOM(epsilon_init = 0.5, n_hidden = 10, max_iter = 250, lambda_value = 0.01),
# 	#			   NNPOM(epsilon_init = 0.5, n_hidden = 20, max_iter = 500, lambda_value = 0.01)]


# 	#Test execution and verification
# 	for expected_prediction, classifier in zip(expected_predictions, classifiers):
# 		classifier.fit(X_train, y_train)
# 		predictions = classifier.predict(X_test)
# 		expected_prediction = np.loadtxt(expected_prediction)
# 		npt.assert_equal(predictions, expected_prediction, "The prediction doesnt match with the desired values")


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


# 	-----	NOT APPLIED	-----
# It doesn't apply to the because it has no internal model
# like in other classifiers like REDSVM or SVOREX.
# def test_nnpom_model_is_not_a_dict(self):
# 	#Test preparation
# 	X_train = self.train_file[:,0:(-1)]
# 	y_train = self.train_file[:,(-1)]

# 	X_test = self.test_file[:,0:(-1)]

# 	classifier = NNPOM(epsilon_init = 0.5, n_hidden = 10, max_iter = 500, lambda_value = 0.01)
# 	classifier.fit(X_train, y_train)

# 	#Test execution and verification
# 	with self.assertRaisesRegex(TypeError, "Model should be a dictionary!"):
# 			classifier.classifier_ = 1
# 			classifier.predict(X_test)


def test_nnpom_predict_not_valid_data(X, y):
    # Test preparation
    classifier = NNPOM(epsilon_init=0.5, n_hidden=10, max_iter=500, lambda_value=0.01)
    classifier.fit(X, y)

    # Test execution and verification
    with pytest.raises(ValueError):
        classifier.predict([])
