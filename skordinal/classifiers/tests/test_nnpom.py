"""Tests for the NNPOM classifier."""

import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from skordinal.classifiers import NNPOM


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
        ("alpha", -1e-5),
    ],
)
def test_nnpom_hyperparameter_value_validation(X, y, param_name, invalid_value):
    """Test that NNPOM raises ValueError for invalid of hyperparameters."""
    classifier = NNPOM(**{param_name: invalid_value})

    with pytest.raises(ValueError, match=rf"The '{param_name}' parameter.*"):
        classifier.fit(X, y)


@pytest.mark.parametrize(
    "param_name, invalid_value",
    [
        ("epsilon_init", "high"),
        ("n_hidden", 5.5),
        ("max_iter", 2.5),
        ("alpha", "tight"),
    ],
)
def test_nnpom_hyperparameter_type_validation(X, y, param_name, invalid_value):
    """Test that NNPOM raises ValueError for invalid types of hyperparameters."""
    classifier = NNPOM(**{param_name: invalid_value})

    with pytest.raises(ValueError, match=rf"The '{param_name}' parameter.*"):
        classifier.fit(X, y)


def test_nnpom_fit_returns_self(X, y):
    """fit should return self for sklearn compatibility."""
    classifier = NNPOM()
    model = classifier.fit(X, y)
    assert model is classifier


def test_nnpom_fit_input_validation(X, y):
    """Test that input data is validated."""
    X_invalid = X[:-1, :-1]
    y_invalid = y[:-1]

    classifier = NNPOM()
    with pytest.raises(ValueError):
        classifier.fit(X, y_invalid)

    with pytest.raises(ValueError):
        classifier.fit([], y)

    with pytest.raises(ValueError):
        classifier.fit(X, [])

    with pytest.raises(ValueError):
        classifier.fit(X_invalid, y)


def test_nnpom_sets_fitted_attributes_after_fit(X, y):
    """Test than NNPOM exposes fitted attributes aligned con sklearn-style."""
    clf = NNPOM(n_hidden=4, max_iter=5)
    clf.fit(X, y)

    for attr in [
        "classes_",
        "n_features_in_",
        "theta1_",
        "theta2_",
        "loss_",
        "n_iter_",
        "n_layers_",
        "n_outputs_",
        "out_activation_",
    ]:
        assert hasattr(clf, attr), f"Missing fitted attribute: {attr}"

    assert isinstance(clf.classes_, np.ndarray) and np.array_equal(
        clf.classes_, np.unique(y)
    )
    assert isinstance(clf.n_features_in_, int) and clf.n_features_in_ == X.shape[1]
    assert isinstance(clf.loss_, (float, np.floating)) and clf.loss_ >= 0
    assert isinstance(clf.n_iter_, int) and 1 <= clf.n_iter_ <= 5
    assert isinstance(clf.n_layers_, int) and clf.n_layers_ == 3
    assert isinstance(clf.n_outputs_, int) and clf.n_outputs_ == len(np.unique(y)) - 1
    assert isinstance(clf.out_activation_, str) and clf.out_activation_ == "logistic"


def test_nnpom_predict_invalid_input_raises_error(X, y):
    """Test that invalid input raises an error."""
    classifier = NNPOM()
    classifier.fit(X, y)

    with pytest.raises(ValueError):
        classifier.predict([])


def test_nnpom_predict_raises_if_not_fitted(X):
    """Test that predict raises NotFittedError if called before fit."""
    classifier = NNPOM()
    with pytest.raises(NotFittedError):
        classifier.predict(X)


def test_nnpom_feature_names_in_when_dataframe(X, y):
    """Test that feature_names_in_ is set when X is a DataFrame."""
    df = pd.DataFrame(X, columns=["f0", "f1"])
    classifier = NNPOM(n_hidden=4, max_iter=5).fit(df, y)

    assert hasattr(classifier, "feature_names_in_")
    np.testing.assert_array_equal(
        classifier.feature_names_in_, np.array(["f0", "f1"], dtype=object)
    )


def test_nnpom_parameter_constraints_match_init_params():
    """Test that _parameter_constraints keys match __init__ parameters."""
    init_params = set(inspect.signature(NNPOM.__init__).parameters) - {"self"}
    assert set(NNPOM._parameter_constraints) == init_params


def test_nnpom_predict_rejects_wrong_n_features(X, y):
    """Test that predict rejects input with mismatched n_features."""
    classifier = NNPOM(n_hidden=4, max_iter=5).fit(X, y)
    with pytest.raises(ValueError):
        classifier.predict(X[:, :-1])
