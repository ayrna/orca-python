"Tests for model selection and estimator loading utilities."

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from orca_python.classifiers import NNOP, NNPOM, REDSVM, SVOREX, OrdinalDecomposition
from orca_python.model_selection import get_classifier_by_name


def test_get_classifier_by_name_correct():
    """Test that get_classifier_by_name returns the correct classifier."""
    # ORCA classifiers
    assert get_classifier_by_name("NNOP") == NNOP
    assert get_classifier_by_name("NNPOM") == NNPOM
    assert get_classifier_by_name("OrdinalDecomposition") == OrdinalDecomposition
    assert get_classifier_by_name("REDSVM") == REDSVM
    assert get_classifier_by_name("SVOREX") == SVOREX

    # Scikit-learn classifiers
    assert get_classifier_by_name("SVC") == SVC
    assert get_classifier_by_name("LogisticRegression") == LogisticRegression


def test_get_classifier_by_name_incorrect():
    """Test that get_classifier_by_name raises ValueError for unknown classifiers."""
    with pytest.raises(ValueError, match="Unknown classifier 'RandomForest'"):
        get_classifier_by_name("RandomForest")

    with pytest.raises(ValueError, match="Unknown classifier 'SVR'"):
        get_classifier_by_name("SVR")
