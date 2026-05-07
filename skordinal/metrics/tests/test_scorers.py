"""Tests for the public scorer API."""

import numpy.testing as npt
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from skordinal.metrics import (
    accuracy_off1,
    amae,
    ccr,
    get_ordinal_scorer,
    gm,
    gmsec,
    list_ordinal_scorers,
    mae,
    mmae,
    ms,
    mze,
    spearman,
    tkendall,
    wkappa,
)


def test_list_ordinal_scorers_is_sorted():
    """list_ordinal_scorers returns names in sorted order."""
    names = list_ordinal_scorers()
    assert names == sorted(names)


def test_list_ordinal_scorers_returns_new_list():
    """Two calls return equal but non-identical list objects."""
    assert list_ordinal_scorers() == list_ordinal_scorers()
    assert list_ordinal_scorers() is not list_ordinal_scorers()


def test_scorers_not_in_public_all():
    """Private symbol _SCORERS is not exported from skordinal.metrics."""
    import skordinal.metrics as m

    assert "_SCORERS" not in m.__all__


@pytest.mark.parametrize(
    "name, metric_fn",
    [
        ("accuracy_off1", accuracy_off1),
        ("ccr", ccr),
        ("gm", gm),
        ("gmsec", gmsec),
        ("ms", ms),
        ("spearman", spearman),
        ("tkendall", tkendall),
        ("wkappa", wkappa),
    ],
)
def test_utility_scorer_sign(name, metric_fn):
    """Utility scorers have sign +1 and wrap the expected metric function."""
    scorer = get_ordinal_scorer(name)
    assert scorer._score_func is metric_fn
    assert scorer._sign == 1


@pytest.mark.parametrize(
    "name, metric_fn",
    [
        ("neg_amae", amae),
        ("neg_mae", mae),
        ("neg_mmae", mmae),
        ("neg_mze", mze),
        ("amae", amae),
        ("mae", mae),
        ("mmae", mmae),
        ("mze", mze),
    ],
)
def test_loss_scorer_sign(name, metric_fn):
    """Loss scorers have sign -1 and wrap the expected metric function."""
    scorer = get_ordinal_scorer(name)
    assert scorer._score_func is metric_fn
    assert scorer._sign == -1


def test_whitespace_stripped():
    """Leading and trailing whitespace in the name is ignored."""
    assert get_ordinal_scorer("  neg_mae  ")._sign == -1


@pytest.mark.parametrize(
    "name, metric_fn",
    [
        ("ccr", ccr),
        ("accuracy_off1", accuracy_off1),
        ("gm", gm),
        ("gmsec", gmsec),
        ("mae", mae),
        ("mmae", mmae),
        ("amae", amae),
        ("ms", ms),
        ("mze", mze),
        ("tkendall", tkendall),
        ("wkappa", wkappa),
        ("spearman", spearman),
    ],
)
def test_scorer_output_matches_metric(name, metric_fn):
    """Scorer's score function returns the same value as calling the metric directly."""
    y_true = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
    y_pred = [1, 3, 3, 1, 2, 3, 1, 2, 2, 1, 3, 1, 1, 2, 2, 2, 3, 3, 1, 3]
    scorer = get_ordinal_scorer(name)
    npt.assert_almost_equal(
        scorer._score_func(y_true, y_pred), metric_fn(y_true, y_pred)
    )


def test_gridcv_with_loss_scorer():
    """Loss scorer integrates with GridSearchCV: best_score_ is non-positive."""
    import numpy as np

    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 3))
    y = np.repeat([0, 1, 2], 10)

    gs = GridSearchCV(
        LogisticRegression(max_iter=200),
        param_grid={"C": [0.1, 1.0]},
        scoring=get_ordinal_scorer("neg_mae"),
        cv=StratifiedKFold(n_splits=2),
    )
    gs.fit(X, y)
    assert gs.best_score_ <= 0


def test_get_ordinal_scorer_type_error():
    """Non-string input raises TypeError."""
    with pytest.raises(TypeError, match="must be a string"):
        get_ordinal_scorer(123)


def test_get_ordinal_scorer_value_error():
    """Unknown name raises ValueError mentioning the requested name."""
    with pytest.raises(ValueError, match="roc_auc"):
        get_ordinal_scorer("roc_auc")
