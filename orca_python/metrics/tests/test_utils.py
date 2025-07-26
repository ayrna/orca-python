"""Tests for the metrics module utilities."""

import numpy.testing as npt
import pytest

from orca_python.metrics import (
    accuracy_off1,
    amae,
    ccr,
    gm,
    gmsec,
    mae,
    mmae,
    ms,
    mze,
    rps,
    spearman,
    tkendall,
    wkappa,
)
from orca_python.metrics.utils import (
    _METRICS,
    compute_metric,
    get_metric_names,
    greater_is_better,
    load_metric_as_scorer,
)


def test_get_metric_names():
    """Test that get_metric_names returns all available metric names."""
    all_metrics = get_metric_names()
    expected_names = list(_METRICS.keys())

    assert type(all_metrics) is list
    assert all_metrics[:3] == ["accuracy_off1", "amae", "ccr"]
    assert "rps" in all_metrics
    npt.assert_array_equal(sorted(all_metrics), sorted(expected_names))


@pytest.mark.parametrize(
    "metric_name, gib",
    [
        ("accuracy_off1", True),
        ("amae", False),
        ("ccr", True),
        ("gm", True),
        ("gmsec", True),
        ("mae", False),
        ("mmae", False),
        ("ms", True),
        ("mze", False),
        ("rps", False),
        ("spearman", True),
        ("tkendall", True),
        ("wkappa", True),
    ],
)
def test_greater_is_better(metric_name, gib):
    """Test that greater_is_better returns the correct boolean for each metric."""
    assert greater_is_better(metric_name) == gib


def test_greater_is_better_invalid_name():
    """Test that greater_is_better raises an error for an invalid metric name."""
    error_msg = "Unrecognized metric name: 'roc_auc'."

    with pytest.raises(KeyError, match=error_msg):
        greater_is_better("roc_auc")


@pytest.mark.parametrize(
    "metric_name, metric",
    [
        ("rps", rps),
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
def test_load_metric_as_scorer(metric_name, metric):
    """Test that load_metric_as_scorer correctly loads the expected metric."""
    metric_func = load_metric_as_scorer(metric_name)

    assert metric_func._score_func == metric
    assert metric_func._sign == (1 if greater_is_better(metric_name) else -1)


@pytest.mark.parametrize(
    "metric_name, metric",
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
def test_correct_metric_output(metric_name, metric):
    """Test that the loaded metric function produces the same output as the
    original metric."""
    y_true = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
    y_pred = [1, 3, 3, 1, 2, 3, 1, 2, 2, 1, 3, 1, 1, 2, 2, 2, 3, 3, 1, 3]
    metric_func = load_metric_as_scorer(metric_name)
    metric_true = metric(y_true, y_pred)
    metric_pred = metric_func._score_func(y_true, y_pred)

    npt.assert_almost_equal(metric_pred, metric_true, decimal=6)


def test_load_metric_invalid_name():
    """Test that loading an invalid metric raises the correct exception."""
    error_msg = "metric_name must be a string."
    with pytest.raises(TypeError, match=error_msg):
        load_metric_as_scorer(123)

    error_msg = "Unrecognized metric name: 'roc_auc'."
    with pytest.raises(KeyError, match=error_msg):
        load_metric_as_scorer("roc_auc")


@pytest.mark.parametrize(
    "metric_name",
    [
        "ccr",
        "accuracy_off1",
        "gm",
        "gmsec",
        "mae",
        "mmae",
        "amae",
        "ms",
        "mze",
        "tkendall",
        "wkappa",
        "spearman",
    ],
)
def test_compute_metric(metric_name) -> None:
    """Test that compute_metric returns the correct metric value."""
    y_true = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
    y_pred = [1, 3, 3, 1, 2, 3, 1, 2, 2, 1, 3, 1, 1, 2, 2, 2, 3, 3, 1, 3]
    metric_value = compute_metric(metric_name, y_true, y_pred)
    metric_func = load_metric_as_scorer(metric_name)
    metric_true = metric_func._score_func(y_true, y_pred)

    npt.assert_almost_equal(metric_value, metric_true, decimal=6)


def test_compute_metric_invalid_name():
    """Test that compute_metric raises an error for an invalid metric name."""
    error_msg = "Unrecognized metric name: 'roc_auc'."

    with pytest.raises(KeyError, match=error_msg):
        compute_metric("roc_auc", [1, 2, 3], [1, 2, 3])
