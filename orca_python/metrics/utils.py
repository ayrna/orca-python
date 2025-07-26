"""Utility functions for accessing and using classification metrics by name."""

from sklearn.metrics import make_scorer

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

# Mapping from metric names to their functions
_METRICS = {
    "accuracy_off1": accuracy_off1,
    "amae": amae,
    "ccr": ccr,
    "gm": gm,
    "gmsec": gmsec,
    "mae": mae,
    "mmae": mmae,
    "ms": ms,
    "mze": mze,
    "rps": rps,
    "spearman": spearman,
    "tkendall": tkendall,
    "wkappa": wkappa,
}

# Indicates whether a higher score means better performance
_GREATER_IS_BETTER = {
    "accuracy_off1": True,
    "amae": False,
    "ccr": True,
    "gm": True,
    "gmsec": True,
    "mae": False,
    "mmae": False,
    "ms": True,
    "mze": False,
    "rps": False,
    "spearman": True,
    "tkendall": True,
    "wkappa": True,
}


def get_metric_names():
    """Get the names of all available metrics.

    These names can be passed to :func:`~orca_python.metrics.compute_metric` to
    compute the metric value.

    Returns
    -------
    list of str
        Names of all available metrics.

    Examples
    --------
    >>> from orca_python.metrics import get_metric_names
    >>> all_metrics = get_metric_names()
    >>> type(all_metrics)
    <class 'list'>
    >>> all_metrics[:3]
    ['accuracy_off1', 'amae', 'ccr']
    >>> "rps" in all_metrics
    True

    """
    return sorted(_METRICS.keys())


def greater_is_better(metric_name):
    """Determine if greater values indicate better classification performance.

    Needed when declaring a new scorer through make_scorer from sklearn.

    Parameters
    ----------
    metric_name : str
        Name of the metric.

    Returns
    -------
    greater_is_better : bool
        True if greater values are better, False otherwise.

    Raises
    ------
    KeyError
        If the metric name is not recognized.

    Examples
    --------
    >>> from orca_python.metrics import greater_is_better
    >>> greater_is_better("ccr")
    True
    >>> greater_is_better("mze")
    False
    >>> greater_is_better("mae")
    False

    """
    try:
        return _GREATER_IS_BETTER[metric_name.lower().strip()]
    except KeyError:
        raise KeyError(f"Unrecognized metric name: '{metric_name}'.")


def load_metric_as_scorer(metric_name):
    """Load a metric function by name and return a scorer compatible with
    sklearn.

    Parameters
    ----------
    metric_name : str
        Name of the metric.

    Returns
    -------
    callable
        A scikit-learn compatible scorer.

    Raises
    ------
    TypeError
        If metric_name is not a string.

    ValueError
        If the metric name is not implemented.

    Examples
    --------
    >>> from orca_python.metrics import load_metric_as_scorer
    >>> scorer = load_metric_as_scorer("ccr")
    >>> type(scorer)
    <class 'sklearn.metrics._scorer._Scorer'>
    >>> load_metric_as_scorer("mae")
    make_scorer(mae, greater_is_better=False, response_method='predict')

    """
    if not isinstance(metric_name, str):
        raise TypeError("metric_name must be a string.")

    metric_name = metric_name.lower().strip()

    try:
        metric_func = _METRICS[metric_name]
    except KeyError:
        raise KeyError(f"Unrecognized metric name: '{metric_name}'.")

    gib = greater_is_better(metric_name)
    scorer = make_scorer(metric_func, greater_is_better=gib)
    scorer.metric_name = metric_name
    return scorer


def compute_metric(metric_name, y_true, y_pred):
    """Compute the value of a metric from true and predicted labels.

    Parameters
    ----------
    metric_name : str
        Name of the metric.

    y_true : np.ndarray, shape (n_samples,)
        Ground truth labels.

    y_pred : np.ndarray, shape (n_samples,)
        Predicted labels.

    Returns
    -------
    float
        Numeric value of the classification metric.

    Raises
    ------
    KeyError
        If the metric name is not recognized.

    Examples
    --------
    >>> from orca_python.metrics import compute_metric
    >>> y_true = [0, 1, 2, 1, 0]
    >>> y_pred = [0, 1, 1, 1, 0]
    >>> compute_metric("ccr", y_true, y_pred)
    0.8
    >>> compute_metric("mae", y_true, y_pred)
    0.2

    """
    try:
        metric_func = _METRICS[metric_name]
    except KeyError:
        raise KeyError(f"Unrecognized metric name: '{metric_name}'.")

    return metric_func(y_true, y_pred)
