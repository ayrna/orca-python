"""Utility functions for accessing and using classification metrics by name."""

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
