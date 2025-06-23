"""Metrics for ordinal classification."""

from __future__ import division

import numpy as np
import scipy.stats
from sklearn.metrics import confusion_matrix, recall_score


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
        True if greater values indicate better classification performance, False otherwise.

    Examples
    --------
    >>> from orca_python.metrics.metrics import greater_is_better
    >>> greater_is_better("ccr")
    True
    >>> greater_is_better("mze")
    False
    >>> greater_is_better("mae")
    False

    """
    greater_is_better_metrics = ["ccr", "ms", "gm", "tkendall", "wkappa", "spearman"]
    if metric_name in greater_is_better_metrics:
        return True
    else:
        return False


def ccr(y_true, y_pred):
    """Calculate the Correctly Classified Ratio.

    Also named Accuracy, it's the percentage of well classified patterns among all
    patterns from a set.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        Ground truth labels.

    y_pred : np.ndarray, shape (n_samples,)
        Predicted labels.

    Returns
    -------
    ccr : float
        Correctly classified ratio.

    Examples
    --------
    >>> import numpy as np
    >>> from orca_python.metrics import ccr
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 0, 0, 1])
    >>> ccr(y_true, y_pred)
    0.5714285714285714

    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    return np.count_nonzero(y_true == y_pred) / float(len(y_true))


def amae(y_true, y_pred):
    """Calculate the Average MAE.

    Mean of the MAE metric among classes.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        Ground truth labels.

    y_pred : np.ndarray, shape (n_samples,)
        Predicted labels.

    Returns
    -------
    amae : float
        Average mean absolute error.

    Examples
    --------
    >>> import numpy as np
    >>> from orca_python.metrics import amae
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 3, 0, 1])
    >>> amae(y_true, y_pred)
    np.float64(0.125)

    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    n_class = cm.shape[0]
    costs = np.reshape(np.tile(range(n_class), n_class), (n_class, n_class))
    costs = np.abs(costs - np.transpose(costs))
    errors = costs * cm
    per_class_maes = np.sum(errors, axis=1) / np.sum(cm, axis=1).astype("double")
    per_class_maes = per_class_maes[~np.isnan(per_class_maes)]
    return np.mean(per_class_maes)


def gm(y_true, y_pred):
    """Calculate the Geometric mean of the sensitivity (accuracy) for each class.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        Ground truth labels.

    y_pred : np.ndarray, shape (n_samples,)
        Predicted labels.

    Returns
    -------
    gm : float
        Geometric mean of the sensitivities.

    Examples
    --------
    >>> import numpy as np
    >>> from orca_python.metrics import gm
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 3, 0, 1])
    >>> gm(y_true, y_pred)
    np.float64(0.8408964152537145)

    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    sum_by_class = np.sum(cm, axis=1)
    sensitivities = np.diag(cm) / sum_by_class.astype("double")
    sensitivities[sum_by_class == 0] = 1
    gm = pow(np.prod(sensitivities), 1.0 / cm.shape[0])
    return gm


def mae(y_true, y_pred):
    """Calculate the Mean Absolute Error.

    Average absolute deviation of the predicted class from the actual true class.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        Ground truth labels.

    y_pred : np.ndarray, shape (n_samples,)
        Predicted labels.

    Returns
    -------
    mae : float
        Mean absolute error.

    Examples
    --------
    >>> import numpy as np
    >>> from orca_python.metrics import mae
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 3, 0, 1])
    >>> mae(y_true, y_pred)
    np.float64(0.2857142857142857)

    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    return abs(y_true - y_pred).sum() / len(y_true)


def mmae(y_true, y_pred):
    """Calculate the Maximum Mean Absolute Error.

    MAE value of the class with higher distance from the true values to the predicted
    ones.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        Ground truth labels.

    y_pred : np.ndarray, shape (n_samples,)
        Predicted labels.

    Returns
    -------
    mmae : float
        Maximum mean absolute error.

    Examples
    --------
    >>> import numpy as np
    >>> from orca_python.metrics import mmae
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 3, 0, 1])
    >>> mmae(y_true, y_pred)
    np.float64(0.5)

    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    n_class = cm.shape[0]
    costs = np.reshape(np.tile(range(n_class), n_class), (n_class, n_class))
    costs = np.abs(costs - np.transpose(costs))
    errors = costs * cm
    per_class_maes = np.sum(errors, axis=1) / np.sum(cm, axis=1).astype("double")
    per_class_maes = per_class_maes[~np.isnan(per_class_maes)]
    return per_class_maes.max()


def ms(y_true, y_pred):
    """Calculate the Minimum Sensitivity.

    Lowest percentage of patterns correctly predicted as belonging to each class, with
    respect to the total number of examples in the corresponding class.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        Ground truth labels.

    y_pred : np.ndarray, shape (n_samples,)
        Predicted labels.

    Returns
    -------
    ms : float
        Minimum sensitivity.

    Examples
    --------
    >>> import numpy as np
    >>> from orca_python.metrics import ms
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 3, 0, 1])
    >>> ms(y_true, y_pred)
    np.float64(0.5)

    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    sensitivities = recall_score(y_true, y_pred, average=None)
    return np.min(sensitivities)


def mze(y_true, y_pred):
    """Calculate the Mean Zero-one Error.

    Better known as error rate, is the complementary measure of CCR.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        Ground truth labels.

    y_pred : np.ndarray, shape (n_samples,)
        Predicted labels.

    Returns
    -------
    mze : float
        Mean zero-one error.

    Examples
    --------
    >>> import numpy as np
    >>> from orca_python.metrics import mze
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 3, 0, 1])
    >>> mze(y_true, y_pred)
    np.float64(0.2857142857142857)

    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    confusion = confusion_matrix(y_true, y_pred)
    return 1 - np.diagonal(confusion).sum() / confusion.sum()


def tkendall(y_true, y_pred):
    """Calculate Kendall's tau.

    A statistic used to measure the association between two measured quantities. It is
    a measure of rank correlation.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.

    y_pred : np.ndarray
        Predicted labels.

    Returns
    -------
    tkendall : float
        Kendall's tau.

    Examples
    --------
    >>> import numpy as np
    >>> from orca_python.metrics import tkendall
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 3, 0, 1])
    >>> tkendall(y_true, y_pred)
    np.float64(0.8140915784106943)

    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    corr, pvalue = scipy.stats.kendalltau(y_true, y_pred)
    return corr


def wkappa(y_true, y_pred):
    """Calculate the Weighted Kappa.

    A modified version of the Kappa statistic calculated to allow assigning
    different weights to different levels of aggregation between two variables.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        Ground truth labels.

    y_pred : np.ndarray, shape (n_samples,)
        Predicted labels.

    Returns
    -------
    wkappa : float
        Weighted Kappa.

    Examples
    --------
    >>> import numpy as np
    >>> from orca_python.metrics import wkappa
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 3, 0, 1])
    >>> wkappa(y_true, y_pred)
    np.float64(0.7586206896551724)

    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    n_class = cm.shape[0]
    costs = np.reshape(np.tile(range(n_class), n_class), (n_class, n_class))
    costs = np.abs(costs - np.transpose(costs))
    f = 1 - costs

    n = cm.sum()
    x = cm / n

    r = x.sum(axis=1)  # Row sum
    s = x.sum(axis=0)  # Col sum
    Ex = r.reshape(-1, 1) * s
    po = (x * f).sum()
    pe = (Ex * f).sum()
    return (po - pe) / (1 - pe)


def spearman(y_true, y_pred):
    """Calculate the Spearman's rank correlation coefficient.

    A non-parametric measure of statistical dependence between two variables.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        Ground truth labels.

    y_pred : np.ndarray, shape (n_samples,)
        Predicted labels.

    Returns
    -------
    spearman : float
        Spearman rank correlation coefficient.

    Examples
    --------
    >>> import numpy as np
    >>> from orca_python.metrics import spearman
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 3, 0, 1])
    >>> spearman(y_true, y_pred)
    np.float64(0.9165444688834581)

    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    n = len(y_true)
    num = (
        (y_true - np.repeat(np.mean(y_true), n))
        * (y_pred - np.repeat(np.mean(y_pred), n))
    ).sum()
    div = np.sqrt(
        (pow(y_true - np.repeat(np.mean(y_true), n), 2)).sum()
        * (pow(y_pred - np.repeat(np.mean(y_pred), n), 2)).sum()
    )

    if num == 0:
        return 0
    else:
        return num / div
