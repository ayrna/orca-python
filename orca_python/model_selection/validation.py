"""Parameter grid preparation and validation utilities for model selection."""

from ast import literal_eval
from copy import deepcopy
from importlib import import_module
from itertools import product

import numpy as np


def check_for_random_state(estimator):
    """Check if the estimator accepts a random_state parameter.

    Parameters
    ----------
    estimator : object
        The estimator class to check.

    Returns
    -------
    bool
        True if the estimator accepts random_state parameter, False otherwise.

    Examples
    --------
    >>> from sklearn.svm import SVC, SVR
    >>> from sklearn.linear_model import LogisticRegression
    >>> check_for_random_state(SVC)
    True
    >>> check_for_random_state(SVR)
    False
    >>> check_for_random_state(LogisticRegression)
    True

    """
    try:
        estimator(random_state=0)
        return True
    except (TypeError, ValueError):
        return False


def is_ensemble(param_grid):
    """Check if the given parameters correspond to an ensemble method.

    Parameters
    ----------
    param_grid : dict
        Dictionary defining hyperparameter search space for model optimization.

    Returns
    -------
    bool
        True if the parameters correspond to an ensemble method, False
        otherwise.

    Examples
    --------
    >>> is_ensemble({"base_classifier": "SVC"})
    True
    >>> is_ensemble({"base_classifier": "SVC", "parameters": {"C": [0.1, 0.2]}})
    True
    >>> is_ensemble({"C": 0.1})
    False

    """
    return isinstance(param_grid, dict) and "base_classifier" in param_grid


def is_searchcv(param_grid):
    """Check if the given parameters require cross-validation search.

    Parameters
    ----------
    param_grid : dict
        Dictionary defining hyperparameter search space for model optimization.

    Returns
    -------
    bool
        True if cross-validation is required, False otherwise.

    """
    if not isinstance(param_grid, dict):
        return False

    has_search_params = any(
        isinstance(value, list) and len(value) > 1 for value in param_grid.values()
    )

    if is_ensemble(param_grid) and "parameters" in param_grid:
        base_params = param_grid["parameters"]

        if isinstance(base_params, dict):
            base_has_search = any(
                isinstance(value, list) and len(value) > 1
                for value in base_params.values()
            )
            return has_search_params or base_has_search
        else:
            if isinstance(base_params, list):
                return has_search_params or len(base_params) > 1
            else:
                return has_search_params

    return has_search_params


def prepare_param_grid(estimator, param_grid, random_state=None):
    """This function processes parameter grids to ensure compatibility with
    scikit-learn's GridSearchCV and handles special cases like ensemble methods and
    random state injection.

    Parameters
    ----------
    estimator : object
        The estimator to prepare parameters for.

    param_grid : dict
        Dictionary defining hyperparameter search space for model optimization.

    random_state : int, RandomState instance or None, optional (default=None)
        Seed for reproducible randomization in model training and probability
        estimation.

    Returns
    -------
    dict
        Prepared parameters dictionary.

    Raises
    ------
    ValueError
        If param_grid is not a dictionary.

    Examples
    --------
    >>> from orca_python.model_selection import get_classifier_by_name
    >>> estimator = get_classifier_by_name("OrdinalDecomposition")
    >>> param_grid = {
    ...     "dtype": "ordered_partitions",
    ...     "decision_method": "frank_hall",
    ...     "base_classifier": "SVC",
    ...     "parameters": {
    ...         "C": [0.1, 1.0],
    ...         "gamma": [0.01, 0.1],
    ...         "probability": ["True"]
    ...     }
    ... }
    >>> prepared_params = prepare_param_grid(estimator, param_grid, random_state=0)
    >>> len(prepared_params["parameters"])
    4

    """
    if not isinstance(param_grid, dict):
        raise ValueError("param_grid must be a dictionary")

    if random_state is None:
        random_state = np.random.get_state()[1][0]

    param_grid = deepcopy(param_grid)
    param_grid = _add_random_state(estimator, param_grid, random_state)

    if is_ensemble(param_grid):
        param_grid["parameters"] = _normalize_param_grid(param_grid["parameters"])
        param_grid = _prepare_parameters_for_ensemble(param_grid, random_state)

    if is_searchcv(param_grid):
        for p_name, p_value in param_grid.items():
            if not isinstance(p_value, list) and not isinstance(p_value, dict):
                param_grid[p_name] = [p_value]
    else:
        for p_name, p_value in param_grid.items():
            if isinstance(p_value, list):
                param_grid[p_name] = p_value[0]

    return param_grid


def _add_random_state(estimator, param_grid, random_state):
    """Add random_state to param_grid if the estimator accepts it and it's
    not already present.

    Parameters
    ----------
    estimator : object
        The estimator to add random_state to.

    param_grid : dict
        Dictionary defining hyperparameter search space for model optimization.

    random_state : int
        Seed for reproducible randomization in model training and probability
        estimation.

    Returns
    -------
    dict
        Prepared parameters dictionary.

    """
    if check_for_random_state(estimator) and "random_state" not in param_grid:
        param_grid["random_state"] = random_state
    return param_grid


def _normalize_param_grid(param_grid):
    """Ensure all values in param_grid are lists (for grid search compatibility).

    Parameters
    ----------
    param_grid : dict
        Dictionary defining hyperparameter search space for model optimization.

    Returns
    -------
    dict
        Dictionary with all values as lists.

    """
    normalized = {}
    for k, v in param_grid.items():
        if isinstance(v, list):
            normalized[k] = v
        else:
            normalized[k] = [v]
    return normalized


def _prepare_parameters_for_ensemble(param_grid, random_state=None):
    """Process the parameters for ensemble methods.

    Parameters
    ----------
    param_grid : dict
        Dictionary defining hyperparameter search space for model optimization.

    random_state : int, RandomState instance or None
        Seed for reproducible randomization in model training and probability
        estimation.

    Returns
    -------
    dict
        Processed parameters dictionary.

    Raises
    ------
    TypeError
        If all parameters for base_classifier are not lists.

    """
    if not is_ensemble(param_grid):
        return param_grid

    from orca_python.model_selection.loaders import get_classifier_by_name

    # base_estimator = get_classifier_by_name(param_grid["base_classifier"])
    # Temporal fix for the issue with dotted paths
    base_id = param_grid["base_classifier"]
    if not isinstance(base_id, str):
        base_estimator = base_id
    else:
        try:
            base_estimator = get_classifier_by_name(base_id)
        except Exception:
            if "." in base_id:
                module_path, class_name = base_id.rsplit(".", 1)
                base_estimator = getattr(import_module(module_path), class_name)
            else:
                raise ValueError(f"Unknown base_classifier identifier: {base_id}. ")

    base_params = param_grid.get("parameters", {})

    if check_for_random_state(base_estimator):
        base_params["random_state"] = [random_state]

    # Creating a list for each parameter.
    # Elements represented as 'parameterName;parameterValue'.
    param_combinations = [
        [f"{k};{v}" for v in values]
        for k, values in _normalize_param_grid(base_params).items()
    ]
    # Creates a list of dictionaries, containing all
    # combinations of given parameters
    combinations = [
        dict(item.split(";") for item in combo)
        for combo in product(*param_combinations)
    ]

    # Returns non-string values back to their normal self
    for combination in combinations:
        for k, v in combination.items():
            try:
                combination[k] = literal_eval(v)
            except ValueError:
                pass

    param_grid["parameters"] = combinations
    return param_grid
