"""Model selection and estimator loading utilities."""

from importlib import import_module

_ORCA_CLASSIFIERS = {
    "NNOP": "orca_python.classifiers.NNOP",
    "NNPOM": "orca_python.classifiers.NNPOM",
    "OrdinalDecomposition": "orca_python.classifiers.OrdinalDecomposition",
    "REDSVM": "orca_python.classifiers.REDSVM",
    "SVOREX": "orca_python.classifiers.SVOREX",
}

_SKLEARN_CLASSIFIERS = {
    "SVC": "sklearn.svm.SVC",
    "LogisticRegression": "sklearn.linear_model.LogisticRegression",
    "RandomForestClassifier": "sklearn.ensemble.RandomForestClassifier",
}

_CLASSIFIERS = {**_ORCA_CLASSIFIERS, **_SKLEARN_CLASSIFIERS}


def get_classifier_by_name(classifier_name):
    """Return a classifier not instantiated matching a given input name.

    Parameters
    ----------
    classifier_name : str
        Name of the classification algorithm being employed.

    Returns
    -------
    classifier : object
        Returns a classifier, either from a scikit-learn module, or from a
        module of this framework.

    Raises
    ------
    ValueError
        If an unknown classifier name is provided.

    Examples
    --------
    >>> get_classifier_by_name("SVOREX")
    <class 'orca_python.classifiers.SVOREX.SVOREX'>
    >>> get_classifier_by_name("REDSVM")
    <class 'orca_python.classifiers.REDSVM.REDSVM'>
    >>> get_classifier_by_name("SVC")
    <class 'sklearn.svm._classes.SVC'>

    """
    if classifier_name not in _CLASSIFIERS:
        raise ValueError(f"Unknown classifier '{classifier_name}'.")

    module_path, class_name = _CLASSIFIERS[classifier_name].rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, class_name)
