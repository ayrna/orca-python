"""Scorer registry for ordinal classification metrics."""

from __future__ import annotations

from typing import Callable

from sklearn.metrics import make_scorer

from ._metrics import (
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

_SCORERS: dict[str, Callable] = {
    # Error metrics (greater_is_better=False).
    "neg_amae": make_scorer(amae, greater_is_better=False),
    "neg_mae": make_scorer(mae, greater_is_better=False),
    "neg_mmae": make_scorer(mmae, greater_is_better=False),
    "neg_mze": make_scorer(mze, greater_is_better=False),
    "neg_rps": make_scorer(rps, greater_is_better=False),
    # Scoring metrics (greater_is_better=True).
    "accuracy_off1": make_scorer(accuracy_off1),
    "ccr": make_scorer(ccr),
    "gm": make_scorer(gm),
    "gmsec": make_scorer(gmsec),
    "ms": make_scorer(ms),
    "spearman": make_scorer(spearman),
    "tkendall": make_scorer(tkendall),
    "wkappa": make_scorer(wkappa),
    # Error metrics without neg_ prefix, for compatibility.
    "amae": make_scorer(amae, greater_is_better=False),
    "mae": make_scorer(mae, greater_is_better=False),
    "mmae": make_scorer(mmae, greater_is_better=False),
    "mze": make_scorer(mze, greater_is_better=False),
    "rps": make_scorer(rps, greater_is_better=False),
}

__all__ = ["get_ordinal_scorer", "list_ordinal_scorers"]


def get_ordinal_scorer(name: str) -> Callable:
    """Return a scikit-learn-compatible scorer by name.

    Parameters
    ----------
    name : str
        Scorer name. Use :func:`list_ordinal_scorers` for the full list.
        Leading and trailing whitespace is stripped before lookup.

    Returns
    -------
    scorer : callable
        A scorer compatible with :class:`~sklearn.model_selection.GridSearchCV`
        and :func:`~sklearn.model_selection.cross_val_score`.

    Raises
    ------
    TypeError
        If ``name`` is not a string.
    ValueError
        If ``name`` is not a registered scorer name.

    Examples
    --------
    >>> from skordinal.metrics import get_ordinal_scorer
    >>> scorer = get_ordinal_scorer("neg_mae")
    >>> callable(scorer)
    True

    """
    if not isinstance(name, str):
        raise TypeError(f"name must be a string, got {type(name)}.")
    key = name.strip()
    if key in _SCORERS:
        return _SCORERS[key]
    raise ValueError(
        f"Unknown scorer name: {name!r}. Available: {list_ordinal_scorers()}."
    )


def list_ordinal_scorers() -> list[str]:
    """Return the sorted list of registered ordinal scorer names.

    Returns
    -------
    names : list of str
        Sorted list of all scorer names accepted by :func:`get_ordinal_scorer`.

    Examples
    --------
    >>> from skordinal.metrics import list_ordinal_scorers
    >>> all_scorers = list_ordinal_scorers()
    >>> type(all_scorers)
    <class 'list'>
    >>> "neg_mae" in all_scorers
    True
    >>> "ccr" in all_scorers
    True

    """
    return sorted(_SCORERS)
