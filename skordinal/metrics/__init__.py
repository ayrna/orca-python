"""Metrics module."""

from sklearn.metrics import accuracy_score, mean_absolute_error

from ._metrics import (
    accuracy_off1,
    average_mean_absolute_error,
    geometric_mean,
    gmsec,
    kendalls_tau,
    maximum_mean_absolute_error,
    mean_zero_one_error,
    minimum_sensitivity,
    ranked_probability_score,
    spearmans_rho,
    weighted_kappa,
)
from ._metrics import (
    amae as amae,
)
from ._metrics import (
    ccr as ccr,
)
from ._metrics import (
    gm as gm,
)
from ._metrics import (
    mae as mae,
)
from ._metrics import (
    mmae as mmae,
)
from ._metrics import (
    ms as ms,
)
from ._metrics import (
    mze as mze,
)
from ._metrics import (
    rps as rps,
)
from ._metrics import (
    spearman as spearman,
)
from ._metrics import (
    tkendall as tkendall,
)
from ._metrics import (
    wkappa as wkappa,
)
from ._scorers import get_ordinal_scorer, list_ordinal_scorers

__all__ = [
    "accuracy_off1",
    "accuracy_score",
    "average_mean_absolute_error",
    "geometric_mean",
    "get_ordinal_scorer",
    "gmsec",
    "kendalls_tau",
    "list_ordinal_scorers",
    "maximum_mean_absolute_error",
    "mean_absolute_error",
    "mean_zero_one_error",
    "minimum_sensitivity",
    "ranked_probability_score",
    "spearmans_rho",
    "weighted_kappa",
]
