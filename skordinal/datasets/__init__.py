"""Bundled ordinal classification datasets."""

from ._loaders import (
    load_balance_scale,
    load_era,
    load_esl,
    load_lev,
    load_swd,
)

__all__ = [
    "load_balance_scale",
    "load_era",
    "load_esl",
    "load_lev",
    "load_swd",
]
