"""Ordinal classification classifiers module."""

from .NNOP import NNOP
from .NNPOM import NNPOM
from .OrdinalDecomposition import OrdinalDecomposition
from .REDSVM import REDSVM
from .SVOREX import SVOREX

__all__ = [
    "NNOP",
    "NNPOM",
    "OrdinalDecomposition",
    "REDSVM",
    "SVOREX",
]
