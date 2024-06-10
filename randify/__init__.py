"""
.. include:: ../README.md
"""

from .randify import randify
from .RandomVariable import RandomVariable
from .utils import pdf, cov
from .plot import plot_pdf

__all__ = ["randify", "RandomVariable", "pdf", "cov", "plot_pdf"]
