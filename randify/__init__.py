"""
.. include:: ../README.md
"""

from .randify import randify
from .RandomVariable import RandomVariable
from .utils import pdf, cdf
from .plot import plot_pdf, plot_cdf

__all__ = ["randify", "RandomVariable", "pdf", "cdf", "plot_pdf", "plot_cdf"]
