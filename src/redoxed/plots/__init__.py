"""
Plotting module for redoxED electrochemical data visualization.

This module provides specialized plotting classes for different types of
electrochemical data including EIS, DRT, polarisation, and residuals plots.
All plotting classes inherit from BasePlot for consistent styling and behavior.
"""

from .base_plot import BasePlot
from .generic_plot import GenericPlot
from .eis_plot import NyquistPlot
from .drt_plot import DRTPlot
from .residuals_plot import ResidualsPlot
from .polarisation_plot import PolarisationPlot
from .cycling_plot import EfficiencyPlot


__all__ = [
    "BasePlot",
    "GenericPlot",
    "NyquistPlot",
    "DRTPlot",
    "ResidualsPlot",
    "PolarisationPlot",
    "EfficiencyPlot",
]
