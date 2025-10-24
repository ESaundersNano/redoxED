"""
Residuals plotting classes.

This module provides specialized plotting classes for visualizing fitting residuals
from electrochemical impedance spectroscopy data analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator
from typing import Optional, Any

from .base_plot import BasePlot


class ResidualsPlot(BasePlot):
    """
    Residuals plot for analyzing fit quality in electrochemical data.

    Creates a plot of residuals vs. frequency with logarithmic x-axis
    for evaluating the quality of impedance data fitting.
    """

    def __init__(self, usetex: Optional[bool] = None, **kwargs: Any) -> None:
        """
        Initialize a residuals plot.

        Args:
            usetex (Optional[bool]): Whether to use LaTeX rendering.
                                   If None, uses global config setting.
            **kwargs: Additional arguments passed to plt.subplots().
        """
        super().__init__(usetex=usetex, **kwargs)

    def _configure_axes(self) -> None:
        """Configure axes specifically for residuals plots."""
        # Set x-axis to log scale
        self.ax.set_xscale("log")

        # Set the x-axis ticks to be evenly spaced in log space
        self.ax.xaxis.set_major_locator(LogLocator(base=10.0))
        self.ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs="auto", numticks=10))

        self.ax.tick_params(
            axis="both",
            which="both",
            bottom=True,
            left=True,
        )

        # Set labels - LaTeX preference already set globally
        self.ax.set_xlabel(r"$f$ / $\mathrm{Hz}$")
        self.ax.set_ylabel(r"$\mathrm{Residual}$ / $\Omega$")

    def add_plot(
        self,
        f: np.ndarray,
        residuals: np.ndarray,
        label: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Add residual data to the plot.

        Args:
            f (np.ndarray): Frequency array.
            residuals (np.ndarray): Residual values array.
            label (Optional[str]): Label for the plot. Defaults to "residual".
            **kwargs: Additional arguments passed to matplotlib plot function.
        """
        if label is None:
            label = "residuals"
        # Use direct matplotlib call
        self.ax.plot(f, residuals, label=label, **kwargs)
