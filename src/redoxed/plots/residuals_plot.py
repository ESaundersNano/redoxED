"""
Residuals plotting classes.

This module provides specialized plotting classes for visualizing fitting residuals
from electrochemical impedance spectroscopy data analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator

from .base_plot import BasePlot

from redoxed.impedance import ResidualsData


class ResidualsPlot(BasePlot):
    """
    Residuals plot for analyzing fit quality in electrochemical data.

    Creates a plot of residuals vs. frequency with logarithmic x-axis
    for evaluating the quality of impedance data fitting.
    """

    def __init__(
        self, usetex: bool | None = None, mode: str = "absolute", **kwargs: object
    ) -> None:
        """
        Initialize a residuals plot.

        Args:
            usetex (bool | None): Whether to use LaTeX rendering.
            mode (str): 'absolute' for Ohms, 'relative' for percent residuals.
            **kwargs: Additional arguments passed to plt.subplots().
        """
        self.mode = mode
        super().__init__(usetex=usetex, **kwargs)
        if mode not in ("absolute", "relative"):
            raise ValueError("mode must be 'absolute' or 'relative'")

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
        if self.mode == "absolute":
            self.ax.set_ylabel(r"$\mathrm{Residual}$ / $\Omega$")
        else:
            self.ax.set_ylabel(r"$\mathrm{Residual}$ / $\%$")

    def add_plot(
        self,
        residuals_data: ResidualsData,
        label: str | None = None,
        **kwargs: object,
    ) -> None:
        """
        Add residual data to the plot from a ResidualsData object.

        Args:
            residuals_data (ResidualsData): Container with frequency and residuals.
            label (str | None): Label for the plot. Defaults to 'residuals'.
            **kwargs: Additional arguments passed to matplotlib plot function.
        """
        if not isinstance(residuals_data, ResidualsData):
            raise TypeError("add_plot expects a ResidualsData object as input.")
        if label is None:
            label = getattr(residuals_data, "label", "residuals")
        f = residuals_data.f
        if self.mode == "absolute":
            y_re = residuals_data.residuals_re
            y_im = residuals_data.residuals_im
        else:
            y_re = (
                residuals_data.residuals_re_rel
                if residuals_data.residuals_re_rel is not None
                else None
            )
            y_im = (
                residuals_data.residuals_im_rel
                if residuals_data.residuals_im_rel is not None
                else None
            )
        if y_re is not None:
            self.ax.plot(f, y_re, label=label + " (Re)", marker="s", **kwargs)
        if y_im is not None:
            self.ax.plot(f, y_im, label=label + " (Im)", marker="^", **kwargs)
