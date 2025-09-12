"""
Distribution of Relaxation Times (DRT) plotting classes.

This module provides specialized plotting classes for visualizing DRT data,
including time constant distributions and frequency-domain representations.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator
from typing import Any

from redoxed.impedance import DRTData
from .base_plot import BasePlot


class DRTPlot(BasePlot):
    """
    Distribution of Relaxation Times (DRT) plot for impedance data.

    Creates a plot of gamma vs tau with logarithmic x-axis and optional
    characteristic frequency axis on top.
    """

    def __init__(
        self, fc_axis: bool = True, usetex: bool | None = None, **kwargs: Any
    ) -> None:
        """
        Initialize a DRT plot.

        Args:
            fc_axis (bool): Whether to add characteristic frequency axis. Defaults to True.
            usetex (bool | None): Whether to use LaTeX rendering.
                                   If None, uses global config setting.
            **kwargs: Additional arguments passed to plt.subplots().
        """
        super().__init__(usetex=usetex, **kwargs)
        self.ax_top: plt.Axes | None = None
        if fc_axis:
            self.add_fc_axis()

    def _configure_axes(self) -> None:
        """Configure axes specifically for DRT plots."""
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
        self.ax.set_xlabel(r"$\tau$ / $\mathrm{s}$")
        self.ax.set_ylabel(r"$\gamma$ / $\Omega$")

    def add_fc_axis(self) -> None:
        """
        Add characteristic frequency axis on top of the plot.

        This creates a twin x-axis showing the characteristic frequency
        fc = 1/(2π·τ) corresponding to the time constants.
        """
        # Clear the existing ax_top if it exists
        if self.ax_top is not None:
            self.ax_top.remove()
            self.ax_top = None

        ax_top = self.ax.twiny()
        lower_tau, upper_tau = self.ax.get_xlim()
        ax_top_lims = (1 / (2 * np.pi * lower_tau), 1 / (2 * np.pi * upper_tau))

        ax_top.set_xlim(ax_top_lims)
        ax_top.set_xscale("log")
        ax_top.xaxis.set_major_locator(LogLocator(base=10.0))
        ax_top.xaxis.set_minor_locator(LogLocator(base=10.0, subs="auto", numticks=10))

        ax_top.tick_params(axis="x", which="both", top=True)
        # Use a simple method call - the context will be handled when needed
        ax_top.set_xlabel(r"$f_{c}$ / $\mathrm{Hz}$")
        ax_top.grid(False)
        self.ax_top = ax_top

    def add_plot(
        self, drt_data: DRTData, label: str | None = None, **kwargs: Any
    ) -> None:
        """
        Add DRT data to the plot.

        Args:
            drt_data (DRTData): The DRT data object to plot.
            label (str | None): Label for the plot. If None, uses the data's label.
            **kwargs: Additional arguments passed to matplotlib plot function.
        """
        if label is None:
            label = drt_data.label
        # Use direct matplotlib call
        self.ax.plot(drt_data.tau, drt_data.gamma, label=label, **kwargs)
