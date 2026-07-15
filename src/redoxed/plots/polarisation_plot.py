"""
Polarisation curve plotting classes.

This module provides specialized plotting classes for visualizing electrochemical
polarisation data, including current density vs. potential plots.
"""

import matplotlib.pyplot as plt
from typing import Any

from redoxed.dc import PolarisationData
from .base_plot import BasePlot


class PolarisationPlot(BasePlot):
    """
    Polarisation curve plot for electrochemical data.

    Creates a plot of voltage vs. current density for polarisation measurements.
    """

    def __init__(self, usetex: bool | None = None, **kwargs: Any) -> None:
        """
        Initialize a polarisation plot.

        Parameters:
            usetex (bool | None): Whether to use LaTeX rendering.
                If None, uses global config setting. Defaults to None.
            **kwargs: Additional arguments passed to BasePlot.
        """
        super().__init__(usetex=usetex, **kwargs)

    def _configure_axes(self) -> None:
        """Configure axes specifically for polarisation plots."""
        self.ax.tick_params(
            axis="both",
            which="both",
            bottom=True,
            left=True,
        )

        # Set labels - LaTeX preference already set globally
        self.ax.set_xlabel(r"$j$ / $\mathrm{mA \, cm^{-2}}$")
        self.ax.set_ylabel(r"$V$ / $\mathrm{V}$")

    def add_plot(
        self,
        polarisation_data: PolarisationData,
        label: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Add polarisation data to the plot.

        Parameters:
            polarisation_data (PolarisationData): The polarisation data object containing
                voltage and current density data to plot.
            label (str | None): Label for the plot series. If None, uses the data's label.
                Defaults to None.
            **kwargs: Additional arguments passed to matplotlib plot function.
        """
        if label is None:
            label = polarisation_data.label
        # Use direct matplotlib call
        self.ax.plot(polarisation_data.j, polarisation_data.V, label=label, **kwargs)
