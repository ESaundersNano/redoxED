"""
Polarisation curve plotting classes.

This module provides specialized plotting classes for visualizing electrochemical
polarisation data, including current density vs. potential plots.
"""

import matplotlib.pyplot as plt
from typing import Optional, Any

from redoxed.dc import PolarisationData
from .base_plot import BasePlot


class PolarisationPlot(BasePlot):
    """
    Polarisation curve plot for electrochemical data.

    Creates a plot of voltage vs. current density for polarisation measurements.
    """

    def __init__(self, usetex: Optional[bool] = None, **kwargs: Any) -> None:
        """
        Initialize a polarisation plot.

        Args:
            usetex (Optional[bool]): Whether to use LaTeX rendering.
                                   If None, uses global config setting.
            **kwargs: Additional arguments passed to plt.subplots().
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
        label: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Add polarisation data to the plot.

        Args:
            polarisation_data (PolarisationData): The polarisation data object to plot.
            label (Optional[str]): Label for the plot. If None, uses the data's label.
            **kwargs: Additional arguments passed to matplotlib plot function.
        """
        if label is None:
            label = polarisation_data.label
        # Use direct matplotlib call
        self.ax.plot(polarisation_data.j, polarisation_data.V, label=label, **kwargs)
