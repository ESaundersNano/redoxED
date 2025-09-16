"""
Cycling data plotting classes.

This module provides specialized plotting classes for visualizing cycling data,
including Nyquist plots and other cycling-specific visualizations.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Any

from redoxed.dc import CyclingData
from .base_plot import BasePlot


class EfficiencyPlot(BasePlot):
    """ """

    def __init__(self, usetex: bool | None = None, **kwargs: Any) -> None:
        """ """
        super().__init__(usetex=usetex, **kwargs)

    def _configure_axes(self) -> None:
        """Configure axes specifically for Nyquist plots."""
        # Ensure equal aspect ratio for proper circle representation
        self.ax.set_aspect("equal", adjustable="datalim")

        self.ax.tick_params(
            axis="both",
            which="both",
            bottom=True,
            left=True,
        )

        # Set labels with proper LaTeX formatting - LaTeX preference already set globally
        self.ax.set_xlabel(r"$Cycle \ No.$")
        self.ax.set_ylabel(r"$Efficiency$ / $\%$")

        self.ax2 = None

    def add_plot(
        self,
        cycling_data: CyclingData,
        label: str | None = None,
        plot_CE: bool = False,
        plot_VE: bool = False,
        plot_EE: bool = True,
        **kwargs: Any,
    ) -> None:
        """ """
        if label is None:
            label = cycling_data.label

        if plot_CE == True:
            self.ax.plot(
                cycling_data.cycle_data["cycle_number"],
                cycling_data.cycle_data["CE"],
                label=label + ": CE",
                marker="s",
                fillstyle="none",
                **kwargs,
            )
        if plot_VE == True:
            self.ax.plot(
                cycling_data.cycle_data["cycle_number"],
                cycling_data.cycle_data["VE"],
                label=label + ": VE",
                marker="v",
                fillstyle="none",
                **kwargs,
            )
        if plot_EE == True:
            self.ax.plot(
                cycling_data.cycle_data["cycle_number"],
                cycling_data.cycle_data["EE"],
                label=label + ": EE",
                marker="o",
                fillstyle="full",
                **kwargs,
            )

    def add_j(
        self, x: np.ndarray, j: np.ndarray, j_in_legend: bool = True, **kwargs: Any
    ) -> None:
        if self.ax2 is None:
            self.ax2 = self.ax.twinx()
            self.ax2.set_ylabel(r"$j$ / $\mathrm{mA \, cm^{-2}}$")
            self.ax2.grid(False)
        self.ax2.plot(x, j, **kwargs)
        if j_in_legend:
            self.ax.plot(np.nan, np.nan, **kwargs)  # make so appears in legend
