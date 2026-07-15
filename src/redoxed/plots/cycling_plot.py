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
    """
    Efficiency trend plot for electrochemical cycling data.

    Plots Coulombic Efficiency (CE), Voltage Efficiency (VE), and Energy Efficiency (EE)
    against cycle number. Optionally displays secondary metrics on a twinned y-axis.

    Attributes:
        ax2 (plt.Axes | None): Secondary y-axis for additional metrics like current density.
    """

    def __init__(self, usetex: bool | None = None, **kwargs: Any) -> None:
        """
        Initialize an efficiency plot.

        Parameters:
            usetex (bool | None): Whether to use LaTeX rendering.
                If None, uses global config setting. Defaults to None.
            **kwargs: Additional arguments passed to BasePlot.
        """
        super().__init__(usetex=usetex, **kwargs)

    def _configure_axes(self) -> None:
        """Configure axes specifically for efficiency plots."""

        self.ax.tick_params(
            axis="both",
            which="both",
            bottom=True,
            left=True,
        )

        # Set labels with proper LaTeX formatting - LaTeX preference already set globally
        self.ax.set_xlabel(r"Cycle No.")
        # self.ax.set_ylabel(r"Efficiency / \%")
        self.ax.set_ylabel(
            r"Efficiency / \%" if self.usetex else "Efficiency / %"
        )  # render % symbol properly depending on usetex

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
        """
        Add cycling data efficiency curves to the plot.

        Parameters:
            cycling_data (CyclingData): The cycling data object containing efficiency metrics.
            label (str | None): Label for the plot series. If None, uses the data's label. Defaults to None.
            plot_CE (bool): Whether to plot Coulombic Efficiency (%). Defaults to False.
            plot_VE (bool): Whether to plot Voltage Efficiency (%). Defaults to False.
            plot_EE (bool): Whether to plot Energy Efficiency (%). Defaults to True.
            **kwargs: Additional arguments passed to matplotlib plot function.
        """
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
        """
        Add current density data on a secondary (twinned) y-axis.

        Parameters:
            x (np.ndarray): X-axis data (typically cycle number).
            j (np.ndarray): Current density data in mA/cm² (milliamps per square centimeter).
            j_in_legend (bool): Whether to include current density in the legend. Defaults to True.
            **kwargs: Additional arguments passed to matplotlib plot function.
        """
        if self.ax2 is None:
            self.ax2 = self.ax.twinx()
            self.ax2.set_ylabel(r"$j$ / $\mathrm{mA \, cm^{-2}}$")
            self.ax2.grid(False)
        self.ax2.plot(x, j, **kwargs)
        if j_in_legend:
            self.ax.plot(np.nan, np.nan, **kwargs)  # make so appears in legend

    def add_Q_discharge(
        self,
        x: np.ndarray,
        Q_discharge: np.ndarray,
        Q_in_legend: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Add discharge capacity data on a secondary (twinned) y-axis.

        Parameters:
            x (np.ndarray): X-axis data (typically cycle number).
            Q_discharge (np.ndarray): Discharge capacity data in mAh (milliamp-hours).
            Q_in_legend (bool): Whether to include capacity in the legend. Defaults to True.
            **kwargs: Additional arguments passed to matplotlib plot function.
        """
        if self.ax2 is None:
            self.ax2 = self.ax.twinx()
            self.ax2.set_ylabel(r"$Q$ / $\mathrm{mAh}$")
            self.ax2.grid(False)
        self.ax2.plot(x, Q_discharge, **kwargs)
        if Q_in_legend:
            self.ax.plot(np.nan, np.nan, **kwargs)  # make so appears in legend
