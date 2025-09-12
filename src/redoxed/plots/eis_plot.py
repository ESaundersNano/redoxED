"""
Electrochemical Impedance Spectroscopy (EIS) plotting classes.

This module provides specialized plotting classes for visualizing EIS data,
including Nyquist plots and other EIS-specific visualizations.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Any

from redoxed.impedance import EISData
from .base_plot import BasePlot


class NyquistPlot(BasePlot):
    """
    Nyquist plot for electrochemical impedance spectroscopy data.

    Creates a plot of -Z'' vs Z' with equal aspect ratio and proper formatting
    for EIS data visualization.
    """

    def __init__(self, usetex: bool | None = None, **kwargs: Any) -> None:
        """
        Initialize a Nyquist plot.

        Args:
            usetex (bool | None): Whether to use LaTeX rendering.
                                   If None, uses global config setting.
            **kwargs: Additional arguments passed to plt.subplots().
        """
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
        self.ax.set_xlabel(r"$Z^{\prime}$ / $\Omega$")
        self.ax.set_ylabel(r"$Z^{\prime\prime}$ / $\Omega$")

    def add_plot(
        self, eis_data: EISData, label: str | None = None, **kwargs: Any
    ) -> None:
        """
        Add EIS data to the Nyquist plot.

        Args:
            eis_data (EISData): The EIS data object to plot.
            label (str | None): Label for the plot. If None, uses the data's label.
            **kwargs: Additional arguments passed to matplotlib plot function.
        """
        if label is None:
            label = eis_data.label
        # Simple matplotlib call - LaTeX preference already set globally
        self.ax.plot(eis_data.Z_re, -eis_data.Z_im, label=label, **kwargs)


# def plot_bode(self, **kwargs):

#     """
#     Plots the Bode plot for the EIS data.

#     Args:
#         **kwargs: Additional keyword arguments for customization.
#     """
#     # Extract figure size if provided
#     figsize = kwargs.pop("figsize", None)  # Use default from .mplstyle if None
#     fig, ax = plt.subplots(1, 1, figsize=figsize)

#     for i, EISData in enumerate(self.EISData_list):
#         if EISData.dataset_type == "measured":
#             ax.plot(
#                 EISData.f,
#                 EISData.Z_mag,
#                 label=EISData.label + " Magnitude",
#                 color=EISData.colour,
#                 marker="o",
#                 fillstyle="none",
#                 linestyle="none",
#                 **kwargs,
#             )
#         elif EISData.dataset_type == "fitted":
#             ax.plot(
#                 EISData.f,
#                 EISData.Z_mag,
#                 label=EISData.label + " Magnitude",
#                 color=EISData.colour,
#                 marker="none",
#                 linestyle="-",
#                 **kwargs,
#             )

#     ax.set_yscale("log")
#     ax.set_xscale("log")

#     # Set labels
#     ax.set_xlabel(r"$f$ / $\mathrm{Hz}$")
#     ax.set_ylabel(r"$\left| Z \right|$ / $\Omega$")

#     # Add legend if enabled
#     if kwargs.pop("legend", True):
#         ax.legend()

#     # Close the plot to avoid spamming in Jupyter Notebook
#     plt.close(fig)

#     return fig, ax

# def plot_bode(self, **kwargs):
#     # Extract figure size if provided
#     figsize = kwargs.pop(
#         "figsize", (3.25, 3)
#     )  # defaults to 1 column width. Pass 6.5 width if want 2 column.
#     # Create the figure and axis
#     fig, ax = plt.subplots(1, 1, figsize=figsize)

#     major_tick_spacing_x = kwargs.pop(
#         "major_tick_spacing_x", None
#     )  # look for tick spacing but leave as default if none found
#     major_tick_spacing_y = kwargs.pop(
#         "major_tick_spacing_y", None
#     )  # look for tick spacing but leave as default if none found
#     if major_tick_spacing_x is not None:
#         ax.xaxis.set_major_locator(MultipleLocator(major_tick_spacing_x))
#     if major_tick_spacing_y is not None:
#         ax.yaxis.set_major_locator(MultipleLocator(major_tick_spacing_y))
#     minor_tick_number = kwargs.pop(
#         "minor_tick_number", None
#     )  # look for tick spacing but leave as default if none found
#     if minor_tick_number is not None:
#         ax.xaxis.set_minor_locator(AutoMinorLocator(minor_tick_number))
#         ax.yaxis.set_minor_locator(AutoMinorLocator(minor_tick_number))
#     grid = kwargs.pop(
#         "grid", False
#     )  # look for grid and return default False if not set
#     legend = kwargs.pop("legend", True)

#     for i, EISData in enumerate(self.EISData_list):
#         if EISData.dataset_type == "measured":
#             ax.plot(
#                 EISData.f,
#                 EISData.Z_mag,
#                 label=EISData.label + " Magnitude",
#                 color=EISData.colour,
#                 marker="o",
#                 fillstyle="none",
#                 linestyle="none",
#                 **kwargs,
#             )
#         elif EISData.dataset_type == "fitted":
#             ax.plot(
#                 EISData.f,
#                 EISData.Z_mag,
#                 label=EISData.label + " Magnitude",
#                 color=EISData.colour,
#                 marker="none",
#                 linestyle="-",
#                 **kwargs,
#             )
#     ax.set_yscale("log")
#     ax.set_xscale("log")
#     # Set labels and title for the second subplot
#     ax.set_xlabel(r"$f$ / $\mathrm{Hz}$", fontsize=10)
#     ax.set_ylabel(r"$\left| Z \right|$ / $\Omega$", fontsize=10)
#     ax.tick_params(
#         axis="both",
#         which="major",
#         labelsize=9,
#         direction="out",
#         length=5,
#         width=1,
#         bottom=True,
#         left=True,
#     )
#     ax.tick_params(
#         axis="both",
#         which="minor",
#         direction="out",
#         length=3,
#         width=1,
#         bottom=True,
#         left=True,
#     )

#     # phase plotting
#     ax_phase = ax.twinx()
#     # ax_phase.set_yscale('linear')
#     for i, EISData in enumerate(self.EISData_list):
#         if EISData.dataset_type == "measured":
#             ax_phase.plot(
#                 EISData.f,
#                 EISData.phases,
#                 label=EISData.label + " Phase",
#                 color=EISData.colour,
#                 marker="^",
#                 fillstyle="none",
#                 linestyle="none",
#                 **kwargs,
#             )
#             # add to legend
#             ax.plot(
#                 np.nan,
#                 np.nan,
#                 marker="^",
#                 linestyle="none",
#                 label=EISData.label + " Phase",
#                 color=EISData.colour,
#             )  # make so appears in legend
#         elif EISData.dataset_type == "fitted":
#             ax_phase.plot(
#                 EISData.f,
#                 EISData.phases,
#                 label=EISData.label + " Phase",
#                 color=EISData.colour,
#                 marker="none",
#                 linestyle="--",
#                 **kwargs,
#             )
#             # add to legend
#             ax.plot(
#                 np.nan,
#                 np.nan,
#                 marker="none",
#                 linestyle="--",
#                 label=EISData.label + " Phase",
#                 color=EISData.colour,
#             )  # make so appears in legend
#     # ax_phase.set_xscale('log')
#     ax_phase.set_ylabel(
#         r"$Phase$ / $\mathrm{Degrees}$"
#     )  # ('Current density (mA/cm2)')
#     ax_phase.tick_params(
#         axis="both",
#         which="major",
#         labelsize=9,
#         direction="out",
#         length=5,
#         width=1,
#         right=True,
#     ),
#     ax_phase.grid(False)
#     # ax_phase.set_ylim(0, None)

#     if legend:
#         ax.legend(fontsize=9, frameon=False)

#     # close the plot so it doesn't spam in jupyter notebook
#     plt.close(fig)

#     return fig, ax
