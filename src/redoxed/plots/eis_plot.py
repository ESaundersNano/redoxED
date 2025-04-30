import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, LogLocator
from pathlib import Path
import warnings

from redoxed.impedance import EISData


class NyquistPlot:
    def __init__(self, **kwargs):
        """ """

        # Load the custom style
        self._load_style()

        # Extract figure size if provided
        # figsize = kwargs.pop("figsize", None)  # Uses default from .mplstyle if None
        fig, ax = plt.subplots(1, 1, **kwargs)

        # Ensure ratio equal
        ax.set_aspect("equal", adjustable="datalim")

        ax.tick_params(
            axis="both",
            which="both",
            bottom=True,
            left=True,
        )

        # Set labels
        ax.set_xlabel(r"$Z'$ / $\Omega$")
        ax.set_ylabel(r"$-Z''$ / $\Omega$")

        # Close the plot to avoid spamming output
        plt.close(fig)

        # Store the figure and axis as attributes
        self.fig = fig
        self.ax = ax

    def _load_style(self, style_path=None):
        """
        Loads the custom Matplotlib style from the .mplstyle file.
        """

        # Apply Seaborn theme as foundation style
        sns.set_theme(
            context="paper", style="whitegrid", palette="muted", font="Times New Roman"
        )

        # Apply custom style if provided
        if style_path is None:
            # Use the default style path if not provided
            style_path = Path(__file__).parent / "redoxed_plot.mplstyle"
        else:
            print(f"Loading style from: {style_path}")
        # Check if the style file exists and apply it on top of Seaborn theme
        if os.path.exists(style_path):
            plt.style.use(style_path)
        else:
            warnings.warn(
                f"Style file '{style_path}' not found. Using default Matplotlib style.",
                UserWarning,
            )

    def add_plot(self, EISData_object: EISData, label: str = None, **kwargs):
        """ """
        if label == None:
            label = EISData_object.label
        self.ax.plot(EISData_object.Z_re, -EISData_object.Z_im, label=label, **kwargs)

    def add_major_ticks(self, major_tick_spacing=None, **kwargs):
        self.ax.xaxis.set_major_locator(MultipleLocator(major_tick_spacing))
        self.ax.yaxis.set_major_locator(MultipleLocator(major_tick_spacing))

    def add_minor_ticks(self, minor_tick_number=None, **kwargs):
        self.ax.xaxis.set_minor_locator(AutoMinorLocator(minor_tick_number))
        self.ax.yaxis.set_minor_locator(AutoMinorLocator(minor_tick_number))

        # Extract optional arguments
        # labelled_frequencies = kwargs.pop("labelled_frequencies", None)
        # freq_label_offset = kwargs.pop("freq_label_offset", (0.05, 0.05))

        #     # Add labels at specified frequencies
        #     if labelled_frequencies is not None:
        #         for freq in labelled_frequencies:
        #             index = np.argmin(np.abs(EISData.f - freq))
        #             x, y = EISData.Z_re[index], -EISData.Z_im[index]
        #             offset_x, offset_y = (
        #                 x + freq_label_offset[0],
        #                 y + freq_label_offset[1],
        #             )
        #             freq_exp = np.log10(freq)
        #             annotation_text = r"$10^{" + f"{freq_exp:.0f}" + "}$"
        #             ax.annotate(
        #                 annotation_text,
        #                 xy=(x, y),
        #                 xytext=(offset_x, offset_y),
        #                 arrowprops=dict(
        #                     arrowstyle="->", connectionstyle="arc", color="black"
        #                 ),
        #             )

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

    # def plot_residuals(self, **kwargs):
    #     # Extract figure size if provided
    #     figsize = kwargs.pop(
    #         "figsize", (3.25, 3)
    #     )  # defaults to 1 column width. Pass 6.5 width if want 2 column.
    #     grid = kwargs.pop(
    #         "grid", False
    #     )  # look for grid and return default False if not set
    #     legend = kwargs.pop("legend", True)

    #     # Create the figure and axis
    #     fig, ax = plt.subplots(1, 1, figsize=figsize)

    #     for i, EISData in enumerate(self.EISData_list):

    #         frequencies = EISData.f
    #         res_real = EISData.residuals_real
    #         res_imag = EISData.residuals_imag

    #         ax.semilogx(
    #             frequencies, res_real, label="$\Delta_{Re}$", linestyle="-", marker="."
    #         )  # plot res_real
    #         ax.semilogx(
    #             frequencies, res_imag, label="$\Delta_{Im}$", linestyle="-", marker="."
    #         )  # plot res_imag

    #     # Set labels and title for the second subplot
    #     ax.set_xlabel(r"$f$ / $\mathrm{Hz}$", fontsize=10)
    #     ax.set_ylabel(r"$Residual$ / $\%$", fontsize=10)
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

    #     ax.grid(grid)
    #     if legend:
    #         ax.legend(fontsize=9, frameon=False)
    #     # ax.legend(fontsize=9, frameon=False, bbox_to_anchor=(0.8, 0.88), loc="upper center") # can move around manually if suppress legend plotting and then do manually with a line like this

    #     # close the plot so it doesn't spam in jupyter notebook
    #     plt.close(fig)
    #     return fig, ax
