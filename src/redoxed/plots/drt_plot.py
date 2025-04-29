import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, LogLocator
from pathlib import Path
import warnings


class DRTPlot:
    def __init__(self, DRT):
        # Check if DRT is a single instance, list, or numpy array
        if isinstance(DRT, (list, np.ndarray)):  # check if list or np.array
            self.DRT_list = np.array(DRT)  # convert list to or keep as np.array
        else:
            self.DRT_list = np.array([DRT])  # make into np.array

        sns.set_theme(
            context="paper", style="whitegrid", palette="muted", font="Times New Roman"
        )  # , font_scale=1) # should maybe make this optional, at least for color

    def plot_DRT(self, **kwargs):

        ### sometimes minor ticks won't appear if range of frequencies is too large somehow.
        # ## Like I think it depends on if there is DRT data there or something.
        # Extract figure size if provided
        figsize = kwargs.pop(
            "figsize", (3.25, 3)
        )  # defaults to 1 column width. Pass 6.5 width if want 2 column.
        # Create the figure and axis
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Set x-axis to log scale
        ax.set_xscale("log")

        # Set the x-axis ticks to be evenly spaced in log space
        ax.xaxis.set_major_locator(
            LogLocator(base=10.0)
        )  # Major ticks at log intervals
        ax.xaxis.set_minor_locator(
            LogLocator(base=10.0, subs="auto", numticks=10)
        )  # Minor ticks at log intervals

        major_tick_spacing_y = kwargs.pop(
            "major_tick_spacing_y", None
        )  # look for tick spacing but leave as default if none found
        if major_tick_spacing_y is not None:
            ax.yaxis.set_major_locator(MultipleLocator(major_tick_spacing_y))
        # minor_tick_number = kwargs.pop('minor_tick_number', None) # look for tick spacing but leave as default if none found
        # if minor_tick_number is not None:
        #     ax.xaxis.set_minor_locator(AutoMinorLocator(minor_tick_number))
        #     ax.yaxis.set_minor_locator(AutoMinorLocator(minor_tick_number))

        grid = kwargs.pop(
            "grid", False
        )  # look for grid and return default False if not set
        legend = kwargs.pop("legend", True)

        fc_axis = kwargs.pop("fc_axis", True)

        for DRT in self.DRT_list:
            ax.plot(DRT.tau, DRT.gamma, label=DRT.label, color=DRT.colour, **kwargs)

        # ax.set_ylim([0.01, None]) # can always edit afterwards like most settings with something like

        # Set labels and title for the second subplot
        ax.set_xlabel(r"$\tau$ / $\mathrm{s}$", fontsize=10)
        ax.set_ylabel(r"$\gamma$ / $\Omega$", fontsize=10)
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=9,
            direction="out",
            length=5,
            width=1,
            bottom=True,
            left=True,
        )
        ax.tick_params(
            axis="both",
            which="minor",
            direction="out",
            length=3,
            width=1,
            bottom=True,
            left=True,
        )

        ax.grid(grid)
        if legend:
            ax.legend(fontsize=9, frameon=False)
        # ax.legend(fontsize=9, frameon=False, bbox_to_anchor=(0.8, 0.88), loc="upper center") # can move around manually if suppress legend plotting and then do manually with a line like this

        if fc_axis:
            ax_top = ax.twiny()
            lower_tau, upper_tau = ax.get_xlim()
            ax_top_lims = (1 / (2 * np.pi * lower_tau), 1 / (2 * np.pi * upper_tau))
            # ax_top_lims = (1/(lower_tau), 1/(upper_tau))
            ax_top.set_xlim(ax_top_lims)
            ax_top.set_xscale("log")
            ax_top.xaxis.set_major_locator(
                LogLocator(base=10.0)
            )  # Major ticks at log intervals
            ax_top.xaxis.set_minor_locator(
                LogLocator(base=10.0, subs="auto", numticks=10)
            )  # Minor ticks at log intervals
            ax_top.tick_params(
                axis="both",
                which="major",
                labelsize=9,
                direction="out",
                length=5,
                width=1,
                top=True,
            )
            ax_top.tick_params(
                axis="both", which="minor", direction="out", length=3, width=1, top=True
            )
            ax_top.set_xlabel(r"$f_{c}$ / $\mathrm{Hz}$", fontsize=10)
            ax_top.grid(False)

        # close the plot so it doesn't spam in jupyter notebook
        plt.close(fig)

        return fig, ax
