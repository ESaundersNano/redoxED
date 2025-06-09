import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, LogLocator
from pathlib import Path
import warnings

# from redoxed.impedance import DRTData


class ResidualsPlot:
    def __init__(self, **kwargs):
        """ """

        # Load the custom style
        self._load_style()

        # Extract figure size if provided
        # figsize = kwargs.pop("figsize", None)  # Uses default from .mplstyle if None
        fig, ax = plt.subplots(1, 1, **kwargs)

        # Set x-axis to log scale
        ax.set_xscale("log")
        # Set the x-axis ticks to be evenly spaced in log space
        ax.xaxis.set_major_locator(
            LogLocator(base=10.0)
        )  # Major ticks at log intervals
        ax.xaxis.set_minor_locator(
            LogLocator(base=10.0, subs="auto", numticks=10)
        )  # Minor ticks at log intervals

        ax.tick_params(
            axis="both",
            which="both",
            bottom=True,
            left=True,
        )

        # Set labels
        ax.set_xlabel(r"$f$ / $\mathrm{Hz}$")
        ax.set_ylabel(r"$residual$ / $\Omega$")

        # Close the plot to avoid spamming output
        plt.close(fig)

        # Store the figure and axis as attributes
        self.fig = fig
        self.ax = ax
        self.ax_top = None

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

    def add_plot(
        self, f: np.ndarray, residual: np.ndarray, label: str = None, **kwargs
    ):
        """ """
        if label == None:
            label = "residual"
        self.ax.plot(f, residual, label=label, **kwargs)

    def add_major_ticks(self, major_tick_spacing=None, **kwargs):
        self.ax.yaxis.set_major_locator(MultipleLocator(major_tick_spacing))

    def add_minor_ticks(self, minor_tick_number=None, **kwargs):
        self.ax.yaxis.set_minor_locator(AutoMinorLocator(minor_tick_number))

        #         major_tick_spacing_y = kwargs.pop(
        #     "major_tick_spacing_y", None
        # )  # look for tick spacing but leave as default if none found
        # if major_tick_spacing_y is not None:
        #     ax.yaxis.set_major_locator(MultipleLocator(major_tick_spacing_y))
        # # minor_tick_number = kwargs.pop('minor_tick_number', None) # look for tick spacing but leave as default if none found
        # # if minor_tick_number is not None:
        # #     ax.xaxis.set_minor_locator(AutoMinorLocator(minor_tick_number))
        # #     ax.yaxis.set_minor_locator(AutoMinorLocator(minor_tick_number))
