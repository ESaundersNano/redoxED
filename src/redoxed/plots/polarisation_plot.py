import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, LogLocator
from pathlib import Path
import warnings

from redoxed.dc import PolarisationData


class PolarisationPlot:
    def __init__(self, **kwargs):
        """ """

        # Load the custom style
        self._load_style()

        # Extract figure size if provided
        # figsize = kwargs.pop("figsize", None)  # Uses default from .mplstyle if None
        fig, ax = plt.subplots(1, 1, **kwargs)

        ax.tick_params(
            axis="both",
            which="both",
            bottom=True,
            left=True,
        )

        # Set labels
        ax.set_xlabel(r"$j$ / $mA \ cm^{-2}$")
        ax.set_ylabel(r"$V$ / $V$")

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

    def add_plot(
        self, PolarisationData_object: PolarisationData, label: str = None, **kwargs
    ):
        """ """
        if label == None:
            label = PolarisationData_object.label
        self.ax.plot(
            PolarisationData_object.j, PolarisationData_object.V, label=label, **kwargs
        )
