"""
Base plotting class for electrochemical data visualization.

This module provides the abstract BasePlot class that serves as the foundation
for all specialized plotting classes in the redoxED package.
"""

from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import subprocess
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, LogLocator
from pathlib import Path
import warnings
from typing import Tuple, Any
import copy
from .. import config


class BasePlot(ABC):
    """
    Abstract base class for all electrochemical plotting classes.

    This class provides common functionality for style loading, figure creation,
    and tick management that is shared across all plotting types.

    Attributes:
        fig (plt.Figure): The matplotlib figure object.
        ax (plt.Axes): The primary matplotlib axes object.
    """

    def __init__(self, usetex: bool | None = None, **kwargs: Any) -> None:
        """
        Initialize the base plot with common setup.

        Configures LaTeX rendering preference, loads custom style, creates figure and axes,
        and applies plot-specific configuration through the abstract _configure_axes() method.

        Parameters:
            usetex (bool | None): Whether to use LaTeX rendering. If None, uses global config
                setting. Defaults to None.
            **kwargs: Additional arguments passed to matplotlib plt.subplots().
        """
        # Use global config if not specified
        if usetex is None:
            usetex = config.USE_LATEX

        # Store usetex preference
        self.usetex = usetex

        # Set LaTeX preference once at the beginning
        self._setup_latex(usetex)

        # Load the custom style
        self._load_style(usetex=usetex)

        # Create figure and axis
        self.fig: plt.Figure
        self.ax: plt.Axes
        self.fig, self.ax = plt.subplots(1, 1, **kwargs)

        # Apply plot-specific configuration
        self._configure_axes()

        # Close the plot to avoid spamming output
        plt.close(self.fig)

    def _setup_latex(self, usetex: bool) -> None:
        """
        Setup LaTeX rendering preference globally.

        Attempts to enable LaTeX if requested. Falls back to matplotlib's mathtext renderer
        with appropriate warnings if LaTeX is not available on the system.

        Parameters:
            usetex (bool): Whether to enable LaTeX rendering.
        """
        if usetex:
            # Try to enable LaTeX
            try:
                subprocess.run(
                    ["latex", "--version"],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

                plt.rcParams.update(
                    {
                        "text.usetex": True,
                        "font.family": "serif",
                        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
                        "text.latex.preamble": r"\usepackage{newtxtext,newtxmath}",
                    }
                )
            except Exception:
                # LaTeX not available, warn and fall back
                plt.rcParams.update(
                    {
                        "text.usetex": False,
                        "font.family": "serif",
                        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
                    }
                )
                warnings.warn("LaTeX not available. Using mathtext.", UserWarning)
        else:
            # Explicitly disable LaTeX
            plt.rcParams.update(
                {
                    "text.usetex": False,
                    "font.family": "sans-serif",
                    "font.sans-serif": [
                        "Arial",
                        "DejaVu Sans",
                        "Liberation Sans",
                        "sans-serif",
                    ],
                }
            )

    def _load_style(self, usetex: bool = False, style_path: str | None = None) -> None:
        """
        Load and apply custom matplotlib style from the .mplstyle file.

        Applies seaborn theme as foundation, then loads custom redoxed_plot.mplstyle file
        for consistent visualization across all plot types.

        Parameters:
            usetex (bool): Whether LaTeX rendering is enabled (affects font selection).
                Defaults to False.
            style_path (str | None): Custom path to style file. If None, uses default redoxed_plot.mplstyle.
                Defaults to None.
        """
        font = "Times New Roman" if usetex else "Arial"

        # Apply Seaborn theme as foundation style
        sns.set_theme(context="paper", style="whitegrid", palette="muted", font=font)

        # Load custom style file
        if style_path is None:
            style_path = Path(__file__).parent / "redoxed_plot.mplstyle"

        if os.path.exists(style_path):
            plt.style.use(style_path)
        else:
            warnings.warn(
                f"Style file '{style_path}' not found. Using default style.",
                UserWarning,
            )

        # Note: LaTeX setup is now handled by the latex_context manager

    @abstractmethod
    def _configure_axes(self) -> None:
        """
        Configure plot-specific axis properties.

        This method must be implemented by subclasses to set:
        - Axis labels
        - Scales (linear/log)
        - Aspect ratios
        - Tick parameters
        - Any other axis-specific configuration
        """
        pass

    def add_major_ticks(
        self, spacing_x: float | None = None, spacing_y: float | None = None
    ) -> None:
        """
        Add and configure major tick marks on the axes.

        Sets major tick spacing for one or both axes using MultipleLocator.

        Parameters:
            spacing_x (float | None): Spacing between major ticks on x-axis in data units.
                If None, x-axis ticks are not modified. Defaults to None.
            spacing_y (float | None): Spacing between major ticks on y-axis in data units.
                If None, y-axis ticks are not modified. Defaults to None.
        """
        if spacing_x is not None:
            self.ax.xaxis.set_major_locator(MultipleLocator(spacing_x))
        if spacing_y is not None:
            self.ax.yaxis.set_major_locator(MultipleLocator(spacing_y))

    def add_minor_ticks(
        self, number_x: int | None = None, number_y: int | None = None
    ) -> None:
        """
        Add and configure minor tick marks on the axes.

        Sets the number of minor tick subdivisions between major ticks using AutoMinorLocator.

        Parameters:
            number_x (int | None): Number of minor tick subdivisions on x-axis (e.g., 2 creates
                one minor tick between major ticks). If None, x-axis minor ticks are not modified.
                Defaults to None.
            number_y (int | None): Number of minor tick subdivisions on y-axis.
                If None, y-axis minor ticks are not modified. Defaults to None.
        """
        if number_x is not None:
            self.ax.xaxis.set_minor_locator(AutoMinorLocator(number_x))
        if number_y is not None:
            self.ax.yaxis.set_minor_locator(AutoMinorLocator(number_y))

    def save(self, filename: str, **kwargs: Any) -> None:
        """
        Save the plot to a file in raster or vector format.

        Parameters:
            filename (str): Path and filename for the output file. Format is determined by extension
                (e.g., '.png' for raster, '.pdf' or '.svg' for vector).
            **kwargs: Additional arguments passed to matplotlib's savefig() function.
        """
        self.fig.savefig(filename, **kwargs)

    def copy(self) -> "BasePlot":
        """
        Return a deep copy of this plot instance.

        Returns:
            BasePlot: A new independent copy of this plot object.
        """
        return copy.deepcopy(self)
