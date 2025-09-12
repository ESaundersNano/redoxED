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

        Args:
            usetex (bool): Whether to use LaTeX rendering. If None, uses global config.
            **kwargs: Additional arguments passed to plt.subplots().
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
        Setup LaTeX rendering preference once.

        Args:
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
        Load the custom Matplotlib style from the .mplstyle file.

        Args:
            usetex (bool): Whether to enable LaTeX rendering. Defaults to False.
            style_path (str | None): Custom path to style file. If None, uses default.
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
        Add major ticks to the axes with specified spacing.

        Args:
            spacing_x (float | None): Spacing for major ticks on x-axis.
                If None, x-axis ticks are not modified.
            spacing_y (float | None): Spacing for major ticks on y-axis.
                If None, y-axis ticks are not modified.
        """
        if spacing_x is not None:
            self.ax.xaxis.set_major_locator(MultipleLocator(spacing_x))
        if spacing_y is not None:
            self.ax.yaxis.set_major_locator(MultipleLocator(spacing_y))

    def add_minor_ticks(
        self, number_x: int | None = None, number_y: int | None = None
    ) -> None:
        """
        Add minor ticks to the axes.

        Args:
            number_x (int | None): Number of minor ticks between major ticks on x-axis.
                If None, x-axis minor ticks are not modified.
            number_y (int | None): Number of minor ticks between major ticks on y-axis.
                If None, y-axis minor ticks are not modified.
        """
        if number_x is not None:
            self.ax.xaxis.set_minor_locator(AutoMinorLocator(number_x))
        if number_y is not None:
            self.ax.yaxis.set_minor_locator(AutoMinorLocator(number_y))

    def show(self) -> None:
        """Display the plot in the current output."""
        plt.figure(self.fig.number)
        plt.show()

    def save(self, filename: str, **kwargs: Any) -> None:
        """
        Save the plot to a file.

        Args:
            filename (str): Path and filename for the saved plot.
            **kwargs: Additional arguments passed to plt.savefig().
        """
        self.fig.savefig(filename, **kwargs)
