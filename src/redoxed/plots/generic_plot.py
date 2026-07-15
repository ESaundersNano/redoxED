import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Tuple, Any
from .base_plot import BasePlot
from redoxed.data_loading import ECData


class GenericPlot(BasePlot):
    """
    Generic data plotter for arbitrary DataFrame columns.

    Plots specified columns from pandas DataFrames with support for dual y-axes.
    Allows plotting y vs x on the primary axis, and optionally y2 vs x on a twinned
    secondary axis. Multiple DataFrame objects can be added to the same plot.

    Attributes:
        x_col (str): Column name for x-axis data.
        y_col (str): Column name for left (primary) y-axis data.
        y2_col (str | None): Column name for right (secondary) y-axis data.
        ax2 (plt.Axes | None): Secondary y-axis created on first y2 data addition.
    """

    def __init__(
        self,
        x_col: str,
        y_col: str,
        y2_col: str | None = None,
        usetex: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a generic data plot.

        Parameters:
            x_col (str): Column name for x-axis data.
            y_col (str): Column name for left y-axis data.
            y2_col (str | None): Column name for right y-axis data. Defaults to None.
            usetex (bool | None): Whether to use LaTeX rendering.
                If None, uses global config setting. Defaults to None.
            **kwargs: Additional arguments passed to BasePlot.
        """
        self.x_col = x_col
        self.y_col = y_col
        self.y2_col = y2_col
        self.ax2 = None  # Will be created when first y2 data is added
        self._axes_configured = False
        super().__init__(**kwargs)

    def add_plot(
        self,
        df: pd.DataFrame,
        label: str | None = None,
        color1: str | None = None,
        color2: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Add DataFrame data to the plot.

        Plots the specified columns (x_col, y_col, and optionally y2_col) from the DataFrame.
        If y2_col is specified, data is plotted on a secondary y-axis.

        Parameters:
            df (pd.DataFrame): The DataFrame containing the data to plot.
            label (str | None): Label for the plot series. If None, defaults to 'data'. Defaults to None.
            color1 (str | None): Color for primary y-axis data. If None, uses default cycle. Defaults to None.
            color2 (str | None): Color for secondary y-axis data. If None, uses default cycle. Defaults to None.
            **kwargs: Additional arguments passed to matplotlib plot function.

        Raises:
            ValueError: If required column(s) are missing from the DataFrame.
        """
        if label is None:
            label = "data"

        # Validate columns exist
        required_cols = [self.x_col, self.y_col]
        if self.y2_col:
            required_cols.append(self.y2_col)
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            available_cols = list(df.columns)
            raise ValueError(
                f"Column(s) {missing_cols} not found. Available columns: {available_cols}"
            )

        # Configure axes labels only once
        if not self._axes_configured:
            self.ax.set_xlabel(self.x_col)
            self.ax.set_ylabel(self.y_col)
            if self.y2_col:
                self.ax2 = self.ax.twinx()
                self.ax2.set_ylabel(self.y2_col)
            self._axes_configured = True

        # Plot primary data
        plot_kwargs = kwargs.copy()
        if color1:
            plot_kwargs["color"] = color1
        self.ax.plot(
            df[self.x_col],
            df[self.y_col],
            # label=f"{label} ({self.y_col})",
            label=label,
            **plot_kwargs,
        )

        # Plot secondary data if column is specified
        if self.y2_col and self.ax2 is not None:
            plot_kwargs_2 = kwargs.copy()
            if color2:
                plot_kwargs_2["color"] = color2
            self.ax2.plot(
                df[self.x_col],
                df[self.y2_col],
                # label=f"{label} ({self.y2_col})",
                label=label,
                **plot_kwargs_2,
            )
            # plot dummy data for legend entry
            self.ax.plot(
                np.nan,
                np.nan,
                # label=f"{label} ({self.y2_col})",
                label=label,
                **plot_kwargs_2,
            )

    def _configure_axes(self) -> None:
        """
        Configure plot-specific axis properties.

        Sets axis labels for primary and secondary axes based on configured columns.
        """
        pass
