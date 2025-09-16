import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Tuple, Any
from .base_plot import BasePlot
from redoxed.data_loading import ECData


class GenericPlot(BasePlot):
    """
    Generic plotter for DataFrame columns.

    Plots y vs x and optionally y2 vs x on a twinned axis.
    Multiple DataFrame objects can be added to the same plot.

    Parameters
    ----------
    x_col : str
        Column name for x-axis.
    y_col : str
        Column name for left y-axis.
    y2_col : str | None
        Column name for right y-axis (optional).
    **kwargs : Any
        Additional arguments passed to BasePlot.
    """

    def __init__(
        self,
        x_col: str,
        y_col: str,
        y2_col: str | None = None,
        **kwargs,
    ):
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
        **kwargs,
    ):
        """
        Add DataFrame to the plot.

        Args:
            df (pd.DataFrame): The DataFrame to plot.
            label (str | None): Label for the plot. If None, uses the data's label.
            color1 (str | None): Color for primary y-axis data. If None, uses default cycle.
            color2 (str | None): Color for secondary y-axis data. If None, uses default cycle.
            **kwargs: Additional arguments passed to matplotlib plot function.
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
            label=f"{label} ({self.y_col})",
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
                label=f"{label} ({self.y2_col})",
                **plot_kwargs_2,
            )
            # plot dummy data for legend entry
            self.ax.plot(
                np.nan,
                np.nan,
                label=f"{label} ({self.y2_col})",
                **plot_kwargs_2,
            )

    def _configure_axes(self) -> None:
        """
        Configure plot-specific axis properties.
        """
        pass
