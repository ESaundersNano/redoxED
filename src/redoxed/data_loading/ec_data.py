"""
Electrochemical data handling module.

This module provides the ECData class for handling general electrochemical
measurement data and converting it to specialized data types.
"""

import pandas as pd
from typing import Any, Optional, Tuple
from collections.abc import Callable
import matplotlib.pyplot as plt

from redoxed.impedance import EISData
from redoxed.dc import PolarisationData
from redoxed.data_loading.data_converters import df_to_EISData, df_to_PolarisationData


class ECData:
    """
    Container for general electrochemical measurement data.

    This class provides a common interface for handling electrochemical data
    from various sources and converting it to specialized data types like
    EISData or PolarisationData.

    Attributes:
        df (pd.DataFrame): The electrochemical measurement data.
        label (str): A descriptive label for the dataset.
    """

    df: pd.DataFrame
    label: str

    def __init__(self, df: pd.DataFrame, label: str) -> None:
        """
        Initialize ECData with measurement data and label.

        Args:
            df (pd.DataFrame): The electrochemical measurement data.
            label (str): A descriptive label for the dataset.

        Raises:
            ValueError: If df is not a DataFrame, is empty, or label is invalid.
        """
        self.df = df
        self.label = label
        self._validate()

    def __repr__(self) -> str:
        """
        Return a string representation of the ECData instance.

        Returns:
            str: String representation showing the label.
        """
        return f"ECData(label={self.label})"

    @property
    def shape(self) -> tuple[int, int]:
        """Return the shape of the DataFrame."""
        return self.df.shape

    @property
    def columns(self) -> pd.Index:
        """Return the column names of the DataFrame."""
        return self.df.columns

    @property
    def size(self) -> int:
        """Return the total number of elements in the DataFrame."""
        return self.df.size

    @property
    def dtypes(self) -> pd.Series:
        """Return the data types of each column."""
        return self.df.dtypes

    def info(self) -> None:
        """Print a concise summary of the dataset."""
        print(f"ECData: {self.label}")
        print(f"Shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        print(f"Data types:")
        for col, dtype in self.df.dtypes.items():
            print(f"  {col}: {dtype}")

    def filter_by_col(
        self,
        col: str = "cycle number",
        condition: Callable[[Any], bool] = lambda value: value == 1,
    ) -> pd.DataFrame:
        """
        Filter the DataFrame based on a condition applied to a column.

        Args:
            col (str): The column to filter on. Defaults to "cycle number".
            condition (Callable[[Any], bool]): A function that takes a column value
                and returns a boolean. Defaults to lambda that checks if value equals 1.

        Returns:
            pd.DataFrame: A filtered DataFrame containing only rows where the condition is True.

        Raises:
            ValueError: If the specified column is not found in the DataFrame.

        Examples:
            >>> ecdata.filter_by_col("voltage", lambda v: v > 1.5)
            >>> ecdata.filter_by_col("cycle number", lambda c: c in [1, 3, 5])
        """
        if col not in self.df.columns:
            available_cols = list(self.df.columns)
            raise ValueError(
                f"Column '{col}' not found. Available columns: {available_cols}"
            )
        return self.df[self.df[col].apply(condition)]

    def to_EISData(
        self,
        converter: Callable[[pd.DataFrame, str, Any], EISData] = df_to_EISData,
        **kwargs: Any,
    ) -> EISData:
        """
        Convert the electrochemical data to EISData format.

        Args:
            converter (Callable[[pd.DataFrame, str, Any], EISData]): Function to convert
                DataFrame to EISData. Defaults to df_to_EISData.
            **kwargs: Additional keyword arguments passed to the converter function.

        Returns:
            EISData: The converted electrochemical impedance spectroscopy data.
        """
        return converter(self.df, self.label, **kwargs)

    def to_PolarisationData(
        self,
        converter: Callable[
            [pd.DataFrame, str, Any], PolarisationData
        ] = df_to_PolarisationData,
        **kwargs: Any,
    ) -> PolarisationData:
        """
        Convert the electrochemical data to PolarisationData format.

        Args:
            converter (Callable[[pd.DataFrame, str, Any], PolarisationData]): Function to convert
                DataFrame to PolarisationData. Defaults to df_to_PolarisationData.
            **kwargs: Additional keyword arguments passed to the converter function.

        Returns:
            PolarisationData: The converted polarisation data.

        Note:
            The default converter is based on the assumed file format of the author
            and may need to be customized for individual users.
        """
        return converter(df=self.df, label=self.label, **kwargs)

    def quick_plot(
        self,
        x_col: str,
        y_col: str,
        y2_col: Optional[str] = None,
        figsize: Tuple[float, float] = (6, 3),
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a quick plot of data columns with optional twin y-axis.

        Args:
            x_col (str): Column name for x-axis data.
            y_col (str): Column name for primary y-axis data.
            y2_col (Optional[str]): Column name for secondary y-axis data (twin axis).
                If None, no secondary axis is created.
            figsize (Tuple[float, float]): Figure size as (width, height). Defaults to (6, 3).

        Returns:
            Tuple[plt.Figure, plt.Axes]: The figure and primary axes objects.

        Raises:
            ValueError: If any specified column is not found in the DataFrame.

        Examples:
            >>> fig, ax = ecdata.quick_plot("time", "voltage")
            >>> fig, ax = ecdata.quick_plot("time", "voltage", "current")
        """
        # Validate columns exist
        required_cols = [x_col, y_col]
        if y2_col is not None:
            required_cols.append(y2_col)

        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            available_cols = list(self.df.columns)
            raise ValueError(
                f"Column(s) {missing_cols} not found. Available columns: {available_cols}"
            )

        # Create the plot
        fig, ax1 = plt.subplots(figsize=figsize)

        # Plot primary data
        color1 = "tab:blue"
        ax1.set_xlabel(x_col)
        ax1.set_ylabel(y_col, color=color1)
        ax1.plot(self.df[x_col], self.df[y_col], color=color1, label=y_col)
        ax1.tick_params(axis="y", labelcolor=color1)

        # Plot secondary data if provided
        ax2 = None
        if y2_col is not None:
            ax2 = ax1.twinx()
            color2 = "tab:red"
            ax2.set_ylabel(y2_col, color=color2)
            ax2.plot(self.df[x_col], self.df[y2_col], color=color2, label=y2_col)
            ax2.tick_params(axis="y", labelcolor=color2)

        # Add grid and layout
        ax1.grid(True, alpha=0.3)
        fig.tight_layout()

        return fig, ax1

    def _validate(self) -> None:
        """
        Validate the input data and label.

        Raises:
            TypeError: If df is not a DataFrame.
            ValueError: If df is empty or label is invalid.
        """
        if not isinstance(self.df, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(self.df)}")
        if len(self.df) == 0:
            raise ValueError("DataFrame must not be empty.")
        if not self.label or not isinstance(self.label, str):
            raise ValueError("Label must be a non-empty string.")
