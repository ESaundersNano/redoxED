"""
Electrochemical data handling module.

This module provides the ECData class for handling general electrochemical
measurement data and converting it to specialized data types.
"""

import pandas as pd
import numpy as np
from typing import Any, Tuple
from collections.abc import Callable
import matplotlib.pyplot as plt
import copy

from redoxed.impedance import EISData
from redoxed.dc import PolarisationData
from redoxed.dc import CyclingData
from redoxed.data_loading.data_converters import (
    df_to_EISData,
    df_to_PolarisationData,
    df_to_CyclingData,
)


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
        return self.df[
            self.df[col].apply(condition)
        ].copy()  # return copy rather than a view

    def to_csv(self, filepath: str) -> None:
        """
        Save the DataFrame to a CSV file.

        Args:
            filepath (str): The path to the output CSV file.
        """
        # Add .csv extension if not present
        if not filepath.lower().endswith(".csv"):
            filepath = filepath + ".csv"
        self.df.to_csv(filepath, index=False)

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

    def average_to_EISData(
        self,
        cycle_range: tuple[int, int] | None = None,
        label: str | None = None,
        converter: Callable[[pd.DataFrame, str, Any], EISData] = df_to_EISData,
        **kwargs: Any,
    ) -> EISData:
        """
        Convert to EISData by averaging impedance data over a range of cycles.

        This method averages EIS measurements across multiple cycles to reduce noise
        and improve data quality. It's particularly useful when you have repeated
        EIS measurements at the same conditions.

        Parameters
        ----------
        cycle_range : tuple[int, int] | None, optional
            A tuple (min_cycle, max_cycle) specifying the range of cycles to average.
            Both endpoints are inclusive. If None, averages over all available cycles.
        label : str | None, optional
            Label for the resulting EISData object. If None, generates a label
            describing the averaged cycles.
        converter : Callable[[pd.DataFrame, str, Any], EISData], optional
            Function to convert DataFrame to EISData. Defaults to df_to_EISData.
        **kwargs : Any
            Additional keyword arguments passed to the converter function.

        Returns
        -------
        EISData
            An EISData object with averaged impedance values over the specified cycles.
            The frequency array is taken from the first valid cycle.

        Raises
        ------
        ValueError
            If no valid cycles are found in the specified range, or if no valid
            EIS data could be extracted from the specified cycles.

        Examples
        --------
        >>> # Average over all cycles
        >>> eis_avg = ecdata.average_to_EISData()

        >>> # Average over cycles 2-5
        >>> eis_avg = ecdata.average_to_EISData(cycle_range=(2, 5), label="cycles 2-5 avg")

        >>> # Average over a single cycle (equivalent to filtering)
        >>> eis_single = ecdata.average_to_EISData(cycle_range=(3, 3), label="cycle 3")
        """

        # Check if cycle number column exists
        if "cycle number" not in self.df.columns:
            raise ValueError(
                "DataFrame must contain 'cycle number' column for cycle averaging"
            )

        # Get all unique cycle numbers
        all_cycles = np.unique(self.df["cycle number"].to_numpy())

        # Filter cycles based on range
        if cycle_range is not None:
            min_cycle, max_cycle = cycle_range
            cycles_to_average = all_cycles[
                (all_cycles >= min_cycle) & (all_cycles <= max_cycle)
            ]
        else:
            cycles_to_average = all_cycles
        # warning if cycle outside available range
        if cycle_range is not None and (
            min_cycle < all_cycles[0] or max_cycle > all_cycles[-1]
        ):
            print(
                f"Warning: Cycle range {cycle_range} is outside available cycles {all_cycles}"
            )

        if len(cycles_to_average) == 0:
            raise ValueError(f"No cycles found in range {cycle_range}")

        # Generate label if not provided
        if label is None:
            if len(cycles_to_average) == len(all_cycles):
                label = f"{self.label} (averaged over all {len(all_cycles)} cycles)"
            elif len(cycles_to_average) == 1:
                label = f"{self.label} (cycle {cycles_to_average[0]})"
            else:
                label = f"{self.label} (averaged over cycles {cycles_to_average[0]}-{cycles_to_average[-1]})"

        # Initialize variables
        Z_sum = None
        f = None
        N = 0

        # Iterate over selected cycles
        for cycle_num in cycles_to_average:
            # Create a copy and filter for this cycle
            ECData_temp = self.copy()
            ECData_temp.df = self.filter_by_col(
                "cycle number", lambda value: value == cycle_num
            )

            try:
                # Convert to EISData
                EISData_temp = ECData_temp.to_EISData(converter=converter, **kwargs)

                # Initialize on first valid cycle
                if Z_sum is None:
                    Z_sum = np.zeros_like(EISData_temp.Z)
                    f_sum = np.zeros_like(EISData_temp.f)

                # Accumulate impedance
                Z_sum += EISData_temp.Z
                f_sum += EISData_temp.f
                N += 1

            except Exception as e:
                print(f"Warning: Error processing cycle {cycle_num}: {e}")
                continue

        if N == 0:
            raise ValueError(
                "No valid EIS data could be extracted from the specified cycles"
            )

        # Calculate average
        Z_avg = Z_sum / N
        f_avg = f_sum / N

        # Create averaged EISData object
        EISData_avg = EISData(f=f_avg, Z=Z_avg, label=label)

        return EISData_avg

    def to_CyclingData(
        self,
        converter: Callable[[pd.DataFrame, str, Any], CyclingData] = df_to_CyclingData,
        **kwargs: Any,
    ) -> CyclingData:
        """
        Convert the electrochemical data to CyclingData format.

        Args:
            converter (Callable[[pd.DataFrame, str, Any], CyclingData]): Function to convert
                DataFrame to CyclingData. Defaults to df_to_CyclingData.
            **kwargs: Additional keyword arguments passed to the converter function.

        Returns:
            CyclingData: The converted cycling data.
        """
        return converter(df=self.df, label=self.label, **kwargs)

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

    def copy(self):
        """Return a deep copy of this ECData instance."""
        return copy.deepcopy(self)
