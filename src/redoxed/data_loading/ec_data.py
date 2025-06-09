import pandas as pd

from redoxed.impedance import EISData
from redoxed.DC import PolarisationData

from redoxed.data_loading.data_converters import df_to_EISData, df_to_PolarisationData

from collections.abc import Callable
from typing import Any


class ECData:
    """ """

    # general data from electrochemical measuremnts for further processing.

    def __init__(self, df: pd.DataFrame, label: str) -> None:
        self.df = df
        self.label = label

    def _validate(self) -> None:
        """ """
        if not isinstance(self.df, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame.")
        if self.df.empty:
            raise ValueError("DataFrame must not be empty.")
        if self.label is None or not isinstance(self.label, str):
            raise ValueError("Label must be a non-empty string.")

    def __repr__(self) -> str:
        return f"ECData(label={self.label})"

    def filter_by_col(
        self, col: str = "cycle number", condition: Callable = lambda value: value == 1
    ) -> pd.DataFrame:
        """
        Filter the DataFrame based on a condition applied to a column.

        Args:
            col (str): The column to filter on.
            condition (Callable): A function or lambda that takes a column value and returns a boolean.

        Returns:
            pd.DataFrame: A filtered DataFrame.

        Raises:
            ValueError: If the column is not found in the DataFrame.
        """
        if col not in self.df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
        return self.df[self.df[col].apply(condition)]

    def to_EISData(
        self,
        converter: Callable[[pd.DataFrame, str, Any], EISData] = df_to_EISData,
        **kwargs: Any,
    ) -> EISData:
        """ """
        return converter(self.df, self.label, **kwargs)

    def to_PolarisationData(
        self,
        converter: Callable[[pd.DataFrame, str, Any], EISData] = df_to_PolarisationData,
        **kwargs: Any,
    ) -> PolarisationData:
        """
        Default converter is df_to_PolarisationData.
        This is based on the assumed file format of the author and so may need to be customised to invidiual users.
        """
        return converter(df=self.df, label=self.label, **kwargs)
