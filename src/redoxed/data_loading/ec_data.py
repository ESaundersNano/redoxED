import pandas as pd

from redoxed.impedance import EISData


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
        self, col: str = "cycle number", condition: callable = lambda value: value == 1
    ) -> pd.DataFrame:
        """
        Filter the DataFrame based on a condition applied to a column.

        Args:
            col (str): The column to filter on.
            condition (callable): A function or lambda that takes a column value and returns a boolean.

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
        column_mapping: dict[str, str | None] | None = None,
    ) -> EISData:
        """ """
        # Set default column names (EClab file assumed)
        default_mapping = {
            "Z_re": "Re(Z)/Ohm",
            "-Z_im": "-Im(Z)/Ohm",
            "f": "freq/Hz",
            "Z": None,
            "Z_im": None,
        }
        if column_mapping is None:
            column_mapping = default_mapping

        # Check if the DataFrame contains the required columns
        # Drop entries with value None
        required_columns = {
            key: value for key, value in column_mapping.items() if value is not None
        }
        # Perform check for required columns
        if not all(col in self.df.columns for col in list(required_columns.values())):
            raise ValueError(
                f"DataFrame did not contain the columns: {required_columns}"
            )
        # Convert the DataFrame columns to extract Z and f
        elif all(key in required_columns for key in ["Z_re", "-Z_im", "f"]):
            Z = (
                self.df[required_columns["Z_re"]].to_numpy()
                - 1j * self.df[required_columns["-Z_im"]].to_numpy()
            )
        elif all(key in required_columns for key in ["Z", "f"]):
            Z = self.df[required_columns["Z"]].to_numpy()
        elif all(key in required_columns for key in ["Z_re", "Z_im", "f"]):
            Z = (
                self.df[required_columns["Z_re"]].to_numpy()
                + 1j * self.df[required_columns["Z_im"]].to_numpy()
            )
        else:
            raise NotImplementedError(
                "Unsupported data format. Try extract necessary data from ECData's df."
            )
        f = self.df[required_columns["f"]].to_numpy()

        # Create and return an instance of EISData
        return EISData(Z, f, label=self.label)
