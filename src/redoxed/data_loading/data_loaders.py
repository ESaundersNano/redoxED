from abc import ABC, abstractmethod
import pandas as pd
import os
import yadg
from galvani import BioLogic
import json
import warnings

from redoxed.data_loading import ECData


class BaseLoader(ABC):
    """Abstract base class for data loaders.

    This class provides a template for loading data from various file formats
    and converting it into an ECData object.

    Attributes:
        fpath (str | None): The absolute file path of the data file.
        ftype (str | None): The file type (extension) of the data file.
        df (pd.DataFrame | None): The loaded data as a pandas DataFrame.
    """

    # Class attribute type annotations
    fpath: str | None
    ftype: str | None
    df: pd.DataFrame | None

    def __init__(self) -> None:
        self.fpath = None
        self.ftype = None
        self.df = None

    @abstractmethod
    def load_data(
        self, fpath: str, method: str | None = None, label: str | None = None
    ) -> ECData:
        """
        Load data from a file and return an ECData object.

        Parameters
        ----------
        fpath : str
            The file path to the data file.
        method : str, optional
            The method to use for loading the data. Defaults to None.
        label : str, optional
            An optional label for the data. Defaults to None.

        Returns
        -------
        ECData
            An ECData object containing the loaded data.
        """
        pass

    def _convert_fpath(self, fpath: str) -> None:
        """Convert the file path to an absolute path and extract the file type.

        Parameters
        ----------
        fpath : str
            The file path to convert.
        """
        self.fpath = os.path.abspath(fpath)
        self.ftype = os.path.splitext(fpath)[1][1:].lower()
        # Get the base filename
        base = os.path.basename(fpath)
        # Split off the extension
        self.name, _ = os.path.splitext(base)


class BiologicLoader(BaseLoader):
    """Loader for Biologic files.

    This class provides functionality to load data from Biologic `.mpr` and `.mpt` files
    and convert it into an ECData object.

    Attributes:
        df (pd.DataFrame | None): The loaded data as a pandas DataFrame.
        metadata (dict | None): Metadata extracted from the file (for `yadg` method).
        variables (list[str] | None): List of variable names extracted from the file (for `yadg` method).
    """

    # Class attribute type annotations
    df: pd.DataFrame | None
    metadata: dict | None
    variables: list[str] | None

    def __init__(self) -> None:
        """Initialize the BiologicLoader."""
        self.df = None
        self.metadata = None
        self.variables = None

    def load_data(
        self, fpath: str, method: str | None = None, label: str | None = None
    ) -> ECData:
        """
        Load data from a Biologic file and return an ECData object.

        Parameters
        ----------
        fpath : str
            The file path to the Biologic data file.
        method : str, optional
            The method to use for loading the data. Supported methods are 'galvani' and 'yadg' for `.mpr` files. Defaults to None.
        label : str, optional
            An optional label for the data. Defaults to None.

        Raises
        ------
        ValueError
            If the file cannot be processed, the method is unsupported, or the file format is invalid.

        Returns
        -------
        ECData
            An ECData object containing the loaded data.
        """
        # Convert the file path to an absolute path and extract the file type
        self._convert_fpath(fpath)

        try:
            if self.ftype == "mpr":
                # Default to 'galvani' method if none is provided
                if method is None:
                    method = "galvani"

                # Attempt to load data using the specified method
                if method == "galvani":
                    try:
                        mpr_file = BioLogic.MPRfile(fpath)
                        self.df = pd.DataFrame(mpr_file.data)

                        metadata_dict = {
                            "timestamp": self._safe_extract_metadata(
                                mpr_file, "timestamp"
                            ),  # seems that timestamp will only be available once the program in EC-lab is finished, but it does represent the starttime of the experiment
                            "startdate": self._safe_extract_metadata(
                                mpr_file, "startdate"
                            ),
                            "enddate": self._safe_extract_metadata(mpr_file, "enddate"),
                            "modules": self._safe_extract_metadata(mpr_file, "modules"),
                        }

                        self.metadata = metadata_dict
                    except Exception as e:
                        raise ValueError(
                            f"Galvani failed to convert the file. "
                            f"Error: {e}. Please open the file in EC-Lab and export it to a compatible format (e.g., mpt)."
                        )
                elif (
                    method == "yadg"
                ):  # yadg generally has more complete metadata and other data within the file, but it is less robust
                    try:
                        dt = yadg.extractors.extract(filetype="eclab.mpr", path=fpath)
                        self.metadata = dt.attrs
                        self.variables = list(dt.data_vars.keys())
                        self.metadata["ole_timestamp"] = json.loads(
                            self.metadata["original_metadata"]
                        )["log"][
                            "ole_timestamp"
                        ]  # days since midnight December 30, 1899

                        # Convert DataTree to DataFrame
                        data_dic: dict[str, pd.Series] = {}
                        for variable in dt.data_vars.values():
                            unit_label = variable.name
                            try:
                                unit = variable.units
                                if unit == "Ω":
                                    unit = "Ohm"
                                elif "·" in unit:
                                    unit = unit.replace("·", ".")
                                unit_label = f"{unit_label}/{unit}"
                            except:
                                unit = None
                                unit_label = unit_label
                            data_dic[unit_label] = variable.data
                        self.df = pd.DataFrame(data_dic)
                    except Exception as e:
                        raise ValueError(
                            f"YADG failed to convert the file. "
                            f"Error: {e}. Please open the file in EC-Lab and export it to a compatible format (e.g., mpt)."
                        )
                else:
                    raise ValueError(
                        "Unsupported method. Please use 'galvani' or 'yadg'."
                    )

            elif self.ftype == "mpt":
                # Attempt to load `.mpt` file using brute force header detection
                try:
                    header_row_guess = 0
                    for i, _ in enumerate(open(self.fpath)):
                        pass
                    text_length = i + 1
                    attempts = min(text_length, 1000)

                    for i in range(attempts):
                        try:
                            df = pd.read_csv(
                                self.fpath,
                                sep="\t",
                                header=header_row_guess,
                                encoding="cp1252",
                            )
                            if "mode" in df.columns or "freq/Hz" in df.columns:
                                break
                            else:
                                raise ValueError("Header row not found.")
                        except Exception:
                            header_row_guess += 1
                            if i == (attempts - 1):
                                raise ValueError(
                                    "Couldn't convert from text. Inspect read_csv parameters or text file header columns."
                                )
                    self.df = df
                except Exception as e:
                    raise ValueError(f"Failed to process mpt file. Error: {e}")

            # If no label is provided, use the file name as the label
            if label is None:
                label = self.name

            # Return the loaded data as an ECData object
            return ECData(df=self.df, label=label)

        except Exception as e:
            raise ValueError(f"Failed to load data from {fpath}. Error: {e}")

    def _safe_extract_metadata(self, mpr_file, attribute_name: str, default_value=None):
        """
        Safely extract metadata attribute with warning if missing.

        Parameters
        ----------
        mpr_file : BioLogic.MPRfile
            The MPR file object.
        attribute_name : str
            Name of the attribute to extract.
        default_value : any, optional
            Value to return if attribute is missing.

        Returns
        -------
        any
            The attribute value or default_value if missing.
        """
        try:
            return getattr(mpr_file, attribute_name)
        except AttributeError:
            warnings.warn(
                f"'{attribute_name}' not available in MPR file metadata",
                UserWarning,
                stacklevel=3,
            )
            return default_value
        except Exception as e:
            warnings.warn(
                f"Could not extract '{attribute_name}' from MPR file: {e}",
                UserWarning,
                stacklevel=3,
            )
            return default_value


class CSVLoader(BaseLoader):
    """Loader for CSV files.

    This class provides functionality to load data from CSV files and convert
    it into an ECData object.

    Attributes:
        df (pd.DataFrame | None): The loaded data as a pandas DataFrame.
    """

    def __init__(self) -> None:
        """Initialize the CSVLoader."""
        self.df = None

    def load_data(
        self, fpath: str, method: str | None = None, label: str | None = None
    ) -> ECData:
        """
        Load data from a CSV file and return an ECData object.

        Parameters
        ----------
        fpath : str
            The file path to the CSV data file.
        method : str, optional
            The method to use for loading the data. Defaults to None.
        label : str, optional
            An optional label for the data. Defaults to None.

        Raises
        ------
        ValueError
            If the file cannot be processed or the file format is unsupported.

        Returns
        -------
        ECData
            An ECData object containing the loaded data.
        """
        # Convert the file path to an absolute path and extract the file type
        self._convert_fpath(fpath)

        try:
            # Ensure the file type is CSV
            if self.ftype != "csv":
                raise ValueError("Unsupported file format. Please use a CSV file.")

            # Load the CSV file into a pandas DataFrame
            self.df = pd.read_csv(fpath)

            # If no label is provided, use the file path as the label
            if label is None:
                label = self.name

            # Return the loaded data as an ECData object
            return ECData(df=self.df, label=label)

        except Exception as e:
            # Raise an error if the file cannot be processed
            raise ValueError(f"Failed to load data from {fpath}. Error: {e}")


# Factory for selecting an appropriate loader
class LoaderFactory:
    """Factory class for selecting an appropriate data loader.

    This class provides a method to return the appropriate loader instance
    based on the file type.
    """

    @staticmethod
    def get_loader(fpath: str) -> BaseLoader:
        """
        Get the appropriate loader for the given file path.

        Parameters
        ----------
        fpath : str
            The file path to the data file.

        Raises
        ------
        ValueError
            If the file format is unsupported.

        Returns
        -------
        BaseLoader
            An instance of the appropriate loader class.
        """
        if fpath.endswith(".mpr"):
            return BiologicLoader()
        if fpath.endswith(".mpt"):
            return BiologicLoader()
        if fpath.endswith(".csv"):
            return CSVLoader()

        else:
            raise ValueError("Unsupported file format!")
