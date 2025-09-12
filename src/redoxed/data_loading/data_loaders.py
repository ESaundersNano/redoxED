from abc import ABC, abstractmethod
import pandas as pd
import os
from typing import Optional
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
        fpath (Optional[str]): The absolute file path of the data file.
        ftype (Optional[str]): The file type (extension) of the data file.
        df (Optional[pd.DataFrame]): The loaded data as a pandas DataFrame.
    """

    # Class attribute type annotations
    fpath: Optional[str]
    ftype: Optional[str]
    df: Optional[pd.DataFrame]

    def __init__(self) -> None:
        self.fpath = None
        self.ftype = None
        self.df = None

    @abstractmethod
    def load_data(
        self, fpath: str, method: Optional[str] = None, label: Optional[str] = None
    ) -> ECData:
        """Load data from a file and return an ECData object.

        Args:
            fpath (str): The file path to the data file.
            method (Optional[str]): The method to use for loading the data. Defaults to None.
            label (Optional[str]): An optional label for the data. Defaults to None.

        Returns:
            ECData: An ECData object containing the loaded data.
        """
        pass

    def _convert_fpath(self, fpath: str) -> None:
        """Convert the file path to an absolute path and extract the file type.

        Args:
            fpath (str): The file path to convert.
        """
        self.fpath = os.path.abspath(fpath)
        self.ftype = os.path.splitext(fpath)[1][1:].lower()

    # def __validate_data__(self):
    #     """
    #     Validate the loaded data.
    #     """
    #     # Check if data arrays are set
    #     if self.Z_re is None or self.Z_im is None or self.f is None:
    #         raise ValueError(
    #             "The Real (Z_re), Imaginary (Z_im) impedance, and frequency (f) arrays must be provided."
    #         )

    #     # Check if all arrays are the same length
    #     if len(self.Z_re) != len(self.Z_im) or len(self.Z_re) != len(self.f):
    #         raise ValueError("Z_re, Z_im, and f must all have the same length.")

    #     # Check if frequencies are positive
    #     if jnp.any(self.f <= 0):
    #         raise ValueError("Frequencies (f) must be strictly positive.")

    #     # Check if arrays are non-empty
    #     if self.Z_re.size == 0 or self.Z_im.size == 0 or self.f.size == 0:
    #         raise ValueError("Z_re, Z_im, and f arrays must not be empty.")

    #     # Check if arrays are numeric
    #     if not jnp.issubdtype(self.Z_re.dtype, jnp.floating):
    #         raise TypeError("Z_re must be a numeric array (floating-point).")
    #     if not jnp.issubdtype(self.Z_im.dtype, jnp.floating):
    #         raise TypeError("Z_im must be a numeric array (floating-point).")
    #     if not jnp.issubdtype(self.f.dtype, jnp.floating):
    #         raise TypeError("f must be a numeric array (floating-point).")

    #     # Check for NaN or infinite values - couldn't be arsed testing these all.
    #     if jnp.any(jnp.isnan(self.Z_re)) or jnp.any(jnp.isinf(self.Z_re)):
    #         raise ValueError("Z_re contains NaN or infinite values.")
    #     if jnp.any(jnp.isnan(self.Z_im)) or jnp.any(jnp.isinf(self.Z_im)):
    #         raise ValueError("Z_im contains NaN or infinite values.")
    #     if jnp.any(jnp.isnan(self.f)) or jnp.any(jnp.isinf(self.f)):
    #         raise ValueError("f contains NaN or infinite values.")

    #     # Check for valid impedance values (complex number conditions)
    #     if jnp.any(self.Z_re == 0) and jnp.any(self.Z_im == 0):
    #         raise ValueError(
    #             "Impedance values should not all be zero in both real and imaginary parts."
    #         )


class BiologicLoader(BaseLoader):
    """Loader for Biologic files.

    This class provides functionality to load data from Biologic `.mpr` and `.mpt` files
    and convert it into an ECData object.

    Attributes:
        df (Optional[pd.DataFrame]): The loaded data as a pandas DataFrame.
        metadata (Optional[dict]): Metadata extracted from the file (for `yadg` method).
        variables (Optional[list[str]]): List of variable names extracted from the file (for `yadg` method).
    """

    # Class attribute type annotations
    df: Optional[pd.DataFrame]
    metadata: Optional[dict]
    variables: Optional[list[str]]

    def __init__(self) -> None:
        """Initialize the BiologicLoader."""
        self.df = None
        self.metadata = None
        self.variables = None

    def load_data(
        self, fpath: str, method: Optional[str] = None, label: Optional[str] = None
    ) -> ECData:
        """Load data from a Biologic file and return an ECData object.

        Args:
            fpath (str): The file path to the Biologic data file.
            method (Optional[str]): The method to use for loading the data. Defaults to None.
                Supported methods are 'galvani' and 'yadg' for `.mpr` files.
            label (Optional[str]): An optional label for the data. Defaults to None.

        Raises:
            ValueError: If the file cannot be processed, the method is unsupported,
                or the file format is invalid.

        Returns:
            ECData: An ECData object containing the loaded data.
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
                ):  # generally has more complete metadata and other data within the file, but it is less robust
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

            # If no label is provided, use the file path as the label
            if label is None:
                label = self.fpath

            # Return the loaded data as an ECData object
            return ECData(df=self.df, label=label)

        except Exception as e:
            raise ValueError(f"Failed to load data from {fpath}. Error: {e}")

    def _safe_extract_metadata(self, mpr_file, attribute_name, default_value=None):
        """Safely extract metadata attribute with warning if missing.

        Args:
            mpr_file: The MPR file object
            attribute_name (str): Name of the attribute to extract
            default_value: Value to return if attribute is missing

        Returns:
            The attribute value or default_value if missing
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
        df (Optional[pd.DataFrame]): The loaded data as a pandas DataFrame.
    """

    def __init__(self) -> None:
        """Initialize the CSVLoader."""
        self.df = None

    def load_data(
        self, fpath: str, method: Optional[str] = None, label: Optional[str] = None
    ) -> ECData:
        """Load data from a CSV file and return an ECData object.

        Args:
            fpath (str): The file path to the CSV data file.
            method (Optional[str]): The method to use for loading the data. Defaults to None.
            label (Optional[str]): An optional label for the data. Defaults to None.

        Raises:
            ValueError: If the file cannot be processed or the file format is unsupported.

        Returns:
            ECData: An ECData object containing the loaded data.
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
                label = self.fpath

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
    def get_loader(fpath: str):
        """Get the appropriate loader for the given file path.

        Args:
            fpath (str): The file path to the data file.

        Raises:
            ValueError: If the file format is unsupported.

        Returns:
            BaseLoader: An instance of the appropriate loader class.
        """
        if fpath.endswith(".mpr"):
            return BiologicLoader()
        if fpath.endswith(".mpt"):
            return BiologicLoader()
        if fpath.endswith(".csv"):
            return CSVLoader()

        else:
            raise ValueError("Unsupported file format!")
