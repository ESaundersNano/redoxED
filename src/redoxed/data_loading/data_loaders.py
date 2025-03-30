from abc import ABC, abstractmethod
import pandas as pd
import os
from typing import Optional
import yadg
from galvani import BioLogic

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
    """_summary_
    Args:
        BaseLoader (_type_): _description_
    """

    def __init__(self) -> None:
        """Initialize the BiologicLoader."""
        self.df = None

    def load_data(
        self, fpath: str, method: Optional[str] = None, label: Optional[str] = None
    ) -> ECData:
        """Load data from a Biologic file and return an ECData object.

        Args:
            fpath (str): The file path to the Biologic data file.
            method (Optional[str]): The method to use for loading the data. Defaults to None.
            label (Optional[str]): An optional label for the data. Defaults to None.

        Raises:
            ValueError: If the file cannot be processed or the method is unsupported.

        Returns:
            ECData: An ECData object containing the loaded data.
        """
        self._convert_fpath(fpath)  # convert fpath to absolute path and get file type
        # Convert file to ECData object
        try:
            if self.ftype == "mpr":
                # set method to galvani if none provided
                if method == None:
                    method = "galvani"
                # attempt to load data using the specified method
                if method == "galvani":
                    try:
                        mpr_file = BioLogic.MPRfile(fpath)
                        self.df = pd.DataFrame(mpr_file.data)
                    except Exception as e:
                        raise ValueError(
                            f"Galvani failed to convert the file. "
                            f"Error: {e}. Please open the file in EC-Lab and export it to a compatible format (e.g., mpt)."
                        )
                elif method == "yadg":
                    try:
                        dt = yadg.extractors.extract(filetype="eclab.mpr", path=fpath)
                        self.metadata = dt.attrs
                        self.variables = list(dt.data_vars.keys())
                        # Convert DataTree to DataFrame
                        data_dic = {}
                        for variable in list(
                            dt.data_vars.values()
                        ):  # unit name conversion for consistency with other loaders
                            label = variable.name
                            try:
                                unit = variable.units
                                if unit == "Ω":
                                    unit = "Ohm"
                                elif "·" in unit:
                                    unit = unit.replace("·", ".")
                                label = label + "/" + unit
                            except Exception as e:
                                raise ValueError(
                                    f"Error processing variable units: {e}"
                                )
                            data_dic[label] = variable.data
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
                # currently defaults to brute force method of conversion
                try:
                    # Initialize guess for header row
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
            # self.__validate_data__()

            # if no label provided, use fpath
            if label is None:
                label = self.fpath
            # return final ECData object
            return ECData(df=self.df, label=label)

        except Exception as e:
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
        else:
            raise ValueError("Unsupported file format!")
