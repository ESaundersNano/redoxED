# from redoxed.impedance.eis_data import EISData

import pandas as pd

import numpy as np

from sklearn.linear_model import LinearRegression

import copy


class PolarisationData:
    """
    Container for electrochemical polarisation (I-V) curve data with linear regression fitting capabilities.

    This class stores potential and current density data from polarisation measurements and provides
    methods to fit Area-Specific Resistance (ASR) by linear regression of the V-j relationship.

    Attributes:
    -----------
        V (np.ndarray): Cell potential data in V (Volts).
        j (np.ndarray): Current density data in mA/cm² (milliamps per square centimeter).
        A (float): Electrode geometric area in cm² (square centimeters).
        label (str | None): Optional label for this dataset.
        ASR (float | None): Area-Specific Resistance in Ω·cm² (Ohm·cm²), computed by calculate_ASR().
        ASR_intercept (float | None): Y-intercept of fitted V-j line in V, computed by calculate_ASR().
        ASR_error (float | None): Standard error of ASR slope estimate, computed by calculate_ASR().

    Unit Convention:
        - Potential: V (Volts)
        - Current density: mA/cm² (milliamps per square centimeter) for input/output
        - Area: cm² (square centimeters)
        - Resistance: Ω·cm² (Ohm·cm²)
    """

    V: np.ndarray
    j: np.ndarray
    A: float
    label: str | None
    ASR: float | None
    ASR_intercept: float | None
    ASR_error: float | None

    def __init__(
        self, V: np.ndarray, j: np.ndarray, A: float, label: str | None = None
    ) -> None:
        """
        Initialize PolarisationData container.

        Parameters:
        -----------
            V (np.ndarray): Cell potential data in V (Volts).
            j (np.ndarray): Current density data in mA/cm² (milliamps per square centimeter).
            A (float): Electrode geometric area in cm² (square centimeters).
            label (str | None): Optional label for this dataset. Defaults to None.
        """
        self.V = V
        self.j = j
        self.A = A
        self.label = label

        # Fitted parameters (computed by calculate_ASR)
        self.ASR = None
        self.ASR_intercept = None
        self.ASR_error = None

        self._validate()

    def _validate(self) -> None:
        """
        Validate input data for consistency and correctness.

        Raises:
        -----------
            ValueError: If V and j arrays have different shapes.
            ValueError: If V or j are not 1D arrays.
            ValueError: If V or j arrays are empty.
        """
        if self.V.shape != self.j.shape:
            raise ValueError("V and j must have the same shape.")
        if len(self.V.shape) != 1:
            raise ValueError("V and j must be 1D arrays.")
        if self.V.size == 0 or self.j.size == 0:
            raise ValueError("V and j arrays must not be empty.")

    def calculate_ASR(
        self, restrict_to_zero: bool = False, j_range: tuple = (-np.inf, np.inf)
    ) -> "PolarisationData":
        """
        Fit Area-Specific Resistance (ASR) from V-j data using linear regression.

        Performs a linear fit to the V vs j relationship. The slope of the fit equals the ASR.
        Updates the object's ASR, ASR_intercept, and ASR_error attributes.

        Parameters:
        -----------
            restrict_to_zero (bool): If True, force the linear fit through the origin (0, 0).
                Defaults to False.
            j_range (tuple): Tuple of (min_j, max_j) in mA/cm² to restrict which data points
                are used in the fit. Data outside this range is excluded. Defaults to (-∞, ∞).

        Returns:
        -----------
            PolarisationData: A new PolarisationData object containing the fitted V values
                and corresponding j values, useful for plotting the fit line.
        """

        # Filter data to specified j_range
        mask = (self.j >= j_range[0]) & (self.j <= j_range[1])
        j_ASR = self.j[mask]
        V_ASR = self.V[mask]

        # Convert current density from mA/cm² to A/cm² for linear regression
        j_ASR = j_ASR / 1000

        x = j_ASR
        y = V_ASR

        # Reshape for scikit-learn (requires 2D feature array)
        x_reshaped = x.reshape(-1, 1)

        # Fit linear model: V = ASR * j + intercept
        model = LinearRegression(fit_intercept=not restrict_to_zero)
        model.fit(x_reshaped, y)

        # Extract fit parameters
        ASR = model.coef_[0]  # Slope in Ω·cm² (Ohm·cm²)
        intercept = model.intercept_ if not restrict_to_zero else 0

        # Calculate predictions and residuals
        y_pred = model.predict(x_reshaped)
        residuals = y - y_pred

        # Calculate standard error of the ASR slope estimate
        n = len(x)
        error = np.sqrt(np.sum(residuals**2) / (n - 2)) / np.sqrt(
            np.sum((x - np.mean(x)) ** 2)
        )

        # Store fitted parameters in this object
        self.ASR = ASR
        self.ASR_intercept = intercept
        self.ASR_error = error

        # Return new PolarisationData object with fitted V values (for plotting fit line)
        # Convert j back to mA/cm² for consistency with input format
        PolarisationData_fit = PolarisationData(V=y_pred, j=x * 1000, A=self.A)

        return PolarisationData_fit

    def copy(self) -> "PolarisationData":
        """
        Return a deep copy of this PolarisationData instance.

        Returns:
        -----------
            PolarisationData: A new independent copy of this object.
        """
        return copy.deepcopy(self)
