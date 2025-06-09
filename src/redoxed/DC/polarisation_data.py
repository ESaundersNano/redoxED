# from redoxed.impedance.eis_data import EISData

import pandas as pd

import numpy as np

from sklearn.linear_model import LinearRegression


class PolarisationData:
    """ """

    V: np.ndarray
    j: np.ndarray
    A: float
    label: str | None
    ASR: float | None
    ASR_intercept: float | None
    ASR_error: float | None

    def __init__(
        self, V: np.ndarray, j: np.ndarray, A: float, label: str = None
    ) -> None:
        """
        Initializes the PolarisationData object.

        Args:
            V (np.ndarray): Potential data.
            j (np.ndarray): Current density data.
            A (float): Area of the electrode.
            label (str, optional): Optional label for the dataset. Defaults to None.
        """
        self.V = V
        self.j = j
        self.A = A
        self.label = label

        self.ASR = None
        self.ASR_intercept = None
        self.ASR_error = None

        self._validate()

    def _validate(self) -> None:
        """
        Validates the input data to ensure consistency and correctness.

        Raises:
            ValueError: If V and j do not have the same shape.
            ValueError: If V and j are not 1D arrays.
            ValueError: If V or j arrays are empty.
        """
        if self.V.shape != self.j.shape:
            raise ValueError("V and j must have the same shape.")
        if len(self.V.shape) != 1:
            raise ValueError("V and j must be 1D arrays.")
        if self.V.size == 0 or self.j.size == 0:
            raise ValueError("V and j arrays must not be empty.")

    def calculate_ASR(self, restrict_to_zero=False, j_range=(-np.inf, np.inf)):
        """
        Fits a gradient to V vs j from the given DataFrame.

        Parameters:
        -----------

        restrict_to_zero : bool, optional, default=False
            If True, the fit will be forced to go through (0, 0).

        j_range : tuple, optional, default=None
            A tuple of the form (min_j, max_j) to restrict the j values for the fit.

        Returns:
        --------
        ASR : float
            The ASR, units Ohms cm2
        intercept : float
            The intercept of the fitted line (if restrict_to_zero=False).
        fit_values : np.ndarray
            The fitted V values.
        """

        # filter based on j_range
        mask = (self.j >= j_range[0]) & (self.j <= j_range[1])
        j_ASR = self.j[mask]
        V_ASR = self.V[mask]

        j_ASR = j_ASR / 1000  # convert from mA/cm2 to A/cm2

        x = j_ASR
        y = V_ASR

        # Reshape x for scikit-learn (since it expects a 2D array for features)
        x_reshaped = x.reshape(-1, 1)

        # Initialize the LinearRegression model
        model = LinearRegression(fit_intercept=not restrict_to_zero)

        # Fit the model
        model.fit(x_reshaped, y)

        # Get the slope and intercept
        ASR = model.coef_[0]  # units are Ohm cm2
        intercept = model.intercept_ if not restrict_to_zero else 0

        # Calculate predicted values
        y_pred = model.predict(x_reshaped)

        # Calculate residuals
        residuals = y - y_pred

        # Calculate standard error of the slope
        n = len(x)
        error = np.sqrt(np.sum(residuals**2) / (n - 2)) / np.sqrt(
            np.sum((x - np.mean(x)) ** 2)
        )

        # pass model fit to class
        PolarisationData_fit = PolarisationData(V=y_pred, j=x * 1000, A=self.A)

        self.ASR = ASR
        self.ASR_intercept = intercept
        self.ASR_error = error

        return PolarisationData_fit
