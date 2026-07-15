import numpy as np
from numpy.typing import NDArray
from numpy import float64
from scipy.integrate import trapezoid
import copy


class DRTData:
    """
    Container for Distribution of Relaxation Times (DRT) analysis data.

    Stores the DRT distribution (gamma vs tau) computed from impedance spectroscopy measurements.
    The DRT provides a description of relaxation time distribution representing the electrochemical
    processes occurring at different time scales.

    Attributes:
        tau (NDArray[float64]): Array of relaxation times in s (seconds).
        gamma (NDArray[float64]): Distribution values (gamma in log-tau space) in Ω (Ohms).
        label (str | None): Optional label for this dataset.
        R0 (float64): Ohmic resistance in Ω (Ohms). NaN if not determined.
        L0 (float64): Inductance in H (Henries). NaN if not determined.
        C0 (float64): Capacitance in F (Farads). NaN if not determined.

    Notes:
        Unless otherwise stated, gamma refers to the DRT in log-tau space.
        All quantities follow SI units except where explicitly noted.
    """

    tau: NDArray[float64]
    gamma: NDArray[float64]
    label: str | None
    R0: float64
    L0: float64
    C0: float64

    def __init__(
        self,
        tau: NDArray[float64],
        gamma: NDArray[float64],
        R0: float64 = np.nan,
        L0: float64 = np.nan,
        C0: float64 = np.nan,
        label: str | None = None,
    ) -> None:
        """
        Initialize DRTData container.

        Parameters:
            tau (NDArray[float64]): Array of relaxation times in s (seconds).
            gamma (NDArray[float64]): Array of gamma values (distribution in log-tau space) in Ω (Ohms).
            R0 (float64): Ohmic resistance in Ω (Ohms). Defaults to NaN.
            L0 (float64): Inductance in H (Henries). Defaults to NaN.
            C0 (float64): Capacitance in F (Farads). Defaults to NaN.
            label (str | None): Optional label for this dataset. Defaults to None.
        """
        self.tau = tau
        self.gamma = gamma
        self.label = label
        self.R0 = R0
        self.L0 = L0
        self.C0 = C0
        self._validate()

    def _validate(self) -> None:
        """
        Validates the input data to ensure consistency and correctness.

        Raises:
            ValueError: If tau and gamma do not have the same shape.
            ValueError: If tau and gamma are not 1D arrays.
            ValueError: If tau or gamma arrays are empty.
        """
        if not isinstance(self.tau, np.ndarray):
            raise TypeError("tau must be a numpy array.")
        if not isinstance(self.gamma, np.ndarray):
            raise TypeError("gamma must be a numpy array.")
        if self.tau.shape != self.gamma.shape:
            raise ValueError("tau and gamma must have the same shape.")
        if len(self.tau.shape) != 1:
            raise ValueError("tau and gamma must be 1D arrays.")
        if self.tau.size == 0 or self.gamma.size == 0:
            raise ValueError("tau and gamma arrays must not be empty.")
        try:
            self.tau = np.asarray(self.tau, dtype=np.float64)
            self.gamma = np.asarray(self.gamma, dtype=np.float64)
        except Exception as e:
            raise ValueError("Failed to convert tau and gamma to float64.") from e

        for param, name in zip([self.R0, self.L0, self.C0], ["R0", "L0", "C0"]):
            if not isinstance(param, (float, np.floating)):
                raise TypeError(f"{name} must be a float or float64.")

    def get_pol_resistance(self) -> float64:
        """
        Calculate the total polarization resistance (Zp) by integrating the DRT.

        Computes the area under the gamma vs ln(tau) curve using trapezoidal integration.
        The polarization resistance is the integral of gamma over the entire tau range
        in log-tau space.

        Returns:
            float64: Total polarization resistance (area under curve) in Ω (Ohms).
        """
        tau = self.tau.copy()
        gamma = self.gamma.copy()
        # Ensure tau is in ascending order
        if not np.all(np.diff(tau) > 0):
            sort_idx = np.argsort(tau)
            tau = tau[sort_idx]
            gamma = gamma[sort_idx]
        ln_tau = np.log(tau)
        pol_resistance = np.trapezoid(gamma, ln_tau)
        return pol_resistance

    def get_DRT_integral(self, tau_min: float, tau_max: float) -> float64:
        """
        Calculate partial polarization resistance over a specified tau range.

        Computes the integral of gamma with respect to log(tau) between two specified
        relaxation time values. Useful for isolating contributions from specific
        electrochemical processes.

        Parameters:
            tau_min (float): Minimum relaxation time in s (seconds) for integration range.
            tau_max (float): Maximum relaxation time in s (seconds) for integration range.

        Returns:
            float64: Partial polarization resistance in Ω (Ohms) between tau_min and tau_max.
                Returns NaN if no tau values fall within the specified range.
        """
        tau = self.tau.copy()
        gamma = self.gamma.copy()

        # Ensure tau is in ascending order
        if not np.all(np.diff(tau) > 0):
            sort_idx = np.argsort(tau)
            tau = tau[sort_idx]
            gamma = gamma[sort_idx]

        # Filter to specified tau range
        mask = (tau >= tau_min) & (tau <= tau_max)
        tau_filtered = tau[mask]
        gamma_filtered = gamma[mask]

        # Handle case where no tau values fall within range
        if len(tau_filtered) == 0:
            return np.nan

        # Calculate integral in log-tau space
        ln_tau = np.log(tau_filtered)
        resistance_integral = np.trapezoid(gamma_filtered, ln_tau)

        return resistance_integral

    def copy(self) -> "DRTData":
        """
        Return a deep copy of this DRTData instance.

        Returns:
            DRTData: A new independent copy of this object.
        """
        return copy.deepcopy(self)
