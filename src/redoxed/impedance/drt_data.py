import numpy as np
from numpy.typing import NDArray
from numpy import float64
from scipy.integrate import trapezoid
import copy


class DRTData:
    """
    A class to store and process Distribution of Relaxation Times (DRT) data.

    Attributes:
        tau (np.ndarray): Array of relaxation times.
        gamma (np.ndarray): Array of gamma values corresponding to tau.
        label (Optional[str]): An optional label for the dataset.

    Notes:
    Unless otherwise stated. It is always assumed that gamma refers to the gamma in log-tau space.
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
        label=None,
    ) -> None:
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
        Calculate the peak area (Zp) for this spectrum.
        Parameters
        ----------


        Returns
        -------
        float64
            Area under the spectrum (Zp value).
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

    def copy(self):
        """Return a deep copy of this DRTData instance."""
        return copy.deepcopy(self)
