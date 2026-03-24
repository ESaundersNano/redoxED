import numpy as np
from numpy import float64, complex128
from numpy.typing import NDArray
import copy


class EISData:
    """
    A class to represent Electrochemical Impedance Spectroscopy (EIS) data.

    Attributes:
        Z (np.ndarray): Complex impedance data.
        f (np.ndarray): Frequency data corresponding to the impedance.
        Z_re (np.ndarray): Real part of the impedance.
        Z_im (np.ndarray): Imaginary part of the impedance.
        label (str | None): Optional label for the dataset.
    """

    Z: NDArray[complex128]
    f: NDArray[float64]
    Z_re: NDArray[float64]
    Z_im: NDArray[float64]
    Z_mag: NDArray[float64]
    Z_phase: NDArray[float64]
    Z_phase_deg: NDArray[float64]
    label: str | None

    def __init__(
        self, Z: NDArray[complex128], f: NDArray[float64], label: str = None
    ) -> None:
        """
        Initializes the EISData object.

        Args:
            Z (np.ndarray): Complex impedance data.
            f (np.ndarray): Frequency data corresponding to the impedance.
            label (str, optional): Optional label for the dataset. Defaults to None.
        """
        self.Z = Z
        self.f = f
        # Crop to f > 0 and calculate Z_quants
        mask = (self.f > 0) & (self.f <= np.inf)
        self.Z = self.Z[mask]
        self.f = self.f[mask]
        self._calculate_z_quants()

        self.label = label  # Optional label for the data

        self._validate()

    def _calculate_z_quants(self) -> None:
        """
        Calculates the real and imaginary parts of the impedance from the complex impedance data.
        """
        self.Z_re = self.Z.real
        self.Z_im = self.Z.imag
        self.Z_mag = np.abs(self.Z)
        self.Z_phase = np.angle(self.Z)
        self.Z_phase_deg = np.angle(self.Z, deg=True)

    def _validate(self) -> None:
        """
        Validates the input data to ensure consistency and correctness.

        Raises:
            ValueError: If Z and f do not have the same shape.
            ValueError: If Z and f are not 1D arrays.
            ValueError: If Z or f arrays are empty.
        """
        if self.Z.shape != self.f.shape:
            raise ValueError("Z and f must have the same shape.")
        if len(self.Z.shape) != 1:
            raise ValueError("Z and f must be 1D arrays.")
        if self.Z.size == 0 or self.f.size == 0:
            raise ValueError("Z and f arrays must not be empty.")
        try:
            self.f = np.asarray(self.f, dtype=np.float64)
            self.Z = np.asarray(self.Z, dtype=np.complex128)
        except Exception as e:
            raise ValueError("Failed to convert tau and gamma to float64.") from e

    def __repr__(self) -> str:
        """
        Returns a string representation of the EISData object.

        Returns:
            str: String representation of the object.
        """
        return f"EISData(Z={self.Z}, f={self.f})"

    def crop_frequency(self, f_min: float, f_max: float) -> None:
        """
        Crops the frequency range of the data to the specified range.

        Args:
            f_min (float): Minimum frequency to retain.
            f_max (float): Maximum frequency to retain.
        """
        mask = (self.f >= f_min) & (self.f <= f_max)
        self.Z = self.Z[mask]
        self.f = self.f[mask]
        self._calculate_z_quants()

    def trim_inductive(self) -> None:
        """
        Trims the inductive part of the data where the imaginary part of the impedance is negative.
        """
        mask = self.Z_im < 0
        self.Z = self.Z[mask]
        self.f = self.f[mask]
        self._calculate_z_quants()

    def sort_frequency_descending(self) -> None:
        """
        Sorts the frequency array (self.f) in descending order and rearranges
        the impedance array (self.Z) to match the new order.
        """
        # Get sorted indices for descending order
        sorted_indices = np.argsort(self.f)[::-1]
        self.f = self.f[sorted_indices]
        self.Z = self.Z[sorted_indices]
        self._calculate_z_quants()

    def estimate_real_intercept(self) -> float:
        """
        Estimates the real intercept of the data by identifying the highest frequency
        points straddling the real axis and performing linear interpolation.

        Returns:
            float: The estimated real intercept (Z_re) where Z_im = 0.

        Raises:
            ValueError: If there are not enough points to perform the interpolation.
        """
        # Find indices where Z_im changes sign
        sign_changes = np.where(np.diff(np.sign(self.Z_im)) != 0)[0]

        if len(sign_changes) == 0:
            raise ValueError("No points straddle the real axis (Z_im = 0).")

        # Determine the indices of the two points for interpolation
        if self.f[0] < self.f[1]:  # Low to high frequency
            idx1 = sign_changes[-1]  # Last sign change
            idx2 = idx1 + 1
        elif self.f[0] > self.f[1]:  # High to low frequency
            idx1 = sign_changes[0]  # First sign change
            idx2 = idx1 + 1

        # Get the real and imaginary parts of the two points
        Z_re1, Z_im1 = self.Z_re[idx1], self.Z_im[idx1]
        Z_re2, Z_im2 = self.Z_re[idx2], self.Z_im[idx2]

        # Perform linear interpolation to find the real intercept (Z_im = 0)
        Z_re_intercept = Z_re1 - (Z_im1 * (Z_re2 - Z_re1)) / (Z_im2 - Z_im1)

        return Z_re_intercept

    def copy(self):
        """Return a deep copy of this EISData instance."""
        return copy.deepcopy(self)
