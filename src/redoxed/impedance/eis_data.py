"""Electrochemical Impedance Spectroscopy (EIS) data container and utilities."""

import numpy as np
from numpy import float64, complex128
from numpy.typing import NDArray
import copy


class EISData:
    """
    Container for Electrochemical Impedance Spectroscopy (EIS) data and derived quantities.

    Stores complex impedance measurements and their frequency points, automatically computing
    and storing real/imaginary components, magnitude, and phase for convenient access.

    Attributes:
        Z (NDArray[complex128]): Complex impedance data in Ω (Ohms).
        f (NDArray[float64]): Frequency data in Hz (Hertz) corresponding to impedance points.
        Z_re (NDArray[float64]): Real part (Z') of impedance in Ω (Ohms).
        Z_im (NDArray[float64]): Imaginary part (Z") of impedance in Ω (Ohms).
        Z_mag (NDArray[float64]): Magnitude |Z| of impedance in Ω (Ohms).
        Z_phase (NDArray[float64]): Phase angle of impedance in radians.
        Z_phase_deg (NDArray[float64]): Phase angle of impedance in degrees.
        label (str | None): Optional label for this dataset.

    Notes:
        Frequency data is automatically filtered to f > 0 during initialization.
        All complex quantities are automatically computed from Z during initialization.
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
        self, Z: NDArray[complex128], f: NDArray[float64], label: str | None = None
    ) -> None:
        """
        Initialize EISData container.

        Parameters:
            Z (NDArray[complex128]): Complex impedance array in Ω (Ohms).
            f (NDArray[float64]): Frequency array in Hz (Hertz).
            label (str | None): Optional label for this dataset. Defaults to None.
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
        Calculate derived impedance quantities from complex impedance data.

        Computes real part (Z'), imaginary part (Z"), magnitude (|Z|), and phase
        from the complex impedance array.
        """
        self.Z_re = self.Z.real
        self.Z_im = self.Z.imag
        self.Z_mag = np.abs(self.Z)
        self.Z_phase = np.angle(self.Z)
        self.Z_phase_deg = np.angle(self.Z, deg=True)

    def _validate(self) -> None:
        """
        Validate input data for consistency and correctness.

        Raises:
            ValueError: If Z and f arrays have different shapes.
            ValueError: If Z or f are not 1D arrays.
            ValueError: If Z or f arrays are empty.
            ValueError: If Z or f cannot be converted to correct dtype.
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
        String representation of the EISData object.

        Returns:
            str: Representation with impedance and frequency array shapes.
        """
        return f"EISData(Z={self.Z}, f={self.f})"

    def crop_frequency(self, f_min: float, f_max: float) -> None:
        """
        Restrict frequency range to specified limits.

        Removes all data points outside the range [f_min, f_max] and recomputes
        all derived impedance quantities.

        Parameters:
            f_min (float): Minimum frequency in Hz (Hertz) to retain.
            f_max (float): Maximum frequency in Hz (Hertz) to retain.
        """
        mask = (self.f >= f_min) & (self.f <= f_max)
        self.Z = self.Z[mask]
        self.f = self.f[mask]
        self._calculate_z_quants()

    def trim_inductive(self) -> None:
        """
        Remove inductive contribution where Z" (imaginary part) is negative.

        Removes all data points where Z_im < 0, typically corresponding to the
        high-frequency inductive tail region of EIS data.
        """
        mask = self.Z_im < 0
        self.Z = self.Z[mask]
        self.f = self.f[mask]
        self._calculate_z_quants()

    def sort_frequency_descending(self) -> None:
        """
        Sort data by frequency in descending order.

        Rearranges all data arrays (Z, f, and derived quantities) in descending
        order of frequency to match typical EIS convention.
        """
        # Get sorted indices for descending order
        sorted_indices = np.argsort(self.f)[::-1]
        self.f = self.f[sorted_indices]
        self.Z = self.Z[sorted_indices]
        self._calculate_z_quants()

    def estimate_real_intercept(self) -> float:
        """
        Estimate real axis intercept (Z_re where Z_im = 0) by linear interpolation.

        Identifies frequency points where Z_im crosses zero and interpolates between
        them to estimate the real impedance at infinite frequency.

        Returns:
            float: Estimated Z_re (real intercept) in Ω (Ohms) where Z_im = 0.

        Raises:
            ValueError: If no data points straddle the real axis (no Z_im sign change).
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

    def copy(self) -> "EISData":
        """
        Return a deep copy of this EISData instance.

        Returns:
            EISData: A new independent copy of this object.
        """
        return copy.deepcopy(self)
