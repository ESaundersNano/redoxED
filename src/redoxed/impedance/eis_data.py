import numpy as np


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

    Z: np.ndarray
    f: np.ndarray
    Z_re: np.ndarray
    Z_im: np.ndarray
    label: str | None

    def __init__(self, Z: np.ndarray, f: np.ndarray, label: str = None) -> None:
        """
        Initializes the EISData object.

        Args:
            Z (np.ndarray): Complex impedance data.
            f (np.ndarray): Frequency data corresponding to the impedance.
            label (str, optional): Optional label for the dataset. Defaults to None.
        """
        self.Z = Z
        self.f = f
        self.Z_re = self.Z.real
        self.Z_im = self.Z.imag
        self.label = label  # Optional label for the data

        self._validate()

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
            #     # Check if data arrays are set

        #     if self.Z_re is None or self.Z_im is None or self.f is None:
        #         raise ValueError(
        #             "The Real (Z_re), Imaginary (Z_im) impedance, and frequency (f) arrays must be provided."
        #         )
        #     # Check if frequencies are positive
        #     if jnp.any(self.f <= 0):
        # should probably check Z are complex and maybe even set type
        #         raise ValueError("Frequencies (f) must be strictly positive.")
        #     # Check for NaN or infinite values

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
        self.Z_re = self.Z.real
        self.Z_im = self.Z.imag

    def trim_inductive(self) -> None:
        """
        Trims the inductive part of the data where the imaginary part of the impedance is negative.
        """
        mask = self.Z_im < 0
        self.Z = self.Z[mask]
        self.f = self.f[mask]
        self.Z_re = self.Z.real
        self.Z_im = self.Z.imag

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
