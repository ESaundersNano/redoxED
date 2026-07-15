"""Container for impedance fitting residuals with cartesian and polar representations."""

import numpy as np


class ResidualsData:
    """
    Container for impedance fitting residuals data.

    Stores residuals computed between observed and predicted impedance spectra in both
    absolute and relative (percentage) forms, with representations in both cartesian
    (real/imaginary) and polar (magnitude/phase) coordinates.

    Attributes:
        f (NDArray[float64] | None): Frequency data in Hz (Hertz).
        residuals (NDArray[complex128] | None): Complex residuals (observed - predicted).
        residuals_rel (NDArray[complex128] | None): Complex residuals as percentage (%).
        residuals_re (NDArray[float64] | None): Real part residuals in Ω (Ohms).
        residuals_im (NDArray[float64] | None): Imaginary part residuals in Ω (Ohms).
        residuals_re_rel (NDArray[float64] | None): Real part residuals as percentage (%).
        residuals_im_rel (NDArray[float64] | None): Imaginary part residuals as percentage (%).
        residuals_mag (NDArray[float64] | None): Magnitude residuals in Ω (Ohms).
        residuals_phase (NDArray[float64] | None): Phase residuals in degrees.
        residuals_mag_rel (NDArray[float64] | None): Magnitude residuals as percentage (%).
        residuals_phase_rel (NDArray[float64] | None): Phase residuals as percentage (%).
        label (str | None): Optional label for this dataset.
    """

    def __init__(
        self,
        f: np.ndarray | None = None,
        residuals: np.ndarray | None = None,
        residuals_rel: np.ndarray | None = None,
        residuals_re: np.ndarray | None = None,
        residuals_im: np.ndarray | None = None,
        residuals_re_rel: np.ndarray | None = None,
        residuals_im_rel: np.ndarray | None = None,
        residuals_mag: np.ndarray | None = None,
        residuals_phase: np.ndarray | None = None,
        residuals_mag_rel: np.ndarray | None = None,
        residuals_phase_rel: np.ndarray | None = None,
        label: str | None = None,
    ) -> None:
        """
        Initialize ResidualsData container.

        Parameters:
            f (NDArray[float64] | None): Frequency data in Hz (Hertz). Defaults to None.
            residuals (NDArray[complex128] | None): Complex residuals. Defaults to None.
            residuals_rel (NDArray[complex128] | None): Complex residuals as percentage. Defaults to None.
            residuals_re (NDArray[float64] | None): Real part residuals in Ω (Ohms). Defaults to None.
            residuals_im (NDArray[float64] | None): Imaginary part residuals in Ω (Ohms). Defaults to None.
            residuals_re_rel (NDArray[float64] | None): Real part residuals as percentage (%). Defaults to None.
            residuals_im_rel (NDArray[float64] | None): Imaginary part residuals as percentage (%). Defaults to None.
            residuals_mag (NDArray[float64] | None): Magnitude residuals in Ω (Ohms). Defaults to None.
            residuals_phase (NDArray[float64] | None): Phase residuals in degrees. Defaults to None.
            residuals_mag_rel (NDArray[float64] | None): Magnitude residuals as percentage (%). Defaults to None.
            residuals_phase_rel (NDArray[float64] | None): Phase residuals as percentage (%). Defaults to None.
            label (str | None): Optional label for this dataset. Defaults to None.
        """
        self.f = f
        self.residuals = residuals
        self.residuals_rel = residuals_rel
        self.residuals_re = residuals_re
        self.residuals_im = residuals_im
        self.residuals_re_rel = residuals_re_rel
        self.residuals_im_rel = residuals_im_rel
        self.residuals_mag = residuals_mag
        self.residuals_phase = residuals_phase
        self.residuals_mag_rel = residuals_mag_rel
        self.residuals_phase_rel = residuals_phase_rel
        self.label = label

    @classmethod
    def calculate_residuals(
        cls, observed, predicted, label: str | None = None
    ) -> "ResidualsData":
        """
        Calculate residuals between observed and predicted impedance data.

        Computes residuals in both absolute (Ohms) and relative (percentage) forms,
        and in both cartesian (real/imaginary) and polar (magnitude/phase) representations.

        Parameters:
            observed: EISData object with observed impedance measurements.
            predicted: EISData object with predicted/fitted impedance values.
            label (str | None): Optional label for the residuals dataset. Defaults to None.

        Returns:
            ResidualsData: New ResidualsData object with all residuals calculated.

        Raises:
            ValueError: If input objects lack required 'f' and 'Z' attributes.
        """
        f = getattr(observed, "f", None)
        Z_obs = getattr(observed, "Z", None)
        Z_pred = getattr(predicted, "Z", None)
        if f is None or Z_obs is None or Z_pred is None:
            raise ValueError("EISData objects must have 'f' and 'Z' attributes.")
        residuals = Z_obs - Z_pred
        residuals_re = np.real(residuals)
        residuals_im = np.imag(residuals)
        abs_Z_obs = np.abs(Z_obs)
        residuals_mag = np.abs(Z_obs) - np.abs(Z_pred)
        residuals_phase = np.angle(Z_obs, deg=True) - np.angle(Z_pred, deg=True)
        residuals_rel = 100 * residuals / abs_Z_obs
        residuals_re_rel = 100 * residuals_re / abs_Z_obs
        residuals_im_rel = 100 * residuals_im / abs_Z_obs
        residuals_mag_rel = 100 * residuals_mag / abs_Z_obs
        residuals_phase_rel = 100 * residuals_phase / np.angle(Z_obs, deg=True)
        return cls(
            f=f,
            residuals=residuals,
            residuals_rel=residuals_rel,
            residuals_re=residuals_re,
            residuals_im=residuals_im,
            residuals_re_rel=residuals_re_rel,
            residuals_im_rel=residuals_im_rel,
            residuals_mag=residuals_mag,
            residuals_phase=residuals_phase,
            residuals_mag_rel=residuals_mag_rel,
            residuals_phase_rel=residuals_phase_rel,
            label=label,
        )

    def __repr__(self) -> str:
        """
        String representation of the ResidualsData object.

        Returns:
            str: Detailed representation showing which residual arrays are populated.
        """
        return (
            f"ResidualsData(label={self.label!r}, "
            f"f={'set' if self.f is not None else 'None'}, "
            f"residuals={'set' if self.residuals is not None else 'None'}, "
            f"residuals_rel={'set' if self.residuals_rel is not None else 'None'}, "
            f"residuals_re={'set' if self.residuals_re is not None else 'None'}, "
            f"residuals_im={'set' if self.residuals_im is not None else 'None'}, "
            f"residuals_re_rel={'set' if self.residuals_re_rel is not None else 'None'}, "
            f"residuals_im_rel={'set' if self.residuals_im_rel is not None else 'None'}, "
            f"residuals_mag={'set' if self.residuals_mag is not None else 'None'}, "
            f"residuals_phase={'set' if self.residuals_phase is not None else 'None'}, "
            f"residuals_mag_rel={'set' if self.residuals_mag_rel is not None else 'None'}, "
            f"residuals_phase_rel={'set' if self.residuals_phase_rel is not None else 'None'})"
        )

    def _validate(self) -> None:
        """
        Validate that all non-None arrays have consistent length.

        Raises:
            ValueError: If arrays with different lengths are present.
        """
        arrays = [
            self.f,
            self.residuals,
            self.residuals_rel,
            self.residuals_re,
            self.residuals_im,
            self.residuals_re_rel,
            self.residuals_im_rel,
            self.residuals_mag,
            self.residuals_phase,
            self.residuals_mag_rel,
            self.residuals_phase_rel,
        ]
        lengths = [arr.shape[0] for arr in arrays if arr is not None]
        if lengths and not all(l == lengths[0] for l in lengths):
            raise ValueError(f"Array lengths do not match: {lengths}")
