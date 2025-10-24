import numpy as np


class ResidualsData:
    """
    Container for residuals data, similar to CyclingData.
    Can be initialized empty or with frequency, residuals, and relative residuals.
    Stores absolute and relative residuals for real and imaginary parts if provided.
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
        label: str | None = None,
    ):
        self.f = f
        self.residuals = residuals
        self.residuals_rel = residuals_rel
        self.residuals_re = residuals_re
        self.residuals_im = residuals_im
        self.residuals_re_rel = residuals_re_rel
        self.residuals_im_rel = residuals_im_rel
        self.label = label

    @classmethod
    def calculate_residuals(cls, observed, predicted, label: str | None = None):
        """
        Calculate residuals from two EISData objects.
        observed: EISData (experimental or measured)
        predicted: EISData (model or fit)
        Returns ResidualsData with all residuals calculated.
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
        residuals_rel = 100 * residuals / abs_Z_obs
        residuals_re_rel = 100 * residuals_re / abs_Z_obs
        residuals_im_rel = 100 * residuals_im / abs_Z_obs
        return cls(
            f=f,
            residuals=residuals,
            residuals_rel=residuals_rel,
            residuals_re=residuals_re,
            residuals_im=residuals_im,
            residuals_re_rel=residuals_re_rel,
            residuals_im_rel=residuals_im_rel,
            label=label,
        )

    def __repr__(self) -> str:
        return (
            f"ResidualsData(label={self.label!r}, "
            f"f={'set' if self.f is not None else 'None'}, "
            f"residuals={'set' if self.residuals is not None else 'None'}, "
            f"residuals_rel={'set' if self.residuals_rel is not None else 'None'}, "
            f"residuals_re={'set' if self.residuals_re is not None else 'None'}, "
            f"residuals_im={'set' if self.residuals_im is not None else 'None'}, "
            f"residuals_re_rel={'set' if self.residuals_re_rel is not None else 'None'}, "
            f"residuals_im_rel={'set' if self.residuals_im_rel is not None else 'None'})"
        )

    def _validate(self) -> None:
        """
        Validate that all provided arrays have matching lengths (if not None).
        Raises ValueError if any mismatch is found.
        """
        arrays = [
            self.f,
            self.residuals,
            self.residuals_rel,
            self.residuals_re,
            self.residuals_im,
            self.residuals_re_rel,
            self.residuals_im_rel,
        ]
        lengths = [arr.shape[0] for arr in arrays if arr is not None]
        if lengths and not all(l == lengths[0] for l in lengths):
            raise ValueError(f"Array lengths do not match: {lengths}")
