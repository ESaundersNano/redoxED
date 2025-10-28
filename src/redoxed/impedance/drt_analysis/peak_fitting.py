from lmfit.minimizer import minimize, MinimizerResult
from lmfit.parameter import Parameters
import numpy as np
from scipy.signal import find_peaks
from redoxed.impedance.drt_analysis import HN_DRT, SG_DRT
from typing import List, Tuple, Dict
from numpy.typing import NDArray
from numpy import float64
from dataclasses import dataclass
from scipy.integrate import quad
import pandas as pd
from redoxed.impedance import DRTData


def find_DRT_peaks(
    DRTData_object: DRTData,
    num_peaks: int | None = None,
    find_peaks_settings: dict | None = dict(
        height=None,
        threshold=None,
        distance=None,
        prominence=None,
        width=None,
        wlen=None,
        rel_height=0.5,
        plateau_size=None,
    ),
) -> List[Tuple[float, float]]:

    if num_peaks is None:
        pass
    elif isinstance(num_peaks, int):
        if num_peaks <= 0:
            raise ValueError("num_peaks must be > 0")
    else:
        raise TypeError("num_peaks must be int or None")

    tau = DRTData_object.tau.copy()
    log_tau = np.log(tau)

    gamma = DRTData_object.gamma.copy()
    # pad for boundary peaks as peaks need to be higher than immediate neighbours
    gamma_ext = np.zeros(len(gamma) + 2, dtype=float64)
    gamma_ext[1:-1] += gamma

    # default peak finding settings
    default_settings = dict(
        height=None,
        threshold=None,
        distance=None,
        prominence=None,
        width=None,
        wlen=None,
        rel_height=0.5,
        plateau_size=None,
    )
    # If user provides settings, update defaults
    if find_peaks_settings is not None:
        settings = default_settings.copy()
        default_settings.update(find_peaks_settings)
    else:
        settings = default_settings

    # find peaks
    peaks, _ = find_peaks(gamma_ext, **settings)

    peaks_array = []
    for i in peaks:
        i -= 1  # adjust for padding
        peaks_array.append((tau[i], gamma[i]))

    # keep tallest num_peaks if given (not perfect as peak area is teh quantity that matters)
    if num_peaks != None:
        peaks_array.sort(key=lambda t: t[1], reverse=True)
        peaks_array = peaks_array[:num_peaks]
    # sort peaks in ascending tau
    peaks_array.sort(key=lambda t: t[0])

    return peaks_array


def _generate_parameters(
    peaks: List[Tuple[float64, float64]],
    peak_type: str,
    skew: bool,
    log_tau0_bound: float | None = None,
) -> Tuple[Parameters, int]:  # noqa: F821

    if len(peaks) < 1:
        raise ValueError(f"Expected to have at least one peak to analyze!")

    if log_tau0_bound is None:
        pass
    elif isinstance(log_tau0_bound, float):
        if log_tau0_bound <= 0:
            raise ValueError("log_tau0_bound must be > 0")
    else:
        raise TypeError("log_tau0_bound must be float or None")

    parameters: Parameters = Parameters()
    num_variables: int = 0

    i: int
    log_tau_pos: float64
    gamma_pos: float64

    for i, (log_tau_pos, gamma_pos) in enumerate(peaks):

        # add log_tau0 (peak position)
        if log_tau0_bound is not None:
            kw = dict(
                name=f"log_tau0_{i}",
                value=log_tau_pos,
                min=log_tau_pos - log_tau0_bound,
                max=log_tau_pos + log_tau0_bound,
            )
            # # adjust limits if they are out of bounds
            # if isclose(kw["min"], 0.0):
            #     kw["min"] = -1e-10
            # if isclose(kw["max"], 1.0):
            #     kw["max"] = 1.0 + 1e-10
            parameters.add(**kw)
        else:
            kw = dict(
                name=f"log_tau0_{i}",
                value=log_tau_pos,
                min=log_tau_pos * 0.99,
                max=log_tau_pos * 1.01,
            )
            # # adjust limits if they are out of bounds
            # if isclose(kw["min"], 0.0):
            #     kw["min"] = -1e-10
            # if isclose(kw["max"], 1.0):
            #     kw["max"] = 1.0 + 1e-10
            parameters.add(**kw)

        if peak_type == "HN":
            parameters.add(
                name=f"Z0_{i}",
                value=gamma_pos,  # Z0 is actually area rather than height, but need something indicative
                min=0.0,
                # no max set
            )

            parameters.add(
                name=f"alpha_{i}",
                value=0.9,
                min=0.0,
                max=1.0,
            )

            parameters.add(
                name=f"beta_{i}",
                value=1,
                min=0.0,
                max=1.0,
                vary=skew,  # allow to vary if skew is permitted
            )

            num_variables += 4 if skew else 3

        elif peak_type == "SG":
            parameters.add(
                name=f"height_{i}",
                value=gamma_pos,
                min=0.0,
                max=2.0 * gamma_pos,  # limit to 2*peak height
            )

            parameters.add(
                name=f"upsilon_{i}",
                value=0.0,
                min=-1.0,
                max=1.0,
                vary=skew,  # allow to vary if skew is permitted
            )

            parameters.add(
                name=f"sigma_{i}",
                value=0.05,
                min=1e-10,
                max=1e2,
            )
            num_variables += 4 if skew else 3

    return (parameters, num_variables)


def _function(
    log_tau: NDArray[float64], parameters: Parameters, peak_type: str  # noqa: F821
) -> NDArray[float64]:
    params = parameters.valuesdict()

    gamma_fit: NDArray[float64] = np.zeros(len(log_tau), dtype=float64)

    if peak_type == "HN":
        tau = np.exp(log_tau)  # required for form of HN
        i = 0
        while True:
            try:
                log_tau0 = params[f"log_tau0_{i}"]
                tau0 = np.exp(log_tau0)
            except KeyError:
                break
            Z0 = params[f"Z0_{i}"]
            alpha = params[f"alpha_{i}"]
            beta = params[f"beta_{i}"]

            gamma_fit += HN_DRT(tau, Z0, tau0, alpha, beta)
            i += 1
    elif peak_type == "SG":
        i = 0
        while True:
            try:
                log_tau0 = params[f"log_tau0_{i}"]
            except KeyError:
                break
            height = params[f"height_{i}"]
            upsilon = params[f"upsilon_{i}"]
            sigma = params[f"sigma_{i}"]

            gamma_fit += SG_DRT(log_tau, height, log_tau0, upsilon, sigma)
            i += 1
    return gamma_fit


def _residual(
    parameters: Parameters,  # noqa: F821
    log_tau: NDArray[float64],
    gamma: NDArray[float64],
    peak_type: str,
) -> NDArray[float64]:
    """Decided against relative residual due to divide by 0 with gamma"""
    return _function(log_tau, parameters, peak_type) - gamma


@dataclass(frozen=True)
class DRTPeak:
    """ """

    label: str
    peak_params: dict

    def get_gamma(self, tau: float | NDArray) -> float | NDArray:
        """ """
        # Convert to numpy array for consistent handling
        tau = np.asarray(tau)

        if self.peak_params["peak_type"] == "HN":
            tau0 = np.exp(self.peak_params["log_tau0"])
            gamma = HN_DRT(
                tau=tau,
                Z0=self.peak_params["Z0"],
                tau0=tau0,
                alpha=self.peak_params["alpha"],
                beta=self.peak_params["beta"],
            )
        elif self.peak_params["peak_type"] == "SG":
            log_tau = np.log(tau)
            gamma = SG_DRT(
                log_tau=log_tau,
                height=self.peak_params["height"],
                log_tau0=self.peak_params["log_tau0"],
                upsilon=self.peak_params["upsilon"],
                sigma=self.peak_params["sigma"],
            )

        return gamma

    def get_Z(self, quad_opts: Dict | None = None) -> float64:
        """ """
        if self.peak_params["peak_type"] == "HN":
            return self.peak_params["Z0"]

        elif self.peak_params["peak_type"] == "SG":
            # Default quad options
            default_opts = {
                "a": -50,
                "b": 50,
                "epsabs": 1e-9,
                "epsrel": 1e-9,
                "limit": 100,
            }
            quad_opts = quad_opts or {}
            opts = {**default_opts, **quad_opts}
            a = opts["a"]
            b = opts["b"]
            quad_args = {k: v for k, v in opts.items() if k not in ["a", "b"]}

            def integrand(log_tau):
                return self.get_gamma(
                    np.exp(log_tau)
                )  # integrating wrt log_tau but made get_gamma input tau for consistency

            Z_component, _ = quad(integrand, a, b, **quad_args)
            return Z_component


@dataclass(frozen=True)
class DRTPeaks:
    """ """

    peaks: List[DRTPeak]
    fit_summary: dict = None

    def __iter__(self):
        return iter(self.peaks)

    def get_num_peaks(self) -> int:
        """
        Get the number of peaks.

        Returns
        -------
        int
        """
        return len(self.peaks)

    def get_gamma(
        self,
        tau: float | NDArray,
        peak_indices: List[int] | None = None,
    ) -> NDArray:
        num_peaks = self.get_num_peaks()
        if peak_indices is None:
            pass
        elif not isinstance(peak_indices, (list, tuple, np.ndarray)):
            raise TypeError(
                f"Expected a list, tuple, or numpy array of integers instead of {peak_indices=}"
            )
        elif not all(isinstance(i, int) for i in peak_indices):
            raise TypeError(
                f"All elements in peak_indices must be integers: {peak_indices=}"
            )
        elif not all(0 <= i < num_peaks for i in peak_indices):
            raise ValueError(f"Expected 0 <= {peak_indices=} < {num_peaks}")

        gamma_fit: NDArray = np.zeros(tau.shape, dtype=float64)

        i: int
        peak: DRTPeak
        for i, peak in enumerate(self.peaks):
            if peak_indices and i not in peak_indices:
                continue
            # If we get here, either peak_indices is None/empty, or i is in peak_indices
            gamma_fit += peak.get_gamma(tau=tau)
        return gamma_fit

    def get_peak_Z(
        self,
        index: int,
    ) -> float64:
        """
        Note only uses the default integration parameters for SG peaks. May be able to improve guess by calling get_Z.
        """
        num_peaks = len(self.peaks)
        if not isinstance(index, int):
            raise TypeError(f"Index must be an integer, got {type(index)}")
        if not (0 <= index < num_peaks):
            raise IndexError(
                f"Index {index} is out of range for peaks (0 to {num_peaks-1})"
            )
        peak: DRTPeak = self.peaks[index]

        return peak.get_Z()

    def to_peaks_df(
        self,
        peak_indices: List[int] | None = None,
    ) -> pd.DataFrame:
        """
        Returns a DataFrame with each row representing a peak.
        Columns: label, all peak_params, tau0 (np.exp(log_tau0)), and Z (peak.get_Z()).
        Note only uses default integration parameters for SG peaks. May be able to refine guess for peak Z using get_Z on DRTPeak objects directly.
        """
        if peak_indices is None:
            selected_peaks = self.peaks
        else:
            selected_peaks = [self.peaks[i] for i in peak_indices]

        rows = []
        for peak in selected_peaks:
            row = {"label": peak.label}
            row.update(peak.peak_params)
            if "log_tau0" in row:
                row["tau0"] = np.exp(row["log_tau0"])
            row["Z"] = peak.get_Z()
            rows.append(row)

        return pd.DataFrame(rows)


def fit_DRT_peaks(
    DRTData_object: DRTData,
    peak_positions: List[float] | NDArray[float64] | None,
    num_peaks: int | None = None,
    peak_type: str = "HN",
    skew: bool = True,
    log_tau0_bound: (
        float | None
    ) = 0.1,  # base e not 10. 0.1 allows about 10 % variation from estimate - None mostly prevents variation
    minimizer_settings: dict = None,  # e.g. input matching default would be {"method": "leastsq", "fit_kws": {"ftol": 1e-8, "xtol": 1e-8, "gtol": 1e-8}}. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html
) -> DRTPeaks:

    tau = DRTData_object.tau.copy()
    gamma = DRTData_object.gamma.copy()

    # Ensure only one of num_peaks or peak_positions is provided
    if (num_peaks is not None) and (peak_positions is not None):
        raise ValueError(
            "Provide either num_peaks or peak_positions, or neither, but not both."
        )

    if peak_type not in {"HN", "SG"}:
        raise ValueError(f"peak_type must be 'HN' or 'SG', got '{peak_type}'")

    # return empty result if gamma empty
    if np.allclose(gamma, 0.0):
        return DRTPeaks(
            peaks=[],
            fit_summary="No fit",
        )

    # convert to log space (base e)
    log_tau: NDArray[float64] = np.log(tau)

    # # pyimpspec chose to use base 10 and scale for optimiser.
    # # Whilst conventional for peak fitting with gaussians, this is awkward for the HN basis.
    # # Therefore, peak fitting is done in ln(tau) space without scaling.
    # # Use of log spacing helps minimiser stability + convergence.

    # find peaks if not provided
    if peak_positions is None:
        peak_positions = find_DRT_peaks(
            DRTData_object=DRTData_object, num_peaks=num_peaks
        )
    elif peak_positions is not None and len(peak_positions) == 0:
        raise ValueError("peak_positions is an empty list/array.")

    # use peak positions (nearest in data)
    peaks: List[Tuple[float64, float64]] = []
    for pos_tuple in peak_positions:
        i = np.argmin(abs(tau - pos_tuple[0]))
        peaks.append((log_tau[i], gamma[i]))

    # generate peak parameters
    parameters: Parameters
    num_variables: int
    parameters, num_variables = _generate_parameters(
        peaks, peak_type=peak_type, skew=skew, log_tau0_bound=log_tau0_bound
    )

    # safeguard that peak fit isn't overfitting (most likely in the case that neither num_peaks nor peak_positions is provided)
    while len(peaks) > 1 and num_variables > gamma.size:
        peaks.sort(key=lambda t: t[1], reverse=True)
        peaks.pop()
        peaks.sort(key=lambda t: t[0])
        parameters, num_variables = _generate_parameters(
            peaks, peak_type=peak_type, skew=skew, log_tau0_bound=log_tau0_bound
        )

    # Set minimizer defaults
    method = "leastsq"
    fit_kws = {}

    if minimizer_settings is not None:
        method = minimizer_settings.get("method", method)
        fit_kws = minimizer_settings.get("fit_kws", fit_kws)

    fit: MinimizerResult = minimize(
        _residual,
        parameters,
        method=method,
        args=(log_tau, gamma, peak_type),
        **fit_kws,
    )
    params: Dict[str, float64] = fit.params.valuesdict()

    # Create fit summary
    fit_summary = {
        "success": fit.success,
        "message": fit.message,
        "chisqr": fit.chisqr,
        "redchi": fit.redchi,
        "nfev": fit.nfev,
    }

    peak_dict = {}  # fill with peak_i: dict of peak properties
    for k, v in params.items():
        if "_" in k:
            prop, idx = k.rsplit("_", 1)
            peak_key = f"peak_{idx}"
            if peak_key not in peak_dict:
                peak_dict[peak_key] = {}
                peak_dict[peak_key]["peak_type"] = peak_type
                peak_dict[peak_key]["peak_number"] = idx
            peak_dict[peak_key][prop] = v

    drt_peaks: List[DRTPeak] = []
    for peak_key, peak_params in peak_dict.items():
        drt_peaks.append(DRTPeak(label=peak_key, peak_params=peak_params))

    return DRTPeaks(
        peaks=drt_peaks,
        fit_summary=fit_summary,
    )
