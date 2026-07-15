"""DRT peak detection and decomposition using Havriliak-Negami and Skewed Gaussian models.

This module provides functionality to identify peaks in Distribution of Relaxation Times (DRT)
spectra and fit them with parametric models. Supports both Havriliak-Negami (HN) and
Skewed Gaussian (SG) peak shapes with nonlinear least squares optimization.

Main functions:
    - find_DRT_peaks(): Detect local maxima in DRT spectrum
    - fit_DRT_peaks(): Optimize peak parameters to fit experimental DRT data
    - _generate_parameters(): Create lmfit Parameters with bounds for optimization
    - _function(): Evaluate fitted DRT spectrum
    - _residual(): Compute residuals for least squares optimization

Main classes:
    - DRTPeak: Single peak with parameters and evaluation methods
    - DRTPeaks: Collection of peaks with aggregate analysis methods

References:
    - Havriliak-Negami impedance: https://doi.org/10.1145/1950
    - DRT decomposition methods for EIS: https://doi.org/10.1016/j.electacta.2015.09.097
"""

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
    """
    Find peaks in the DRT spectrum using scipy.signal.find_peaks.

    Identifies local maxima in the gamma (DRT) spectrum and returns corresponding
    (tau, gamma) tuples. Optionally filters to keep only the most prominent peaks.

    Parameters:
        DRTData_object (DRTData): DRT data object containing tau and gamma arrays.
        num_peaks (int | None): Number of tallest peaks to keep. If None, keeps all found peaks. Default: None.
        find_peaks_settings (dict | None): Dictionary of settings for scipy.signal.find_peaks:
            - height (float | tuple): Peak height requirement(s)
            - threshold (float): Minimum difference to neighboring samples
            - distance (int): Minimum sample distance between peaks
            - prominence (float | tuple): Peak prominence requirement(s)
            - width (float | tuple): Peak width requirement(s)
            - wlen (int): Window length for evaluation
            - rel_height (float): Relative height for width calculation (default: 0.5)
            - plateau_size (int): Size of flat region at peak
            Default: All None except rel_height=0.5 (no restrictions)

    Returns:
        List[Tuple[float, float]]: List of (tau in s, gamma in Ω) tuples for each detected peak,
            sorted in ascending tau order.

    Notes:
        - Peaks are padded internally for boundary handling (scipy.signal.find_peaks requirement)
        - Padding ensures boundary peaks are properly detected
        - If num_peaks provided, keeps the num_peaks tallest peaks (by gamma value)
    """

    if num_peaks is None:
        pass
    elif isinstance(num_peaks, int):
        if num_peaks <= 0:
            raise ValueError("num_peaks must be > 0")
    else:
        raise TypeError("num_peaks must be int or None")

    tau = DRTData_object.tau.copy().astype(np.float64)
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
    assym_bound: float | None = None,
    dispersion_bounds: tuple | None = None,
) -> Tuple[Parameters, int]:
    """
    Generate lmfit.Parameters for each peak with bounds and initial guesses.

    Creates parameter objects for nonlinear least squares optimization, including
    peak position, height, width, and shape parameters. Initializes bounds based
    on peak properties and user constraints.

    Parameters:
        peaks (List[Tuple[float64, float64]]): List of (log(tau) in s, gamma in Ω) tuples for each peak.
        peak_type (str): Peak model type: "HN" (Havriliak-Negami) or "SG" (Skewed Gaussian).
        skew (bool): If True, allow skew/asymmetry parameter to vary during optimization.
        log_tau0_bound (float | None): Relative bound for log_tau0 parameter position (unitless fraction).
            If None, uses 1% bounds. If provided, bounds are ±log_tau0_bound relative to initial value.
            Example: 0.1 = 10% bounds on either side.
        assym_bound (float | None): Bound for asymmetry parameters (unitless, 0-1).
            If None, uses defaults: HN beta ∈ [0.5, 1], SG upsilon ∈ [-1, 1].
            For HN: sets beta_min = 1 - assym_bound, beta_max = 1.
            For SG: sets upsilon bounds to [-assym_bound, +assym_bound].
        dispersion_bounds (tuple | None): Bounds for dispersion/width parameters (unitless).
            If None, uses defaults: HN alpha ∈ [0.1, 1], SG sigma > 0.01 (ln units).

    Returns:
        Tuple[Parameters, int]: (lmfit.Parameters object with all peak parameters, count of free variables).

    Raises:
        ValueError: If peaks list is empty, or bound parameters are out of valid ranges.
        TypeError: If parameter types are incorrect.

    Notes:
        - Parameters are numbered sequentially (peak_0_log_tau0, peak_0_height, etc.)
        - Initial parameter values derived from detected peak positions and heights
        - Parameters linked to prevent negative heights/widths via constraints
    """

    if len(peaks) < 1:
        raise ValueError(f"Expected to have at least one peak to analyze!")

    if log_tau0_bound is None:
        pass
    elif isinstance(log_tau0_bound, float):
        if log_tau0_bound <= 0:
            raise ValueError("log_tau0_bound must be > 0")
    else:
        raise TypeError("log_tau0_bound must be float or None")

    if assym_bound is None:
        pass
    elif isinstance(assym_bound, (float, int)):
        if assym_bound < 0:
            raise ValueError("assym_bound must be >= 0")
        if assym_bound > 1:
            raise ValueError("assym_bound must be <= 1")
    else:
        raise TypeError("assym_bound must be float or None")

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

            if dispersion_bounds is not None:
                parameters.add(
                    name=f"alpha_{i}",
                    value=0.9,
                    min=dispersion_bounds[0],
                    max=dispersion_bounds[1],
                )
            else:
                parameters.add(
                    name=f"alpha_{i}",
                    value=0.9,
                    min=0.0,
                    max=1.0,
                )

            if assym_bound is not None:
                parameters.add(
                    name=f"beta_{i}",
                    value=1,
                    min=1.0 - assym_bound,
                    max=1.0,
                    vary=skew,  # allow to vary if skew is permitted
                )
            else:
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

            if assym_bound is not None:
                parameters.add(
                    name=f"upsilon_{i}",
                    value=0.0,
                    min=-assym_bound,
                    max=assym_bound,
                    vary=skew,  # allow to vary if skew is permitted
                )
            else:
                parameters.add(
                    name=f"upsilon_{i}",
                    value=0.0,
                    min=-1.0,
                    max=1.0,
                    vary=skew,  # allow to vary if skew is permitted
                )

            if dispersion_bounds is not None:
                parameters.add(
                    name=f"sigma_{i}",
                    value=0.05,
                    min=dispersion_bounds[0],
                    max=dispersion_bounds[1],
                )
            else:
                parameters.add(
                    name=f"sigma_{i}",
                    value=0.05,
                    min=1e-10,
                    max=1e2,
                )
            num_variables += 4 if skew else 3

    return (parameters, num_variables)


def _function(
    log_tau: NDArray[float64], parameters: Parameters, peak_type: str
) -> NDArray[float64]:
    """
    Compute the fitted DRT spectrum for given parameters and peak type.

    Combines multiple peaks (Havriliak-Negami or Skewed Gaussian) to generate
    the complete fitted DRT spectrum at specified relaxation times.

    Parameters:
        log_tau (NDArray[float64]): Array of log10(tau) values in s (seconds) where DRT is evaluated.
        parameters (Parameters): lmfit.Parameters object containing peak parameters
            (log_tau0, Z0/height, alpha/sigma, beta/upsilon for each peak).
        peak_type (str): Peak model type: "HN" (Havriliak-Negami) or "SG" (Skewed Gaussian).

    Returns:
        NDArray[float64]: Fitted gamma values in Ω (Ohms), same shape as log_tau.

    Notes:
        - For HN peaks: converts log_tau to tau (linear scale) for HN_DRT function
        - For SG peaks: operates in log-scale directly
        - Multiple peaks are summed to produce total DRT
        - Parameters indexed sequentially (peak_0, peak_1, etc.)
    """
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
    parameters: Parameters,
    log_tau: NDArray[float64],
    gamma: NDArray[float64],
    peak_type: str,
) -> NDArray[float64]:
    """
    Calculate residuals between experimental and fitted DRT spectra.

    Computes absolute difference |gamma_fit - gamma_exp| for least squares minimization.
    Uses absolute residuals rather than relative to avoid divide-by-zero issues.

    Parameters:
        parameters (Parameters): lmfit.Parameters object with current peak parameter values.
        log_tau (NDArray[float64]): Array of log10(tau) in s (seconds) where DRT is evaluated.
        gamma (NDArray[float64]): Experimental gamma values in Ω (Ohms).
        peak_type (str): Peak model type: "HN" (Havriliak-Negami) or "SG" (Skewed Gaussian).

    Returns:
        NDArray[float64]: Absolute residuals |gamma_fit - gamma_exp| in Ω, same shape as input arrays.

    Notes:
        - Called repeatedly during optimization by lmfit.minimize
        - Uses absolute value to avoid sign bias in fitting
        - Returned array squared and summed by lmfit to compute cost function
    """
    return np.abs(_function(log_tau, parameters, peak_type) - gamma)


@dataclass(frozen=True)
class DRTPeak:
    """
    Represents a single DRT peak with its parameters and evaluation methods.

    Container for all parameters describing one peak in the DRT decomposition,
    supporting both Havriliak-Negami and Skewed Gaussian models. Provides methods
    to evaluate gamma (DRT value) at specified relaxation times and compute peak area.

    Attributes:
        label (str): Descriptive label for the peak (e.g., "Peak 0", "HF process").
        peak_params (dict): Dictionary containing peak parameters:
            - "peak_type" (str): "HN" or "SG"
            - "log_tau0" (float): log10(tau_0) in s
            - "Z0" (float): HN peak height in Ω (for HN only)
            - "alpha" (float): HN dispersion parameter (for HN, unitless)
            - "beta" (float): HN asymmetry parameter (for HN, unitless)
            - "height" (float): SG peak height in Ω (for SG only)
            - "upsilon" (float): SG skewness parameter (for SG, unitless)
            - "sigma" (float): SG width in log-scale (for SG, unitless)
    """

    label: str
    peak_params: dict

    def get_gamma(self, tau: float | NDArray) -> float | NDArray:
        """
        Calculate gamma (DRT value) for this peak at given relaxation time(s).

        Evaluates the peak using the appropriate model (HN or SG) and stored parameters.

        Parameters:
            tau (float | NDArray[float64]): Relaxation time(s) in s (seconds) at which to evaluate gamma.

        Returns:
            float | NDArray[float64]: Gamma value(s) in Ω (Ohms). Type matches input (float or array).

        Notes:
            - For HN peaks: converts log_tau0 to linear tau for HN_DRT evaluation
            - For SG peaks: converts tau to log-scale for SG_DRT evaluation
            - Scalar input returns scalar output; array input returns array output
        """
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
        """
        Calculate the peak area (integral) for this peak.

        For Havriliak-Negami peaks, returns Z0 directly. For Skewed Gaussian peaks,
        integrates gamma over log(tau) domain to compute equivalent peak area.

        Parameters:
            quad_opts (dict | None): Dictionary of options for scipy.integrate.quad numerical integration.
                Only used for SG peaks. Default: None (uses preset values).
                May need adjustment if integration warnings occur; reduce 'a' and 'b' limits if interval too large.
                Valid keys:
                    - 'epsabs' (float): Absolute error tolerance (default: 1e-9)
                    - 'epsrel' (float): Relative error tolerance (default: 1e-9)
                    - 'limit' (int): Maximum number of subintervals (default: 100)
                    - 'a' (float): Lower integration limit log(tau) (default: -20, ~1e-8 s)
                    - 'b' (float): Upper integration limit log(tau) (default: 20, ~1e8 s)
                Example: {'epsabs': 1e-8, 'epsrel': 1e-8, 'limit': 200, 'a': -20, 'b': 5}

        Returns:
            float64: Peak area Z in Ω (Ohms). For HN peaks, equals Z0. For SG peaks, integral result.

        Notes:
            - HN: Direct return (constant time complexity)
            - SG: Requires numerical integration (requires ~100 function evaluations by default)
        """
        if self.peak_params["peak_type"] == "HN":
            return self.peak_params["Z0"]
        elif self.peak_params["peak_type"] == "SG":
            # Default quad options for integration
            default_opts = {
                "a": -20,
                "b": 20,
                "epsabs": 1e-9,
                "epsrel": 1e-9,
                "limit": 100,
            }
            quad_opts = quad_opts or {}
            opts = {**default_opts, **quad_opts}
            a = opts["a"]
            b = opts["b"]
            quad_args = {k: v for k, v in opts.items() if k not in ["a", "b"]}

            def integrand(log_tau: float) -> float:
                return self.get_gamma(
                    np.exp(log_tau)
                )  # wrt log_tau integration but get_gamma needs tau

            Z_component, _ = quad(integrand, a, b, **quad_args)
            return Z_component


@dataclass(frozen=True)
class DRTPeaks:
    """
    Container for a set of DRT peaks and their fit results.

    Aggregates multiple DRTPeak objects with fit quality metrics. Provides methods
    to evaluate total DRT (summed over peaks) and compute peak properties at specified
    relaxation times.

    Attributes:
        peaks (List[DRTPeak]): List of DRTPeak objects for each identified peak.
        fit_summary (dict | None): Dictionary containing fit quality metrics and optimization results.
            Typical keys: "success", "message", "nfev", "nvarys", "chi_square", "reduced_chi_square", etc.
    """

    peaks: List[DRTPeak]
    fit_summary: dict = None

    def __iter__(self) -> DRTPeak:
        """Allow iteration over contained peaks."""
        return iter(self.peaks)

    def get_num_peaks(self) -> int:
        """
        Get the number of peaks in this collection.

        Returns:
            int: Count of peaks in the peaks list.
        """
        return len(self.peaks)

    def get_gamma(
        self,
        tau: float | NDArray,
        peak_indices: List[int] | None = None,
    ) -> NDArray:
        """
        Sum gamma (DRT) values for selected peaks at given relaxation time(s).

        Evaluates each peak and sums their gamma values at specified tau values.
        Allows selective inclusion of peaks by index.

        Parameters:
            tau (float | NDArray[float64]): Relaxation time(s) in s (seconds) at which to evaluate gamma.
            peak_indices (List[int] | None): Indices of peaks to include in sum. If None, includes all peaks.
                Indices range from 0 to get_num_peaks()-1. Default: None (all peaks).

        Returns:
            NDArray[float64]: Summed gamma values in Ω (Ohms), shape matching tau input.

        Raises:
            TypeError: If peak_indices is not list/tuple/array or contains non-integers.
            ValueError: If peak_indices contains indices outside valid range [0, num_peaks).

        Notes:
            - Output shape matches input tau shape
            - For scalar tau, returns shape () array (compatible with arithmetic)
            - Peaks summed in order; peak_indices order does not affect result
        """
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

        gamma_fit: NDArray = np.zeros(np.asarray(tau).shape, dtype=float64)
        for i, peak in enumerate(self.peaks):
            if peak_indices and i not in peak_indices:
                continue
            # If we get here, either peak_indices is None/empty, or i is in peak_indices
            gamma_fit += peak.get_gamma(tau=tau)
        return gamma_fit

    def get_peak_Z(
        self,
        index: int,
        quad_opts: dict | None = None,
    ) -> float64:
        """
        Get the area (Z value) for a specific peak by index.

        Retrieves peak from collection and computes its area. For SG peaks, may require
        numerical integration (uses default or provided options).

        Parameters:
            index (int): Index of peak to query (0-based, range 0 to num_peaks-1).
            quad_opts (dict | None): scipy.integrate.quad options for SG peak integration. Default: None.

        Returns:
            float64: Peak area Z in Ω (Ohms).

        Raises:
            TypeError: If index is not an integer.
            IndexError: If index outside valid range [0, num_peaks).
        """
        num_peaks = len(self.peaks)
        if not isinstance(index, int):
            raise TypeError(f"Index must be an integer, got {type(index)}")
        if not (0 <= index < num_peaks):
            raise IndexError(
                f"Index {index} is out of range for peaks (0 to {num_peaks-1})"
            )
        peak: DRTPeak = self.peaks[index]
        return peak.get_Z(quad_opts=quad_opts)

    def to_peaks_df(
        self,
        peak_indices: List[int] | None = None,
        quad_opts: dict | None = None,
    ) -> pd.DataFrame:
        """
        Export peaks to DataFrame with each row representing one peak.

        Columns include: label, all peak_params keys, tau0 (linear from log_tau0), and Z (computed area).
        Useful for analysis, visualization, and export to files.

        Parameters:
            peak_indices (List[int] | None): Indices of peaks to include. If None, includes all peaks. Default: None.
            quad_opts (dict | None): scipy.integrate.quad options for SG peak area calculation. Default: None (uses defaults).

        Returns:
            pd.DataFrame: DataFrame with one row per peak. Columns:
                - "label" (str): Peak label/identifier
                - All peak_params keys ("peak_type", "log_tau0", "Z0"/"height", "alpha"/"sigma", "beta"/"upsilon")
                - "tau0" (float): Linear tau0 in s (computed from log_tau0)
                - "Z" (float64): Peak area in Ω (Ohms)

        Notes:
            - SG peak areas use default quad options (may be overridden via quad_opts)
            - Row order matches peak order (or provided peak_indices order)
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
            row["Z"] = peak.get_Z(quad_opts=quad_opts)
            rows.append(row)
        return pd.DataFrame(rows)


def fit_DRT_peaks(
    DRTData_object: DRTData,
    peak_positions: List[float] | NDArray[float64] | None,
    num_peaks: int | None = None,
    peak_type: str = "HN",
    skew: bool = True,
    log_tau0_bound: float | None = 0.1,
    assym_bound: float | None = None,
    dispersion_bounds: tuple | None = None,
    minimizer_settings: dict = None,
) -> DRTPeaks:
    """
    Fit DRT peaks to provided data using nonlinear least squares optimization.

    Automatically detects peaks (if needed), initializes parameters with bounds, and
    optimizes to fit selected peak models (Havriliak-Negami or Skewed Gaussian) to DRT data.
    Adjusts peak count if number of variables exceeds data points (prevents overfitting).

    Parameters:
        DRTData_object (DRTData): DRT data object containing tau and gamma arrays.
        peak_positions (List[Tuple[float, float]] | NDArray[float64] | None): Initial peak positions as (tau in s, gamma in Omega) tuples.
            If None, peaks are detected automatically via find_DRT_peaks. Default: None.
        num_peaks (int | None): Target number of peaks to fit. If None, uses all detected peaks.
            Only used if peak_positions is None. Default: None.
        peak_type (str): Peak model type: "HN" (Havriliak-Negami) or "SG" (Skewed Gaussian). Default: "HN".
        skew (bool): If True, allow asymmetry parameter (beta for HN, upsilon for SG) to vary. Default: True.
        log_tau0_bound (float | None): Relative bound for peak position parameter (unitless fraction).
            Default: 0.1 (approximately ±10% bounds). If None, uses 1% bounds. Example: 0.2 = ±20%.
        assym_bound (float | None): Bound for asymmetry parameters (unitless, 0-1 range).
            If None, uses defaults: HN beta in [0.5, 1], SG upsilon in [-1, 1].
            For HN: sets beta_min = 1 - assym_bound, beta_max = 1.
            For SG: sets upsilon bounds to [-assym_bound, +assym_bound].
            Must satisfy 0 <= assym_bound <= 1. Default: None.
        dispersion_bounds (tuple | None): Bounds for dispersion parameters (unitless).
            If None, uses defaults: HN alpha in [0, 1], SG sigma in [1e-10, 1e2].
            Default: None.
        minimizer_settings (dict | None): Optimization settings dictionary with keys:
            - "method" (str): Minimizer method ("leastsq", "nelder", "lbfgsb", "differential_evolution", etc.)
            - "fit_kws" (dict): Additional keywords for lmfit.minimize (ftol, xtol, etc.)
            Note: "leastsq" uses L2 norm (prefers peak height); "nelder"-type use L1 norm (prefers peak area).
            "nelder" recommended for convergence near initial conditions. Default: None (uses leastsq).

    Returns:
        DRTPeaks: Fitted peaks (DRTPeak objects) and fit summary (success indicator, residuals, etc.).

    Raises:
        ValueError: If both num_peaks and peak_positions provided, peak_type not in ["HN", "SG"],
            or peak_positions is empty list.

    Notes:
        - Automatically reduces num_peaks if num_variables > len(gamma) (prevents overfitting)
        - Peaks sorted by height before removal if overfitting detected
        - Uses natural logarithm (base e) for tau spacing (more natural for HN basis)
        - Fits absolute residuals |gamma_fit - gamma_exp| to avoid divide-by-zero issues
    """

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
        peaks,
        peak_type=peak_type,
        skew=skew,
        log_tau0_bound=log_tau0_bound,
        assym_bound=assym_bound,
        dispersion_bounds=dispersion_bounds,
    )

    # safeguard that peak fit isn't overfitting (most likely in the case that neither num_peaks nor peak_positions is provided)
    while len(peaks) > 1 and num_variables > gamma.size:
        peaks.sort(key=lambda t: t[1], reverse=True)
        peaks.pop()
        peaks.sort(key=lambda t: t[0])
        parameters, num_variables = _generate_parameters(
            peaks,
            peak_type=peak_type,
            skew=skew,
            log_tau0_bound=log_tau0_bound,
            assym_bound=assym_bound,
        )

    # Set minimizer defaults
    method: str = "nelder"
    fit_kws: dict = {}

    if minimizer_settings is not None:
        method = minimizer_settings.get("method", method)
        fit_kws = minimizer_settings.get("fit_kws", fit_kws)

    # Perform minimization
    fit: MinimizerResult = minimize(
        _residual,
        parameters,
        method=method,
        args=(log_tau, gamma, peak_type),
        **fit_kws,
    )
    params: Dict[str, float64] = fit.params.valuesdict()

    # Create fit summary dictionary
    fit_summary: dict = {
        "success": fit.success,
        "message": fit.message,
        "chisqr": fit.chisqr,
        "redchi": fit.redchi,
        "nfev": fit.nfev,
        "method": fit.method,
        "kws sent": fit.call_kws,
    }

    # Organize peak parameters by peak index
    peak_dict: Dict[str, dict] = {}
    for k, v in params.items():
        if "_" in k:
            prop, idx = k.rsplit("_", 1)
            peak_key = f"peak_{idx}"
            if peak_key not in peak_dict:
                peak_dict[peak_key] = {}
                peak_dict[peak_key]["peak_type"] = peak_type
                peak_dict[peak_key]["peak_number"] = int(idx)
            peak_dict[peak_key][prop] = v

    drt_peaks: List[DRTPeak] = []
    for peak_key, peak_params in peak_dict.items():
        drt_peaks.append(DRTPeak(label=peak_key, peak_params=peak_params))

    return DRTPeaks(
        peaks=drt_peaks,
        fit_summary=fit_summary,
    )
