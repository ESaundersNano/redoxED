"""Reconstruction of impedance from Distribution of Relaxation Times (DRT) data."""

from scipy.integrate import quad, trapezoid
import numpy as np


def Z_from_DRT(
    f,
    gamma,
    tau=None,
    R0=np.nan,
    L0=np.nan,
    C0=np.nan,
    quad_opts=None,
):
    """
    Reconstruct complex impedance from Distribution of Relaxation Times (DRT).

    Integrates the DRT distribution (gamma) over relaxation times to compute impedance
    at specified frequency points. Supports both analytical (callable) and vectorized
    (array) representations of gamma.

    Parameters:
        f (float | NDArray[float64]): Frequency data in Hz (Hertz). Can be scalar or array.
        gamma (callable | NDArray[float64]): DRT distribution. If callable, gamma(tau) returns
            DRT value for tau. If array, vector of gamma values corresponding to tau array.
        tau (NDArray[float64] | None): Array of relaxation times in s (seconds). Required if
            gamma is a vectorized array. Defaults to None.
        R0 (float): Ohmic resistance in Ω (Ohms) to add in series. Defaults to NaN (not added).
        L0 (float): Series inductance in H (Henries). Defaults to NaN (not added).
        C0 (float): Series capacitance in F (Farads). Defaults to NaN (not added).
        quad_opts (dict | None): Integration options dictionary. Supports keys:
            - 'epsabs': Absolute error tolerance (default: 1e-9)
            - 'epsrel': Relative error tolerance (default: 1e-9)
            - 'limit': Maximum subintervals (default: 100)
            - 'a': Lower integration limit in log-space (default: -20)
            - 'b': Upper integration limit in log-space (default: 20)
            Defaults to None (uses default options).

    Returns:
        complex | NDArray[complex128]: Complex impedance Z(ω) in Ω (Ohms) at each frequency.
            Returns scalar if input f is scalar, array if f is array.

    Raises:
        ValueError: If gamma is vectorized but tau is not provided.

    Notes:
        - Integration performed over log-tau space using change of variables: tau = exp(ln_tau)
        - Vectorized integration using trapezoidal rule is more robust for singular gamma
        - Suggested quad_opts limits: {'a': -20, 'b': 20} for typical peak sizes
    """
    # Default quad options
    default_opts = {"a": -20, "b": 20, "epsabs": 1e-9, "epsrel": 1e-9, "limit": 100}
    quad_opts = quad_opts or {}
    opts = {**default_opts, **quad_opts}
    a = opts["a"]
    b = opts["b"]
    quad_args = {k: v for k, v in opts.items() if k not in ["a", "b"]}

    f = np.atleast_1d(f)
    Z_DRT = np.zeros(f.shape, dtype=complex)
    if callable(gamma):
        for i, fi in enumerate(f):

            def real_integrand(ln_tau):
                tau_val = np.exp(ln_tau)
                denom = 1 + (2 * np.pi * fi * tau_val) ** 2
                return gamma(tau_val) / denom

            def imag_integrand(ln_tau):
                tau_val = np.exp(ln_tau)
                denom = 1 + (2 * np.pi * fi * tau_val) ** 2
                return -gamma(tau_val) * (2 * np.pi * fi * tau_val) / denom

            real_integral, _ = quad(real_integrand, a, b, **quad_args)
            imag_integral, _ = quad(imag_integrand, a, b, **quad_args)
            Z_DRT[i] = real_integral + 1j * imag_integral
    else:
        if tau is None:
            raise ValueError("tau array must be provided when gamma is vectorized.")
        # Ensure tau is in ascending order
        if not np.all(np.diff(tau) > 0):
            sort_idx = np.argsort(tau)
            tau = tau[sort_idx]
            gamma = gamma[sort_idx]
        ln_tau = np.log(tau)
        for i, fi in enumerate(f):
            kernel = gamma / (1 + 1j * 2 * np.pi * fi * tau)
            Z_DRT[i] = trapezoid(kernel, ln_tau)
    if not np.isnan(R0):
        Z_DRT += R0
    if not np.isnan(L0):
        Z_DRT += 1j * 2 * np.pi * f * L0
    if not np.isnan(C0):
        Z_DRT += 1 / (1j * 2 * np.pi * f * C0)
    return Z_DRT if Z_DRT.size > 1 else Z_DRT[0]
