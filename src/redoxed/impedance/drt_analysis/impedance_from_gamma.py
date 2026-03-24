from scipy.integrate import quad, trapezoid
import numpy as np


def Z_from_DRT(f, gamma, tau=None, R0=np.nan, L0=np.nan, C0=np.nan, quad_opts=None):
    """
    Reconstruct impedance Z_DRT(f) from DRT using either analytic or vectorized gamma.

    Parameters
    ----------
    f : float or np.ndarray
        Frequency (Hz), can be scalar or array.
    gamma : callable or np.ndarray
        If callable: gamma(tau) returns DRT value for tau.
        If array: vector of gamma values corresponding to tau.
    tau : np.ndarray, optional
        Array of tau values (required if gamma is array).
    R0 : float, optional
        Ohmic resistance (default 0.0)
    L0 : float, optional
        Inductance (default 0.0)
    quad_opts : dict | None, optional
        Dictionary of options to pass to scipy.integrate.quad for integrals.
        May need to reduce a and b limits if getting integration warnings. Likely you just start getting numeric instability when interval is too large.
        With typical peak sizes, limits of -20 to 5 are more than reasonable (1e-8 to 1e2 tau).
        If None, uses:
            {'epsabs': 1e-9, 'epsrel': 1e-9, 'limit': 100, 'a': -20, 'b': 20}
        Valid keys:
            - 'epsabs': Absolute error tolerance (default: 1e-9)
            - 'epsrel': Relative error tolerance (default: 1e-9)
            - 'limit': Maximum number of subintervals (default: 100)
            - 'a': Lower integration limit (default: -50)
            - 'b': Upper integration limit (default: 50)
        Example:
            quad_opts = {'epsabs': 1e-8, 'epsrel': 1e-8, 'limit': 200, 'a': -20, 'b': 5}

    Returns
    -------
    Z_DRT : complex or np.ndarray
        Reconstructed impedance at each frequency.

    Notes
    -----
    If analytic gamma contains a singularity, integration will likely fail.
    Vectorised gamma with trapezoidal integration is more robust in such cases.

    """
    f = np.atleast_1d(f)
    Z_DRT = np.zeros(f.shape, dtype=complex)
    # Default quad options
    default_opts = {"a": -20, "b": 20, "epsabs": 1e-9, "epsrel": 1e-9, "limit": 100}
    quad_opts = quad_opts or {}
    opts = {**default_opts, **quad_opts}
    a = opts["a"]
    b = opts["b"]
    quad_args = {k: v for k, v in opts.items() if k not in ["a", "b"]}
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
