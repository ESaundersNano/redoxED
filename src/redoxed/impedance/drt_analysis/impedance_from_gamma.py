from scipy.integrate import quad, trapezoid
import numpy as np


def Z_from_DRT(f, gamma, tau=None, R0=0.0, L0=0.0, quad_opts=None):
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
    quad_opts : dict, optional
        Dictionary of options for scipy.integrate.quad. Supported keys:
        - 'a', 'b': integration limits (default -50, 50)
        - 'epsabs', 'epsrel': error tolerances
        - 'limit': max subdivisions (default 100)
        If a key is provided, its value will override the default. If not provided, defaults are used.

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
    default_opts = {"a": -50, "b": 50, "epsabs": 1e-9, "epsrel": 1e-9, "limit": 100}
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
            integral = real_integral + 1j * imag_integral
            Z_DRT[i] = R0 + 1j * 2 * np.pi * fi * L0 + integral
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
            Z_DRT[i] = R0 + 1j * 2 * np.pi * fi * L0 + trapezoid(kernel, ln_tau)
    return Z_DRT if Z_DRT.size > 1 else Z_DRT[0]
