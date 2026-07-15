"""Utility functions for DRT analysis including Havriliak-Negami impedance."""

import numpy as np
from numpy import float64, complex128
from numpy.typing import NDArray


def extend_logspace_f(
    f: NDArray[float64], new_f_min: float, new_f_max: float
) -> NDArray[float64]:
    """
    Extend a log-spaced frequency vector to new limits maintaining uniform spacing.

    Extends a descending, uniformly log-spaced frequency array to specified limits
    while maintaining the original logarithmic spacing.

    Parameters:
        f (NDArray[float64]): Original frequency vector in Hz (Hertz), descending, uniformly log-spaced.
        new_f_min (float): New minimum frequency in Hz (Hertz) (lowest value).
        new_f_max (float): New maximum frequency in Hz (Hertz) (highest value).

    Returns:
        NDArray[float64]: Extended frequency vector in Hz (Hertz), descending, uniformly log-spaced.

    Raises:
        ValueError: If f is not in descending order.
        ValueError: If f is not uniformly log-spaced (within 1% tolerance).
    """
    f = np.asarray(f)
    if not np.all(np.diff(f) < 0):
        raise ValueError("f must be in descending order.")

    log_f = np.log10(f)
    log_diffs = np.diff(log_f)
    log_diff_mean = np.mean(log_diffs)
    log_diff_std = np.std(log_diffs)
    rel_std = log_diff_std / abs(log_diff_mean)
    if rel_std > 0.01:
        raise ValueError("f must be uniformly log-spaced (within tolerance).")
    log_step = log_diff_mean

    # Extend above (higher frequencies)
    extra_high = []
    next_log = np.log10(f[0]) - log_step
    while 10**next_log <= new_f_max:
        extra_high.append(10**next_log)
        next_log -= log_step

    # Extend below (lower frequencies)
    extra_low = []
    next_log = np.log10(f[-1]) + log_step
    while 10**next_log >= new_f_min:
        extra_low.append(10**next_log)
        next_log += log_step

    # Concatenate: extra_high (reversed) + original + extra_low
    extended_f = np.concatenate([extra_high[::-1], f, extra_low])
    return extended_f


def HN_Z(
    omega: float | NDArray[float64],
    Z0: float64,
    tau0: float64,
    alpha: float64,
    beta: float64,
) -> complex | NDArray[complex128]:
    """
    Calculate Havriliak-Negami (HN) impedance function.

    Implements the generalized HN model that encompasses multiple impedance response
    models (Debye, Cole-Cole, Cole-Davidson, Gerischer) through shape parameters.

    Parameters:
        omega (float | NDArray[float64]): Angular frequency in rad/s (radians per second).
        Z0 (float64): Characteristic impedance magnitude in Ω (Ohms).
        tau0 (float64): Characteristic relaxation time in s (seconds).
        alpha (float64): Symmetric broadening parameter (0 < alpha ≤ 1).
            Controls peak width symmetry: alpha=1 for sharp, alpha<1 for broad.
        beta (float64): Asymmetric broadening parameter (0 < beta ≤ 1).
            Controls peak asymmetry: beta=1 for symmetric, beta<1 for asymmetric.

    Returns:
        complex | NDArray[complex128]: Complex impedance Z(ω) in Ω (Ohms).

    Notes:
        Equation: Z(ω) = Z0 / (1 + (jωτ0)^α)^β
        Special cases:
        - alpha=1, beta=1: Debye (single RC element)
        - alpha<1, beta=1: Cole-Cole (ZARC element)
        - alpha=1, beta<1: Cole-Davidson
        - alpha=0.5, beta=1: Gerischer (semi-infinite diffusion)
        Reference: Boukamp & Rolle (2018), https://doi.org/10.1016/j.ssi.2017.11.021
    """
    # Enforce parameter bounds
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1]")
    if not (0 < beta <= 1):
        raise ValueError("beta must be in (0, 1]")
    # Convert to numpy array for consistent handling
    omega = np.asarray(omega)

    # Calculate (jωτ0)^α
    jwt0_alpha = (1j * omega * tau0) ** alpha

    # Calculate Z(ω) = Z0 / (1 + (jωτ0)^α)^β
    Z = Z0 / (1 + jwt0_alpha) ** beta

    return Z


def HN_DRT(
    tau: float | NDArray[float64],
    Z0: float64,
    tau0: float64,
    alpha: float64,
    beta: float64,
) -> float | NDArray[float64]:
    """
    Calculate the Havriliak-Negami Distribution of Relaxation Times (DRT).

    Parameters
    ----------
    tau : float or np.ndarray
        Time constant(s) (s)
    Z0 : float
        Characteristic impedance magnitude (Ohm)
    tau0 : float
        Characteristic time constant (s)
    alpha : float
        Shape parameter (0 < alpha <= 1)
        Controls the symmetric broadening of the peak
    beta : float
        Shape parameter (0 < beta <= 1)
        Controls the asymmetric broadening of the peak

    Returns
    -------
    float or np.ndarray
        DRT γ(τ) (Ohm)

    Notes
    -----
    Implements equation for Havriliak Negami DRT:
    γ(τ) = (Z0 * (τ/τ0)^(αβ) * sin(β*θ)) /
           (π * ((τ/τ0)^(2α) + 2*(τ/τ0)^α * cos(πα) + 1)^(β/2))
    where:
    θ = π/2 - arctan( ((τ/τ0)^α + cos(πα)) / sin(πα))
    From:
    Boukamp, B. A.; Rolle, A. "Use of a Distribution Function of Relaxation Times (DFRT) in Impedance Analysis of SOFC Electrodes." Solid State Ionics 2018, 314, 103–111. https://doi.org/10.1016/j.ssi.2017.11.021
    See Eq. 12.

    For numerical stability, the implementation uses a rearranged form of θ using arctan2.
    """
    # Enforce parameter bounds
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in [0, 1]")
    if not (0 < beta <= 1):
        raise ValueError("beta must be in [0, 1]")

    # Convert to numpy array for consistent handling
    tau = np.asarray(tau)

    # Calculate normalized time constant
    tau_ratio = tau / tau0
    tau_ratio_alpha = tau_ratio**alpha

    # Calculate theta
    cos_pi_alpha = np.cos(np.pi * alpha)
    sin_pi_alpha = np.sin(np.pi * alpha)
    theta = np.arctan2((sin_pi_alpha), (tau_ratio_alpha + cos_pi_alpha))  # see notes

    # Calculate numerator: Z0 * (τ/τ0)^(αβ) * sin(β*θ)
    numerator = Z0 * (tau_ratio ** (alpha * beta)) * np.sin(beta * theta)

    # Calculate denominator: π * ((τ/τ0)^(2α) + 2*(τ/τ0)^α * cos(πα) + 1)^(β/2)
    denominator_base = tau_ratio ** (2 * alpha) + 2 * tau_ratio_alpha * cos_pi_alpha + 1
    denominator = np.pi * (denominator_base ** (beta / 2))

    # Calculate γ(τ)
    gamma = numerator / denominator

    return gamma


def GFLW_Z(
    omega: float | NDArray[float64],
    Z0: float64,
    tau0: float64,
    zeta: float64,
) -> complex | NDArray[complex128]:
    """
    Calculate the generalized finite-length Warburg impedance.

    Parameters
    ----------
    omega : float or np.ndarray
        Angular frequency (rad/s)
    Z0 : float
        Characteristic impedance magnitude (Ohm)
    tau0 : float
        Characteristic time constant (s)
    zeta : float
        Exponent parameter (0 < zeta <= 0.5)

    Returns
    -------
    complex or np.ndarray
        Complex impedance Z(ω)

    Notes
    -----
    Implements the generalized finite-length Warburg form:
    Z(ω) = [Z0 / (jωτ0)^ζ] * tanh[(jωτ0)^ζ]
    From:
    Boukamp, B. A. Distribution (Function) of Relaxation Times, Successor to Complex Nonlinear Least Squares Analysis of Electrochemical Impedance Spectroscopy? J. Phys. Energy 2020, 2 (4), 042001. https://doi.org/10.1088/2515-7655/aba9e0.

    """
    # Enforce parameter bounds
    if not (0 < zeta <= 0.5):
        raise ValueError("zeta must be in (0, 0.5]")

    # Convert to numpy array for consistent handling
    omega = np.asarray(omega)

    # Calculate x = (jωτ0)^ζ
    x = (1j * omega * tau0) ** zeta

    # Compute tanh(x)/x with stable handling at x=0 (limit is 1)
    ratio = np.empty_like(x, dtype=np.complex128)
    np.divide(np.tanh(x), x, out=ratio, where=x != 0)
    ratio = np.where(x == 0, 1.0 + 0.0j, ratio)

    # Calculate Z(ω)
    Z = Z0 * ratio

    return Z


def GFLW_DRT(
    tau: float | NDArray[float64],
    Z0: float64,
    tau0: float64,
    zeta: float64,
) -> float | NDArray[float64]:
    """
    Calculate the generalized finite-length Warburg Distribution of Relaxation Times (DRT).

    Parameters
    ----------
    tau : float or np.ndarray
        Time constant(s) (s)
    Z0 : float
        Characteristic impedance magnitude (Ohm)
    tau0 : float
        Characteristic time constant (s)
    zeta : float
        Exponent parameter (0 < zeta <= 0.5)

    Returns
    -------
    float or np.ndarray
        DRT γ(τ) (Ohm)

    Notes
    -----
    Implements generalized finite-length Warburg DRT equations:
        γ(τ) = (Z0 / π X) *
            [sin(πζ)(1 - Y^2) - 2Y cos(πζ) sin(2X sin(πζ))] /
            [1 + 2Y cos(2X sin(πζ)) + Y^2]
    X = (τ0/τ)^ζ
    Y = exp[-2X cos(πζ)]
    From:
    Boukamp, B. A. Distribution (Function) of Relaxation Times, Successor to Complex Nonlinear Least Squares Analysis of Electrochemical Impedance Spectroscopy? J. Phys. Energy 2020, 2 (4), 042001. https://doi.org/10.1088/2515-7655/aba9e0.

    """
    # Enforce parameter bounds
    if not (0 < zeta <= 0.5):
        raise ValueError("zeta must be in (0, 0.5]")

    # Convert to numpy array for consistent handling
    tau = np.asarray(tau)

    # Calculate auxiliary variables
    sin_pi_zeta = np.sin(np.pi * zeta)
    cos_pi_zeta = np.cos(np.pi * zeta)
    X = (tau0 / tau) ** zeta
    Y = np.exp(-2 * X * cos_pi_zeta)

    # Calculate second-fraction numerator and denominator terms
    gamma_num = sin_pi_zeta * (1 - Y**2) - 2 * Y * cos_pi_zeta * np.sin(
        2 * X * sin_pi_zeta
    )
    gamma_den = 1 + 2 * Y * np.cos(2 * X * sin_pi_zeta) + Y**2

    # Calculate γ(τ) = (Z0/π) * (gamma_num/gamma_den)
    gamma = (Z0 / (np.pi * X)) * (gamma_num / gamma_den)

    return gamma


def SG_DRT(
    log_tau: float64 | NDArray[float64],
    height: float64,
    log_tau0: float64,
    upsilon: float64,
    sigma: float64,
) -> float64 | NDArray[float64]:
    """
    Calculate the Skewed Gaussian Distribution of Relaxation Times (DRT).

    Parameters
    ----------
    log_tau : float or np.ndarray
        Logarithm (base 10 or natural) of time constant(s).
    height : float
        Peak height (Ohm).
    log_tau0 : float
        Peak position (center) in log space.
    upsilon : float
        Skew parameter (controls asymmetry).
    sigma : float
        Standard deviation (width) in log space.

    Returns
    -------
    float or np.ndarray
        DRT value(s) (Ohm).

    Notes
    -----
    Implements a skewed normal distribution:
        SG_DRT(log_tau) = height * exp(-((log_tau - log_tau0) * (1 + upsilon * sign(log_tau - log_tau0)))**2 / (2 * sigma**2))
    From:
    Hahn, M.; Schindler, S.; Triebs, L.-C.; Danzer, M. A. Optimized Process Parameters for a Reproducible Distribution of Relaxation Times Analysis of Electrochemical Systems. Batteries 2019, 5 (2), 43. https://doi.org/10.3390/batteries5020043.
    See Eq. 17.
    """
    if not (-1 <= upsilon <= 1):
        raise ValueError("upsilon must be in [-1, 1] for sensible peaks")

    log_tau = np.asarray(log_tau)
    dp = log_tau - log_tau0
    den = 2 * sigma**2
    num = (dp * (1 + upsilon * np.sign(dp))) ** 2
    return height * np.exp(-num / den)
