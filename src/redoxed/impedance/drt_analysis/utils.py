import numpy as np


def HN_Z(
    omega: float | np.ndarray, Z0: float, tau0: float, alpha: float, beta: float
) -> complex | np.ndarray:
    """
    Calculate the Havriliak-Negami impedance.

    Parameters
    ----------
    omega : float or np.ndarray
        Angular frequency (rad/s)
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
    complex or np.ndarray
        Complex impedance Z(ω)

    Notes
    -----
    Implements equation for Havriliak Negami impedance:
    Z(ω) = Z0 / (1 + (jωτ0)^α)^β
    From:
    Boukamp, B. A.; Rolle, A. "Use of a Distribution Function of Relaxation Times (DFRT) in Impedance Analysis of SOFC Electrodes." Solid State Ionics 2018, 314, 103–111. https://doi.org/10.1016/j.ssi.2017.11.021
    See Eq. 8.

    Special cases:
    - α=1, β=1: Debye (single time constant) AKA RC element
    - α<1, β=1: Cole-Cole (symmetric peak broadening) AKA RQ or ZARC element
    - α=1, β<1: Cole-Davidson (asymmetric peak broadening)
    - α=0.5, β=1: Gerischer element (semi-infinite diffusion)
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
    tau: float | np.ndarray, Z0: float, tau0: float, alpha: float, beta: float
) -> float | np.ndarray:
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


def SG_DRT(
    log_tau: float | np.ndarray,
    height: float,
    log_tau0: float,
    upsilon: float,
    sigma: float,
) -> float | np.ndarray:
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
