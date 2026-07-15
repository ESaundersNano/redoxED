"""Radial Basis Function (RBF) implementations and utilities for DRT discretization.

Provides RBF types (Matérn, Cauchy, Gaussian, inverse-quadratic, inverse-quadric, piecewise-linear)
and computes inner products of RBF derivatives for discretization matrix assembly in DRT solver.
Supports both standard and offset RBF formulations with derivative orders 1 and 2.

References:
    - Radial basis function theory for function approximation
    - RBF-DRT methods: https://doi.org/10.1016/j.electacta.2015.09.097
"""

from typing import Callable, Dict, List
import numpy as np
from scipy.integrate import quad

_RBF_TYPES: List[str] = [
    "c0-matern",
    "c2-matern",
    "c4-matern",
    "c6-matern",
    "cauchy",
    "gaussian",
    "inverse-quadratic",
    "inverse-quadric",
    "piecewise-linear",
]

_RBF_FUNCTIONS: Dict[str, Callable] = {
    "c0-matern": lambda x, mu: np.exp(-abs(mu * x)),
    "c2-matern": lambda x, mu: np.exp(-abs(mu * x)) * (1 + abs(mu * x)),
    "c4-matern": lambda x, mu: 1
    / 3
    * np.exp(-abs(mu * x))
    * (3 + 3 * abs(mu * x) + abs(mu * x) ** 2),
    "c6-matern": lambda x, mu: 1
    / 15
    * np.exp(-abs(mu * x))
    * (15 + 15 * abs(mu * x) + 6 * abs(mu * x) ** 2 + abs(mu * x) ** 3),
    "cauchy": lambda x, mu: 1 / (1 + abs(mu * x)),
    "gaussian": lambda x, mu: np.exp(-((mu * x) ** 2)),
    "inverse-quadratic": lambda x, mu: 1 / (1 + (mu * x) ** 2),
    "inverse-quadric": lambda x, mu: 1 / np.sqrt(1 + (mu * x) ** 2),
    "piecewise-linear": lambda x, mu: 0.0,
}

_OFFSET_RBF_FUNCTIONS: Dict[str, Callable] = {
    "gaussian": lambda x: np.exp(-((x) ** 2)) - 0.5,
    "c0-matern": lambda x: np.exp(-abs(x)) - 0.5,
    "c2-matern": lambda x: np.exp(-abs(x)) * (1 + abs(x)) - 0.5,
    "c4-matern": lambda x: 1 / 3 * np.exp(-abs(x)) * (3 + 3 * abs(x) + abs(x) ** 2)
    - 0.5,
    "c6-matern": lambda x: 1
    / 15
    * np.exp(-abs(x))
    * (15 + 15 * abs(x) + 6 * abs(x) ** 2 + abs(x) ** 3)
    - 0.5,
    "inverse-quadratic": lambda x: 1 / (1 + (x) ** 2) - 0.5,
    "inverse-quadric": lambda x: 1 / np.sqrt(1 + (x) ** 2) - 0.5,
    "cauchy": lambda x: 1 / (1 + abs(x)) - 0.5,
    "piecewise-linear": lambda x: 0.0,
}


def _inner_product_rbf(
    tau_p: float,
    tau_q: float,
    mu: float,
    derivative_order: int,
    rbf_type: str,
) -> float:
    """
    Calculate the inner product of RBF derivatives for discretization matrix.

    Computes ∫ φ'(ln(τ/τ_p)) * ψ'(ln(τ/τ_q)) dτ/τ for RBF derivatives,
    which is used to construct the regularization/smoothing matrix M in DRT solver.

    Parameters:
        tau_p (float): First relaxation time in s (seconds).
        tau_q (float): Second relaxation time in s (seconds).
        mu (float): RBF shape parameter (controls width/localization).
        derivative_order (int): Order of derivative (1 or 2) for regularization.
        rbf_type (str): RBF type from _RBF_TYPES list:
            - "c0-matern": Matérn C0 (exponential)
            - "c2-matern": Matérn C2 (smoother exponential)
            - "c4-matern": Matérn C4 (smoother still)
            - "c6-matern": Matérn C6 (highest smoothness)
            - "cauchy": Cauchy function
            - "gaussian": Gaussian (RBF)
            - "inverse-quadratic": 1/(1 + (μx)²)
            - "inverse-quadric": 1/√(1 + (μx)²)
            - "piecewise-linear": Piecewise linear

    Returns:
        float: Inner product value (dimensionless).

    Raises:
        NotImplementedError: If derivative_order not supported for RBF type.
        ValueError: If rbf_type not recognized.

    Notes:
        - For each RBF type, analytical expressions are used where available
        - For inverse-quadric, numerical integration via scipy.integrate.quad is used
        - Parameter a = μ * ln(τ_p / τ_q) is used internally for log-space calculations
        - Different Matérn orders (C0, C2, C4, C6) provide varying smoothness properties
    """
    a: float = mu * np.log(tau_p / tau_q)

    if rbf_type == "c0-matern":
        if derivative_order == 1:
            return mu * (1 - abs(a)) * np.exp(-abs(a))
        elif derivative_order == 2:
            return mu**3 * (1 + abs(a)) * np.exp(-abs(a))

    elif rbf_type == "c2-matern":
        if derivative_order == 1:
            return mu / 6 * (3 + 3 * abs(a) - abs(a) ** 3) * np.exp(-abs(a))
        elif derivative_order == 2:
            return (
                mu**3
                / 6
                * (3 + 3 * abs(a) - 6 * abs(a) ** 2 + abs(a) ** 3)
                * np.exp(-abs(a))
            )

    elif rbf_type == "c4-matern":
        if derivative_order == 1:
            return (
                mu
                / 30
                * (
                    105
                    + 105 * abs(a)
                    + 30 * abs(a) ** 2
                    - 5 * abs(a) ** 3
                    - 5 * abs(a) ** 4
                    - abs(a) ** 5
                )
                * np.exp(-abs(a))
            )
        elif derivative_order == 2:
            return (
                mu**3
                / 30
                * (45 + 45 * abs(a) - 15 * abs(a) ** 3 - 5 * abs(a) ** 4 + abs(a) ** 5)
                * np.exp(-abs(a))
            )

    elif rbf_type == "c6-matern":
        if derivative_order == 1:
            return (
                mu
                / 140
                * (
                    10395
                    + 10395 * abs(a)
                    + 3780 * abs(a) ** 2
                    + 315 * abs(a) ** 3
                    - 210 * abs(a) ** 4
                    - 84 * abs(a) ** 5
                    - 14 * abs(a) ** 6
                    - abs(a) ** 7
                )
                * np.exp(-abs(a))
            )
        elif derivative_order == 2:
            return (
                mu**3
                / 140
                * (
                    2835
                    + 2835 * abs(a)
                    + 630 * abs(a) ** 2
                    - 315 * abs(a) ** 3
                    - 210 * abs(a) ** 4
                    - 42 * abs(a) ** 5
                    + abs(a) ** 7
                )
                * np.exp(-abs(a))
            )

    elif rbf_type == "cauchy":
        if a == 0:
            if derivative_order == 1:
                return 2 / 3 * mu
            elif derivative_order == 2:
                return 8 / 5 * mu**3
        else:
            numerator: float
            denominator: float
            if derivative_order == 1:
                numerator = abs(a) * (2 + abs(a)) * (
                    4 + 3 * abs(a) * (2 + abs(a))
                ) - 2 * (1 + abs(a)) ** 2 * (4 + abs(a) * (2 + abs(a))) * np.log(
                    1 + abs(a)
                )
                denominator = abs(a) ** 3 * (1 + abs(a)) * (2 + abs(a)) ** 3
                return 4 * mu * numerator / denominator
            elif derivative_order == 2:
                numerator = abs(a) * (2 + abs(a)) * (
                    -96
                    + abs(a)
                    * (2 + abs(a))
                    * (-30 + abs(a) * (2 + abs(a)))
                    * (4 + abs(a) * (2 + abs(a)))
                ) + 12 * (1 + abs(a)) ** 2 * (
                    16 + abs(a) * (2 + abs(a)) * (12 + abs(a) * (2 + abs(a)))
                ) * np.log(
                    1 + abs(a)
                )
                denominator = abs(a) ** 5 * (1 + abs(a)) * (2 + abs(a)) ** 5

                return 8 * mu**3 * numerator / denominator

    elif rbf_type == "gaussian":
        if derivative_order == 1:
            return mu * (1 - a**2) * np.exp(-(a**2 / 2)) * np.sqrt(np.pi / 2)
        elif derivative_order == 2:
            return (
                mu**3 * (3 - 6 * a**2 + a**4) * np.exp(-(a**2 / 2)) * np.sqrt(np.pi / 2)
            )

    elif rbf_type == "inverse-quadratic":
        if derivative_order == 1:
            return 4 * mu * (4 - 3 * a**2) * np.pi / ((4 + a**2) ** 3)
        elif derivative_order == 2:
            return (
                48 * (16 + 5 * a**2 * (-8 + a**2)) * np.pi * mu**3 / ((4 + a**2) ** 5)
            )

    elif rbf_type == "inverse-quadric":

        y_i: float = np.log(tau_p)
        y_j: float = np.log(tau_q)
        rbf_i: Callable = lambda y: 1 / np.sqrt(1 + (mu * (y - y_i)) ** 2)
        rbf_j: Callable = lambda y: 1 / np.sqrt(1 + (mu * (y - y_j)) ** 2)

        delta: float
        sqr_drbf_dy: Callable
        if derivative_order == 1:
            delta = 1e-8
            sqr_drbf_dy = (
                lambda y: 1
                / (2 * delta)
                * (rbf_i(y + delta) - rbf_i(y - delta))
                * 1
                / (2 * delta)
                * (rbf_j(y + delta) - rbf_j(y - delta))
            )

        elif derivative_order == 2:
            delta = 1e-4
            sqr_drbf_dy = (
                lambda y: 1
                / (delta**2)
                * (rbf_i(y + delta) - 2 * rbf_i(y) + rbf_i(y - delta))
                * 1
                / (delta**2)
                * (rbf_j(y + delta) - 2 * rbf_j(y) + rbf_j(y - delta))
            )
        else:
            raise NotImplementedError(f"Unsupported {derivative_order=}")

        return quad(sqr_drbf_dy, -50, 50, epsabs=1e-9, epsrel=1e-9)[0]

    if rbf_type in _RBF_TYPES:
        raise NotImplementedError(f"Unsupported RBF type: {rbf_type}")

    raise ValueError(f"Unknown/invalid RBF type {rbf_type}")
