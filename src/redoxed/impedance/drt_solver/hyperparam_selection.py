"""Hyperparameter (regularization parameter) selection methods for DRT solver.

Implements multiple criteria for automatic selection of Tikhonov regularization
parameter (lambda) including Generalized Cross-Validation (GCV), modified GCV (mGCV),
robust GCV (rGCV), and L-curve methods. Based on pyDRTtools and pyimpspec.

Based on code from https://github.com/ciuccislab/pyDRTtools and https://github.com/vyrjana/pyimpspec
References:
    - Electrochimica Acta 2015, 10.1016/j.electacta.2015.09.097 (RBF-DRT)
    - J. Electrochem. Soc. 2022, 10.1149/1945-7111/acbca4 (Hyperparameter selection)
    - ACS ElectroChem. 2015, 10.1021/acselectrochem.5c00334 (DRTtools)
"""

from typing import Callable, Dict
from numpy.typing import NDArray
from numpy import array, concatenate, exp, eye, trace, zeros, float64

import numpy as np

from numpy.linalg import (
    cholesky,
    norm,
    inv,
)

from scipy.optimize import (
    OptimizeResult,
    minimize,
    differential_evolution,
    basinhopping,
)

from .utils import _is_positive_definite, _nearest_positive_definite
from .qp import _quad_format, _solve_qp_cvxopt


def _gcv_wrapper(func: Callable) -> Callable:
    """Decorator wrapping cost functions with matrix preparation for GCV-based methods.

    Prepares concatenated impedance and discretization matrices for generalized
    cross-validation and related regularization parameter selection methods.

    Parameters:
        func (Callable): Cost function computing GCV or variant score from processed matrices.

    Returns:
        Callable: Wrapper function accepting ln_lambda, impedance, and discretization matrices.

    Notes:
        Implementation follows Eq. 5 and 13 from DOI: 10.1149/1945-7111/acbca4
    """

    def wrapper(
        ln_lambda: float64,
        A_re: NDArray[float64],
        A_im: NDArray[float64],
        Z_re: NDArray[float64],
        Z_im: NDArray[float64],
        M: NDArray[float64],
        show_lam: bool = False,
    ):
        lambda_value: float64 = exp(ln_lambda)
        if show_lam:
            print(f"Lambda value: {lambda_value}")

        # See eq. 5 in https://doi.org/10.1149/1945-7111/acbca4
        A: NDArray[float64] = concatenate((A_re, A_im), axis=0)
        Z: NDArray[float64] = concatenate((Z_re, Z_im), axis=0)

        # See eq. 13 in https://doi.org/10.1149/1945-7111/acbca4
        A_agm: NDArray[float64] = A.T @ A + lambda_value * M

        if not _is_positive_definite(A_agm):
            A_agm = _nearest_positive_definite(A_agm)

        # Cholesky transform to invert A_agm
        L_agm: NDArray[float64] = cholesky(A_agm)
        inv_L_agm: NDArray[float64] = inv(L_agm)

        # Inverse of A_agm
        # See eq. 13 in https://doi.org/10.1149/1945-7111/acbca4
        inv_A_agm: NDArray[float64] = inv_L_agm.T @ inv_L_agm
        A_GCV: NDArray[float64] = A @ inv_A_agm @ A.T

        return func(
            M=Z_re.shape[0],
            I=eye(2 * Z_re.shape[0]),
            K=A_GCV,
            Z_exp=Z,
        )

    return wrapper


@_gcv_wrapper
def _compute_generalized_cross_validation(
    M: int,
    I: NDArray[float64],
    K: NDArray[float64],
    Z_exp: NDArray[float64],
) -> float64:
    """Compute generalized cross-validation (GCV) score for regularization parameter selection.

    Calculates log of GCV score; logarithmic transformation prevents underflow for large M.

    Parameters:
        M (int): Number of experimental data points (impedance measurements).
        I (NDArray[float64]): Identity matrix in shape (2*M, 2*M).
        K (NDArray[float64]): Influence matrix from regularized least squares.
        Z_exp (NDArray[float64]): Stacked experimental impedance vector (real + imaginary) in Ω.

    Returns:
        float64: Log of GCV score (dimensionless). Lower values indicate better regularization.

    References:
        - Wahba, G. (1985). A comparison of GCV and GML. Ann. Statist. 13, 1378-1402.
        - DOI: 10.1149/1945-7111/acbca4 Eq. 13
    """
    # See eq. 13 in https://doi.org/10.1149/1945-7111/acbca4
    num: float64 = (norm((I - K) @ Z_exp) ** 2) / (2 * M)
    den: float64 = (trace(I - K) / (2 * M)) ** 2
    score: float64 = num / den
    return np.log(score)


@_gcv_wrapper
def _compute_modified_gcv(
    M: int,
    I: NDArray[float64],
    K: NDArray[float64],
    Z_exp: NDArray[float64],
) -> float64:
    """Compute modified GCV (mGCV) score with stabilization parameter rho.

    Calculates log of mGCV score; logarithmic transformation prevents underflow for large M.
    Includes stabilization parameter rho based on data size for improved robustness.

    Parameters:
        M (int): Number of experimental data points (impedance measurements).
        I (NDArray[float64]): Identity matrix in shape (2*M, 2*M).
        K (NDArray[float64]): Influence matrix from regularized least squares.
        Z_exp (NDArray[float64]): Stacked experimental impedance vector (real + imaginary) in Ω.

    Returns:
        float64: Log of mGCV score (dimensionless). Lower values indicate better regularization.

    Notes:
        Stabilization parameter rho:
        - rho = 2.0 for M >= 50
        - rho = 1.3 for M < 50

    References:
        - Kim, Y.J., Gu, C. (2004). Smoothing spline Gaussian regression. J. Royal Statist. Soc. 66, 337-356.
        - DOI: 10.1149/1945-7111/acbca4 Eq. 14-15
    """
    # the stabilization parameter, rho, is computed as described by Kim et al.
    # See eq. 15 in https://doi.org/10.1149/1945-7111/acbca4
    rho: float = 2.0 if M >= 50 else 1.3

    # See eq. 14 in https://doi.org/10.1149/1945-7111/acbca4
    num: float64 = (norm((I - K) @ Z_exp) ** 2) / (2 * M)
    den: float64 = (trace(I - rho * K) / (2 * M)) ** 2
    score: float64 = num / den
    return np.log(score)


@_gcv_wrapper
def _compute_robust_gcv(
    M: int,
    I: NDArray[float64],
    K: NDArray[float64],
    Z_exp: NDArray[float64],
) -> float64:
    """Compute robust GCV (rGCV) score with adaptive scaling parameter xi.

    Calculates log of rGCV score; logarithmic transformation prevents underflow for large M.
    Includes adaptive scaling parameter xi for increased robustness to outliers.

    Parameters:
        M (int): Number of experimental data points (impedance measurements).
        I (NDArray[float64]): Identity matrix in shape (2*M, 2*M).
        K (NDArray[float64]): Influence matrix from regularized least squares.
        Z_exp (NDArray[float64]): Stacked experimental impedance vector (real + imaginary) in Ω.

    Returns:
        float64: Log of rGCV score (dimensionless). Lower values indicate better regularization.

    Notes:
        Scaling parameter xi:
        - xi = 0.3 for M >= 50
        - xi = 0.2 for M < 50
        mu_2 is computed as trace(K.T @ K) / (2*M)

    References:
        - Lukas, M.A., de Hoog, F.R., Anderssen, R.S. (2016). Practical use of robust GCV. Comput. Statist. 31, 269-289.
        - DOI: 10.1149/1945-7111/acbca4 Eq. 16
    """
    # See eq. 13 in https://doi.org/10.1149/1945-7111/acbca4
    num: float64 = (norm((I - K) @ Z_exp) ** 2) / (2 * M)
    den: float64 = (trace(I - K) / (2 * M)) ** 2
    gcv_score: float64 = num / den

    # The robust parameter, xsi, is computed as described in Lukas et al.
    # See eq. 16 in https://doi.org/10.1149/1945-7111/acbca4
    xi: float = 0.3 if M >= 50 else 0.2
    mu_2: float64 = trace(K.T @ K) / (2 * M)
    score: float = (xi + (1 - xi) * mu_2) * gcv_score

    return np.log(score)


# # TODO: This seems to be giving different answers compared to pyDRTtools
# # for some reason. (copied from pyimpspec)
# def _compute_re_im_cross_validation(
#     ln_lambda: float64,
#     A_re: NDArray[float64],
#     A_im: NDArray[float64],
#     Z_re: NDArray[float64],
#     Z_im: NDArray[float64],
#     M: NDArray[float64],
# ) -> float64:
#     """
#     This function computes the score for real-imaginary discrepancy (re-im).
#     Inputs:
#         ln_lambda: regularization parameter
#         A_re: discretization matrix for the real part of the impedance
#         A_im: discretization matrix for the real part of the impedance
#         Z_re: vector of the real parts of the impedance
#         Z_im: vector of the imaginary parts of the impedance
#         M: differentiation matrix
#     """
#     lambda_value: float64 = exp(ln_lambda)

#     # Non-negativity constraint on the DRT gmma
#     # + 1 if a resistor or an inductor is included in the DRT model
#     h: NDArray[float64] = zeros([Z_re.shape[0] + 1])
#     G: NDArray[float64] = -eye(h.shape[0])

#     # quadratic programming through cvxopt
#     H_re: NDArray[float64]
#     c_re: NDArray[float64]
#     gamma_ridge_re: NDArray[float64]
#     H_re, c_re = _quad_format(A_re, Z_re, M, lambda_value)
#     gamma_ridge_re = _solve_qp_cvxopt(H_re, c_re, G=G, h=h)

#     H_im: NDArray[float64]
#     c_im: NDArray[float64]
#     gamma_ridge_im: NDArray[float64]
#     H_im, c_im = _quad_format(A_im, Z_im, M, lambda_value)
#     gamma_ridge_im = _solve_qp_cvxopt(H_im, c_im, G=G, h=h)

#     # stacking the resistance R and inductance L on top of gamma_ridge_im and gamma_ridge_re, repectively
#     gamma_ridge_re_cv: NDArray[float64] = concatenate(
#         (array([0, gamma_ridge_re[1]]), gamma_ridge_im[2:])
#     )
#     gamma_ridge_im_cv: NDArray[float64] = concatenate(
#         (array([gamma_ridge_im[0], 0]), gamma_ridge_re[2:])
#     )

#     # See eq. 13 in https://doi.org/10.1016/j.electacta.2014.09.058
#     # or eq. (17) in https://doi.org/10.1149/1945-7111/acbca4
#     re_im_cv_score: float64 = (
#         norm(Z_re - A_re @ gamma_ridge_re_cv) ** 2
#         + norm(Z_im - A_im @ gamma_ridge_im_cv) ** 2
#     )

#     return re_im_cv_score


# TODO: Refactor and add type hints
def _compute_L_curve(
    ln_lambda: float64,
    A_re: NDArray[float64],
    A_im: NDArray[float64],
    Z_re: NDArray[float64],
    Z_im: NDArray[float64],
    M: NDArray[float64],
    show_lam: bool = False,
) -> float64:
    """
    This function computes the score for L curve (LC)

    Reference: P.C. Hansen, D.P. O’Leary, The use of the L-curve in the regularization of discrete ill-posed problems, SIAM J. Sci. Comput. 14 (1993) 1487–1503.
    """

    lambda_value = exp(ln_lambda)

    if show_lam:
        print(f"Lambda value: {lambda_value}")

    A = concatenate(
        (A_re, A_im), axis=0
    )  # matrix A with A_re and A_im; # see (5) in [4]
    Z = concatenate((Z_re, Z_im), axis=0)  # stacked impedance

    # numerator eta_num of the first derivative of eta = log(||Z_exp - Ax||^2)
    A_agm = A.T @ A + lambda_value * M  # see (13) in [4]
    if not _is_positive_definite(A_agm):
        A_agm = _nearest_positive_definite(A_agm)

    L_agm = cholesky(A_agm)  # Cholesky transform to inverse A_agm
    inv_L_agm = inv(L_agm)
    inv_A_agm = inv_L_agm.T @ inv_L_agm  # inverse of A_agm
    A_LC = A @ ((inv_A_agm.T @ inv_A_agm) @ inv_A_agm) @ A.T
    eta_num = Z.T @ A_LC @ Z

    # denominator eta_denom of the first derivative of eta
    A_agm_d = A @ A.T + lambda_value * eye(A.shape[0])
    if not _is_positive_definite(A_agm_d):
        A_agm_d = _nearest_positive_definite(A_agm_d)

    L_agm_d = cholesky(A_agm_d)  # Cholesky transform to inverse A_agm_d
    inv_L_agm_d = inv(L_agm_d)
    inv_A_agm_d = inv_L_agm_d.T @ inv_L_agm_d
    eta_denom = lambda_value * Z.T @ (inv_A_agm_d.T @ inv_A_agm_d) @ Z

    # derivative of eta
    eta_prime = eta_num / eta_denom

    # numerator theta_num of the first derivative of theta = log(lambda*||Lx||^2)
    theta_num = eta_num

    # denominator theta_denom of the first derivative of theta
    A_LC_d = A @ (inv_A_agm.T @ inv_A_agm) @ A.T
    theta_denom = Z.T @ A_LC_d @ Z

    # derivative of theta
    theta_prime = -(theta_num) / theta_denom

    # numerator LC_num of the LC score in (19) in [4]
    a_sq = (eta_num / (eta_denom * theta_denom)) ** 2
    p = (Z.T @ (inv_A_agm_d.T @ inv_A_agm_d) @ Z) * theta_denom
    m = (
        2 * lambda_value * Z.T @ ((inv_A_agm_d.T @ inv_A_agm_d) @ inv_A_agm_d) @ Z
    ) * theta_denom
    q = (2 * lambda_value * Z.T @ (inv_A_agm_d.T @ inv_A_agm_d) @ Z) * eta_num
    LC_num = a_sq * (p + m - q)

    # denominator LC_denom of the LC score
    LC_denom = ((eta_prime) ** 2 + (theta_prime) ** 2) ** (3 / 2)

    # LC score ; see (19) in [4]
    LC_score = np.abs(LC_num / LC_denom)

    return LC_score


_LAMBDA_SELECTION_METHODS: Dict[str, Callable] = {
    "fixed": None,  # Fixed lambda
    "gcv": _compute_generalized_cross_validation,  # Generalized cross-validation
    "mgcv": _compute_modified_gcv,  # Modified GCV
    "rgcv": _compute_robust_gcv,  # Robust GCV
    # "re-im": _compute_re_im_cross_validation,  # Real-imaginary cross-validation # disabled because I am not sure how additional series C will change calculation
    # "kf": _compute_,  # k-fold GCV  # TODO: Implement? Requires scikit-learn
    "lc": _compute_L_curve,  # L-curve
}


def _pick_lambda(
    A_re: NDArray[float64],
    A_im: NDArray[float64],
    Z_re: NDArray[float64],
    Z_im: NDArray[float64],
    M: NDArray[float64],
    lambda_value0: float,
    lambda_selection: str,
    method: str = "basinhopping",
    bounds: list = None,
    options: dict = None,
) -> float:

    if lambda_selection == "fixed":
        lambda_value = lambda_value0
    else:
        if bounds is None:
            bounds = [(np.log(1e-9), np.log(1e0))]
        if options is None:
            options = {"disp": False}

        show_lam = options["disp"]

        # Choose optimization strategy based on method
        if method == "differential_evolution":
            result = differential_evolution(
                _LAMBDA_SELECTION_METHODS[lambda_selection],
                bounds=bounds,
                args=(A_re, A_im, Z_re, Z_im, M, show_lam),
                maxiter=options.get("maxiter", 100),
                atol=options.get("atol", 0),
                tol=options.get("tol", 0.01),
                disp=options.get("disp", False),
                seed=options.get("seed", 42),
            )
            lambda_value = float(np.exp(result.x[0]))

        elif method == "basinhopping":
            minimizer_kwargs = {
                "method": options.get("local_method", "L-BFGS-B"),
                "args": (A_re, A_im, Z_re, Z_im, M, show_lam),
                "bounds": bounds,
            }
            result = basinhopping(
                _LAMBDA_SELECTION_METHODS[lambda_selection],
                np.log(lambda_value0),
                minimizer_kwargs=minimizer_kwargs,
                niter=options.get("niter", 50),
                disp=options.get("disp", False),
            )
            lambda_value = float(np.exp(result.x[0]))

        else:
            # Standard local optimization (e.g., "L-BFGS-B", "SLSQP", "Powell")
            result = minimize(
                _LAMBDA_SELECTION_METHODS[lambda_selection],
                np.log(lambda_value0),
                args=(A_re, A_im, Z_re, Z_im, M, show_lam),
                method=method,
                bounds=bounds,
                options=options,
            )
            lambda_value = float(np.exp(result.x)[0])

    return lambda_value
