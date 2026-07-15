# Based on code from https://github.com/ciuccislab/pyDRTtools and https://github.com/vyrjana/pyimpspec
# This module uses Tikhonov regularization and either radial basis function or piecewise linear discretization
# - 10.1016/j.electacta.2015.09.097 # radial basis functions for DRT
# - 10.1149/1945-7111/acbca4 # hyperparameter selection
# - 10.1021/acselectrochem.5c00334 # DRTtools

"""
Quadratic Programming (QP) solver utilities for DRT computation.

Provides functions to set up and solve QP problems using CVXOPT/KVXOPT.
The standard QP problem is formulated as:
    minimize    (1/2) x^T H x + c^T x
    subject to  G x <= h  (inequality constraints)
                A x = b   (equality constraints)

For DRT computation, the least squares problem with Tikhonov regularization:
    minimize ||Ax - b||^2 + λ||Mx||^2
is reformulated as a QP problem where:
    H = 2(A^T A + λM)
    c = -2b^T A

References:
    - CVXOPT: https://cvxopt.org/
    - KVXOPT: https://github.com/rajatk/kvxopt
"""

from redoxed import config

if getattr(config, "QP_SOLVER", "cvxopt") == "kvxopt":
    from kvxopt import matrix, solvers
else:
    from cvxopt import matrix, solvers

from typing import List, Tuple
from numpy.typing import NDArray

from numpy import array, float64


def _quad_format(
    A: NDArray[float64],
    b: NDArray[float64],
    M: NDArray[float64],
    lambda_value: float,
) -> Tuple[NDArray[float64], NDArray[float64]]:
    """
    Convert a regularized least squares problem to standard QP format.

    Transforms the optimization problem:
        minimize ||Ax - b||^2 + λ x^T M x

    into standard QP form:
        minimize (1/2) x^T H x + c^T x

    where:
        H = 2(A^T A + λM)
        c = -2b^T A

    The factor of 2 accounts for the (1/2) in the standard QP formulation.
    The matrix H is symmetrized to ensure numerical stability.

    Parameters
    ----------
    A : NDArray[float64]
        Discretisation matrix relating DRT coefficients to impedance data.
    b : NDArray[float64]
        Observed data vector (impedance measurements).
    M : NDArray[float64]
        Regularization matrix (typically a derivative operator).
    lambda_value : float
        Regularization parameter controlling smoothness vs. fit quality.

    Returns
    -------
    Tuple[NDArray[float64], NDArray[float64]]
        H : Quadratic term matrix (symmetric, positive semi-definite)
        c : Linear term vector
    """
    H: NDArray[float64] = 2 * (A.T @ A + lambda_value * M)
    H = (H.T + H) / 2  # Symmetrize for numerical stability
    c: NDArray[float64] = -2 * b.T @ A

    return (
        H,
        c,
    )


def _quad_format_combined(
    A_re: NDArray[float64],
    A_im: NDArray[float64],
    b_re: NDArray[float64],
    b_im: NDArray[float64],
    M: NDArray[float64],
    lambda_value: float,
) -> Tuple[NDArray[float64], NDArray[float64]]:
    """
    Convert a combined real-imaginary regularized least squares problem to QP format.

    Transforms the combined optimization problem:
        minimize ||A_re x - b_re||^2 + ||A_im x - b_im||^2 + λ x^T M x

    into standard QP form:
        minimize (1/2) x^T H x + c^T x

    where:
        H = 2((A_re^T A_re + A_im^T A_im) + λM)
        c = -2(b_re^T A_re + b_im^T A_im)

    This formulation allows fitting both real and imaginary parts of impedance
    simultaneously with a shared regularization parameter.

    Parameters
    ----------
    A_re : NDArray[float64]
        Discretisation matrix for real part of impedance.
    A_im : NDArray[float64]
        Discretisation matrix for imaginary part of impedance.
    b_re : NDArray[float64]
        Observed real part of impedance.
    b_im : NDArray[float64]
        Observed imaginary part of impedance.
    M : NDArray[float64]
        Regularization matrix (typically a derivative operator).
    lambda_value : float
        Regularization parameter controlling smoothness vs. fit quality.

    Returns
    -------
    Tuple[NDArray[float64], NDArray[float64]]
        H : Quadratic term matrix (symmetric, positive semi-definite)
        c : Linear term vector
    """
    H: NDArray[float64] = 2 * ((A_re.T @ A_re + A_im.T @ A_im) + lambda_value * M)
    H = (H.T + H) / 2  # Symmetrize for numerical stability
    c: NDArray[float64] = -2 * (b_im.T @ A_im + b_re.T @ A_re)

    return (
        H,
        c,
    )


def _solve_qp_cvxopt(
    H: NDArray[float64],
    c: NDArray[float64],
    G: NDArray[float64] | None = None,
    h: NDArray[float64] | None = None,
    A: NDArray[float64] | None = None,
    b: NDArray[float64] | None = None,
    solver_options: dict | None = None,
) -> NDArray[float64]:
    """
    Solve a quadratic programming problem using CVXOPT/KVXOPT.

    Solves the standard QP problem:
        minimize    (1/2) x^T H x + c^T x
        subject to  G x <= h  (inequality constraints, optional)
                    A x = b   (equality constraints, optional)

    Constraint handling:
    -------------------
    - If G and h are provided: applies inequality constraints G x <= h
      (typically used for non-negativity: -I x <= 0, i.e., x >= 0)
    - If A and b are provided: applies equality constraints A x = b
      (typically used for sum constraints, e.g., sum(x) = R_pol)
    - If no constraints: solves unconstrained QP

    Mathematical equivalence:
    ------------------------
    For DRT with non-negativity constraint:
        G = -I (negative identity matrix)
        h = 0  (zero vector)
        Result: -I x <= 0  =>  x >= 0 (all elements non-negative)

    Parameters
    ----------
    H : NDArray[float64]
        Quadratic term matrix (must be symmetric, positive semi-definite).
    c : NDArray[float64]
        Linear term vector.
    G : NDArray[float64] | None, optional
        Inequality constraint matrix. Shape (m, n) where n is size of x.
        Mathematical constraint: G x <= h
    h : NDArray[float64] | None, optional
        Inequality constraint bound vector. Shape (m,).
        Mathematical constraint: G x <= h
    A : NDArray[float64] | None, optional
        Equality constraint matrix. Shape (p, n) where n is size of x.
        Mathematical constraint: A x = b
    b : NDArray[float64] | None, optional
        Equality constraint bound vector. Shape (p,).
        Mathematical constraint: A x = b

    Returns
    -------
    NDArray[float64]
        Solution vector x that minimizes the objective function.

    Raises
    ------
    ValueError
        If the solver fails to find an optimal solution.

    Notes
    -----
    Solver tolerances are set to 1e-15 for both absolute and relative convergence
    to ensure high-precision solutions for ill-conditioned DRT problems.
    """

    args: List[matrix] = [matrix(H), matrix(c)]

    # Add inequality constraints G x <= h if provided
    if G is not None:
        args.extend([matrix(G), matrix(h)])

    # Add equality constraints A x = b if provided
    if A is not None:
        args.extend([matrix(A), matrix(b)])

    # Set solver tolerances (defaults or user-supplied)
    if solver_options is not None:
        for k, v in solver_options.items():
            solvers.options[k] = v
    else:
        solvers.options["abstol"] = 1e-15
        solvers.options["reltol"] = 1e-15

    # Solve the QP problem
    solution: dict = solvers.qp(
        *args,
        options={"show_progress": False},
    )

    # Check if solution converged
    if "optimal" not in solution["status"]:
        raise ValueError("Failed to find optimal solution")

    # Extract and reshape solution vector
    return array(solution["x"]).reshape((H.shape[1],))
