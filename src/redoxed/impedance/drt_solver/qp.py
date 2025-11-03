try:
    from cvxopt import (
        matrix,
        solvers,
    )
except ImportError:
    from kvxopt import (
        matrix,
        solvers,
    )

from typing import List, Tuple
from numpy.typing import NDArray, float64

from numpy import array


def _quad_format(
    A: NDArray[float64],
    b: NDArray[float64],
    M: NDArray[float64],
    lambda_value: float,
) -> Tuple[NDArray[float64], NDArray[float64]]:
    H: NDArray[float64] = 2 * (A.T @ A + lambda_value * M)
    H = (H.T + H) / 2
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
    H: NDArray[float64] = 2 * ((A_re.T @ A_re + A_im.T @ A_im) + lambda_value * M)
    H = (H.T + H) / 2
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
) -> NDArray[float64]:

    args: List[matrix] = [matrix(H), matrix(c)]

    if G is not None:
        if not _is_floating_array(h):
            raise TypeError(f"Expected an NDArray[floating] instead of {h=}")

        args.extend([matrix(G), matrix(h)])

    if A is not None:
        if not _is_floating_array(b):
            raise TypeError(f"Expected an NDArray[floating] instead of {b=}")

        args.extend([matrix(A), matrix(b)])

    solvers.options["abstol"] = 1e-15
    solvers.options["reltol"] = 1e-15
    solution: dict = solvers.qp(
        *args,
        options={"show_progress": False},
    )

    if "optimal" not in solution["status"]:
        raise ValueError("Failed to find optimal solution")

    return array(solution["x"]).reshape((H.shape[1],))
