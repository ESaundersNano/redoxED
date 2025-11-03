from numpy.linalg import (
    LinAlgError,
    cholesky,
    svd,
    eigvals,
    norm,
)

from numpy import (
    diag,
    dot,
    eye,
    spacing,
    min as array_min,
)

from typing import List, Tuple
from numpy.typing import NDArray, float64

from numpy import array


def _nearest_positive_definite(A: NDArray) -> NDArray:
    """
    Find the nearest positive definite matrix of the input matrix A.

    Based on John D'Errico's "nearestSPD" (https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd).
    Ported by the developers of pyDRTtools.

    See also:
    - N.J. Higham, "Computing a nearest symmetric positive semidefinite matrix" (1988, https://doi.org/10.1016/0024-3795(88)90223-6)
    """

    B: NDArray = (A + A.T) / 2
    Sigma_mat: NDArray
    V: NDArray
    _, Sigma_mat, V = svd(B)

    H: NDArray = dot(V.T, dot(diag(Sigma_mat), V))

    A_nPD: NDArray = (B + H) / 2
    A_symm: NDArray = (A_nPD + A_nPD.T) / 2

    k: int = 1
    I: NDArray = eye(A_symm.shape[0])

    while not _is_positive_definite(A_symm):
        # The MATLAB function chol accepts matrices with eigenvalue = 0,
        # but NumPy does not so we replace the MATLAB function eps(min_eig)
        # with the following one
        eps: float = spacing(norm(A_symm))
        min_eig: float = min(0.0, array_min(eigvals(A_symm).real))
        A_symm += I * (-min_eig * k**2 + eps)
        k += 1

    return A_symm


def _is_positive_definite(matrix: NDArray) -> bool:
    """
    Check if a matrix is positive definite.

    Args:
        matrix (NDArray): The matrix to check.

    Returns:
        bool: True if the matrix is positive definite, False otherwise.
    """
    try:
        cholesky(matrix)
        return True
    except LinAlgError:
        return False
