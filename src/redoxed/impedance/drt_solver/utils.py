# Based on code from https://github.com/ciuccislab/pyDRTtools and https://github.com/vyrjana/pyimpspec
# This module uses Tikhonov regularization and either radial basis function or piecewise linear discretization
# - 10.1016/j.electacta.2015.09.097 # radial basis functions for DRT
# - 10.1149/1945-7111/acbca4 # hyperparameter selection
# - 10.1021/acselectrochem.5c00334 # DRTtools

from numpy.linalg import (
    LinAlgError,
    cholesky,
    svd,
    eigvals,
    norm,
)

from numpy import diag, dot, eye, spacing, min as array_min, float64

from typing import List, Dict, Any
from numpy.typing import NDArray

from multiprocessing import cpu_count
from os import environ
from numpy import (
    __config__ as numpy_config,
)


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


NUM_PROCS_OVERRIDE: int = -1


def get_default_num_procs() -> int:
    """
    Taken from pyimpspec in case I want multiprocessing in DRT solver.

    Get the default number of parallel processes that pyimpspec would try to use.
    NumPy may be using libraries that are multithreaded, which can lead to poor performance or system responsiveness when combined with pyimpspec's use of multiple processes.
    This function attempts to return a reasonable number of processes depending on the detected libraries (and relevant environment variables):

    - OpenBLAS (``OPENBLAS_NUM_THREADS``)
    - MKL (``MKL_NUM_THREADS``)

    If none of the libraries listed above are detected because some other library is used, then the value returned by ``multiprocessing.cpu_count()`` is used.

    Returns
    -------
    int
    """
    if NUM_PROCS_OVERRIDE > 0:
        return NUM_PROCS_OVERRIDE
    num_cores: int = cpu_count()

    multithreaded: Dict[str, List[str]] = {
        "openblas": ["OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS", "OMP_NUM_THREADS"],
        "mkl": ["MKL_NUM_THREADS"],
    }
    libraries = set()

    if hasattr(numpy_config, "CONFIG"):
        key: str
        obj: Any
        for key, obj in numpy_config.CONFIG.items():
            if not isinstance(obj, dict):
                continue

            blas_config: dict = obj.get("blas", {})
            if not blas_config or not blas_config.get("found", False):
                continue

            lib: str
            for lib in multithreaded:
                if lib in blas_config.get("name", ""):
                    libraries.add(lib)

    name: str
    for name in libraries:
        envs: List[str]
        for lib, envs in multithreaded.items():
            if lib in name:
                num_threads: int = -1
                for env in envs:
                    try:
                        num_threads = int(environ.get(env, ""))
                    except ValueError:
                        continue
                    else:
                        break

                if num_threads < 0:
                    # Assume that the library will use as many threads as there
                    # are cores available to the system.
                    return 1
                elif num_threads == 1:
                    return num_cores
                else:
                    num_procs: int = num_cores // num_threads
                    return num_procs if num_procs > 1 else 1

    return num_cores
