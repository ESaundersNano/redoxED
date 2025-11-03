# pyimpspec is licensed under the GPLv3 or later (https://www.gnu.org/licenses/gpl-3.0.html).
# Copyright 2025 pyimpspec developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# The licenses of pyimpspec's dependencies and/or sources of portions of code are included in
# the LICENSES folder.

# This module uses Tikhonov regularization and either radial basis function or piecewise linear discretization
# - 10.1016/j.electacta.2015.09.097
# - 10.1016/j.electacta.2015.03.123
# - 10.1016/j.electacta.2017.07.050
# Based on code from https://github.com/ciuccislab/pyDRTtools.
# pyDRTtools commit: 1653298d52183c36ec941197ae59399b9dc85579

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from redoxed.data_loading.data_loaders import LoaderFactory, BiologicLoader, CSVLoader
from redoxed.plots import NyquistPlot, DRTPlot, ResidualsPlot, PolarisationPlot
from redoxed.impedance import EISData, DRTData, ResidualsData
from redoxed.impedance.drt_analysis import (
    HN_Z,
    HN_DRT,
    SG_DRT,
    fit_DRT_peaks,
    find_DRT_peaks,
    Z_from_DRT,
)
from redoxed.dc import PolarisationData

from lmfit.minimizer import minimize, MinimizerResult
from lmfit.parameter import Parameters
import numpy as np
from scipy.signal import find_peaks
from redoxed.impedance.drt_analysis import HN_DRT, SG_DRT
from typing import List, Tuple, Dict, Callable
from numpy.typing import NDArray
from numpy import float64
from dataclasses import dataclass
from scipy.integrate import quad
import pandas as pd
from redoxed.impedance import DRTData

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

_RBF_SHAPES: List[str] = ["fwhm", "factor"]

_MODES: List[str] = ["complex", "real", "imaginary"]

_CROSS_VALIDATION_METHODS: Dict[str, Callable] = {
    "gcv": _compute_generalized_cross_validation,  # Generalized cross-validation
    "mgcv": _compute_modified_gcv,  # Modified GCV
    "rgcv": _compute_robust_gcv,  # Robust GCV
    "re-im": _compute_re_im_cross_validation,  # Real-imaginary cross-validation
    # "kf": _compute_,  # k-fold GCV  # TODO: Implement? Requires scikit-learn
    "lc": _compute_L_curve,  # L-curve
}


class DRT_rr_solver:
    """
    A class to solve the Distribution of Relaxation Times (DRT) using
    the Regularized Regression with Radial Basis Functions (RR-RBF) method.
    """

    def __init__(
        self,
        data: EISData,
        mode: str = "complex",
        lambda_value: float = 0,
        cross_validation: str = "gcv",
        rbf_type: str = "gaussian",
        derivative_order: int = 1,
        rbf_shape: str = "fwhm",
        shape_coeff: float = 0.5,
        inductance: bool = True,
        num_procs: int = -1,
    ) -> None:
        self.data = data
        self.mode = mode
        self.lambda_value = lambda_value
        self.cross_validation = cross_validation
        self.rbf_type = rbf_type
        self.derivative_order = derivative_order
        self.rbf_shape = rbf_shape
        self.shape_coeff = shape_coeff
        self.inductance = inductance
        self.num_procs = num_procs

    def calculate_drt(self) -> DRTData:
        # Implement the DRT calculation logic here
        pass

        """
        Calculates the distribution of relaxation times (DRT) for a given data set using regularisation and radial basis (or piecewise linear) discretization (TR-RBF method).

        References:

        - `Wan, T. H., Saccoccio, M., Chen, C., and Ciucci, F., 2015, Electrochim. Acta, 184, 483-499 <https://doi.org/10.1016/j.electacta.2015.09.097>`_
        - `Effat, M. B. and Ciucci, F., 2017, Electrochim. Acta, 247, 1117-1129 <https://doi.org/10.1016/j.electacta.2017.07.050>`_

        Parameters
        ----------

        """

        data = self.data
        mode = self.mode
        lambda_value = self.lambda_value
        cross_validation = self.cross_validation
        rbf_type = self.rbf_type
        derivative_order = self.derivative_order
        rbf_shape = self.rbf_shape
        shape_coeff = self.shape_coeff
        inductance = self.inductance
        num_procs = self.num_procs

        if not isinstance(mode, str):
            raise TypeError(f"Expected a string instead of {mode=}")
        elif mode not in _MODES:
            raise ValueError("Valid mode values: '" + "', '".join(_MODES))

        if not isinstance(cross_validation, str):
            raise TypeError(f"Expected a string or None instead of {cross_validation=}")
        elif not (
            cross_validation == "" or cross_validation in _CROSS_VALIDATION_METHODS
        ):
            raise ValueError(
                "Valid cross-validation methods include:\n- "
                + "\n- ".join(_CROSS_VALIDATION_METHODS.keys())
            )
        elif cross_validation != "" and not (1e-7 <= lambda_value < 1.0):
            if lambda_value <= 0.0:
                lambda_value = 1e-3
            else:
                # These are the bounds that are currently used by the _pick_lambda function.
                raise ValueError(f"Expected 1e-7 <= {lambda_value=} < 1.0")

        if not _is_floating(lambda_value):
            raise TypeError(f"Expected a float instead of {lambda_value=}")
        elif not lambda_value > 0.0:
            raise ValueError(
                f"Expected a value greater than zero instead of {lambda_value=}"
            )

        if not isinstance(rbf_type, str):
            raise TypeError(f"Expected a string instead of {rbf_type}")
        elif rbf_type not in _RBF_TYPES:
            raise ValueError("Valid rbf_type values: '" + "', '".join(_RBF_TYPES))

        if not _is_integer(derivative_order):
            raise TypeError(f"Expected an integer instead of {derivative_order=}")
        elif not (1 <= derivative_order <= 2):
            raise ValueError("Valid derivative_order values: 1, 2")

        if not isinstance(rbf_shape, str):
            raise TypeError(f"Expected a string instead of {rbf_shape=}")
        elif rbf_shape not in _RBF_SHAPES:
            raise ValueError("Valid rbf_shape values: '" + "', '".join(_RBF_SHAPES))

        if not _is_floating(shape_coeff):
            raise TypeError(f"Expected a float instead of {shape_coeff=}")
        elif shape_coeff <= 0.0:
            raise ValueError("The shape coefficient must be greater than 0.0")

        if not _is_boolean(inductance):
            raise TypeError(f"Expected a boolean instead of {inductance=}")

        if not _is_boolean(credible_intervals):
            raise TypeError(f"Expected a boolean instead of {credible_intervals=}")

        if not _is_integer(num_samples):
            raise TypeError(f"Expected an integer instead of {num_samples=}")
        elif credible_intervals and num_samples < 1000:
            raise ValueError(
                "The number of samples must be greater than or equal to 1000"
            )

        if not _is_integer(timeout):
            raise TypeError(f"Expected an integer instead of {timeout=}")

        if not _is_integer(num_procs):
            raise TypeError(f"Expected an integer instead of {num_procs=}")
        elif num_procs < 1:
            num_procs = max((get_default_num_procs() - abs(num_procs), 1))

        # TODO: Figure out if f and Z need to be altered depending on the value
        # of the 'inductance' argument!
        f: Frequencies = data.get_frequencies()
        if len(f) < 1:
            raise ValueError(
                f"There are no unmasked data points in the '{data.get_label()}' data set parsed from '{data.get_path()}'"
            )

        Z_exp: ComplexImpedances = data.get_impedances()

        tau: TimeConstants = 1 / f
        tau_fine: TimeConstants = logspace(
            log(tau.min()) - 0.5, log(tau.max()) + 0.5, 10 * f.shape[0]
        )
        num_freqs: int = f.size
        num_taus: int = tau.size
        epsilon: float = _compute_epsilon(f, rbf_shape, shape_coeff, rbf_type)

        num_steps: int = 0
        num_steps += 3  # A_re, A_im, and M matrices
        if credible_intervals:
            num_steps += num_samples

        prog: Progress
        with Progress("Preparing matrices", total=num_steps + 1) as prog:
            i: int
            args = [
                (
                    f,
                    tau,
                    epsilon,
                    True,
                    rbf_type,
                ),
                (
                    f,
                    tau,
                    epsilon,
                    False,
                    rbf_type,
                ),
            ]

            A_re: NDArray[float64]
            A_im: NDArray[float64]
            if num_procs > 1:
                with get_context(method="spawn").Pool(2) as pool:
                    for i, res in enumerate(pool.imap(_assemble_A_matrix, args)):
                        if i == 0:
                            A_re = res
                        else:
                            A_im = res
                        prog.increment()
            else:
                A_re = _assemble_A_matrix(args[0])
                prog.increment()
                A_im = _assemble_A_matrix(args[1])
                prog.increment()

            M: NDArray[float64] = _assemble_M_matrix(
                tau,
                epsilon,
                derivative_order,
                rbf_type,
            )

            b_re: NDArray[float64] = Z_exp.real
            b_im: NDArray[float64] = Z_exp.imag

            num_RL: int = -1
            if mode == "complex":
                A_re, A_im, M, num_RL = _prepare_complex_matrices(
                    A_re,
                    A_im,
                    M,
                    f,
                    num_freqs,
                    num_taus,
                    inductance,
                )
            elif mode == "real":
                A_re, A_im, M, num_RL = _prepare_real_matrices(
                    A_re,
                    A_im,
                    M,
                    num_freqs,
                    num_taus,
                )
            elif mode == "imaginary":
                A_re, A_im, M, num_RL = _prepare_imaginary_matrices(
                    A_re,
                    A_im,
                    M,
                    f,
                    num_freqs,
                    num_taus,
                    inductance,
                )

            if cross_validation != "":
                prog.set_message("Picking lambda value")
                lambda_value = _pick_lambda(
                    A_re,
                    A_im,
                    b_re,
                    b_im,
                    M,
                    lambda_value,
                    cross_validation,
                )

            prog.increment()
            prog.set_message("Calculating DRT")

            H: NDArray[float64]
            c: NDArray[float64]
            if mode == "complex":
                H, c = _quad_format_combined(
                    A_re,
                    A_im,
                    b_re,
                    b_im,
                    M,
                    lambda_value,
                )
            elif mode == "real":
                H, c = _quad_format(
                    A_re,
                    b_re,
                    M,
                    lambda_value,
                )
            elif mode == "imaginary":
                H, c = _quad_format(
                    A_im,
                    b_im,
                    M,
                    lambda_value,
                )

            if not (0 <= num_RL <= 2, num_RL):
                raise ValueError(f"Expected 0 <= {num_RL=} = 2")

            # Enforce positivity constraint
            h: NDArray[float64] = zeros(b_re.shape[0] + num_RL)
            G: NDArray[float64] = -eye(h.shape[0])
            x: NDArray[float64] = _solve_qp_cvxopt(
                H,
                c,
                G=G,
                h=h,
            )

            Z_fit: ComplexImpedances = array(
                list(map(lambda _: complex(*_), zip(A_re @ x, A_im @ x))),
                dtype=ComplexImpedance,
            )

        sigma_re_im: float
        if mode == "complex":
            sigma_re_im = std(concatenate([Z_fit.real - b_re, Z_fit.imag - b_im]))

        elif mode == "real":
            sigma_re_im = std(Z_fit.real - b_re)

        elif mode == "imaginary":
            sigma_re_im = std(Z_fit.imag - b_im)

        inv_V: NDArray[float64] = 1 / sigma_re_im**2 * eye(num_freqs)

        Sigma_inv: NDArray[float64]
        mu_numerator: NDArray[float64]
        if mode == "complex":
            Sigma_inv = (
                (A_re.T @ inv_V @ A_re)
                + (A_im.T @ inv_V @ A_im)
                + (lambda_value / sigma_re_im**2) * M
            )
            mu_numerator = A_re.T @ inv_V @ b_re + A_im.T @ inv_V @ b_im

        elif mode == "real":
            Sigma_inv = (A_re.T @ inv_V @ A_re) + (lambda_value / sigma_re_im**2) * M
            mu_numerator = A_re.T @ inv_V @ b_re

        elif mode == "imaginary":
            Sigma_inv = (A_im.T @ inv_V @ A_im) + (lambda_value / sigma_re_im**2) * M
            mu_numerator = A_im.T @ inv_V @ b_im

        Sigma_inv = (Sigma_inv + Sigma_inv.T) / 2
        if not _is_positive_definite(Sigma_inv):
            Sigma_inv = _nearest_positive_definite(Sigma_inv)

        L_Sigma_inv: NDArray[float64] = cholesky(Sigma_inv)
        mu: NDArray[float64] = solve(
            L_Sigma_inv.T,
            solve(L_Sigma_inv, mu_numerator),
        )

        # These L and R values are used by pyDRTtools when exporting a DRT report
        # as a CSV file.
        L: float
        R: float
        if num_RL == 0:
            L, R = 0.0, 0.0
        elif num_RL == 1:
            if mode == "imaginary":
                L, R = x[0], 0.0
            else:
                L, R = 0.0, x[0]
        elif num_RL == 2:
            L, R = x[0:2]

        x = x[num_RL:]
        time_constants: TimeConstants
        time_constants, gamma = _x_to_gamma(x, tau_fine, tau, epsilon, rbf_type)

        if credible_intervals:
            prog.set_message("Calculating credible intervals")
            mean_gamma, lower_gamma, upper_gamma = _calculate_credible_intervals(
                num_RL,
                num_samples,
                mu,
                Sigma_inv,
                x,
                tau_fine,
                tau,
                epsilon,
                rbf_type,
                timeout,
                prog,
            )
        else:
            mean_gamma, lower_gamma, upper_gamma = (
                array([]),  # Mean
                array([]),  # Lower bound
                array([]),  # Upper bound
            )

        return TRRBFResult(
            time_constants=time_constants,
            gammas=gamma,
            frequencies=f,
            impedances=Z_fit,
            residuals=_calculate_residuals(Z_exp, Z_fit),
            mean_gammas=mean_gamma,
            lower_bounds=lower_gamma,
            upper_bounds=upper_gamma,
            pseudo_chisqr=_calculate_pseudo_chisqr(Z_exp, Z_fit),
            lambda_value=lambda_value,
        )
