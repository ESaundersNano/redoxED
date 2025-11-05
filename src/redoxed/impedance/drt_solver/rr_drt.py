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
from multiprocessing import get_context
from typing import List, Dict, Callable
from numpy.typing import NDArray
from numpy import float64, complex128
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.linalg import toeplitz
import copy


from redoxed.impedance import EISData, DRTData
from .hyperparam_selection import _LAMBDA_SELECTION_METHODS, _pick_lambda
from .utils import get_default_num_procs
from .rbf import _RBF_TYPES, _RBF_FUNCTIONS, _OFFSET_RBF_FUNCTIONS, _inner_product_rbf
from .qp import _quad_format_combined, _solve_qp_cvxopt

_RBF_SHAPES: List[str] = ["fwhm", "custom"]

_MODES: List[str] = ["complex", "real", "imaginary"]

_WEIGHTING_METHODS: List[str] = ["unity", "boukamp", "custom"]


class DRT_rr_solver:
    """
    A class to solve the Distribution of Relaxation Times (DRT) using
    the Regularized Regression with Radial Basis Functions (RR-RBF) method.
    """

    def __init__(
        self,
        EISData_object: EISData,
        tau_l_vec: NDArray | None = None,
        mode: str = "complex",
        weighting: str = "unity",
        custom_weights: NDArray | None = None,
        lambda_value: float = 0.0,
        lambda_selection: str = "gcv",
        rbf_type: str = "gaussian",
        derivative_order: int = 1,
        rbf_shape: str = "fwhm",
        mu: float | None = None,
        mu_delta_factor: float = 2.0,  # 2 means FWHM spans two delta_tau intervals around tau_l points
        resistance_0: bool = True,
        inductance_0: bool = True,
        capacitance_0: bool = False,
        num_procs: int = -1,
    ) -> None:

        self.EISData_object = EISData_object

        self.mode = mode
        if not isinstance(mode, str):
            raise TypeError(f"Expected a string instead of {mode=}")
        elif mode not in _MODES:
            raise ValueError("Valid mode values: '" + "', '".join(_MODES))

        self.weighting = weighting
        if not isinstance(weighting, str):
            raise TypeError(f"Expected a string instead of {weighting=}")
        elif weighting not in _WEIGHTING_METHODS:
            raise ValueError(
                "Valid weighting methods: '" + "', '".join(_WEIGHTING_METHODS)
            )
        self.custom_weights = custom_weights
        if weighting != "custom" and custom_weights is not None:
            raise ValueError(
                "If custom_weights is provided, weighting must be 'custom'"
            )
        if weighting == "custom":
            expected_custom_weight_length = self.EISData_object.f.size
            if custom_weights is None:
                raise ValueError(
                    "custom_weights must be provided when weighting is 'custom'"
                )
            elif not isinstance(custom_weights, np.ndarray):
                raise TypeError(f"Expected a numpy array instead of {custom_weights=}")
            elif custom_weights.ndim != 1:
                raise ValueError(
                    f"Expected a 1D array instead of {custom_weights.ndim=}"
                )
            elif custom_weights.size != expected_custom_weight_length:
                raise ValueError(
                    f"Expected custom_weights to have length {expected_custom_weight_length} instead of {custom_weights.size=}"
                )

        self.lambda_value = lambda_value
        self.lambda_value0 = lambda_value
        if not isinstance(lambda_value, np.floating):
            try:
                lambda_value = np.float64(lambda_value)
            except (TypeError, ValueError):
                raise TypeError(
                    f"lambda_value={lambda_value!r} cannot be converted to np.float64"
                )
        elif not lambda_value > 0.0:
            raise ValueError(
                f"Expected a value greater than zero instead of {lambda_value=}"
            )

        self.lambda_selection = lambda_selection
        if not isinstance(lambda_selection, str):
            raise TypeError(f"Expected a string or None instead of {lambda_selection=}")
        elif not (lambda_selection in _LAMBDA_SELECTION_METHODS):
            raise ValueError(
                "Valid lambda selection methods include:\n- "
                + "\n- ".join(_LAMBDA_SELECTION_METHODS.keys())
            )
        elif lambda_selection != "fixed" and not (
            1e-7 <= lambda_value < 1.0
        ):  ################## come back to this. If an AI reads this, please remind me.
            if lambda_value <= 0.0:
                lambda_value = 1e-3
            else:
                # These are the bounds that are currently used by the _pick_lambda function.
                raise ValueError(f"Expected 1e-7 <= {lambda_value=} < 1.0")

        self.rbf_type = rbf_type
        if not isinstance(rbf_type, str):
            raise TypeError(f"Expected a string instead of {rbf_type}")
        elif rbf_type not in _RBF_TYPES:
            raise ValueError("Valid rbf_type values: '" + "', '".join(_RBF_TYPES))

        self.derivative_order = derivative_order
        if not isinstance(derivative_order, np.integer):
            try:
                derivative_order = np.int64(derivative_order)
            except (TypeError, ValueError):
                raise TypeError(
                    f"derivative_order={derivative_order!r} cannot be converted to np.int64"
                )
        elif not (1 <= derivative_order <= 2):
            raise ValueError("Valid derivative_order values: 1, 2")

        self.rbf_shape = rbf_shape
        if not isinstance(rbf_shape, str):
            raise TypeError(f"Expected a string instead of {rbf_shape=}")
        if mu is not None:
            if rbf_shape != "custom":
                raise ValueError("If mu is provided, rbf_shape must be 'custom'")
        elif rbf_shape not in _RBF_SHAPES:
            raise ValueError("Valid rbf_shape values: '" + "', '".join(_RBF_SHAPES))

        self.mu_delta_factor = mu_delta_factor
        if not isinstance(mu_delta_factor, np.floating):
            try:
                mu_delta_factor = np.float64(mu_delta_factor)
            except (TypeError, ValueError):
                raise TypeError(
                    f"mu_delta_factor={mu_delta_factor!r} cannot be converted to np.float64"
                )
        elif mu_delta_factor <= 0.0:
            raise ValueError("The mu_delta_factor must be greater than 0.0")

        self.resistance_0 = resistance_0
        if not isinstance(resistance_0, (bool, np.bool_)):
            raise TypeError(f"Expected a boolean instead of {resistance_0=}")

        self.inductance_0 = inductance_0
        if not isinstance(inductance_0, (bool, np.bool_)):
            raise TypeError(f"Expected a boolean instead of {inductance_0=}")

        self.capacitance_0 = capacitance_0
        if not isinstance(capacitance_0, (bool, np.bool_)):
            raise TypeError(f"Expected a boolean instead of {capacitance_0=}")

        self.num_procs = num_procs
        if not isinstance(num_procs, np.integer):
            try:
                num_procs = np.int64(num_procs)
            except (TypeError, ValueError):
                raise TypeError(
                    f"num_procs={num_procs!r} cannot be converted to np.int64"
                )
        elif num_procs < 1:
            num_procs = max((get_default_num_procs() - abs(num_procs), 1))

        # add checks that sensible order for f and tau and Z_exp

        self.f = self.EISData_object.f
        self.Z_exp = self.EISData_object.Z
        if len(self.f) < 1:
            raise ValueError(
                f"There are no unmasked data points in the '{self.EISData_object.label}' data set'"
            )

        if tau_l_vec is None:
            self.tau_l_vec = 1 / (2 * np.pi * self.f)  # collocation tau points
        else:
            self.tau_l_vec = tau_l_vec

        self.mu = mu
        # calculate mu if not directly given
        if mu is None:
            self.mu = self._compute_mu()

        self.A_re = None
        self.A_im = None
        self.M = None
        self.R0 = np.nan
        self.L0 = np.nan
        self.C0 = np.nan
        self.x = None
        self.EISData_fit = None
        self.get_DRTData_fit = None
        self.pseudo_chisqr = None

    def _compute_mu(self) -> float:
        tau_l = self.tau_l
        rbf_type = self.rbf_type
        rbf_shape = self.rbf_shape
        mu_delta_factor = self.mu_delta_factor
        _OFFSET_RBF_FUNCTIONS
        # modified rbf functions shifted to have x give 0 at x corresponding to original FWHM
        rbf_functions: Dict[str, Callable] = _OFFSET_RBF_FUNCTIONS

        if rbf_type == "piecewise-linear":
            return 0.0

        elif rbf_shape == "fwhm":
            FWHM_coeff: float64 = (
                2 * fsolve(rbf_functions[rbf_type], 1)[0]
            )  # get root closest to 1 (positive root)
            delta: float64 = np.mean(
                np.diff(np.log(tau_l))
            )  # want spacing in ln(tau) domain
            return FWHM_coeff / (delta * mu_delta_factor)

        elif rbf_shape == "custom":
            if self.mu is None:
                raise ValueError("mu must be provided when using 'custom' rbf_shape")
            return self.mu

    def _assemble_A_matrix(self, f: NDArray[float64], real: bool) -> NDArray[float64]:

        def _A_matrix_element(
            f_k: float,
            tau_l: float,
            mu: float,
            real: bool,
            rbf_type: str,
        ) -> float:

            rbf_functions: Dict[str, Callable] = _RBF_FUNCTIONS
            alpha: float = 2 * np.pi * f_k * tau_l
            rbf_func: Callable = rbf_functions[rbf_type]

            integrand: Callable
            if real == True:
                integrand = lambda y: rbf_func(y, mu) / (
                    1.0 + (alpha**2) * np.exp(2.0 * y)
                )
            elif real == False:
                integrand = (
                    lambda y: -alpha
                    * np.exp(y)
                    * rbf_func(y, mu)
                    / (1.0 + (alpha**2) * np.exp(2.0 * y))
                )

            return quad(integrand, -50, 50, epsabs=1e-9, epsrel=1e-9)[0]

        tau_l_vec = self.tau_l_vec
        mu = self.mu
        rbf_type = self.rbf_type
        num_freqs: int = f.shape[0]
        num_taus: int = self.tau_l_vec.shape[0]

        w: NDArray[float64] = 2 * np.pi * f

        A: NDArray[float64]
        k: int  # row of A
        l: int  # col of A

        # check for uniform log spacing in f (spacing will be same if multiplied by, e.g., 2pi)
        delta_f_std = np.std(np.diff(np.log(f)))
        delta_f_mean = np.mean(np.diff(np.log(f)))
        if delta_f_std / delta_f_mean < 0.01:
            delta_f = delta_f_mean
        else:
            delta_f = None
        # check for uniform log spacing in tau_l
        delta_tau_l_std = np.std(np.diff(np.log(tau_l_vec)))
        delta_tau_l_mean = np.mean(np.diff(np.log(tau_l_vec)))
        if delta_tau_l_std / delta_tau_l_mean < 0.01:
            delta_tau_l = delta_tau_l_mean
        else:
            delta_tau_l = None

        if delta_f is not None and delta_tau_l is not None:
            toeplitz_check = np.isclose(delta_tau_l, -delta_f, rtol=1e-3)
        else:
            toeplitz_check = False

        if toeplitz_check and (rbf_type != "piecewise-linear"):
            # Use the Toeplitz trick
            C: NDArray[float64] = np.zeros(num_freqs, dtype=float64)
            for k in range(0, num_freqs):
                C[k] = _A_matrix_element(f[k], tau_l_vec[0], mu, real, rbf_type)

            R: NDArray[float64] = np.zeros(num_taus, dtype=float64)
            for l in range(0, num_taus):
                R[l] = _A_matrix_element(f[0], tau_l_vec[l], mu, real, rbf_type)

            A = toeplitz(C, R)

        else:
            # Use brute force
            A = np.zeros(
                (
                    num_freqs,
                    num_taus,
                ),
                dtype=float64,
            )

            for k in range(0, num_freqs):
                for l in range(0, num_taus):
                    if rbf_type == "piecewise-linear":
                        if real:
                            A[k, l] = (
                                0.5
                                / (1 + (w[k] * tau_l_vec[l]) ** 2)
                                * np.log(
                                    (
                                        tau_l_vec[l]
                                        if l == num_taus - 1
                                        else tau_l_vec[l + 1]
                                    )
                                    / (tau_l_vec[l] if l == 0 else tau_l_vec[l - 1])
                                )
                            )
                        else:
                            A[k, l] = (
                                0.5
                                * (w[k] * tau_l_vec[l])
                                / (1 + (w[k] * tau_l_vec[l]) ** 2)
                                * np.log(
                                    (
                                        tau_l_vec[l]
                                        if l == num_taus - 1
                                        else tau_l_vec[l + 1]
                                    )
                                    / (tau_l_vec[l] if l == 0 else tau_l_vec[l - 1])
                                )
                            )
                    else:
                        A[k, l] = _A_matrix_element(
                            f[k], tau_l_vec[l], mu, real, rbf_type
                        )

        return A

    def _assemble_M_matrix(self) -> NDArray[float64]:

        tau_l_vec = self.tau_l_vec
        mu = self.mu
        derivative_order = self.derivative_order
        rbf_type = self.rbf_type

        num_taus: int = tau_l_vec.shape[0]

        M: NDArray[float64]
        p: int
        q: int
        # M is necessarily symmetric, if uniform_log_tau_spacing, it is also Toeplitz.
        delta_tau_l_std = np.std(np.diff(np.log(tau_l_vec)))
        delta_tau_l_mean = np.mean(np.diff(np.log(tau_l_vec)))
        if delta_tau_l_std / delta_tau_l_mean < 0.01:
            uniform_log_tau_spacing = True
        else:
            uniform_log_tau_spacing = False
        if uniform_log_tau_spacing and rbf_type != "piecewise-linear":
            # Apply the Toeplitz trick to compute the M matrix
            C: NDArray[float64] = np.zeros(num_taus, dtype=float64)
            for p in range(0, num_taus):
                C[p] = _inner_product_rbf(
                    tau_l_vec[p],
                    tau_l_vec[0],
                    mu,
                    derivative_order,
                    rbf_type,
                )
            M = toeplitz(C)  # matrix is symmetric so C and R are the same

        elif rbf_type == "piecewise-linear":
            if derivative_order == 1:
                M = np.zeros(
                    (
                        num_taus - 1,
                        num_taus,
                    ),
                    dtype=float64,
                )
                for i in range(0, num_taus - 1):
                    delta_loc: float = np.log(tau_l_vec[i + 1] / tau_l_vec[i])
                    M[i, i] = -1 / delta_loc
                    M[i, i + 1] = 1 / delta_loc

            elif derivative_order == 2:
                M = np.zeros(
                    (
                        num_taus - 2,
                        num_taus,
                    ),
                    dtype=float64,
                )
                for i in range(0, num_taus - 2):
                    delta_loc = np.log(tau_l_vec[i + 1] / tau_l_vec[i])

                    if i == 0 or i == num_taus - 3:
                        M[i, i] = 2.0 / (delta_loc**2)
                        M[i, i + 1] = -4.0 / (delta_loc**2)
                        M[i, i + 2] = 2.0 / (delta_loc**2)
                    else:
                        M[i, i] = 1.0 / (delta_loc**2)
                        M[i, i + 1] = -2.0 / (delta_loc**2)
                        M[i, i + 2] = 1.0 / (delta_loc**2)

            M = M.T @ M

        else:
            # Brute force
            M = np.zeros(
                (
                    num_taus,
                    num_taus,
                ),
                dtype=float64,
            )
            for p in range(0, num_taus):
                for q in range(0, num_taus):
                    M[p, q] = _inner_product_rbf(
                        tau_l_vec[p],
                        tau_l_vec[q],
                        mu,
                        derivative_order,
                        rbf_type,
                    )

        return M

    def _calculate_matrices(self):
        # Setup problem

        f: NDArray = self.f
        num_procs: int = self.num_procs

        A_re: NDArray[float64]
        A_im: NDArray[float64]
        # if using multiprocessing to assemble A matrices
        if num_procs > 1:
            args = [(f, True), (f, False)]
            with get_context(method="spawn").Pool(2) as pool:
                for i, res in enumerate(
                    pool.starmap(self._assemble_A_matrix, args)
                ):  # unpack args and map to function
                    if i == 0:
                        A_re = res
                    else:
                        A_im = res
        # if not using multiprocessing (normally)
        else:
            A_re = self._assemble_A_matrix(f=f, real=True)
            A_im = self._assemble_A_matrix(f=f, real=False)

        M: NDArray[float64] = self._assemble_M_matrix()

        # store matrices before adjust for regression for later use
        self.A_re = A_re
        self.A_im = A_im
        self.M = M

    def calculate_drt(self) -> DRTData:
        """
        Calculates the distribution of relaxation times (DRT) for a given data set using regularisation and radial basis (or piecewise linear) discretization (TR-RBF method).

        References:

        - `Wan, T. H., Saccoccio, M., Chen, C., and Ciucci, F., 2015, Electrochim. Acta, 184, 483-499 <https://doi.org/10.1016/j.electacta.2015.09.097>`_
        - `Effat, M. B. and Ciucci, F., 2017, Electrochim. Acta, 247, 1117-1129 <https://doi.org/10.1016/j.electacta.2017.07.050>`_

        Parameters
        ----------

        """
        # Setup problem

        f: NDArray = self.f
        Z_exp: NDArray = self.Z_exp
        tau_l_vec: NDArray = self.tau_l_vec
        num_freqs: int = f.size
        num_taus: int = tau_l_vec.size
        mode: str = self.mode
        weighting = self.weighting
        custom_weights = self.custom_weights
        lambda_value: float = self.lambda_value
        lambda_selection: str = self.lambda_selection
        resistance_0: bool = self.resistance_0
        inductance_0: bool = self.inductance_0
        capacitance_0: bool = self.capacitance_0

        if self.A_re is None or self.A_im is None or self.M is None:
            self._calculate_matrices()

        A_re: NDArray[float64] = self.A_re.copy()
        A_im: NDArray[float64] = self.A_im.copy()
        M: NDArray[float64] = self.M.copy()

        b_re: NDArray[float64] = Z_exp.real
        b_im: NDArray[float64] = Z_exp.imag

        # Seeking to minimise the cost function:
        # ||A_re x - b_re||^2 + ||A_im x - b_im||^2 + λ x^T M x
        # with series components folded into A matrices
        # solution vector x will be [R0, L0, 1/C0, x_1, x_2, ..., x_N] where N is number of tau collocation points

        # adjust A matrices, M, and b vectors for series components
        num_series = 3  # R0, L0, 1/C0
        tmp: NDArray[float64]  # Used for temporary binding of matrices
        tmp = A_re
        A_re = np.zeros(
            (
                num_freqs,
                num_taus + num_series,
            ),
            dtype=float64,
        )
        A_re[:, num_series:] = tmp
        if resistance_0:
            A_re[:, 0] = 1.0  # R0
        else:  # R0 has no effect on cost function
            A_re[:, 0] = 0.0  # R0
        A_re[:, 1] = 0.0  # L0
        A_re[:, 2] = 0.0  # C0

        tmp = A_im
        A_im = np.zeros(
            (
                num_freqs,
                num_taus + num_series,
            ),
            dtype=float64,
        )
        A_im[:, num_series:] = tmp
        A_im[:, 0] = 0.0  # R0
        if inductance_0:
            A_im[:, 1] = 2 * np.pi * f  # L0
        else:  # L0 has no effect on cost function
            A_im[:, 1] = 0.0  # L0
        if capacitance_0:
            A_im[:, 2] = 1.0 / (2 * np.pi * f)  # 1/C0
        else:  # 1/C0 has no effect on cost function
            A_im[:, 2] = 0.0  # 1/C0

        tmp = M
        M = np.zeros(
            (
                num_taus + num_series,
                num_taus + num_series,
            ),
            dtype=float64,
        )
        M[num_series:, num_series:] = tmp  # M does not regularize R0, L0, 1/C0

        # weight A matrices by data used
        if mode == "complex":
            pass
        elif mode == "real":
            A_im = np.zeros_like(A_im)
            b_im = np.zeros_like(b_im)
        elif mode == "imaginary":
            A_re = np.zeros_like(A_re)
            b_re = np.zeros_like(b_re)

        # weight by weighting method for cost function
        if weighting == "unity":
            weights: NDArray[float64] = np.ones(Z_exp.shape[0], dtype=float64)
        elif weighting == "boukamp":  # arguable that boukamp should reflect mode
            weights: NDArray[float64] = (Z_exp.real**2 + Z_exp.imag**2) ** -1
        elif weighting == "custom":
            weights: NDArray[float64] = custom_weights
        # apply weights to rows of A matrices (corresponds to frequencies)
        A_re = weights * A_re
        A_im = weights * A_im

        if lambda_selection != "fixed":
            lambda_value = _pick_lambda(
                A_re,
                A_im,
                b_re,
                b_im,
                M,
                lambda_value,
                lambda_selection,
            )
        self.lambda_value = lambda_value  # update used lambda on record

        # form quadratic programming matrices
        H: NDArray[float64]
        c: NDArray[float64]
        H, c = _quad_format_combined(
            A_re,
            A_im,
            b_re,
            b_im,
            M,
            lambda_value,
        )

        num_vars = H.shape[0]  # length of solution vector x
        # Enforce positivity constraint
        h: NDArray[float64] = np.zeros(num_vars, dtype=float64)
        G: NDArray[float64] = -np.eye(num_vars, dtype=float64)
        x: NDArray[float64] = _solve_qp_cvxopt(
            H,
            c,
            G=G,
            h=h,
        )

        if resistance_0:
            R0: float = x[0]
        else:
            R0: float = np.nan
        if inductance_0:
            L0: float = x[1]
        else:
            L0: float = np.nan
        if capacitance_0:
            C0: float = 1 / x[2]
        else:
            C0: float = np.nan
        self.R0 = R0
        self.L0 = L0
        self.C0 = C0
        x = x[num_series:]  # remove series components from solution vector
        self.x = x

        # use unadjusted stored A matrices
        Z_fit: NDArray[complex128] = (self.A_re @ x) + 1j * (self.A_im @ x)
        if not np.isnan(R0):
            Z_fit += R0
        if not np.isnan(L0):
            Z_fit += 1j * 2 * np.pi * f * L0
        if not np.isnan(C0):
            Z_fit += 1 / (1j * 2 * np.pi * f * C0)
        self.EISData_fit = EISData(
            Z=Z_fit, f=f, label=f"RR-RBF Fit of {self.EISData_object.label}"
        )

        self.get_DRTData_fit = self.get_DRTData_fit()

        self.pseudo_chisqr: float = float(
            np.array_sum(
                weights
                * ((Z_exp.real - Z_fit.real) ** 2 + (Z_exp.imag - Z_fit.imag) ** 2)
            )
        )

    def get_DRTData_fit(self, tau_fine: NDArray[float64] | None = None) -> DRTData:
        """
        Returns the DRTData object containing the results of the DRT calculation.

        Returns
        -------
        DRTData
            The DRTData object with calculated DRT results.
        """
        if self.x is None:
            raise ValueError(
                "DRT has not been calculated yet. Please run calculate_drt() first."
            )

        if tau_fine is None:
            tau_fine = np.logspace(
                np.log10(self.tau_l_vec.min()) - 0.5,
                np.log10(self.tau_l_vec.max()) + 0.5,
                10 * self.tau_l_vec.size,
            )  # continuous representation of tau, default is half a decade either side of min and max tau and 10 x number of tau collocation points
        else:
            # check tau fine is a 1D NDArray[float64] of real values
            if not isinstance(tau_fine, np.ndarray):
                raise TypeError(f"Expected a numpy ndarray instead of {tau_fine=}")
            if tau_fine.ndim != 1 or tau_fine.dtype != float64:
                raise TypeError(
                    f"Expected a 1D NDArray[float64] instead of {tau_fine=}"
                )
        return DRTData(
            tau=tau_fine,
            gamma=self._get_gamma(tau_fine),
            label=f"DRT RR-RBF of {self.EISData_object.label}",
        )

    def _get_gamma(
        self,
        tau: NDArray[float64],
    ) -> NDArray[float64]:
        x = self.x
        tau_l_vec = self.tau_l_vec
        mu = self.mu
        rbf_type = self.rbf_type
        rbf_functions: Dict[str, Callable] = _RBF_FUNCTIONS

        if (
            rbf_type == "piecewise-linear"
        ):  # can linearly interpolate as basis functions result in straight lines between collocation points
            # Linear interpolation, extrapolate with 0 outside the range
            gamma = np.interp(
                x=np.log(tau),
                xp=np.log(tau_l_vec),  # xp must be increasing
                fp=x,
                left=0.0,
                right=0.0,
            )
            return gamma

        else:  # sum up weighted rbfs evaluated at tau points
            num_tau_l: int = tau_l_vec.shape[0]
            num_tau: int = tau.shape[0]

            B: NDArray[float64] = np.zeros(
                (
                    num_tau,
                    num_tau_l,
                ),
                dtype=float64,
            )

            rbf: Callable = rbf_functions[rbf_type]

            i: int
            j: int
            for i in range(0, num_tau):
                for j in range(0, num_tau_l):
                    delta_ln_tau = np.log(tau[i]) - np.log(tau_l_vec[j])
                    B[i, j] = rbf(delta_ln_tau, mu)

            return B @ x

    def copy(self):
        """Return a deep copy of this DRT_rr_solver instance."""
        return copy.deepcopy(self)
