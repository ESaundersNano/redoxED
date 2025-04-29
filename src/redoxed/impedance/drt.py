from redoxed.impedance import EISData

import pandas as pd

from nanodrt.dataobjects.dataobject import EISDataObject
from nanodrt.calculators import ImpedanceCalculator
from nanodrt.radialbasisfunctions import GaussianRBF
from nanodrt.optimizers import ImpedanceOptimizer
import jax.numpy as jnp


class DRT:
    def __init__(self, **kwargs):
        self.tau = []
        self.gamma = []
        self.label = kwargs.pop("label", None)

    def fit_DRT_nanodrt(self, EISData_object, mu=None, tau=None, lambda_value=0):
        # load the data into the nanodrt object
        data_object = EISDataObject(
            Z_re=EISData_object.Z_re, Z_im=EISData_object.Z_im, f=EISData_object.f
        )
        # Create instance of RBF
        config = {"mu": mu}
        rbf = GaussianRBF(config)

        tau_values = tau  # tau values for the DRT
        freq_values = data_object.f  # frequency values from the EIS data
        lambda_value = lambda_value

        # Create ImpedanceCalculator instance
        calculator = ImpedanceCalculator(
            rbf, tau_values, freq_values, dy=0.001, y_max=10, y_min=-10
        )
        # initial guess assuming all ones
        x_values = jnp.ones(len(tau_values) + 2)

        # run
        optimizer = ImpedanceOptimizer(calculator, data_object)
        optimizer.add_regularisation(lambda_value=1e-3)
        results = optimizer.run(
            x_values, method="scipy", config={"method": "BFGS", "maxiter": 1000}
        )  # lowered from 10000
        Z = calculator.simulate(
            results.params[:-2], results.params[-2], results.params[-1]
        )
        # plt.plot(x.real[:-3], -x.imag[:-3], '-o')
        # plt.plot(data_object.Z_re, -data_object.Z_im, 'o')
        # plt.show()
        gamma = calculator.simulate_gamma(results.params[:-2])
        # plt.plot(jnp.log(tau_values), gamma, "-o")

        self.tau = tau_values
        self.gamma = gamma
        self.R_0 = results.params[-2]
        self.L_0 = results.params[-1]
        self.lambda_value = 1e-3

        df_temp = {
            "freq/Hz": freq_values,
            "Re(Z)/Ohm": Z.real,
            "-Im(Z)/Ohm": -Z.imag,
        }  # prepare df
        df_temp = pd.DataFrame(df_temp)  # convert to pd df
        self.EIS_data_fit = EISData(
            df_temp, dataset_type="fitted"
        )  # store as EIS_data object
        # (
        #     self.EIS_data_fit.residuals_real,
        #     self.EIS_data_fit.residuals_imag,
        #     self.EIS_data_fit.residuals_real_rmse,
        #     self.EIS_data_fit.residuals_imag_rmse,
        # ) = EISData_object.calculate_residuals(self.EIS_data_fit)


"""
    def fit_DRT(
        self, EIS_data_object, DRT_method="pyDRTtools_ridge_regression", **kwargs
    ):

        if DRT_method == "pyDRTtools_ridge_regression":
            # Load measurement data into EIS_object
            pyDRTtools_EIS_object = pyDRTtools.runs.EIS_object(
                EIS_data_object.frequencies, EIS_data_object.Z_Re, EIS_data_object.Z_Im
            )
            # this includes tau definition of
            # # self.tau = 1/freq # we assume that the collocation points equal to 1/freq as default
            # # self.tau_fine  = np.logspace(log10(self.tau.min())-0.5,log10(self.tau.max())+0.5,10*freq.shape[0])

            solver_dict = {"DRT_method": DRT_method}

            rbf_type = kwargs.pop("rbf_type", "Gaussian")
            solver_dict["rbf_type"] = rbf_type

            data_used = kwargs.pop("data_used", "Combined Re-Im Data")
            solver_dict["data_used"] = data_used

            fit_series_inductance = kwargs.pop("fit_series_inductance", True)
            if fit_series_inductance:
                induct_used = 1
            else:
                induct_used = 0
            solver_dict["fit_series_inductance"] = fit_series_inductance

            der_used = kwargs.pop("der_used", "1st order")
            solver_dict["der_used"] = der_used

            lambda_selection = kwargs.pop("lambda_selection", "fixed")
            if lambda_selection == "fixed":
                cv_type = "custom"
            else:
                cv_type = lambda_selection  # e.g., GCV
            solver_dict["lambda_selection"] = lambda_selection
            # print("cv_type", cv_type)

            lambda_0 = kwargs.pop("lambda_0", 1e-9)  # initial guess
            # if lambda_0 is not None:
            reg_param = lambda_0
            # print("reg_param", reg_param)
            solver_dict["lambda_0"] = lambda_0

            coeff = kwargs.pop("coeff", 0.5)
            solver_dict["coeff"] = coeff
            mu = kwargs.pop("mu", None)
            solver_dict["mu"] = mu

            shape_control = kwargs.pop("shape_control", "FWHM Coefficient")
            solver_dict["shape_control"] = shape_control
            if (
                shape_control == "FWHM Coefficient"
            ):  # default makes coeff scalar such that the full width at half maximum (FWHM) of the RBF is equal to 1/coeff times the average relaxation time spacing in logarithm scale
                coeff = coeff
            elif (
                self.mu is not None
            ):  # anything else treats coeff as the shape factor, mu, of the rbfs
                coeff = mu
            else:
                raise ValueError(
                    "must select shape_control properly or provide a shape factor, mu"
                )

            # ridge regression DRT
            fitted_entry = pyDRTtools.runs.simple_run(
                entry=pyDRTtools_EIS_object,
                rbf_type=rbf_type,
                data_used=data_used,
                induct_used=induct_used,
                der_used=der_used,
                cv_type=cv_type,
                reg_param=reg_param,
                shape_control=shape_control,
                coeff=coeff,
            )
            ### If want to mute the console, can use something like this with simple_run nested in
            # # Redirecting stdout to capture the output
            # with io.StringIO() as output, contextlib.redirect_stdout(output):

            self.pyDRTtools_fit = fitted_entry  # store in case need to double check what data fitted to, access .x, .tau (the tau discretisation actually used in the fitting), .A_re, .A_im, .M, or other useful results such as residuals
            self.tau = (
                fitted_entry.out_tau_vec
            )  # is tau_fine from DRTtools, not tau used for fitting x.
            self.gamma = fitted_entry.gamma
            self.R_0 = fitted_entry.R
            self.L_0 = fitted_entry.L
            self.lambda_value = fitted_entry.lambda_value

            df_temp = {
                "freq/Hz": fitted_entry.freq,
                "Re(Z)/Ohm": fitted_entry.mu_Z_re,
                "-Im(Z)/Ohm": -fitted_entry.mu_Z_im,
            }  # prepare df
            df_temp = pd.DataFrame(df_temp)  # convert to pd df
            self.EIS_data_fit = EIS_data(
                df_temp, dataset_type="fitted"
            )  # store as EIS_data object
            (
                self.EIS_data_fit.residuals_real,
                self.EIS_data_fit.residuals_imag,
                self.EIS_data_fit.residuals_real_rmse,
                self.EIS_data_fit.residuals_imag_rmse,
            ) = EIS_data_object.calculate_residuals(self.EIS_data_fit)

    def peak_fit(self, N_peaks, dynamic_threshold):

        # Step 1: define the necessary quantities before the subsequent optimizations
        N_peaks = int(N_peaks)

        # # upper and lower log tau values
        # log_tau_min = np.min(np.log(entry.out_tau_vec))
        # log_tau_max = np.max(np.log(entry.out_tau_vec))

        # Find peaks in the gamma spectrum with a lower threshold for height
        # dynamic_threshold = 0.05 * np.mean(self.gamma)  # Lower threshold for small peak detection
        peak_indices, _ = find_peaks(self.gamma, height=dynamic_threshold, distance=5)

        # Adjust N_peaks if fewer peaks are found than specified
        # N_peaks = min(len(peak_indices), N_peaks)
        num_peaks_found = len(peak_indices)

        if N_peaks > num_peaks_found:
            print(f"Warning: Adjusting N_peaks to {num_peaks_found}.")
            N_peaks = num_peaks_found
        elif N_peaks < num_peaks_found:
            print(f"Warning: Adjusting N_peaks to {num_peaks_found}.")
            N_peaks = num_peaks_found
        tau_pos = self.tau[peak_indices]
        fc_pos = 1 / (2 * np.pi * tau_pos)
        gamma_pos = self.gamma[peak_indices]

        df_temp = {
            "indices": peak_indices,
            "tau_pos": tau_pos,
            "fc_pos": fc_pos,
            "gamma_pos": gamma_pos,
        }
        df_temp = pd.DataFrame(df_temp)
        self.peak_analysis = df_temp

    def peak_analysis2(self, num_peaks, peak_positions, disallow_skew):
        time_constants = self.tau
        gammas = self.gamma
        peaks = _analyze_peaks(
            time_constants, gammas, num_peaks, peak_positions, disallow_skew
        )
        return peaks
"""
