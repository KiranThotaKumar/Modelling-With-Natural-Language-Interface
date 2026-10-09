import numpy as np
from dataclasses import dataclass
from scipy import stats, special
from typing import Dict, Any, Optional, List, Tuple, Union, Callable, Iterable, Literal
from physics.single_qubit import single_qubit_model_physics
from stats_visuals import metrics, plotting, posteriors, comparison

@dataclass
class SingleQubitData:
    times: np.ndarray                 # shape (n_times,)
    measurements: np.ndarray           # shape (n_observables, n_times) or (n_times,)
    errors: Optional[np.ndarray] = None  # shape matches observables
    metadata: Optional[dict] = None

@dataclass
class SingleQubitModelConfig:
    ntimes: int
    tmax: float   
    noise_std:float

class SingleQubitModel:
    """
    """
    def __init__(self, config: SingleQubitModelConfig, data: SingleQubitData):
        self.config = config
        self.data = data

    def model_on_data(self, params: dict) -> np.ndarray:
        if self.data is None:
            raise ValueError("No data attached to model.")
        return self.model(self.data.times, params)

    def model(self, times: np.ndarray, params: dict) -> np.ndarray:
        return single_qubit_model_physics(self.data.times, params)
    
    def log_prior(self, theta: np.ndarray) -> float:

        """
        Compute the log prior for model parameters.
    
        Parameters
        ----------
        theta : np.ndarray
            Parameter vector       
        Returns
        -------
        float
            Log-prior value.
        """

        if theta.size < 5:
            print("Returned at 1")
            return -np.inf
        omega_r, detuning, gamma, amp, offset = theta[:5]
        if not (0.0 <= omega_r <= 1e3):
            print("Returned at 2", omega_r)
            return -np.inf
        if not (-1e3 <= detuning <= 1e3):
            print("Returned at 3", detuning)
            return -np.inf
        if not (0.0 <= gamma <= 1e2):
            print("Returned at 4", gamma)
            return -np.inf
        if not (-1e6 <= amp <= 1e6):
            print("Returned at 5", amp)
            return -np.inf
        if not (-1e6 <= offset <= 1e6):
            print("Returned at 6", offset)
            return -np.inf

        lp = 0.0
        lp += stats.expon.logpdf(gamma, scale=1.0)
        
        return lp

    def log_likelihood(self, theta: np.ndarray) -> float:
        
        data = self.data

        params = {
        "omega_r": theta[0],
        "detuning": theta[1],
        "gamma": theta[2],
        "amp": theta[3],
        "offset": theta[4]
        }

        model = self.model_on_data(params)
        
        resid = data.measurements - model
        var = data.errors ** 2
        var = np.maximum(var, 1e-12)
        ll = -0.5 * np.sum(resid ** 2 / var + np.log(2 * np.pi * var))
        return ll

    def log_posterior(self, theta: np.ndarray) -> float:
        """
        Compute total log-posterior = log-prior + log-likelihood.
        """            

        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf

        ll = self.log_likelihood(theta)
        if not np.isfinite(ll):
            return -np.inf

        return lp + ll

    def inference(self, samples, log_prob, burnin_frac=0.2):
        """
        Parameter inference: observed data → parameter estimates.
        Returns a dictionary of posterior summaries.
        """
        samples_flat, log_prob_flat = posteriors.flatten_posterior(samples, log_prob, burnin_frac=0.2)
        theta_map = posteriors.map_estimate(samples_flat, log_prob_flat)
        theta_mean = posteriors.posterior_mean(samples_flat)
        posterior = {
            "samples_flat": samples_flat,
            "log_prob_flat": log_prob_flat,
            "theta_map": theta_map,
            "theta_mean": theta_mean,
        }

        return posterior

    def spec_from_theta(self, theta: np.ndarray = None):
 
        omega_r, detuning, gamma, amp, offset = theta[:5]
        
        params = {
            "omega_r": omega_r,
            "detuning": detuning,
            "gamma": gamma,
            "amp": amp,
            "offset": offset
        }

        return single_qubit_model_physics(self.data.times, params)


    def compare_to_data(self, samples, log_prob):

        posterior = self.inference(samples, log_prob)

        theta_map = posterior["theta_map"]
        theta_mean = posterior["theta_mean"]

        times = self.data.times
        y_obs = self.data.measurements

        y_map = self.spec_from_theta(theta_map)
        y_mean = self.spec_from_theta(theta_mean)

        score = metrics.rmse(y_obs, y_map)
        correlation_value = metrics.correlation(y_obs, y_map)

        plotting.plot_overlay(self.data.times, self.data.measurements, y_map, y_mean, labels = ["Measured", "MAP", "Mean"])
        plotting.plot_residuals(self.data.times, res = self.data.measurements - y_map)
        comparison.plot_compare_spectra(y_obs, y_mean, self.data.times)

        # for i in range(posterior["samples_flat"].shape[1]):
        #     print(f"param {i} ESS ~ {self.simple_ess(posterior["samples_flat"][:, i])}")

        generated_spectra = []
        # Generate spectra from samples
        
        nsamp_plot = 400
        indices = np.random.choice(posterior["samples_flat"].shape[0], size=nsamp_plot, replace=False)

        all_spectra = np.zeros((nsamp_plot, len(times)))
        for i, idx in enumerate(indices):
            theta_i = posterior["samples_flat"][idx]
            all_spectra[i] = self.spec_from_theta(theta=theta_i)

        median_spec = np.median(all_spectra, axis=0)

        lower = np.percentile(all_spectra, 2.5, axis=0)
        upper = np.percentile(all_spectra, 97.5, axis=0)

        plotting.plot_posterior_band(times, median_spec, lower, upper, self.data.measurements, self.data.errors)

        #chi2
        nparams = posterior["samples_flat"].shape[1]        
        dof = len(self.data.measurements) - nparams        
        chi_square = metrics.chi2(self.data.measurements, median_spec, self.data.errors, dof, nparams)
        print("chi_square = ", chi_square)