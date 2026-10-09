import models
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple, Union, Callable, Iterable, Literal
from physics.hydrogen import hydrogen_model_spectrum_physics
from instrument.convolution import instrument_convolve
from scipy import stats, special
import logging
from stats_visuals import metrics, plotting, posteriors, comparison


@dataclass
class HydrogenModelConfig:
    transitions: List[Tuple[int, int]]
    line_energies: np.ndarray   # matched/measured energies
    line_strengths: np.ndarray  # relative (or normalized) strengths

@dataclass
class HydrogenData:
    energies: np.ndarray
    counts: np.ndarray
    errors: np.ndarray

class HydrogenModel:
    """
    Hydrogen atomic spectrum model.

    Responsibilities:
    1) Forward modeling: generate a hydrogen spectrum from physical parameters.
    2) Inference: estimate physical parameters from observed spectral data.

    Inputs:
    - Modeling: transitions, amplitudes, background, instrumental resolution.
    - Inference: observed spectrum (energy vs intensity).

    Outputs:
    - Modeling: synthetic spectrum (energy vs intensity).
    - Inference: estimated parameters + uncertainties.

    Notes:
    - Designed for Z = 1 (Hydrogen).
    - Extensible to Z > 1, fine and hyperfine structure.
    - Contains no NLP, routing, or intent logic.
    """

    def __init__(self, config: HydrogenModelConfig, data: HydrogenData):
        self.config = config
        self.data = data

    def model_on_data(self, params: dict) -> np.ndarray:
        if self.data is None:
            raise ValueError("No data attached to model.")
        return self.model(self.data.energies, params)

    def model(self, energies: np.ndarray, params: dict) -> np.ndarray:
        return hydrogen_model_spectrum_physics(energies, params)
        
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

    # helper to generate spectrum from theta
    def spec_from_theta(self, theta: np.ndarray = None):
 
        sigma_instr = theta[0]
        background = theta[1]
        scale = theta[2]
        transitions = self.config.transitions
        nlines = len(transitions)
        line_rel = theta[3:3 + nlines]
        amplitudes = np.exp(line_rel) * scale
        params = {
            "transitions": transitions,
            "amplitudes": amplitudes,
            "sigma_instr": sigma_instr,
            "background": background,
            "scale": scale,

        }
        return hydrogen_model_spectrum_physics(self.data.energies, params)

    def compare_to_data(self, samples, log_prob):

        posterior = self.inference(samples, log_prob)

        theta_map = posterior["theta_map"]
        theta_mean = posterior["theta_mean"]

        energies = self.data.energies
        y_obs = self.data.counts

        y_map = self.spec_from_theta(theta_map)
        y_mean = self.spec_from_theta(theta_mean)

        score = metrics.rmse(y_obs, y_map)
        correlation_value = metrics.correlation(y_obs, y_map)

        print("RMSE score and correlation_value are:", score, correlation_value)
        
        plotting.plot_overlay(self.data.energies, self.data.counts, y_map, y_mean, labels = ["Measured", "MAP", "Mean"])
        plotting.plot_residuals(self.data.energies, res = self.data.counts - y_map)
        comparison.plot_compare_spectra(y_obs, y_mean, self.data.energies)

        # for i in range(posterior["samples_flat"].shape[1]):
        #     print(f"param {i} ESS ~ {self.simple_ess(posterior["samples_flat"][:, i])}")

        generated_spectra = []
        # Generate spectra from samples
        
        nsamp_plot = 500
        indices = np.random.choice(posterior["samples_flat"].shape[0], size=nsamp_plot, replace=False)

        all_spectra = np.zeros((nsamp_plot, len(energies)))
        for i, idx in enumerate(indices):
            theta_i = posterior["samples_flat"][idx]
            all_spectra[i] = self.spec_from_theta(theta=theta_i)

        median_spec = np.median(all_spectra, axis=0)
        lower = np.percentile(all_spectra, 2.5, axis=0)
        upper = np.percentile(all_spectra, 97.5, axis=0)
        plotting.plot_posterior_band(energies, median_spec, lower, upper, self.data.counts, self.data.errors)

        #chi2
        nparams = posterior["samples_flat"].shape[1]        
        dof = len(self.data.counts) - nparams        
        chi_square = metrics.chi2(self.data.counts, median_spec, self.data.errors, dof, nparams)

        neg_amp_sam = 0
        for sample in posterior["samples_flat"]:
            sigma_instr, background, temp_scale, *log_amplitudes = sample
            amplitudes = np.exp(log_amplitudes)

            if(amplitudes[0] <  0):
                neg_amp_sam +=1
        print("No.Of Negative Amplitude Samples:", neg_amp_sam)

        return score

    def log_prior(self, theta: np.ndarray) -> float:
        if theta.size < 3:
            return -np.inf
        sigma_instr, background, scale = theta[0], theta[1], theta[2]
        if not (1e-1 <= scale <= 1e7):
            return -np.inf
        if not (1e-2 <= sigma_instr <= 1e3):
            return -np.inf
        if not (0.0 <= background <= 1e6):
            return -np.inf
        lp = 0.0
        lp += stats.norm.logpdf(np.log(scale), loc=np.log(1e4), scale=3.0)
        lp += stats.expon.logpdf(sigma_instr, scale=10.0)
        return lp

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

    
    def log_likelihood(self, theta: np.ndarray) -> float:
        model_config = self.config
        data = self.data
        transitions = model_config.transitions
        nlines = len(transitions)

        if transitions is None:
            transitions = [(2, 1), (3, 1), (4, 1)]
            logging.warning("hydrogen_log_likelihood:  transitions are not available; using default values.")
        
        expect_len = 3 + nlines
        if theta.size < expect_len:
            return -np.inf
        
        sigma_instr = theta[0]
        background = theta[1]
        scale = theta[2]
        line_rel = theta[3:3 + nlines]
        # Convert log-relative amplitudes to linear amplitudes
        line_rel = np.clip(line_rel, -10, 10)
        
        amplitudes = np.exp(line_rel) * scale
        # Logistic transform (keeps amplitude finite and positive)
        #A_max = 10.0
        #amplitudes = scale * (A_max / (1.0 + np.exp(-line_rel)))
        amplitudes = np.maximum(amplitudes, 1e-12)
   
        params = {
            "transitions": transitions,
            "amplitudes": amplitudes,
            "sigma_instr": sigma_instr,
            "background": background,
            "scale": scale
        }

        params["sigma_instr"] = theta[0]
        params["background"] = theta[1]
        params["scale"]  = theta[2]
        params["transitions"] = model_config.transitions
        params["amplitudes"] = amplitudes        
        model = self.model_on_data(params)
    
        # Avoid zeros or negatives in Poisson likelihood
        lam = np.maximum(model, 1e-12)
    
        k = data.counts
        ll = np.sum(k * np.log(lam) - lam - special.gammaln(k + 1.0))  #Poisson log-likelihood:
    
        # Safety check (prevents NaNs from propagating to emcee)
        if not np.isfinite(ll):
            return -np.inf

        return ll
