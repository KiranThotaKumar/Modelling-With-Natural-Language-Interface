
import numpy as np
from dataclasses import dataclass
from scipy import stats, special
from typing import Dict, Any, Optional, List, Tuple, Union, Callable, Iterable, Literal
from stats_visuals import metrics, plotting, posteriors, comparison
from default_inits.default_theta_params import build_default_theta_full
from physics.multi_qubit import multi_qubit_model_physics
from models.multi_qubit_param_builder import theta_to_model_params_full
from stats_visuals.plotting import compareModelWithMeasured, plot_multi_qubit_posterior_bands, compute_multi_qubit_stats, plot_multi_qubit_residuals, plot_multi_qubit_timeseries, plot_single_channel_overlay
from stats_visuals.metrics import ensure_shape_time_nqubits


@dataclass
class MultiQubitData:
    times: np.ndarray                 # shape (n_times,)
    measurements: np.ndarray           # shape (n_observables, n_times) or (n_times,)
    errors: Optional[np.ndarray] = None  # shape matches observables
    metadata: Optional[dict] = None

@dataclass
class MultiQubitModelConfig:
    n_qubits: int = None
    ntimes: int = None
    tmax: float = None
    tlist:np.ndarray = None
    noise_std:float = None
    n_qubit_guard:int = None
    observables_spec:List[Tuple[int,str]] = None
    local_fields:Dict[int, Tuple] = None
    psi0:np.ndarray = None
    params:Dict[int, Tuple] = None
    c_ops:List[np.ndarray] = None
    zz_couplings:List[Tuple[int,int,float]] = None
    custom_terms:Dict = None
    simplify_result:bool = None
    open_system:bool = None
    couplings: dict = None
    fast: bool  = None
    qubit_params_override:dict = None
    spectrum_mode: bool = None
    sigma_instr: float = None
    background: float = None
    theta:np.ndarray = None    


class MultiQubitModel:
    """
    """
    def __init__(self, config: MultiQubitModelConfig, data: MultiQubitData):
        self.config = config
        self.data = data

    def model_on_data(self, params: dict) -> np.ndarray:
        if self.data is None:
            raise ValueError("No data attached to model.")
        return self.model(self.data.times, params)

    def model(self, times: np.ndarray, params: dict) -> np.ndarray:
        return multi_qubit_model_physics(self.data.times, params)

    def log_prior(self, theta: np.ndarray) -> float:

        """
        Compute the log prior for model parameters.
    
        Parameters
        ----------
        theta : np.ndarray
            Parameter vector (amplitudes, frequencies, etc.).

        Returns
        -------
        float
            Log-prior value.
        """


        # --- New full quantum model prior ---
        # Example: allow correlated amplitudes via multivariate Gaussian
        n = len(theta)
        mu_vec = np.zeros(n)
        cov = 0.1 * np.eye(n) + 0.9 * np.ones((n, n)) * 0.01  # small correlations
        inv_cov = np.linalg.inv(cov)
        log_det = np.log(np.linalg.det(cov))
        diff = theta - mu_vec
        logp = -0.5 * (diff @ inv_cov @ diff) - 0.5 * log_det - (n / 2) * np.log(2 * np.pi)
        #print("log prior is:", logp)
        return logp

    def log_likelihood(self, theta: np.ndarray) -> float:
        """
        Placeholder full-mode likelihood:
        - Use multi_qubit_model_physics(mode = 'auto', theta, model_grid, ...) to compute y_pred
        - Compute Gaussian log-likelihood with instrument noise sigma (from theta or kwargs)
        """               
        data = self.data
        # Compute model prediction (user will replace with full quantum evolution)        
        h_range_from_metadata = data.metadata.get('_h_range_for_theta', 1.0)
        amp_range_from_metadata = data.metadata.get('_amp_range_for_theta', 1.0)
                
        params_for_sim = theta_to_model_params_full(theta, nqubits=data.metadata['n_qubits'], h_range=h_range_from_metadata, amp_range=amp_range_from_metadata, noise_range=0.02)

        
        #y_pred = multi_qubit_model_physics(mode=mode, times=model_grid, theta=theta,
            #qubit_params_override=params_for_sim, fast=True, h_range=h_range_from_metadata, amp_range=amp_range_from_metadata, couplings=data.metadata.get('zz_couplings', None),**kwargs)
        
        params = {
            "n_qubits": data.metadata['n_qubits'],
            "theta": theta,
            "qubit_params_override": params_for_sim,
            "fast": True,
            "h_range": h_range_from_metadata,
            "amp_range": amp_range_from_metadata,
            "couplings":data.metadata.get('zz_couplings', None),
            "spectrum_mode": False, # need to get from args or model config
            "sigma_instr": 0.5, # need to get from args or model config
            "debug_compare": False, # need to get from args or model config
            "background": 0.0
        }
        y_pred = self.model_on_data(params)

        #noise_sigma = self.config.noise_sigma
        #if noise_sigma is None:            
            #params = theta_to_model_params_full(theta, **kwargs)
            #noise_sigma = params.get("instrument", {}).get("noise_sigma", 1e-2)
            # Use params_for_sim, which is already generated, to get noise_sigma
            #noise_sigma = params_for_sim.get("instrument", {}).get("noise_sigma", 1e-2)
                
        #CodeChange Below code is added
        noise_sigma = 0.01 # Need to be commented
        # flatten arrays for numeric stability
        #compareModelWithMeasured(y_pred, (data.observables).T)

        y_obs_arr = np.asarray(data.measurements).ravel()
        #y_obs_arr = np.asarray(data.observables).T.ravel()
        y_pred_arr = np.asarray(y_pred).T.ravel()

        if y_obs_arr.shape != y_pred_arr.shape:
          raise ValueError(f"Shape mismatch: y_obs {y_obs_arr.shape} vs y_pred {y_pred_arr.shape}")

        var = float(noise_sigma) ** 2
        # Gaussian log-likelihood:
        resid = y_obs_arr - y_pred_arr

        #np.savetxt('y_obs_arr.txt', y_obs_arr, fmt='%0.6f')
        #np.savetxt('y_pred_arr.txt', y_pred_arr, fmt='%0.6f')
        #np.savetxt('resid.txt', resid, fmt='%0.6f')

        ll = -0.5 * (np.sum(resid ** 2) / var + y_obs_arr.size * np.log(2 * np.pi * var))
        #print("noise_sigma SumResidSquare, and ll are:", noise_sigma, np.sum(resid ** 2), ll)
        return float(ll)

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
        
        couplings_from_metadata = self.data.metadata.get('zz_couplings', None)
        h_range_from_metadata = self.data.metadata.get('_h_range_for_theta', 1.0)        
        amp_range_from_metadata = self.data.metadata.get('_amp_range_for_theta', 1.0)

        params = {
            "n_qubits": self.data.metadata['n_qubits'],
            "theta": theta,
            "qubit_params_override": None,
            "fast": True,
            "h_range": h_range_from_metadata,
            "amp_range": amp_range_from_metadata,
            "couplings":couplings_from_metadata,
            "spectrum_mode": False, # need to get from args or model config
            "sigma_instr": 0.5, # need to get from args or model config
            "debug_compare": False, # need to get from args or model config
            "background": 0.0 # need to get from args or model config
        }
        y_pred = self.model_on_data(params)
        return y_pred

    def compare_to_data(self, samples, log_prob):
        
        #data_mcmc = np.load("multi_qubit_mcmc_output.npz", allow_pickle=True)

        #samples = data_mcmc["samples"]
        #log_prob = data_mcmc["log_prob"]
        #y_synth = data_mcmc.get("y_synthetic", None)
        #times = data_mcmc.get("times", None)
        #param_names = data_mcmc.get("param_names", None)
        times = self.data.times
        
        posterior = self.inference(samples, log_prob)

        y_obs  = self.data.measurements      # (T, n_qubits)
        times  = self.data.times
        y_map  = self.spec_from_theta(posterior["theta_map"])
        y_mean = self.spec_from_theta(posterior["theta_mean"])

        #Ensure data to be in the shape (T, n_qubits)
        y_obs = ensure_shape_time_nqubits(y_obs)
        y_map = ensure_shape_time_nqubits(y_map)
        y_mean = ensure_shape_time_nqubits(y_mean)

        plot_multi_qubit_timeseries(times, y_obs, y_map, y_mean)
        plot_multi_qubit_residuals(times, y_obs, y_map)

        stats = compute_multi_qubit_stats(y_obs, y_map)
        print(stats)

        nsteps, nwalkers, ndim = samples.shape
        nsamp_plot = int(nsteps * 0.5) # Taking 50% of the samples
        
        all_signals = np.zeros((nsamp_plot, len(times), self.data.metadata['n_qubits']))
        indices = np.random.choice(posterior["samples_flat"].shape[0], size=nsamp_plot, replace=False)

        for i, idx in enumerate(indices):
            theta_i = posterior["samples_flat"][idx]
            all_signals[i] = self.spec_from_theta(theta_i)

        plot_multi_qubit_posterior_bands(times, all_signals, y_obs=y_obs)


