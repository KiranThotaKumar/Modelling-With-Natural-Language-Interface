import numpy as np
from typing import Dict, Any, Optional

try:
    import emcee
except Exception:
    emcee = None

#def initialize_walkers(nwalkers: int, ndim: int, theta0: np.ndarray, spread: float = 1e-2) -> np.ndarray:
#    p0 = theta0 + spread * (np.abs(theta0) + 1.0) * np.random.randn(nwalkers, ndim)
#    np.savetxt('Walkers.txt', p0, fmt='%0.6f') 
#    return p0
def initialize_walkers(nwalkers: int, ndim: int, theta0: np.ndarray, spread: float = 1e-2) -> np.ndarray:
    """
    Initializes the starting positions for MCMC walkers with conditional randomness.

    Args:
        nwalkers (int): The number of MCMC walkers.
        ndim (int): The number of dimensions (parameters).
        theta0 (np.ndarray): The initial guess for the parameters (1D array).
        spread (float): For non-zero parameters, this controls the relative spread (e.g., 0.01 for 1%).
                        For zero parameters, they are initialized to 1e-5.

    Returns:
        np.ndarray: A 2D array of shape (nwalkers, ndim) with the initial positions.
    """
    theta0 = np.asarray(theta0)
    p0 = np.zeros((nwalkers, ndim), dtype=float)

    for i in range(nwalkers):
        for j in range(ndim):
            value = theta0[j]
            if value == 0:
                # Initialize around 0 with symmetric spread, as theta parameters often span positive/negative values.
                # Ensure it's not exactly zero to avoid potential numerical issues if used as a denominator.
                ## Assign the zero  values a negligeble value
                almost_zero = 1e-12
                p0[i, j] = almost_zero * np.random.uniform(-1, 1)
                if p0[i, j] == 0:
                    p0[i, j] = 1e-12 # A very small non-zero value
            else:
                # Existing logic for non-zero values
                random_factor = 1 + spread * np.random.uniform(-1, 1)
                p0[i, j] = value * random_factor

    np.savetxt('Walkers.txt', p0, fmt='%0.6f')
    return p0

def run_emcee(model, ndim: int, nwalkers: int, nsteps: int, p0: np.ndarray, nprocs: Optional[int] = None,
              progress: bool = True):
    #ensure_emcee_available()
    #nsteps = 10
    
    try:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lambda th: model.log_posterior(th), threads=nprocs)
        state = sampler.run_mcmc(p0, 10, progress=progress)
        sampler.reset()
        sampler.run_mcmc(state, nsteps, progress=progress)
    except TypeError:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lambda th: model.log_posterior(th))
        sampler.run_mcmc(p0, nsteps, progress=progress)
    samples = sampler.get_chain(flat=True)
    log_prob = sampler.get_log_prob(flat=True)

    # assumes sampler is emcee.EnsembleSampler and has run
    samples = sampler.get_chain()  # shape (nsteps, nwalkers, ndim)
    acc_frac = np.mean(sampler.acceptance_fraction)
    print("Acceptance fraction (mean):", acc_frac)

    # compute tau robustly
    try:
        tau = sampler.get_autocorr_time(tol=0)
        print("Integrated autocorrelation times:", tau)
        nsteps, nwalkers, ndim = samples.shape
        burn = int(0.2 * nsteps)
        ess_per_param = (nwalkers * (nsteps - burn)) / tau
        print("Approx ESS per param:", ess_per_param)
    except Exception as e:
        print("Autocorr time not reliable yet (too short):", e)

    return samples, log_prob