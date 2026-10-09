import os
import numpy as np
from models.multi_qubit_model import MultiQubitData, MultiQubitModelConfig, MultiQubitModel
from inference.mcmc.emcee_utils import initialize_walkers, run_emcee
from models.multi_qubit_param_builder import prepare_theta_from_measured

def load_results(filename: str):
    d = np.load(filename, allow_pickle=True)
    samples = d['samples']
    log_prob = d['log_prob']
    metadata = json.loads(str(d['metadata']))
    return samples, log_prob, metadata


def multi_qubit_inference_pipeline(conf, data, n_qubits=2):
    """
    Uses a single source of truth for theta initialization (build_default_theta_full)
    and properly initializes MCMC walkers for linear independence.
    See the previous version of this function for any  single qubit issues
    """
    conf = {}
    theta0, paramsDummy = prepare_theta_from_measured(data)
    ndim = len(theta0)
    print("thetaZero:", theta0)
    
    # # --- Output paths ---
    # outbase = os.path.join(conf.get('outdir', 'results'), f"multi_qubit_mcmc_{n_qubits}q")
    # outnpz = outbase + '.npz'

    # # --- Resume logic ---
    # if conf.get('resume', False) and os.path.exists(outnpz):
    #     samples, log_prob, metadata = load_results(outnpz)
    #     print('[INFO] Loaded results from disk (resume mode)')
    # else:
    # --- MCMC walker initialization ---
    nwalkers = int(conf.get('nwalkers', 64))
    rng = np.random.default_rng(seed=int(conf.get('seed', 42)))
    # Initialize walkers for MCMC
    #CodeChange spread=0.02 to 0.1
    p0 = initialize_walkers(nwalkers, ndim, theta0, spread=0.001)  # spread adjustable
        
    #Model creation for MCMC
    multi_qubit_data = MultiQubitData(data.times, data.measurements, data.errors, data.metadata)
    #multi_qubit_model_config = MultiQubitModelConfig(conf["ntimes"], conf["tmax"], conf["noise_std"])
    multi_qubit_model_config = MultiQubitModelConfig(1000, 20, 0.02)

    multi_qubit_model = MultiQubitModel(multi_qubit_model_config, multi_qubit_data)

    # --- Run emcee ---
    samples, log_prob = run_emcee(
        multi_qubit_model,
        ndim,
        nwalkers,
        #int(conf.get('nsteps', 500)),
        10,
        p0,
        nprocs=int(conf.get('nprocs', 1))
    )
    print(samples.shape)

    for i in range(samples.shape[-1]):
        std = np.std(samples[:,:,i])
        print(i, std)

    # --- Comparison / plotting ---
    multi_qubit_model.compare_to_data(samples, log_prob)

    # --- Save metadata ---
    metadata = {
        'model': "multi_qubit", #which,
        'n_qubits': 2,
        'nwalkers': nwalkers,
        'nsteps': int(conf.get('nsteps', 500)),
        'seed': int(conf.get('seed', 42))
    }
    # save_results(outnpz, samples, log_prob, metadata)
    flat_samples = samples.reshape(-1, samples.shape[-1]) if samples.ndim == 3 else samples
    return flat_samples, log_prob, metadata




    