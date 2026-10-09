from default_inits.default_theta_params import build_default_theta
from inference.mcmc.emcee_utils import  initialize_walkers, run_emcee
from inference.mcmc.emcee_utils import  initialize_walkers, run_emcee
from models.single_qubit_model import SingleQubitModel, SingleQubitModelConfig

import logging
import json
import numpy as np
from typing import Dict, Any
import os


def single_qubit_inference_pipeline(single_qubit_model_config, data):
    """
    single-qubit inference using UnifiedModel.
    Supports n_qubits = 1 
    
    Uses a single source of truth for theta initialization (build_default_theta_full)
    and properly initializes MCMC walkers for linear independence.
    See the previous version of this function for any  single qubit issues
    """
    # Needs cleaning
 
    #theta0 = build_default_theta(args)
    theta0 = np.array([
        2.0, #omega_r
        0.0, #detuning
        0.1, #gamma
        1.0, #amp
        0.0 #offset
        ])

    ndim = theta0.size

    #configSQB = SingleQubitModelConfig(args.ntimes, args.tmax, args.noise_std)

    # 2. Instantiate the class (create an object)
    # Note the parentheses () which call the __init__ method
    sqb_model = SingleQubitModel(single_qubit_model_config, data)

    # outbase = os.path.join(conf.get('outdir', 'results'), f"{which}_mcmc_{args.n_qubits}q")
    # outnpz = outbase + '.npz'

    # # Resume logic
    # if conf.get('resume', False) and os.path.exists(outnpz):
    #     samples, log_prob, metadata = load_results(outnpz)
    #     print('[INFO] Loaded results from disk (resume mode)')
    # else:
    
    #Quick fix for conf, needs to be addressed properly
    conf = {}
    nwalkers = 64
    nsteps = 500
    nprocs = 1
    p0 = initialize_walkers(nwalkers, ndim, theta0, spread=0.05)
    samples, log_prob = run_emcee(sqb_model, ndim, nwalkers, nsteps, p0, nprocs)
    metadata = {'model': "single_qubit", #which,
                    'n_qubits': 1,
                    'nwalkers': int(conf.get('nwalkers', 64)),
                    'nsteps': int(conf.get('nsteps', 500)),
                    'seed': int(conf.get('seed', 42))}
    #     #save_results(outnpz, samples, log_prob, metadata)

    # # Code to test multiqubit
    signal_data=data
    print("Dimension of samples is:", samples.shape)
    sqb_model.compare_to_data(samples, log_prob)

    flat_samples = samples.reshape(-1, samples.shape[-1]) if samples.ndim == 3 else samples
    return flat_samples, log_prob, metadata

def save_results(filename: str, samples: np.ndarray, log_prob: np.ndarray, metadata: Dict[str, Any]):
    np.savez_compressed(filename, samples=samples, log_prob=log_prob, metadata=json.dumps(metadata))
    logging.info(f"Saved results to {filename}")


def load_results(filename: str):
    d = np.load(filename, allow_pickle=True)
    samples = d['samples']
    log_prob = d['log_prob']
    metadata = json.loads(str(d['metadata']))
    return samples, log_prob, metadata