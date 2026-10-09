
#inference\pipelines\hydrogen_pipeline.py

from gettext import npgettext
from models.hydrogen_model import HydrogenModel, HydrogenModelConfig, HydrogenData
from models.hydrogen_param_builder import build_hydrogen_params_from_measured
from inference.mcmc.emcee_utils import  initialize_walkers, run_emcee
import logging
import json
import numpy as np
from typing import Dict, Any
import os

def hydrogen_inference_pipeline(conf, data_filepath):
    """
    Hydrogen inference pipeline using legacy UnifiedModel.
    """
    which = "hydrogen"
    
    with np.load(data_filepath) as data:
        energies = data["energies"]
        counts   = data["counts"]
        errors   = data["errors"]
        
        hydrogen_data = HydrogenData(
            energies=energies,
            counts=counts,
            errors=errors,
        )

    theta0, model_config  = build_hydrogen_params_from_measured(hydrogen_data)
    model = HydrogenModel(model_config, hydrogen_data)

    ndim = theta0.size

    outbase = os.path.join(conf.get('outdir', 'results'), f"{which}_mcmc")
    outnpz = outbase + '.npz'

    # Resume logic
    if conf.get('resume', False) and os.path.exists(outnpz):
        samples, log_prob, metadata = load_results(outnpz)
        print('[INFO] Loaded results from disk (resume mode)')
    else:
        p0 = initialize_walkers(int(conf.get('nwalkers', 64)), ndim, theta0, spread=0.05)
        samples, log_prob = run_emcee(model, ndim,
                                      int(conf.get('nwalkers', 64)),
                                      int(conf.get('nsteps', 500)),
                                      p0,
                                      nprocs=int(conf.get('nprocs', 1)))
        metadata = {'model': which,
                    'nwalkers': int(conf.get('nwalkers', 64)),
                    'nsteps': int(conf.get('nsteps', 500)),
                    'seed': int(conf.get('seed', 42))}
        #save_results(outnpz, samples, log_prob, metadata)
    
    # Compare spectra using MAP    
    
    model.compare_to_data(samples, log_prob)
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