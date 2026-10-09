import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple, Union, Callable, Iterable, Literal
from models.hydrogen_model import HydrogenData
import logging

@dataclass
class HydrogenSyntheticConfig:
    nbins: int = 2048
    emin: float = 5.0
    emax: float = 15.0
    sigma_instr: float = 0.25
    background_rate: float = 10.0
    random_seed: Optional[int] = 42

def generate_synthetic_hydrogen_data(
    config: HydrogenSyntheticConfig,
    transitions: Optional[List[Tuple[int, int]]] = None,
    amplitudes: Optional[List[float]] = None
) -> HydrogenData:

    rng = np.random.default_rng(config.random_seed)
    energies = np.linspace(config.emin, config.emax, config.nbins)
    if transitions is None:
        logging.warning("generate_synthetic_hydrogen_data:  transitions are not available; using default values.")
        transitions = [(2, 1), (3, 1), (4, 1)]
                
    if amplitudes is None:
        amplitudes = [0.5, 0.3, 0.2][:len(transitions)]
        #amplitudes = [0.6, 0.4][:len(transitions)]

    Rydberg = 13.598285
    spectrum = np.zeros_like(energies)
    for (n_u, n_l), amp in zip(transitions, amplitudes):
        E_u = -Rydberg / (n_u ** 2)
        E_l = -Rydberg / (n_l ** 2)
        E_line = E_u - E_l  # positive photon energy (eV)
        line_profile = amp * np.exp(-0.5 * ((energies - E_line) / config.sigma_instr) ** 2) / (config.sigma_instr * np.sqrt(2 * np.pi))
        spectrum += line_profile
        
    total_scale = 1e5 / np.sum(spectrum)
    spectrum *= total_scale
    spectrum += config.background_rate / config.nbins

    counts = rng.poisson(spectrum)
    errors = np.sqrt(np.maximum(counts, 1.0))
    hydrogen_synth_data = HydrogenData(energies=energies, counts=counts.astype(float), errors=errors)
    

    save_hydrogen_data_npz(hydrogen_synth_data, "synthetic_hydrogen.npz")

    return hydrogen_synth_data

def save_hydrogen_data_npz(hydrogen_data, filepath):
    """
    Save HydrogenData to a .npz file.

    Parameters
    ----------
    hydrogen_data : HydrogenData
        Object with attributes: energies, counts, errors
    filepath : str
        Path without or with .npz extension
    """
    np.savez(
        filepath,
        energies=hydrogen_data.energies,
        counts=hydrogen_data.counts,
        errors=hydrogen_data.errors,
    )
def load_hydrogen_data_npz(filepath):
    """
    Load HydrogenData from a .npz file.
    """
    with np.load(filepath) as data:
        energies = data["energies"]
        counts   = data["counts"]
        errors   = data["errors"]

    return HydrogenData(
        energies=energies,
        counts=counts,
        errors=errors,
    )
