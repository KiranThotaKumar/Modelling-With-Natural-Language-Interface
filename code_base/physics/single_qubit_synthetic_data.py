
import numpy as np
from dataclasses import dataclass
from models.single_qubit_model import SingleQubitData
from typing import Dict, Any, Optional, List, Tuple, Union, Callable, Iterable, Literal


@dataclass
class SingleQubitSyntheticConfig:
    ntimes: int = 200
    tmax: float = 20.0    
    omega_r: float = 2.0
    detuning: float = 0.0
    gamma: float = 0.15
    amp: float = 1.0
    offset: float = 0.0
    noise_std: float = 0.02
    random_seed: Optional[int] = 42

# def generate_singlequbit_measured_data(config: SingleQubitSyntheticConfig) -> SingleQubitData:

#     rng = np.random.default_rng(config.random_seed)
#     times = np.linspace(0.0, config.tmax, config.ntimes)
#     omega_eff = np.sqrt(config.omega_r ** 2 + config.detuning ** 2)
#     observables_clean = config.offset + config.amp * np.exp(-config.gamma * times) * np.cos(omega_eff * times)
#     noise = rng.normal(scale=config.noise_std, size=times.shape)
#     observables = observables_clean + noise
#     errors = np.full_like(times, config.noise_std)
#     return SingleQubitData(times=times, observables=observables, errors=errors, metadata=None)

def generate_singlequbit_measured_data(config: SingleQubitSyntheticConfig) -> SingleQubitData:

    rng = np.random.default_rng(config.random_seed)
    times = np.linspace(0.0, config.tmax, config.ntimes)
    omega_eff = np.sqrt(config.omega_r ** 2 + config.detuning ** 2)
    measurements_clean = config.offset + config.amp * np.exp(-config.gamma * times) * np.cos(omega_eff * times)
    noise = rng.normal(scale=config.noise_std, size=times.shape)
    measurements = measurements_clean + noise
    errors = np.full_like(times, config.noise_std)

    metadata={
        "source": "synthetic",
        "noise_model": "gaussian"
    }

    return SingleQubitData(times=times, measurements=measurements, errors=errors, metadata=metadata)


def save_single_qubit_data_npz(single_qubit_data, filepath):

    np.savez(
        filepath,
        times=single_qubit_data.times,
        measurements=single_qubit_data.measurements,
        errors=single_qubit_data.errors,
        metadata=single_qubit_data.metadata
    )


def load_single_qubit_data_npz(filepath):

    with np.load(filepath, allow_pickle=True) as data:

        times = data["times"]
        measurements = data["measurements"]

        errors = (
            data["errors"]
            if "errors" in data.files
            else None
        )

        metadata = (
            data["metadata"].item()
            if "metadata" in data.files
            else None
        )

    return SingleQubitData(
        times=times,
        measurements=measurements,
        errors=errors,
        metadata=metadata
    )