#execution\domain\defaults.py

import numpy

DEFAULT_HYDROGEN_OBSERVATION = {
    "sigma_instr": 0.1,
    "background": 0.01,
    "scale": 1.0,
}

# Sensible defaults (can be refined later)
series_energy_ranges = {
    "lyman":   (10.0, 14.0),
    "balmer":  (1.5, 3.5),
    "paschen": (0.5, 1.5),
    "brackett":(0.1, 0.6),
    "pfund":   (0.05, 0.3),
}

def default_amplitudes(n: int) -> list[float]:
    return [1.0 / n] * n

