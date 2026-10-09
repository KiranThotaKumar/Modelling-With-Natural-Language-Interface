
#core.domains.hydrogen.hydrogen_physics.py

import numpy as np
from instrument.convolution import instrument_convolve

RYDBERG_EV = 13.598285


def hydrogen_line_energy(n_u: int, n_l: int) -> float:
    """
    Compute hydrogen transition energy in eV.
    """
    E_u = -RYDBERG_EV / (n_u ** 2)
    E_l = -RYDBERG_EV / (n_l ** 2)
    return E_u - E_l


def hydrogen_forward_spectrum(
    energies: np.ndarray,
    transitions,
    amplitudes,
    sigma_instr: float,
    background: float,
):
    """
    Canonical hydrogen forward model.

    Parameters
    ----------
    energies : array
        Energy grid (eV)
    transitions : list[(n_u, n_l)]
    amplitudes : array-like
        Linear amplitudes per line
    sigma_instr : float
        Instrumental Gaussian sigma
    background : float
        Additive background level

    Returns
    -------
    spectrum : ndarray
    """

    spec = np.zeros_like(energies)

    for (n_u, n_l), amp in zip(transitions, amplitudes):
        E_line = hydrogen_line_energy(n_u, n_l)

        intrinsic_sigma = max(0.1, sigma_instr * 0.1)

        line = (
            amp
            * np.exp(-0.5 * ((energies - E_line) / intrinsic_sigma) ** 2)
            / (intrinsic_sigma * np.sqrt(2 * np.pi))
        )

        spec += line

    spec = instrument_convolve(energies, spec, sigma_instr)

    spec = np.maximum(spec, 0.0)
    spec += background

    return spec