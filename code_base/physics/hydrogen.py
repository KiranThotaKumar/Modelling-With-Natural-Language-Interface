
import numpy as np
from instrument.convolution import instrument_convolve

def hydrogen_model_spectrum_physics(energies: np.ndarray, params:dict) -> np.ndarray:
    Rydberg = 13.598285
    spec = np.zeros_like(energies)
    for (n_u, n_l), amp in zip(params["transitions"], params["amplitudes"]):
        E_u = -Rydberg / (n_u ** 2)
        E_l = -Rydberg / (n_l ** 2)
        E_line = (E_u - E_l)
        #print("Energy of the line for lower {n_l} and upper {n_u} is: ", n_l, n_u, E_line)
        # Optional: resolve per-line sigma based on resolving power
        intrinsic_sigma = max(0.1, params["sigma_instr"] * 0.1)
        line = amp * np.exp(-0.5 * ((energies - E_line) / intrinsic_sigma) ** 2) / (intrinsic_sigma * np.sqrt(2 * np.pi))
        #has_negative = (line < 0).any()
        #if(has_negative):
            #print("Line has negative values", has_negative, n_u, n_l, amp)
        spec += line
    
    # Convolution for instrumental response
    spec = instrument_convolve(energies, spec, params["sigma_instr"])

    # Scale and add background (scale lines first, then add background)
    spec = np.maximum(spec, 0.0)
    spec += params["background"]
    return spec

def hydrogen_transition_energies(n_max=7):
    transitions = []
    energies = []

    for n in range(2, n_max + 1):
        E = 13.6 * (1.0 - 1.0 / n**2)
        transitions.append((n, 1))
        energies.append(E)

    return np.array(energies), transitions

def match_lines_to_hydrogen_transitions(line_energies, n_max=7):
    theo_energies, transitions = hydrogen_transition_energies(n_max)

    matched = []
    residuals = []

    for E_meas in line_energies:
        idx = np.argmin(np.abs(theo_energies - E_meas))
        matched.append(transitions[idx])
        residuals.append(E_meas - theo_energies[idx])

    return matched, np.array(residuals)
