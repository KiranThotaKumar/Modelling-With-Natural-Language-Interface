#Modelling\default_inits\default_theta_params.py

import numpy as np

def build_default_theta(args) -> np.ndarray:

    if args.mode == 'hydrogen':        
        # Much smaller instrument sigma (in same units as energy, e.g. eV)
        nlines = len(args.transitions)
        rels = args.amplitudes
        #rels = [0.6, 0.4][:nlines]
        log_rels = np.log(rels) # Taking logs to avoid neggative amplitudes        
        return np.array([args.sigma_instr, args.background, args.scale] + list(log_rels))

    elif args.mode == "singlequbit":
        return np.array([args.omega_r, args.detuning, args.gamma, args.amp, args.offset])  # basic default       

    elif args.mode == "multiqubit":    
        return build_default_theta_full(args)

    else:
        raise ValueError(f"Unsupported type : {which}")

def build_default_theta_full(which: str, nlines: int = 3, nqubits: int = 2, rng=None, **kwargs) -> np.ndarray:
    """
    Build default theta for multi-qubit MCMC.
    Includes small random perturbations for each walker to avoid ad-hoc p0 modifications.
    
    Parameters:
    - rng: np.random.Generator instance for reproducible noise
    """

    if rng is None:
        rng = np.random.default_rng()

    n_qubit_params = 7
    n_couplings = nqubits * (nqubits - 1) // 2
    n_instrument = 6
    total_len = nqubits * n_qubit_params + n_couplings + n_instrument

    theta = np.zeros(total_len)

    # --- Per-qubit defaults with small randomized spread ---
    for q in range(nqubits):
        base = q * n_qubit_params

        # deterministic defaults
        theta[base + 0] = 5.0 + 0.01*q + 0.005*q       # freq
        theta[base + 1] = 1e-3                          # gamma_relax
        theta[base + 2] = 5e-4                          # gamma_dephase

        # h_x, h_y, h_z: reduced amplitudes + small random
        theta[base + 3] = 0.2 + 0.05*rng.standard_normal()  # h_x
        theta[base + 4] = 0.15 + 0.05*rng.standard_normal() # h_y
        theta[base + 5] = 0.1 + 0.05*rng.standard_normal()  # h_z

        # drive_phase with small random perturbation
        theta[base + 6] = 0.15 + 0.05*rng.standard_normal()  

    # --- Couplings ---
    start_coupl = nqubits * n_qubit_params
    for i in range(n_couplings):
        theta[start_coupl + i] = 0.05 + 0.02*rng.standard_normal()

    # --- Instrument defaults ---
    inst_idx = start_coupl + n_couplings
    theta[inst_idx + 0] = 0.01 + 0.005*rng.standard_normal()  # noise_sigma
    theta[inst_idx + 1] = 1.0 + 0.05*rng.standard_normal()    # gain
    theta[inst_idx + 2] = 0.0 + 0.05*rng.standard_normal()    # global phase
    theta[inst_idx + 3] = 0.01 + 0.005*rng.standard_normal()  # time delay
    theta[inst_idx + 4] = 0.0 + 0.01*rng.standard_normal()    # amp slope
    theta[inst_idx + 5] = 0.0 + 0.01*rng.standard_normal()    # DC offset

    print(f"[DEBUG] Built default theta with randomization (len={len(theta)}): {theta}")
    return theta