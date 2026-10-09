from __future__ import annotations  # Enables postponement of type hint evaluation
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # This block is ONLY executed by type checkers, not at runtime
    from models.multi_qubit_model import MultiQubitData

import numpy as np
from scipy.signal import find_peaks
from signal_analysis.peak_finding import smooth_counts, estimate_fwhm, cluster_peaks, reduce_clusters



def softplus(x: float) -> float:
    """Calculates softplus function, ln(1 + e^x)."""
    return np.log1p(np.exp(x))

def inv_softplus(y: float) -> float:
    """Calculates inverse softplus function, ln(e^y - 1). Handles small y for numerical stability.
    Note: This is generally for y > 0. For y=0, we define inv_softplus(0) = -inf which maps to 0 after softplus.
    """
    if y <= 1e-12: # Handle cases where gamma is effectively zero
        return -100.0 # A very small number to map to near-zero after softplus(x*5.0)
    return np.log(np.exp(y) - 1.0)

def params_dict_to_theta_full(params: dict, nqubits: int) -> np.ndarray:
    """
    Inverse of theta_to_model_params_full.
    Converts a full-physics params dict into a flattened theta vector.
    Fully consistent with multi_qubit_model_physics() and expects:
      - 7 parameters per qubit
      - C(nqubits,2) couplings
      - 6 instrument parameters
    """

    # Physical scaling ranges (must match theta_to_model_params_full)
    # Retrieved from `params` dict, which should contain autoscaled ranges
    freq_offset = params.get("freq_offset", 0.1)
    freq_range  = params.get("freq_range", 2.0)
    gamma_range = params.get("_gamma_range_for_theta", 0.02)
    h_range     = params.get("_h_range_for_theta", 1.0)
    J_range     = params.get("_J_range_for_theta", 0.05)
    noise_range = params.get("_noise_range_for_theta", 0.005)
    phase_range = params.get("phase_range", np.pi/2)
    amp_range = params.get("_amp_range_for_theta", 1.0)
    dc_range    = params.get("dc_range", 0.0)
    time_range  = params.get("time_range", 1.0)

    theta = []

    # ---------------------------------------------------
    # 1. Per-qubit parameters (7 each)
    # ---------------------------------------------------
    # Expected order:
    #   freq, relax, dephase, h_x, h_y, h_z, drive_phase
    # ---------------------------------------------------
    for q in range(nqubits):
        qd = params["qubits"][q]

        raw_freq  = (qd["freq"] - freq_offset) / freq_range

        # Use linear mapping for gamma parameters, as they can be zero
        raw_relax   = qd["gamma_relax"] / gamma_range
        raw_dephase = qd["gamma_dephase"] / gamma_range

        raw_hx = qd["h_x"] / h_range
        raw_hy = qd["h_y"] / h_range
        raw_hz = qd["h_z"] / h_range

        raw_phase = qd["drive_phase"] / phase_range

        theta.extend([
            np.clip(raw_freq,   -1, 1),
            np.clip(raw_relax,   -1, 1),
            np.clip(raw_dephase, -1, 1),
            np.clip(raw_hx,      -1, 1),
            np.clip(raw_hy,      -1, 1),
            np.clip(raw_hz,      -1, 1),
            np.clip(raw_phase,   -1, 1),
        ])

    # ---------------------------------------------------
    # 2. Couplings (C(n,2) order)
    # ---------------------------------------------------
    # Ensure correct unpacking of couplings if it's a list of dicts/tuples
    processed_couplings = {}
    for k, v in params["couplings"].items():
        if isinstance(k, tuple) and len(k) == 2: # Already (i,j) tuple
            processed_couplings[k] = v
        elif isinstance(k, dict) and 'i' in k and 'j' in k: # if it came as a list of dicts
            processed_couplings[(k['i'], k['j'])] = k.get('strength', v)

    for i in range(nqubits):
        for j in range(i+1, nqubits):
            # Ensure the key is a tuple (i,j) for lookup
            J = processed_couplings.get((i, j), 0.0)
            raw_J = J / J_range
            theta.append(np.clip(raw_J, -1, 1))

    # ---------------------------------------------------
    # 3. Instrument parameters (last 6 terms)
    # ---------------------------------------------------
    inst = params["instrument"]

    # Use linear mapping for noise_sigma
    raw_noise  = inst["noise_sigma"] / noise_range

    # Gain still uses softplus due to its positive-definite nature
    raw_gain   = inv_softplus(inst["gain"] / amp_range) / 5.0

    raw_gphase = inst["global_phase_offset"] / phase_range
    raw_tdelay = inst["time_delay"] / time_range
    raw_slope  = inst["amp_slope"] / amp_range
    raw_dc     = 0.0 if dc_range == 0 else inst["dc_offset"] / dc_range

    theta.extend([
        np.clip(raw_noise,  -1, 1),
        np.clip(raw_gain,   -1, 1),
        np.clip(raw_gphase, -1, 1),
        np.clip(raw_tdelay, -1, 1),
        np.clip(raw_slope,  -1, 1),
        np.clip(raw_dc,     -1, 1),
    ])

    return np.array(theta, dtype=float)

def estimate_period_from_trace(tvec, y, peak_height_threshold: float = 0.1):
    # robust period estimate: FFT peak + refine with time-domain peaks
    y0 = y - np.mean(y)
    N = len(y0)
    if N < 5:
        print(f"[DEBUG_PERIOD] Returning None: N={N} < 5")
        return None
    dt = tvec[1] - tvec[0]
    yf = np.fft.rfft(y0 * np.hanning(N))
    freqs = np.fft.rfftfreq(N, dt)
    mag = np.abs(yf)
    mag[0] = 0.0
    idx = np.argmax(mag)
    if mag[idx] <= 0:
        print(f"[DEBUG_PERIOD] Returning None: mag[idx]={mag[idx]} <= 0")
        return None
    f0 = freqs[idx]
    if f0 <= 0:
        print(f"[DEBUG_PERIOD] Returning None: f0={f0} <= 0")
        return None
    T_fft = 1.0 / f0
    # refine: find peaks in time domain
    # MODIFICATION: Use a more robust height threshold for peak detection
    # A fraction of the peak-to-peak amplitude, with a floor based on std dev
    peak_to_peak_amp = np.max(y0) - np.min(y0)
    min_peak_height = peak_to_peak_amp * peak_height_threshold
    # Ensure min_peak_height is not extremely small; add a floor based on overall signal variation
    min_peak_height = max(min_peak_height, np.std(y0) * 0.05) # Smaller floor to catch more peaks

    # Adjust distance based on the FFT-estimated period for more accurate peak spacing
    # This prevents finding multiple peaks within a single cycle or missing peaks
    min_peak_distance_steps = max(2, int(T_fft / (2 * dt))) # Distance should be less than half a period
    
    peaks, props = find_peaks(y0, height=min_peak_height, distance=min_peak_distance_steps)
    print(f"[DEBUG_PERIOD] Detected {len(peaks)} peaks. T_fft={T_fft:.4f}")
    if len(peaks) >= 3:
        tpeaks = tvec[peaks]
        spacings = np.diff(tpeaks)
        T_med = np.median(spacings)
        print(f"[DEBUG_PERIOD] Returning T_med={T_med:.4f}")
        return float(T_med)
    print(f"[DEBUG_PERIOD] Returning T_fft={T_fft:.4f}")
    return float(T_fft)

def convert_legacy_to_full_physics(params_legacy, n_qubits):
    """
    Convert legacy Ising-parameter metadata into the new full-physics param dict.
    This keeps the values consistent and fills missing physics terms with defaults.
    """

    # --------------- Extract legacy --------------------
    local_fields = params_legacy.get("local_fields", {})
    zz_list = params_legacy.get("zz_couplings", [])

    # --------------- Build full-physics dict --------------------
    params_full = {
        "nqubits": n_qubits,
        "qubits": [],
        "couplings": {},
        "instrument": {
            # Reasonable defaults (no noise, no gain distortions)
            "noise_sigma": 0.0,
            "gain": 1.0,
            "global_phase_offset": 0.0,
            "time_delay": 0.0,
            "amp_slope": 0.0,
            "dc_offset": 0.0,
        }
    }

    # ------------ Per-qubit ---------------------------
    for q in range(n_qubits):
        # Legacy fields: (hx, hy, hz)
        hx, hy, hz = local_fields[q]

        # Map legacy → new fields
        params_full["qubits"].append({
            #CodeChange from 4.5 to 0.1
            "freq": 0.1 + hx,            # simple mapping, adjustable
            "gamma_relax": 0.01,         # safe small nonzero default
            "gamma_dephase": 0.01,
            "h_x": hx,
            "h_y": hy, 
            "h_z": hz,
            "drive_phase": 0.0,
        })

    # ------------ Couplings ---------------------------
    for (i, j, Jij) in zz_list:
        params_full["couplings"][(i, j)] = Jij

    #print("FULL MODEL PARAMS:", params_full)

    return params_full


def prepare_theta_from_measured(data: "MultiQubitData") -> tuple[np.ndarray, dict]:
    """
    Convert measured MultiQubitData into theta suitable for MCMC input.

    Parameters
    ----------
    data : MultiQubitData
        Measured data including times, observables, and metadata with params.

    Returns
    -------
    theta_measured : np.ndarray
        Flattened theta array ready for MCMC.
    params_measured : dict
        The parameters dictionary *after* autoscaling (h_x,y,z, J, gamma, noise) has been applied.
        This is the effective input to params_dict_to_theta_full.
    """
    md = data.metadata
    n_qubits = md['n_qubits']

    # Initial state
    psi0 = getattr(md, "psi0", None)
    if psi0 is None:
        psi0 = np.zeros((2 ** n_qubits,), dtype=complex)
        psi0[1 << (n_qubits - 1 - 0)] = 1.0  # replicate measured initial state

    # Extract measured parameters & convert legacy → full physics
    # IMPORTANT: Make a deepcopy here if this `params_measured` is modified in-place,
    # but we want to return the state *after* autoscaling.
    params_measured = convert_legacy_to_full_physics(md["params"], n_qubits)

    # Observables shape handling
    tvec = np.asarray(data.times)
    obs = np.asarray(data.measurements)
    if obs.shape[0] == len(tvec) and obs.shape[1] != len(tvec):
        obs = obs
    elif obs.shape[0] == len(tvec):
        obs = obs.T
    elif obs.shape[1] == len(tvec):
        obs = obs
    else:
        obs = obs.T

    # Auto-scale physical drives per qubit
    h_scales = []
    for i in range(n_qubits):
        y = obs[i]
        T_meas = estimate_period_from_trace(tvec, y)
        if T_meas is None:
            # print(f"[AUTOSCALE] Q{i}: no clear period found, skipping autoscale.")
            h_scales.append(1.0)
            continue
        omega_meas = 2.0 * np.pi / T_meas
        q = params_measured['qubits'][i]

        model_hamiltonian_field_magnitude = np.sqrt(q.get('h_x',0.0)**2 + q.get('h_y',0.0)**2 + q.get('h_z',0.0)**2)
        if model_hamiltonian_field_magnitude <= 0:
            # print(f"[AUTOSCALE] Q{i}: model Hamiltonian magnitude zero, skipping autoscale.")
            h_scales.append(1.0)
            continue

        target_hamiltonian_field_magnitude = omega_meas / 2.0

        h_scale = float(target_hamiltonian_field_magnitude / model_hamiltonian_field_magnitude)
        # print(f"[AUTOSCALE] Q{i}: T_meas={T_meas:.6g}, omega_meas={omega_meas:.6g}, model_hamiltonian_field_magnitude={model_hamiltonian_field_magnitude:.6g}, target_hamiltonian_field_magnitude={target_hamiltonian_field_magnitude:.6g}, h_scale={h_scale:.6g}")
        h_scales.append(h_scale)
        # Apply scale
        params_measured['qubits'][i]['h_x'] *= h_scale
        params_measured['qubits'][i]['h_y'] *= h_scale
        params_measured['qubits'][i]['h_z'] *= h_scale

    # Compute safe h_range_for_theta
    all_h = [abs(h) for q in params_measured['qubits'] for h in (q.get('h_x',0.0), q.get('h_y',0.0), q.get('h_z',0.0))]
    max_h = max(all_h) if all_h else 1.0
    h_range_for_theta = max(1.0, max_h * 1.1)  # 10% margin

    print(f"[AUTOSCALE] chosen _h_range_for_theta = {h_range_for_theta:.6g} (max_true_H_mag={max_h:.6g})")
    data.metadata['_h_range_for_theta'] = h_range_for_theta
    params_measured['_h_range_for_theta'] = h_range_for_theta # Store in params_measured for pass-through

    # Auto-scale amplitude for instrument gain
    peak_to_peak_amplitude = np.max(obs) - np.min(obs)
    amp_range_for_theta = max(1.0, peak_to_peak_amplitude * 1.1) # 10% margin, minimum 1.0
    print(f"[AUTOSCALE] chosen _amp_range_for_theta = {amp_range_for_theta:.6g} (peak_to_peak={peak_to_peak_amplitude:.6g})")
    data.metadata['_amp_range_for_theta'] = amp_range_for_theta
    params_measured['_amp_range_for_theta'] = amp_range_for_theta # Store in params_measured for pass-through

    # Determine J_range (couplings)
    all_J = [abs(strength) for k, strength in params_measured.get('couplings', {}).items()]
    max_J = max(all_J) if all_J else 0.05 # Default if no couplings
    J_range_for_theta = max(0.05, max_J * 1.1) # 10% margin, min 0.05
    print(f"[AUTOSCALE] chosen J_range = {J_range_for_theta:.6g} (max_true_J={max_J:.6g})")
    data.metadata['_J_range_for_theta'] = J_range_for_theta
    params_measured['_J_range_for_theta'] = J_range_for_theta # Store in params_measured for pass-through

    # Determine gamma_range (relaxation/dephasing)
    # Assuming small default values for gamma, so make range small but adaptable
    all_gamma = []
    for q in params_measured['qubits']:
        all_gamma.append(q.get('gamma_relax', 0.0))
        all_gamma.append(q.get('gamma_dephase', 0.0))
    max_gamma = max(all_gamma) if all_gamma else 0.01 # Default if no gamma
    gamma_range_for_theta = max(0.01, max_gamma * 2.0) # Larger margin for gamma
    print(f"[AUTOSCALE] chosen gamma_range = {gamma_range_for_theta:.6g}")
    data.metadata['_gamma_range_for_theta'] = gamma_range_for_theta
    params_measured['_gamma_range_for_theta'] = gamma_range_for_theta # Store in params_measured for pass-through

    # Determine noise_range
    # Assuming noise_sigma might be 0. Use a sensible default range.
    noise_sigma_true = params_measured.get('instrument', {}).get('noise_sigma', 0.0)
    noise_range_for_theta = max(0.005, noise_sigma_true * 2.0) # Larger margin
    print(f"[AUTOSCALE] chosen noise_range = {noise_range_for_theta:.6g}")
    data.metadata['_noise_range_for_theta'] = noise_range_for_theta
    params_measured['_noise_range_for_theta'] = noise_range_for_theta # Store in params_measured for pass-through


    # Convert to theta
    theta_measured = params_dict_to_theta_full(params_measured, n_qubits)

    return theta_measured, params_measured


def theta_to_model_params_full(theta: np.ndarray, nqubits: int = 2, **kwargs) -> dict:
    """
    Compatibility-preserving, MCMC-stable mapping for multi-qubit physics.
    Maps theta in [-1,1] ranges to realistic amplitudes, couplings, relaxation rates, and instrument parameters.
    """

    # --- Physical ranges (fetched from kwargs to ensure consistency) ---
    freq_offset = kwargs.get("freq_offset", 0.1)
    freq_range  = kwargs.get("freq_range", 2.0)
    gamma_range = kwargs.get("_gamma_range_for_theta", 0.02)
    h_range = kwargs.get("h_range", 1.0) # This should come from _h_range_for_theta in practice
    J_range = kwargs.get("_J_range_for_theta", 0.05)
    noise_range = kwargs.get("_noise_range_for_theta", 0.005)
    phase_range = kwargs.get("phase_range", np.pi/2)
    amp_range = kwargs.get("amp_range", 1.0) # This should come from _amp_range_for_theta in practice
    dc_range = kwargs.get("dc_range", 0.0)
    time_range = kwargs.get("time_range", 1.0)

    per_qubit_params = 7
    n_couplings_expected = nqubits * (nqubits - 1) // 2
    expect_len = nqubits * per_qubit_params + n_couplings_expected + 6
    if len(theta) < expect_len:
        raise ValueError(f"theta length {len(theta)} ({theta.shape}) < expected {expect_len}")

    params = {"nqubits": nqubits, "qubits": [], "couplings": {}, "instrument": {}, "raw_theta": theta.copy()}
    idx = 0

    # --- Per-qubit parameters ---
    for q in range(nqubits):
        raw_freq = np.clip(theta[idx], -1, 1); idx += 1
        raw_relax = np.clip(theta[idx], -1, 1); idx += 1
        raw_dephase = np.clip(theta[idx], -1, 1); idx += 1
        raw_hx = np.clip(theta[idx], -1, 1); idx += 1
        raw_hy = np.clip(theta[idx], -1, 1); idx += 1
        raw_hz = np.clip(theta[idx], -1, 1); idx += 1
        raw_drive_phase = np.clip(theta[idx], -1, 1); idx += 1

        freq = freq_offset + freq_range * raw_freq

        # Use linear mapping for gamma parameters
        gamma_relax = gamma_range * raw_relax
        gamma_dephase = gamma_range * raw_dephase

        h_x = h_range * raw_hx
        h_y = h_range * raw_hy
        h_z = h_range * raw_hz
        drive_phase = phase_range * raw_drive_phase

        params["qubits"].append({
            "freq": freq,
            "gamma_relax": np.maximum(gamma_relax, 0.0), # Ensure non-negative
            "gamma_dephase": np.maximum(gamma_dephase, 0.0), # Ensure non-negative
            "h_x": h_x,
            "h_y": h_y,
            "h_z": h_z,
            "drive_phase": drive_phase
        })

    # --- Couplings ---
    couplings_list = []
    for _ in range(n_couplings_expected):
        raw_J = np.clip(theta[idx], -1, 1); idx += 1
        couplings_list.append(J_range * raw_J)

    coupling_dict = {}
    count = 0
    for i in range(nqubits):
        for j in range(i + 1, nqubits):
            coupling_dict[(i, j)] = couplings_list[count]
            count += 1
    params["couplings"] = coupling_dict

    # --- Instrument parameters ---
    raw_noise = np.clip(theta[idx], -1, 1); idx += 1
    raw_gain = np.clip(theta[idx], -1, 1); idx += 1
    raw_global_phase = np.clip(theta[idx], -1, 1); idx += 1
    raw_time_delay = np.clip(theta[idx], -1, 1); idx += 1
    raw_amp_slope = np.clip(theta[idx], -1, 1); idx += 1
    raw_dc_offset = np.clip(theta[idx], -1, 1); idx += 1

    params["instrument"].update({
        "noise_sigma": np.maximum(noise_range * raw_noise, 0.0), # Use linear, ensure non-negative
        "gain": softplus(raw_gain * 5.0) * amp_range, # Keep softplus for gain
        "global_phase_offset": phase_range * raw_global_phase,
        "time_delay": time_range * raw_time_delay,
        "amp_slope": amp_range * raw_amp_slope,
        "dc_offset": dc_range * raw_dc_offset
    })

    return params