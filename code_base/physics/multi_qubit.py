import numpy as np
import qutip as qt
#from models.multi_qubit_model import MultiQubitModelConfig
from models.multi_qubit_param_builder import theta_to_model_params_full


def multi_qubit_model_physics(times:np.ndarray, config_params:dict ) -> np.ndarray:
    """
    Multi-qubit physics model with optional speedup and spectral-domain computation.

    Notes:
    - Keeps signature compatible with your existing code.
    - If `qubit_params_override` is provided, it is used directly (no theta->params conversion).
    - `debug_compare` (bool, in kwargs) toggles a fast-vs-slow comparison test.
    """

    spectrum_mode = config_params["spectrum_mode"]
    sigma_instr = config_params["sigma_instr"]
    background = config_params["background"]

    # optional debug flag: compare fast vs slow implementations for exactness
    debug_compare = config_params["debug_compare"]
    #model_grid = config_params["model_grid"]
    fast = config_params["fast"]

    # ensure kwargs receives h_range and tlist for downstream conversion functions
    kwargs = {}
    kwargs["h_range"] = config_params["h_range"]
    kwargs["amp_range"] = config_params["amp_range"]
    kwargs["tlist"] = times
    

    # Convert theta -> qubit params (or use override)
    if config_params["qubit_params_override"] is not None:
        params = config_params["qubit_params_override"]
    else:
        params = theta_to_model_params_full(config_params["theta"], **kwargs)

    n_qubits = int(params.get("nqubits", config_params["n_qubits"]))

    # -------------------------------
    # Build operator lists (fast)
    # -------------------------------
    if fast:
        sigmaz_list = [qt.tensor([qt.sigmaz() if k == idx else qt.qeye(2) for k in range(n_qubits)])
                        for idx in range(n_qubits)]
        sigmax_list = [qt.tensor([qt.sigmax() if k == idx else qt.qeye(2) for k in range(n_qubits)])
                        for idx in range(n_qubits)]
        sigmay_list = [qt.tensor([qt.sigmay() if k == idx else qt.qeye(2) for k in range(n_qubits)])
                        for idx in range(n_qubits)]
    else:
        # For slow mode we will construct per-qubit tensor operators when needed
        sigmaz_list = None
        sigmax_list = None
        sigmay_list = None

    # -------------------------------
    # Defensive guard: one-time build flag
    # -------------------------------
    # Prevent accidental multiple builds in the same call path (helpful during debugging)
    if getattr(multi_qubit_model_physics, "_hamiltonian_built_flag", False):
        # Reset the flag for next call — keep it non-blocking; in debug you may raise
        multi_qubit_model_physics._hamiltonian_built_flag = False

    # -------------------------------
    # 2.1 Build H from term lists (components) and sum once (guard A)
    # -------------------------------
    H_terms = []

    # Collect per-qubit terms so we sum once. Also print per-qubit debug ω (guard D).
    per_qubit_omega_info = []

    for idx_q, q in enumerate(params["qubits"]):
        freq = float(q.get("freq", 0.0))
        h_x = float(q.get("h_x", 0.0))
        h_y = float(q.get("h_y", 0.0))
        h_z = float(q.get("h_z", 0.0))

        # compute model-convention omega for printing:
        # If the fast branch multiplies h by 2*pi, show the same convention.
        # Assumption: fast uses 2*pi multiplication for the local drives (we keep that).
        omega_model = np.sqrt((h_x)**2 + (h_y)**2 + (h_z)**2) if (h_x or h_y or h_z) else 0.0
        per_qubit_omega_info.append((idx_q, h_x, h_y, h_z, omega_model))

        if fast:
            # in fast branch we will apply 2*pi scaling to local drives (as per our convention)
            if h_x != 0.0:
                H_terms.append(h_x * sigmax_list[idx_q] )
            if h_y != 0.0:
                H_terms.append(h_y * sigmay_list[idx_q] )
            if h_z != 0.0:
                H_terms.append(h_z * sigmaz_list[idx_q] )

            # NOTE: we exclude freq term (per your request B). Keep as commented ref:
            # H_terms.append( (2*np.pi * 0.5 * freq) * sigmaz_list[idx_q] )  # optionally include
        else:
            # slow / explicit branch: build full tensor for each op
            ops_x = [qt.qeye(2)] * n_qubits
            ops_y = [qt.qeye(2)] * n_qubits
            ops_z = [qt.qeye(2)] * n_qubits
            ops_x[idx_q] = qt.sigmax()
            ops_y[idx_q] = qt.sigmay()
            ops_z[idx_q] = qt.sigmaz()

            # Use 2*pi scaling in slow branch as well (keep conventions identical)
            if h_x != 0.0:
                H_terms.append(h_x * qt.tensor(ops_x))
            if h_y != 0.0:
                H_terms.append(h_y * qt.tensor(ops_y))
            if h_z != 0.0:
                H_terms.append(h_z * qt.tensor(ops_z))

            # commented out freq term for reference:
            # H_terms.append((2*np.pi * 0.5 * freq) * qt.tensor(ops_z))

    # -------------------------------
    # 2.2 Handle couplings (collect then apply 2π)
    # -------------------------------
    raw_couplings = config_params["couplings"] if config_params["couplings"] is not None else params.get("couplings", [])
    all_couplings = []

    if isinstance(raw_couplings, (float, int)):
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                all_couplings.append((i, j, float(raw_couplings)))
    elif isinstance(raw_couplings, dict):
        for (i, j), g in raw_couplings.items():
            all_couplings.append((int(i), int(j), float(g)))
    elif isinstance(raw_couplings, (list, tuple)):
        for item in raw_couplings:
            if isinstance(item, (float, int)):
                for i in range(n_qubits):
                    for j in range(i + 1, n_qubits):
                        all_couplings.append((i, j, float(item)))
            elif isinstance(item, (list, tuple)):
                if len(item) == 3:
                    all_couplings.append((int(item[0]), int(item[1]), float(item[2])))
                elif len(item) == 2:
                    all_couplings.append((int(item[0]), int(item[1]), 0.05))

    # apply couplings (2*pi scaled if modeling linear->angular)
    for (i, j, g) in all_couplings:
        if fast:
            H_terms.append( float(g) * sigmaz_list[i] * sigmaz_list[j])
        else:
            ops = [qt.qeye(2)] * n_qubits
            ops[i] = qt.sigmaz()
            ops[j] = qt.sigmaz()
            H_terms.append(float(g) * qt.tensor(ops))  # slow branch preserves same operator but no extra scaling here
                
    # -------------------------------
    # Final H: sum exactly once (guard A)
    # -------------------------------
    if len(H_terms) == 0:
        H = 0
    else:
        # sum terms safely; ensure Qobj addition is used
        H = H_terms[0]
        for term in H_terms[1:]:
            H = H + term

    # mark built (guard C)
    multi_qubit_model_physics._hamiltonian_built_flag = True

    # -------------------------------
    # Small numeric sanity check: Frobenius norm / magnitude (guard B)
    # -------------------------------
    try:
        #H_qobj = H  # whatever QuTiP Qobj
        #H_full = H_qobj.full()
        #max_imag = np.max(np.abs(H_full.imag))
        #print("[DEBUG] max imaginary part in H:", max_imag)
        #if max_imag > 1e-10:
            #print("[WARNING] Non-negligible imaginary parts in Hamiltonian.", max_imag)
        #H_mat = H_full.real  # safe if imag is tiny

        H_mat = np.array(H.full(), dtype=float)
        frob = np.linalg.norm(H_mat)
        # print for debugging; comment/remove in production
        #print(f"[H CHECK] H Frobenius norm: {frob:.6g}")
        # optional assert threshold (tunable)
        if frob > 1e6:
            print("[H WARNING] Hamiltonian norm unusually large, check for duplicated terms/units.")
    except Exception:
        # Qobj.full might fail for symbolic or other unusual cases; skip the check
        pass

    # -------------------------------
    # 2.3 Initial state & observables
    # -------------------------------
    psi0 = qt.tensor([qt.basis(2, 1) if k == 0 else qt.basis(2, 0) for k in range(n_qubits)])
    e_ops = sigmaz_list if fast else [qt.tensor([qt.sigmaz() if k == idx else qt.qeye(2) for k in range(n_qubits)])
                                        for idx in range(n_qubits)]

    # print initial per-qubit <Z> expected by model & per-qubit omega debug
    for qidx, op in enumerate(e_ops if fast else e_ops):
        init_val = qt.expect(op, psi0)
        hx, hy, hz = params["qubits"][qidx]["h_x"], params["qubits"][qidx]["h_y"], params["qubits"][qidx]["h_z"]
        # omega_model (angular) used in H:
        omega_model_print = np.sqrt((hx)**2 + (hy)**2 + (hz)**2) if (hx or hy or hz) else 0.0
        #print(f"[DEBUG] model initial <Z> qubit {qidx} = {init_val}, hx={hx}, hy={hy}, hz={hz}, omega_model={omega_model_print:.6g}")

    # -------------------------------
    # 2.4 Spectrum mode
    # -------------------------------
    if spectrum_mode:
        eigvals = H.eigenenergies()
        transitions = [eigvals[i] - eigvals[j] for i in range(len(eigvals)) for j in range(i)]
        transitions = np.array(transitions)
        #freq_grid = np.linspace(np.min(transitions) - 5, np.max(transitions) + 5, len(times)) # To avoid negative frequencies, clipped to zero
        freq_grid = np.linspace(0.0, np.max(transitions) + 0.5, len(times))
        y_freq = np.zeros_like(freq_grid)
        for tr in transitions:
            y_freq += np.exp(-0.5 * ((freq_grid - tr) / sigma_instr)**2)
        print(freq_grid[:10])
        print(freq_grid[-10:])

        print("Spectrum min:", np.min(y_freq))
        print("Spectrum max:", np.max(y_freq))
        print("Spectrum argmax:", np.argmax(y_freq))
        final_spec = background + y_freq / np.max(y_freq)

        print("Final min:", final_spec.min())
        print("Final max:", final_spec.max())

        return freq_grid, background + y_freq / np.max(y_freq)

    # -------------------------------
    # 2.5 Test evolution (dense and period-aware) - replaces tiny 0.1,5 test
    # -------------------------------
    # Compute expected (model) minimal Rabi omega for window choice
    rabi_omegas = []
    for q in params["qubits"]:
        hx, hy, hz = float(q.get("h_x", 0.0)), float(q.get("h_y", 0.0)), float(q.get("h_z", 0.0))
        if hx == 0 and hy == 0 and hz == 0:
            continue
        # match the H scaling convention: we used 2*pi*hx in H, so include that here
        omega = np.sqrt((hx)**2 + (hy)**2 + (hz)**2)
        rabi_omegas.append(omega)
    if len(rabi_omegas) == 0:
        t_test = np.linspace(0.0, min(1.0, times[-1] if len(times)>0 else 1.0), 201)
    else:
        omega_min = min(rabi_omegas)
        T_rabi = 2.0 * np.pi / omega_min if omega_min > 0 else 1.0
        # cover a few periods with good resolution
        t_test = np.linspace(0.0, max(3.0*T_rabi, 0.1), 401)

    res_test = qt.sesolve(H, psi0, t_test, e_ops=e_ops if fast else e_ops)
    obs_test = np.array(res_test.expect)  # shape (n_qubits, len(t_test))
    #print("[DEBUG] t_test span, dt:", t_test[0], t_test[1], t_test[1]-t_test[0])
    #print("[DEBUG] test obs min/max per qubit:", obs_test.min(axis=1), obs_test.max(axis=1))

    # -------------------------------
    # 2.6 Time-domain evolution (actual)
    # -------------------------------
        
    result = qt.sesolve(H, psi0, times, e_ops=e_ops)
    obs_array = np.array(result.expect)
    #print("[DEBUG] actual obs min/max per qubit:", obs_array.min(axis=1), obs_array.max(axis=1))

    # -------------------------------
    # Optional: compare fast vs slow implementations (debug_compare)
    # -------------------------------
    if debug_compare:
        try:
            # Build slow H explicitly for comparison (re-use same params but non-fast operator construction)
            # This intentionally mirrors the non-fast branch above, but we do it here for check only.
            H_slow_terms = []
            for idx_q, q in enumerate(params["qubits"]):
                hx, hy, hz = float(q.get("h_x", 0.0)), float(q.get("h_y", 0.0)), float(q.get("h_z", 0.0))
                # explicit tensor ops
                ops_x = [qt.qeye(2)] * n_qubits
                ops_y = [qt.qeye(2)] * n_qubits
                ops_z = [qt.qeye(2)] * n_qubits
                ops_x[idx_q] = qt.sigmax()
                ops_y[idx_q] = qt.sigmay()
                ops_z[idx_q] = qt.sigmaz()
                if hx != 0.0:
                    H_slow_terms.append( hx* qt.tensor(ops_x))
                if hy != 0.0:
                    H_slow_terms.append( hy * qt.tensor(ops_y))
                if hz != 0.0:
                    H_slow_terms.append(hz * qt.tensor(ops_z))
            for (i,j,g) in all_couplings:
                H_slow_terms.append(( float(g)) * (qt.tensor([qt.sigmaz() if k==i else qt.qeye(2) for k in range(n_qubits)]) *
                                                                qt.tensor([qt.sigmaz() if k==j else qt.qeye(2) for k in range(n_qubits)])))
            if len(H_slow_terms) == 0:
                H_slow = 0
            else:
                H_slow = H_slow_terms[0]
                for term in H_slow_terms[1:]:
                    H_slow = H_slow + term

            res_fast = qt.sesolve(H, psi0, t_test, e_ops=e_ops)
            res_slow = qt.sesolve(H_slow, psi0, t_test, e_ops=e_ops)
            obs_fast = np.array(res_fast.expect)
            obs_slow = np.array(res_slow.expect)
            diff = np.abs(obs_fast - obs_slow)
            maxdiff = diff.max()
            #print(f"[COMPARE] fast-vs-slow max abs diff = {maxdiff:.3e}")
            if maxdiff > 1e-6:
                print("[COMPARE] WARNING: fast and slow differ. Investigate H construction and conversions.")
        except Exception as ex:
            print("[COMPARE] Exception during fast/slow comparison:", ex)

    # -------------------------------
    # 2.7 Instrument effects & return
    # -------------------------------
    gain = params.get("instrument", {}).get("gain", 1.0)
    dc = params.get("instrument", {}).get("dc_offset", 0.0)
    y_time = gain * obs_array.T + dc
    ############Comparison with Measured############
    y_pred = y_time
    # try:
    #     y_obs = np.loadtxt('Measured.txt')
    #     compareModelWithMeasured(y_pred, y_obs)
    # except FileNotFoundError:
    #     raise FileNotFoundError("Error: 'Measured.txt' not found.")
    # #except Exception as e:
    #     #raise Exception(f"Error loading 'Measured.txt': {e}")
    # # Determine number of qubits
    # if y_obs.shape != y_pred.shape:
    #     raise ValueError(f"Shape mismatch between measured ({y_obs.shape}) and predicted ({y_pred.shape}) data.")
    # n_qubits = y_obs.shape[1]

    # rmse_per_qubit = [np.sqrt(np.mean((y_obs[:, i] - y_pred[:, i])**2)) for i in range(n_qubits)]
    # corr_per_qubit = [np.corrcoef(y_obs[:, i], y_pred[:, i])[0,1] for i in range(n_qubits)]
    # for q in range(n_qubits):
    #     current_rmse = rmse_per_qubit[q]
    #     print(f"Qubit {q}: RMSE = {current_rmse:.4f}, Correlation = {corr_per_qubit[q]:.4f}")
    #     if current_rmse > 0.05:
    #         print("These  params need correction:", params)
    #         raise ValueError(f"RMSE for Qubit {q} ({current_rmse:.4f}) exceeds the threshold of 0.05. Exiting program.")
    ################################################
    # print("obs min/max:",
    # obs_array.min(),
    # obs_array.max())

    # print("gain:",
    #         gain)

    # print("dc:",
    #         dc)

    # print("y min/max:",
    #         y_time.min(),
    #         y_time.max())
    return y_time  # shape = (T, n_qubits)