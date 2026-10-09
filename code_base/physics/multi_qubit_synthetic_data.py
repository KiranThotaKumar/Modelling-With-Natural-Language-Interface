
import logging
import qutip as qt
import numpy as np
from dataclasses import dataclass, field
from models.single_qubit_model import SingleQubitData
from typing import Dict, Any, Optional, List, Tuple, Union, Callable, Iterable, Literal
from physics.symbolic_multi_qubit_hamiltonian import build_symbolic_multi_qubit_hamiltonian, _multi_qubit_operator
from physics.numeric_multi_qubit_hamiltonian import _to_numeric_hamiltonian
from physics.multi_qubit_time_evolution import unitary_evolution, open_system_evolution
from models.multi_qubit_model import MultiQubitData

@dataclass
class MultiQubitSyntheticConfig:
    ntimes: int = 1001
    tlist:np.ndarray = None
    n_qubits:int = 2
    n_qubit_guard: int = 7
    tmax: float = 20.0
    observables_spec: List[Tuple[int,str]] = field(default_factory=lambda: [(0, 'Z')])
    local_fields: Dict[int, Tuple] = None
    psi0: np.ndarray = None
    params: Dict = None
    c_ops: List[np.ndarray] = None,    
    zz_couplings: List[Tuple[int,int,float]] = None
    custom_terms: List[dict] = None
    simplify_result: bool = True
    open_system:bool = True
    

def _default_initial_state(n_qubits: int, excited_qubits: Optional[List[int]] = None):
    """
    Return state-vector psi0 for computational basis with selected excited qubits.
    excited_qubits: list of indices to put in |1> (0-based); default None -> all ground |00..0>
    """
    N = 2**n_qubits
    idx = 0
    if excited_qubits:
        for q in excited_qubits:
            idx |= (1 << (n_qubits - 1 - q))   # bit ordering convention (adjust to your basis)
    psi0 = np.zeros((N,), dtype=complex)
    psi0[idx] = 1.0
    return psi0

def _ensure_obs_shape(obs_trace):

    # Extract expectation values if obs_trace is a QuTiP Result
    if isinstance(obs_trace, qt.solver.Result):
        obs_trace = obs_trace.expect

    arr = np.atleast_2d(np.array(obs_trace, dtype=float))  # force 2D

    # If single observable, ensure first axis = n_obs
    n_obs, n_times = arr.shape
    if n_times == 1 and n_obs > 1:
        # Possibly shape is (n_times, n_obs) -> transpose
        arr = arr.T
    elif n_obs == 1 and n_times == 1:
        # single value, expand to (1,1)
        arr = arr.reshape(1, 1)
    #Code Changed - Commented below code
    #elif n_obs < n_times:
        # assume shape is (n_times, n_obs) -> transpose
        #arr = arr.T

    return arr

def generate_multiqubit_measured_data(config: MultiQubitSyntheticConfig) -> MultiQubitData:
    """
    Build H for n_qubits, evolve, and return observables.

    - local_fields: {qubit_idx: (hx, hy, hz)} where each entry can be float, sympy expr, or callable t->float
    - zz_couplings: [(i, j, J_ij), ...]
    - observables_spec: [(qubit_idx, 'Z'), ...] default: [(0,'Z')]
    - psi0: initial state vector (dim=2**n_qubits). If None -> all ground |00..0>
    - c_ops: collapse operators (numeric matrices) for Lindblad
    - open_system: use open_system_evolution if True
    """
    n_qubits = config.n_qubits
    n_qubit_guard = config.n_qubit_guard

    # safety guard
    if n_qubits > n_qubit_guard:
        raise ValueError(f"n_qubits={n_qubits} > {n_qubit_guard}. Dense simulation expensive. "
                         "Use sparse/tensor methods for larger systems.")

    if config.tlist is None:
        tlist = np.linspace(0.0, config.tmax, config.ntimes)
    else:
        tlist = config.tlist

    if config.observables_spec is None:
        logging.warning("observables_spec is None: using: [(0, 'Z')]")
        observables_spec = [(0, 'Z')]
    else:
        observables_spec = config.observables_spec

    local_fields = config.local_fields
    zz_couplings = config.zz_couplings
    custom_terms = config.custom_terms
    simplify_result = config.simplify_result
    open_system = config.open_system

    # 1) Build symbolic H using your builder (returns sympy Matrix or numeric)
    H_sym = build_symbolic_multi_qubit_hamiltonian(
        n_qubits=n_qubits,
        local_fields=local_fields,
        zz_couplings=zz_couplings,
        custom_terms=custom_terms,
        simplify_result=simplify_result
    )

    # 2) numeric H
    #Need to check how params is used
    params = None
    H_num = _to_numeric_hamiltonian(H_sym, params or {})

    # 3) default psi0
    if config.psi0 is None:
        #psi0 = np.zeros((2**n_qubits,), dtype=complex)
        #psi0[0] = 1.0  # |00..0>
        psi0 = _default_initial_state(n_qubits) #excited_qubits=None
    else:
        psi0 = config.psi0

    # 4) build observables numeric
    obs_matrices = []
    for qb, op_name in observables_spec:
        Om = _multi_qubit_operator(n_qubits, [(qb, op_name)])  # expects sympy or numeric matrix
        Om_num = np.array(Om, dtype=complex)
        obs_matrices.append(Om_num)

    # 5) evolve and get expectation values
    if open_system:
        # expects: open_system_evolution(H_num, psi0, tlist, c_ops=c_ops, observables=obs_matrices)
        results = open_system_evolution(H_num, psi0, tlist, c_ops=config.c_ops, observables=obs_matrices)
    else:
        results = unitary_evolution(H_num, psi0, tlist, observables=obs_matrices)        
        #print("Shape of results: ", results)
    obs_trace = _ensure_obs_shape(results)  # (n_obs, n_times)
    # If params are not explicitly provided, create from local_fields & couplings
    if params is None:
        params = {
            "local_fields": local_fields,
            "zz_couplings": zz_couplings
        }

    print("Shape of obs_trace: ", obs_trace.shape)

    md = {
        'n_qubits': n_qubits,
        'local_fields': local_fields,
        'zz_couplings': zz_couplings,
        'params': params,
        'observables_spec': observables_spec,
        'open_system': open_system
    }
    # md = {
    #     "schema_version": 1,

    #     "system": {
    #         "n_qubits": n_qubits,
    #         "open_system": open_system,
    #     },

    #     "hamiltonian": {
    #         "local_fields": local_fields,
    #         "zz_couplings": zz_couplings,
    #     },

    #     "measurement": {
    #         "observables_spec": observables_spec,
    #     }
    # }

    return MultiQubitData(times=tlist, measurements=obs_trace, errors=None, metadata=md)


def save_multi_qubit_data_npz(multi_qubit_data, filepath):

    np.savez(
        filepath,
        times=multi_qubit_data.times,
        measurements=multi_qubit_data.measurements,
        errors = (
            multi_qubit_data.errors
            if multi_qubit_data.errors is not None
            else np.array([])
        ),
        metadata=multi_qubit_data.metadata
    )


def load_multi_qubit_data_npz(filepath):

    with np.load(filepath, allow_pickle=True) as data:

        times = data["times"]
        measurements = data["measurements"]

        errors = data["errors"]

        if errors.size == 0:
            errors = None

        metadata = (
            data["metadata"].item()
            if "metadata" in data.files
            else None
        )

    return MultiQubitData(
        times=times,
        measurements=measurements,
        errors=errors,
        metadata=metadata
    )