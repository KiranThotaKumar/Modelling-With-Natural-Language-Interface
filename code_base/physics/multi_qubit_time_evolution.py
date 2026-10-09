import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union, Callable, Iterable, Literal
import sympy as sp
import qutip as qt

# -------------------------------------------------------------
#                 Time Evolution Methods
# -------------------------------------------------------------
def unitary_evolution(H: np.ndarray, psi0: np.ndarray, tlist: np.ndarray, observables: Optional[List[np.ndarray]] = None):
    """
    Perform unitary (closed system) evolution using sesolve.

    Args:
        H: Hamiltonian (np.ndarray or qutip.Qobj)
        psi0: Initial state vector (np.ndarray)
        tlist: Array of times
        observables: Optional list of operators to compute expectation values

    Returns:
        qutip.Result: Contains states and expectation values
    """
    H_qobj = qt.Qobj(H) if isinstance(H, np.ndarray) else H
    psi0_qobj = qt.Qobj(psi0) if isinstance(psi0, np.ndarray) else psi0
    obs_qobj = [qt.Qobj(O) for O in observables] if observables else []
    return qt.sesolve(H_qobj, psi0_qobj, tlist, e_ops=obs_qobj)


def open_system_evolution(H: np.ndarray, psi0: np.ndarray, tlist: np.ndarray,
                          c_ops: Optional[List[np.ndarray]] = None, observables: Optional[List[np.ndarray]] = None):
    """
    Perform open system evolution using the Lindblad master equation (mesolve).

    Args:
        H: Hamiltonian (np.ndarray or qutip.Qobj)
        psi0: Initial state vector or density matrix (np.ndarray)
        tlist: Array of times
        c_ops: List of collapse operators (np.ndarray or qutip.Qobj)
        observables: Optional list of operators to compute expectation values

    Returns:
        qutip.Result: Contains states and expectation values
    """
    H_qobj = qt.Qobj(H) if isinstance(H, np.ndarray) else H
    rho0_qobj = qt.Qobj(psi0) if isinstance(psi0, np.ndarray) else psi0
    c_ops_qobj = [qt.Qobj(c) for c in c_ops] if c_ops else []
    obs_qobj = [qt.Qobj(O) for O in observables] if observables else []
    return qt.mesolve(H_qobj, rho0_qobj, tlist, c_ops=c_ops_qobj, e_ops=obs_qobj)

