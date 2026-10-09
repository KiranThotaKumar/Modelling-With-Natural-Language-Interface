import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union, Callable, Iterable, Literal
import sympy as sp


def _to_numeric_hamiltonian(H_sym, param_subs: Dict = None):
    """
    Convert sympy matrix to numeric numpy.ndarray using parameter substitutions.
    If H_sym is already numeric, return as-is.
    """
    try:
        import sympy as sp
    except Exception:
        sp = None

    if sp and isinstance(H_sym, sp.Matrix):
        H_num = np.array(H_sym.subs(param_subs).evalf(), dtype=complex)
        # Expanded code below
         # subs and evalf to numeric
        #H_sub = H_sym.subs(param_subs)
        #H_eval = sp.N(H_sub)  # numeric
        #H_num = np.asarray(H_eval.tolist(), dtype=complex)
        return H_num
    else:
        # assume H_sym is already numeric (np.ndarray)
        return np.array(H_sym, dtype=complex)

