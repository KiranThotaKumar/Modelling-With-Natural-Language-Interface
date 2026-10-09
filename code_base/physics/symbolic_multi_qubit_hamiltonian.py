import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union, Callable, Iterable, Literal
import sympy as sp

# Basic Pauli matrices (SymPy)
PAULI = {
    'I': sp.eye(2),
    'X': sp.Matrix([[0, 1], [1, 0]]),
    'Y': sp.Matrix([[0, -sp.I], [sp.I, 0]]),
    'Z': sp.Matrix([[1, 0], [0, -1]])
}

def _single_qubit_operator(n_qubits: int, target: int, op_char: str) -> sp.Matrix:
    """
    Return the full 2^N x 2^N operator corresponding to applying `op_char`
    (one of 'I','X','Y','Z') on qubit `target` (0-indexed), identity elsewhere.
    """
    if op_char not in PAULI:
        raise ValueError(f"Unknown pauli op '{op_char}'")
    ops = []
    for q in range(n_qubits):
        ops.append(PAULI[op_char] if q == target else PAULI['I'])
    # Kronecker (tensor) product left-to-right
    result = ops[0]
    for mat in ops[1:]:
        result = sp.kronecker_product(result, mat)
    return sp.simplify(result)


def _multi_qubit_operator(n_qubits: int, ops_spec: List[Tuple[int, str]]) -> sp.Matrix:
    """
    ops_spec: list of (qubit_index, 'X'/'Y'/'Z')
    Returns the tensor product operator for applying those Pauli's on respective qubits.
    Example: [(0,'Z'), (1,'Z')] => Z0 ⊗ Z1 ⊗ I ⊗ ...
    """
    # Start with identity on all qubits, then multiply each single-qubit op
    # Efficient composition by building tensor list then kronecker
    ops = []
    # Build operator list for each position
    for q in range(n_qubits):
        # find op for this qubit if provided
        op_here = next((op for (idx, op) in ops_spec if idx == q), 'I')
        if op_here not in PAULI:
            raise ValueError(f"Unknown pauli op '{op_here}'")
        ops.append(PAULI[op_here])
    result = ops[0]
    for mat in ops[1:]:
        result = sp.kronecker_product(result, mat)
    return sp.simplify(result)


def build_symbolic_multi_qubit_hamiltonian(
    n_qubits: int,
    *,
    local_fields: Optional[Dict[int, Tuple[Union[sp.Expr, float], Union[sp.Expr, float], Union[sp.Expr, float]]]] = None,
    zz_couplings: Optional[List[Tuple[int, int, Union[sp.Expr, float]]]] = None,
    custom_terms: Optional[List[Dict]] = None,
    simplify_result: bool = True) -> sp.Matrix:
    """
    Build a symbolic multi-qubit Hamiltonian as a SymPy Matrix.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (N).
    local_fields : dict (optional)
        Mapping qubit_index -> (hx, hy, hz).
        Each component can be a sympy.Symbol, sympy.Expr, or numeric.
        If a component is zero or None, it is ignored.
        Example: {0: (h0x, 0, h0z), 1: (0, 0, h1z)}
    zz_couplings : list of tuples (optional)
        Each tuple (i, j, J_ij) creates a coupling term J_ij * Z_i * Z_j.
        Indices i,j are 0-based.
    custom_terms : list of dicts (optional)
        Each dict must provide:
            - 'ops': [(index, 'X'/'Y'/'Z'), ...]
            - 'coeff': sympy.Expr or numeric
        Example: {'ops': [(0,'X'), (1,'Y')], 'coeff': g_xy}
    simplify_result : bool
        Whether to simplify the final matrix (default True).

    Returns
    -------
    H : sympy.Matrix
        The 2^N x 2^N symbolic Hamiltonian matrix.
    """

     # Convert linear → angular
    TWOPI = 2 * sp.pi

    if n_qubits < 1:
        raise ValueError("n_qubits must be >= 1")

    # Start with zero matrix of size 2^N
    dim = 2 ** n_qubits
    H = sp.zeros(dim)

    print(f"Type of local_fields: {type(local_fields)}")

    # Local field terms: hx * X_i + hy * Y_i + hz * Z_i
    if local_fields:
        for qubit_idx, (hx, hy, hz) in local_fields.items():
            if hx is not None and hx != 0:
                H += sp.sympify(hx) * _single_qubit_operator(n_qubits, qubit_idx, 'X')
            if hy is not None and hy != 0:
                H += sp.sympify(hy) * _single_qubit_operator(n_qubits, qubit_idx, 'Y')
            if hz is not None and hz != 0:
                H += sp.sympify(hz) * _single_qubit_operator(n_qubits, qubit_idx, 'Z')

    # ZZ couplings J_ij * Z_i * Z_j
    if zz_couplings:
        for (i, j, J) in zz_couplings:
            H += sp.sympify(J) * _multi_qubit_operator(n_qubits, [(i, 'Z'), (j, 'Z')])

    # Custom terms: linear combination of tensor products
    if custom_terms:
        for term in custom_terms:
            ops = term.get('ops')
            coeff = sp.sympify(term.get('coeff', 1))
            if not ops or not isinstance(ops, list):
                raise ValueError("Each custom term requires 'ops' as list of (index, 'X'|'Y'|'Z')")
            H += coeff * _multi_qubit_operator(n_qubits, ops)

    return sp.simplify(H) if simplify_result else H