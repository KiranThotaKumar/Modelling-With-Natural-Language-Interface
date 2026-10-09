#physics\multi_qubit_builder\multi_qubit_initial_params_builder.py


import numpy as np


# ============================================================
# Initial State Builder
# ============================================================

def build_initial_state(
    initial_state_family: str,
    n_qubits: int
):
    """
    Build initial quantum state vector psi0.

    Returns
    -------
    np.ndarray
        Complex state vector of size 2^n_qubits
    """

    dim = 2 ** n_qubits

    # --------------------------------------------------------
    # Ground state |000...0>
    # --------------------------------------------------------
    if initial_state_family == "ground":

        psi0 = np.zeros(dim, dtype=np.complex128)
        psi0[0] = 1.0
        return psi0

    # --------------------------------------------------------
    # Excited state |111...1>
    # --------------------------------------------------------
    elif initial_state_family == "excited":

        psi0 = np.zeros(dim, dtype=np.complex128)
        psi0[-1] = 1.0
        return psi0

    # --------------------------------------------------------
    # Equal superposition
    # --------------------------------------------------------
    elif initial_state_family == "superposition":

        psi0 = np.ones(dim, dtype=np.complex128)
        psi0 /= np.sqrt(dim)
        return psi0

    # --------------------------------------------------------
    # Bell state (2-qubit only)
    # (|00> + |11>) / sqrt(2)
    # --------------------------------------------------------
    elif initial_state_family == "bell":

        if n_qubits != 2:
            raise ValueError(
                "Bell state currently supported only for 2 qubits."
            )

        psi0 = np.zeros(4, dtype=np.complex128)

        psi0[0] = 1 / np.sqrt(2)
        psi0[3] = 1 / np.sqrt(2)

        return psi0

    # --------------------------------------------------------
    # GHZ state
    # (|000...0> + |111...1>) / sqrt(2)
    # --------------------------------------------------------
    elif initial_state_family == "ghz":

        psi0 = np.zeros(dim, dtype=np.complex128)

        psi0[0] = 1 / np.sqrt(2)
        psi0[-1] = 1 / np.sqrt(2)

        return psi0

    # --------------------------------------------------------
    # Default fallback
    # --------------------------------------------------------
    else:

        psi0 = np.zeros(dim, dtype=np.complex128)
        psi0[0] = 1.0

        return psi0


# ============================================================
# Coupling Topology Builder
# ============================================================

def build_topology(
    coupling_topology: str,
    n_qubits: int,
    coupling_strength: float
):
    """
    Build ZZ coupling list.

    Returns
    -------
    List[Tuple[int, int, float]]

    Example:
    [
        (0, 1, J),
        (1, 2, J)
    ]
    """

    zz_couplings = []

    # --------------------------------------------------------
    # No couplings
    # --------------------------------------------------------
    if coupling_topology == "none":

        return zz_couplings

    # --------------------------------------------------------
    # Nearest-neighbor chain
    # --------------------------------------------------------
    elif coupling_topology in ["nearest_neighbor", "chain"]:

        for i in range(n_qubits - 1):

            zz_couplings.append(
                (i, i + 1, coupling_strength)
            )

    # --------------------------------------------------------
    # Ring topology
    # --------------------------------------------------------
    elif coupling_topology == "ring":

        for i in range(n_qubits - 1):

            zz_couplings.append(
                (i, i + 1, coupling_strength)
            )

        zz_couplings.append(
            (n_qubits - 1, 0, coupling_strength)
        )

    # --------------------------------------------------------
    # All-to-all coupling
    # --------------------------------------------------------
    elif coupling_topology == "all_to_all":

        for i in range(n_qubits):

            for j in range(i + 1, n_qubits):

                zz_couplings.append(
                    (i, j, coupling_strength)
                )

    # --------------------------------------------------------
    # Default fallback
    # --------------------------------------------------------
    else:

        return []

    return zz_couplings


# ============================================================
# Observable Builder
# ============================================================

def build_observable(
    observable_type: str,
    n_qubits: int
):
    """
    Build observables specification.

    Returns
    -------
    List[Tuple[int, str]]

    Example:
    [
        (0, "z"),
        (1, "z")
    ]
    """

    observables_spec = []

    # --------------------------------------------------------
    # Pauli-Z observables
    # --------------------------------------------------------
    if observable_type == "pauli_z":

        for i in range(n_qubits):

            observables_spec.append((i, "z"))

    # --------------------------------------------------------
    # Pauli-X observables
    # --------------------------------------------------------
    elif observable_type == "pauli_x":

        for i in range(n_qubits):

            observables_spec.append((i, "x"))

    # --------------------------------------------------------
    # Population measurement
    # --------------------------------------------------------
    elif observable_type == "population":

        for i in range(n_qubits):

            observables_spec.append((i, "population"))

    # --------------------------------------------------------
    # Coherence
    # --------------------------------------------------------
    elif observable_type == "coherence":

        for i in range(n_qubits):

            observables_spec.append((i, "coherence"))

    # --------------------------------------------------------
    # Default fallback
    # --------------------------------------------------------
    else:

        for i in range(n_qubits):

            observables_spec.append((i, "z"))

    return observables_spec