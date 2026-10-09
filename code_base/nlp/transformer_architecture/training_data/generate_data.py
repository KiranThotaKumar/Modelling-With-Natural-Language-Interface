
#MergingWithNLP\Modelling\nlp\transformer_architecture\training_data\generate_data.py

import json
import random
import uuid
from pathlib import Path
from physics.multi_qubit_builder.multi_qubit_initial_params_builder import build_initial_state, build_topology, build_observable

# -------------------------------
# Utility Functions
# -------------------------------

def uid():
    return str(uuid.uuid4())


def sample_uniform(a, b, precision=1):
    
    # Dropping precision to 1 or 2 decimals restricts the vocabulary size safely
    return round(random.uniform(a, b), precision)


def sample_choice(options):
    return random.choice(options)


CONVERSATIONAL_PREFIXES = [
    "", 
    "Please ", 
    "Kindly ", 
    "Can you ", 
    "Can you please ", 
    "Show me ", 
    "Please show me ",
    "Could you get me "
]

def add_conversational_noise(text):
    """Randomly injects prefixes while keeping the text lowercased cleanly."""
    prefix = random.choice(CONVERSATIONAL_PREFIXES)
    # Ensure proper spacing and lowercase compatibility
    full_text = f"{prefix}{text}"
    # Remove accidental double spaces if empty string was picked
    return " ".join(full_text.split())


TEMPLATES_INFERENCE = [
    "Infer {domain_name} parameters from file {file}",
    "Estimate parameters of {domain_name} system using {file}",
    "Perform parameter inference for {domain_name} using data file {file}",
    "Recover system parameters from {file} for {domain_name}",
]

# Only values explicitly mentioned in a prompt belong in a generated record's
# `slots` field.  Runtime defaults live separately so they are not treated as
# text-prediction targets during training.  Add a new domain/action by adding
# its defaults here and a corresponding generator below.
RUNTIME_DEFAULTS = {
    "hydrogen": {
        "forward": {
            "spectrum_mode": "absorption",
            "sigma_instr": 0.04,
            "background": 0.0,
            "scale": 1.0,
        },
    },
    "single_qubit": {
        "forward": {
            "omega_r": 1.0,
            "detuning": 0.0,
            "amplitude": 1.0,
            "gamma": 0.05,
            "offset": 0.0,
            "tmax": 10.0,
            "ntimes": 500,
        },
    },
    "multi_qubit": {
        "forward": {
            "coupling_strength": 0.5,
            "omega_r": 2.0,
            "detuning": 0.0,
            "gamma": 0.0,
            "noise_std": 0.0,
            "sigma_instr": 0.04,
            "background": 0.0,
            "tmax": 10.0,
            "spectrum_mode": "none",
            "evolution_mode": "time",
            "open_system_mode": "closed",
            "coupling_topology": "nearest_neighbor",
            "initial_state_family": "ground",
            "observable_type": "population",
            "n_qubits": 2,
            "ntimes": 500,
        },
    },
}

HYDROGEN_SERIES_ENERGY_RANGES = {
    "Lyman": (10.0, 14.0),
    "Balmer": (1.5, 3.5),
    "Paschen": (0.5, 1.5),
    "Brackett": (0.1, 0.6),
    "Pfund": (0.05, 0.3),
}


def resolve_runtime_slots(domain, action, extracted_slots):
    """Combine model-extracted slots with deterministic simulator defaults."""
    resolved = dict(RUNTIME_DEFAULTS.get(domain, {}).get(action, {}))
    resolved.update(extracted_slots)

    # Hydrogen's default energy range is determined by the explicitly named
    # spectral series, not randomly sampled for an underspecified prompt.
    if domain == "hydrogen" and action == "forward" and "series" in resolved:
        default_emin, default_emax = HYDROGEN_SERIES_ENERGY_RANGES[resolved["series"]]
        resolved.setdefault("emin", default_emin)
        resolved.setdefault("emax", default_emax)

    # An explicitly requested open system needs a deterministic non-zero
    # damping default unless the user also supplied gamma.
    if domain == "multi_qubit" and action == "forward":
        if resolved["open_system_mode"] == "open" and "gamma" not in extracted_slots:
            resolved["gamma"] = 0.05

    return resolved


def make_record(input_text, domain, action, explicit_slots):
    """Create a training record containing only slots inferable from text."""
    return {
        "input_text": input_text,
        "domain": domain,
        "action": action,
        "slots": explicit_slots,
    }

# -------------------------------
# Hydrogen Forward
# -------------------------------

def generate_hydrogen_forward():
    TEMPLATES_WITH_RANGE = [
        "Generate {series} hydrogen spectrum from {emin} to {emax} eV",
        "Compute {series} spectrum between {emin} and {emax} eV",
        "Simulate hydrogen {series} lines in range {emin}-{emax} eV",
        "Plot {series} transitions for hydrogen from {emin} to {emax} eV",
    ]

    TEMPLATES_NO_RANGE = [
        "Generate {series} spectrum",
        "Compute hydrogen {series} series",
        "Show {series} lines of hydrogen",
        "Simulate {series} transitions",
        "Plot hydrogen {series}",
    ]
    series = sample_choice(["Lyman", "Balmer", "Paschen", "Brackett", "Pfund"])

    emin_series, emax_series = HYDROGEN_SERIES_ENERGY_RANGES[series]

    use_range = random.choice([True, False])

    if use_range:
        emin = sample_uniform(
        max(1e-3, emin_series * 0.9),
        emin_series
        )
        emax = sample_uniform(
            emax_series,
            emax_series * 1.1
        )

        template = sample_choice(TEMPLATES_WITH_RANGE)
        input_text = template.format(series=series, emin=emin, emax=emax)
        explicit_slots = {"series": series, "emin": emin, "emax": emax}
    else:
        template = sample_choice(TEMPLATES_NO_RANGE)
        input_text = template.format(series=series)
        explicit_slots = {"series": series}

    input_text = add_conversational_noise(input_text)
    return make_record(input_text, "hydrogen", "forward", explicit_slots)


# -------------------------------
# Hydrogen Inference
# -------------------------------

def generate_hydrogen_inference(index):
    file_name = f"synthetic_hydrogen_{index}.npz"
    
    template = random.choice(TEMPLATES_INFERENCE)
    input_text = template.format(
        domain_name="hydrogen",
        file=file_name
    )

    input_text = add_conversational_noise(input_text)
    return make_record(input_text, "hydrogen", "infer_parameters", {"file_name": file_name})

# -------------------------------
# Single Qubit
# -------------------------------
def generate_single_qubit():

    n_qubits = 1
    omega_r = sample_uniform(0.5, 4.0)
    detuning = sample_uniform(-1.0, 1.0)
    amplitude = sample_uniform(0.7, 1.3)
    gamma = sample_uniform(0.01, 0.15)
    offset = sample_uniform(-0.2, 0.2)
    tmax = sample_uniform(5.0, 25.0)
    ntimes = sample_choice([400, 500, 600, 700])


    templates = [
        (
            f"Simulate single qubit with omega {omega_r} detuning {detuning} amplitude {amplitude} gamma {gamma} offset {offset} up to time {tmax}",
            {"omega_r": omega_r, "detuning": detuning, "amplitude": amplitude, "gamma": gamma, "offset": offset, "tmax": tmax},
        ),
        (
            f"Run single qubit evolution with omega {omega_r}, detuning {detuning} amplitude {amplitude} gamma {gamma} offset {offset} until time {tmax}",
            {"omega_r": omega_r, "detuning": detuning, "amplitude": amplitude, "gamma": gamma, "offset": offset, "tmax": tmax},
        ),
        ("Simulate single qubit dynamics", {}),
        ("Time evolution of a single qubit", {}),
        (f"Simulate 1 qubit up to time {tmax}", {"tmax": tmax}),
        ("Generate evolution for one qubit", {}),
    ]

    input_text, explicit_slots = sample_choice(templates)
    input_text = add_conversational_noise(input_text)
    return make_record(input_text, "single_qubit", "forward", explicit_slots)

def generate_single_qubit_inference(index):

    file_name = f"synthetic_single_qubit_{index}.npz"
    
    template = random.choice(TEMPLATES_INFERENCE)
    input_text = template.format(
        domain_name="single qubit",
        file=file_name
    )
    
    input_text = add_conversational_noise(input_text)
    return make_record(input_text, "single_qubit", "infer_parameters", {"file_name": file_name})

# -------------------------------
# Multi Qubit (n=2)
# -------------------------------


def generate_multi_qubit_forward():

    # ========================================================
    # Fixed system size for V1
    # ========================================================

    n_qubits = 2

    # ========================================================
    # Continuous parameters
    # ========================================================

    coupling_strength = sample_uniform(0.05, 1.0)

    omega_r = sample_uniform(0.5, 5.0)

    detuning = sample_uniform(-1.0, 1.0)

    tmax = sample_uniform(5.0, 25.0)

    sigma_instr = sample_uniform(0.01, 0.08)

    background = sample_uniform(0.0, 0.1)

    noise_std = sample_uniform(0.0, 0.05)

    # ========================================================
    # Categorical parameters
    # ========================================================

    evolution_mode = "time"

    open_system_mode = sample_choice([
        "open",
        "closed"
    ])

    coupling_topology = sample_choice([
        "nearest_neighbor",
        "chain"
    ])

    initial_state_family = sample_choice([
        "ground",
        "superposition",
        "bell"
    ])

    observable_type = sample_choice([
        "population",
        "pauli_z"
    ])

    # ========================================================
    # Physics consistency
    # ========================================================

    if open_system_mode == "open":

        gamma = sample_uniform(0.01, 0.2)

    else:

        gamma = sample_uniform(0.0, 0.005)

    # ========================================================
    # Query templates
    # ========================================================

    templates_full = [
        (
            f"Simulate {initial_state_family} evolution "
            f"in an {open_system_mode} two qubit system "
            f"with {coupling_topology} coupling strength {coupling_strength} "
            f"up to time {tmax}",
            {
                "initial_state_family": initial_state_family,
                "open_system_mode": open_system_mode,
                "coupling_topology": coupling_topology,
                "coupling_strength": coupling_strength,
                "tmax": tmax,
            },
        ),
        (
            f"Run two qubit {observable_type} dynamics "
            f"with Rabi frequency {omega_r} and detuning {detuning}",
            {"observable_type": observable_type, "omega_r": omega_r, "detuning": detuning},
        ),
        (
            f"Show {initial_state_family} state evolution "
            f"for coupled qubits with gamma {gamma}",
            {"initial_state_family": initial_state_family, "gamma": gamma},
        ),
    ]

    templates_partial = [
        (
            f"{open_system_mode} two qubit evolution with {coupling_topology} coupling",
            {"open_system_mode": open_system_mode, "coupling_topology": coupling_topology},
        ),
        ("Open system two qubit dynamics", {"open_system_mode": "open"}),
        ("Bell state evolution of two qubits", {"initial_state_family": "bell"}),
        ("Two qubit population dynamics", {"observable_type": "population"}),
    ]

    templates_minimal = [
        ("Simulate two qubit dynamics", {}),
        ("Time evolution of coupled qubits", {}),
        ("Quantum evolution of two qubits", {}),
    ]

    # ========================================================
    # Query sampling
    # ========================================================

    r = random.random()
    
    # --------------------------------------------------------
    # FULL semantic queries
    # --------------------------------------------------------

    if r < 0.50:
        input_text, explicit_slots = sample_choice(templates_full)

    # --------------------------------------------------------
    # PARTIAL semantic queries
    # --------------------------------------------------------

    elif r < 0.85:
        input_text, explicit_slots = sample_choice(templates_partial)

    # --------------------------------------------------------
    # MINIMAL queries
    # Use canonical/default semantic settings
    # to reduce ambiguity
    # --------------------------------------------------------

    else:

        
        input_text, explicit_slots = sample_choice(templates_minimal)


    # Validate the deterministic runtime configuration without putting its
    # defaulted values into the model-training target.
    runtime_slots = resolve_runtime_slots("multi_qubit", "forward", explicit_slots)

    try:

        psi0 = build_initial_state(
            runtime_slots["initial_state_family"],
            runtime_slots["n_qubits"]
        )

        zz_couplings = build_topology(
            runtime_slots["coupling_topology"],
            runtime_slots["n_qubits"],
            runtime_slots["coupling_strength"]
        )

        observables_spec = build_observable(
            runtime_slots["observable_type"],
            runtime_slots["n_qubits"]
        )

    except Exception:

        return generate_multi_qubit_forward()


    # Example update within generate_multi_qubit_forward:
    input_text = add_conversational_noise(input_text)

    return make_record(input_text, "multi_qubit", "forward", explicit_slots)


def generate_multi_qubit_inference(index):

    file_name = f"synthetic_multi_qubit_{index}.npz"
    
    template = random.choice(TEMPLATES_INFERENCE)
    input_text = template.format(
        domain_name="two qubit",
        file=file_name
    )
        
    input_text = add_conversational_noise(input_text)
    return make_record(input_text, "multi_qubit", "infer_parameters", {"file_name": file_name})


# -------------------------------
# Master Generator
# -------------------------------

def generate_dataset(n_per_domain=200, output_file="synthetic_dataset_hyd_qubits.jsonl"):
    
    dataset = []
    
    for i in range(n_per_domain):
        dataset.append(generate_hydrogen_forward())
        dataset.append(generate_hydrogen_inference(i))
        dataset.append(generate_single_qubit())
        dataset.append(generate_single_qubit_inference(i))
        dataset.append(generate_multi_qubit_forward())        
        dataset.append(generate_multi_qubit_inference(i))

    random.shuffle(dataset)

    with open(output_file, "w") as f:
        for record in dataset:
            f.write(json.dumps(record) + "\n")
    
    print(f"Dataset written to {output_file}")
    print(f"Total records: {len(dataset)}")


if __name__ == "__main__":
    generate_dataset(n_per_domain=200)   # → 1200 total
