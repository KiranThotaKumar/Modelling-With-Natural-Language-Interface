#Modelling\nlp\transformer_architecture\training_constants.py

import numpy
from dataclasses import dataclass


@dataclass
class TrainingConstants:

    # --- 1. File-Based Dataset Loading ---
    file_name_synthetic_dataset = "synthetic_dataset_hyd_qubits.jsonl"

    max_sequence_length = 25  # Safe fixed maximum context size based on long samples


    # --- 1. Global Model Parameters & Instantiation ---
    d_model = 64
    num_heads = 4
    d_ff = 256
    num_encoder_layers = 2
    num_decoder_layers = 2


    # Hyperparameters
    learning_rate = 0.005
    num_epochs = 5
    batch_size = 8

    # Task Weight Balancing Matrix Blueprint (Applying Request 3)
    domain_loss_weight = 1.0   # De-escalated since it converges near zero rapidly
    action_loss_weight = 1.0
    slot_loss_weight = 3.0     # Scaled up to guide physical bounds regression

    # Initial hyperparameter boundaries
    base_lr = 0.002
    min_lr = 0.0001


    slot_schema = ["file_name", "series", "emin", "emax", "observable_type", "omega_r","detuning", "amplitude", "gamma", "offset", "tmax",  
                   "open_system_mode", "coupling_topology","coupling_strength", "initial_state_family"]


    slot_value_map = {"series": {"Balmer": 1, "Paschen": 2, "Pfund": 3, "Lyman": 4, "Brackett": 5},
                          "observable_type": {"population": 1, "pauli_z": 2, "spectrum": 3},
                          "open_system_mode": {"open": 1, "closed": 2},
                          "coupling_topology": {"chain": 1, "nearest_neighbor": 2},
                          "initial_state_family": {"bell": 1, "ground": 2, "superposition": 3},
                     }


train_constants = TrainingConstants()

