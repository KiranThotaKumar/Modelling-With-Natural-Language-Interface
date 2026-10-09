from physics.hydrogen_synthetic_data import generate_synthetic_hydrogen_data
from physics.single_qubit_synthetic_data import generate_singlequbit_measured_data, SingleQubitSyntheticConfig
from inference.pipelines.hydrogen_pipeline import hydrogen_inference_pipeline
from models.hydrogen_model import HydrogenData
from physics.hydrogen_synthetic_data import HydrogenSyntheticConfig
import os
import numpy as np
from models.single_qubit_model import SingleQubitModelConfig, SingleQubitData

from inference.pipelines.singlequbit_pipeline import single_qubit_inference_pipeline
from inference.pipelines.multiqubit_pipeline import multi_qubit_inference_pipeline

from physics.multi_qubit_synthetic_data import MultiQubitSyntheticConfig, generate_multiqubit_measured_data, save_multi_qubit_data_npz, load_multi_qubit_data_npz
from models.multi_qubit_model import MultiQubitData
from physics.single_qubit_synthetic_data import load_single_qubit_data_npz
from core.contracts.scientific_intent import ScientificIntent

def load_or_generate_hydrogen_data(args) -> HydrogenData:
    if os.path.exists(args.data_file):
        return load_hydrogen_data(args.data_file)
    else:
        config = HydrogenSyntheticConfig
        config.emin = args.emin
        config.emax = args.emax
        config.random_seed = args.random_seed
        config.nbins = args.nbins
        config.sigma_instr = args.sigma_instr
        config.background_rate = args.background_rate

        transitions = args.transitions
        amplitudes = args.amplitudes
        
        data = generate_synthetic_hydrogen_data(config, transitions, amplitudes)
            
        return data

def run_hydrogen_inference(args):
    """
    Entry point for hydrogen MCMC inference.
    """

    data = load_or_generate_hydrogen_data(args)    
    flat_samples, log_prob, metadata = hydrogen_inference_pipeline(vars(args), data)
    print("[DONE] Hydrogen inference finished.")
    return flat_samples, log_prob, metadata

def load_or_generate_single_qubit_data(file_name) -> SingleQubitData:
    if os.path.exists(file_name):
        return load_single_qubit_data_npz(file_name)
    else:
        raise ValueError("File Name needs to be specified")
        # config = SingleQubitSyntheticConfig

        # config.ntimes = args.ntimes
        # config.tmax = args.tmax
        # config.omega_r = args.omega_r
        # config.detuning = args.detuning
        # config.gamma = args.gamma
        # config.amp = args.amp
        # config.offset = args.offset
        # config.noise_std = args.noise_std
        # config.random_seed = args.random_seed

        
        data = generate_singlequbit_measured_data(config)
            
        return data

def run_singlequbit_inference(intent: ScientificIntent):
    """
    Entry point for single-qubit MCMC inference (legacy/full mode).
    """
    data_signal = load_or_generate_single_qubit_data(intent.parameters["file_name"])

    data = SingleQubitData(
        times=data_signal.times,        # from generator
        measurements=data_signal.measurements,      # from generator
        errors=data_signal.errors,      # from generator
        metadata=data_signal.metadata,      # from generator
        # include other fields as needed (noise, metadata, etc.)
    )
    #configSQB = SingleQubitModelConfig(200, 20, 0.02)

    single_qubit_model_config = SingleQubitModelConfig(intent.parameters["ntimes"], intent.parameters["tmax"], intent.parameters["noise_std"])
    

    #flat_samples, log_prob, metadata = single_qubit_inference_pipeline(vars(args), data)
    flat_samples, log_prob, metadata = single_qubit_inference_pipeline(single_qubit_model_config, data)
    print("[DONE] Single-qubit inference finished.")
    return flat_samples, log_prob, metadata

def load_or_generate_multi_qubit_data(data_file, args) -> MultiQubitData:
    
    if os.path.exists(data_file):
        #return load_synthetic_data(data_file)
        return load_multi_qubit_data_npz(data_file)
    else:        
        # # config = MultiQubitSyntheticConfig(args.n_qubits, args.ntimes, args.tmax, args.tlist, args.noise_std,
        # #     args.n_qubit_guard, args.observables_spec, args.local_fields, args.psi0, args.params, args.c_ops, 
        # #     args.zz_couplings, args.custom_terms, args.simplify_result, args.open_system, args.couplings,
        # #     args.fast, args.qubit_params_override, args.spectrum_mode, args.sigma_instr, args.background,
        # #     args.theta
        # # )                                           
        config = MultiQubitSyntheticConfig(args.ntimes, args.tlist, args.n_qubits, 
            args.n_qubit_guard, args.tmax,
            args.observables_spec, args.local_fields, args.psi0, args.params, args.c_ops, 
            args.zz_couplings, args.custom_terms,
            args.simplify_result, args.open_system
        )                             

        data = generate_multiqubit_measured_data(config)
        save_multi_qubit_data_npz(data, "synthetic_multi_qubit_data.npz")
        return data

def run_multiqubit_inference(intent: ScientificIntent = None, args=None):
    """
    Entry point for multi-qubit MCMC inference.
    """
    n_qubits = 2
    #data_signal = load_or_generate_multi_qubit_data(intent.parameters["file_name"])
    data_signal = load_or_generate_multi_qubit_data(intent.parameters["file_name"], args)
    gen_mode = "full"
    print(f"[DEBUG] Running Multi-qubit inference, n_qubits: {n_qubits}")   

    data = MultiQubitData(
        times=data_signal.times,        # from generator
        measurements=data_signal.measurements,      # from generator
        errors=data_signal.errors,      # from generator
        metadata=data_signal.metadata,      # from generator
        # include other fields as needed (noise, metadata, etc.)
    )

    #flat_samples, log_prob, metadata = multi_qubit_inference_pipeline(vars(args), data, n_qubits=n_qubits)
    flat_samples, log_prob, metadata = multi_qubit_inference_pipeline(args, data, n_qubits=n_qubits)
    print(f"[DONE] Multi-qubit inference finished.")
    return flat_samples, log_prob, metadata

def load_hydrogen_data(data_file):
    from pathlib import Path

    p = Path(data_file)

    if p.exists() and p.is_file():
        print(f"File found: {data_file}. Proceeding to load data.")
        # Example: Reading the content of a text file
        try:
            with open(p, 'r') as f:
                data = f.read()
                print("Data loaded successfully.")
                # Process your data here
        except IOError as e:
            print(f"Error reading file: {e}")
    else:
        print(f"File not found or is not a regular file: {data_file}")

    return data

def load_synthetic_data(data_file):
    from pathlib import Path

    p = Path(data_file)

    if p.exists() and p.is_file():
        print(f"File found: {data_file}. Proceeding to load data.")
        # Example: Reading the content of a text file
        try:
            with open(p, 'r') as f:
                data = f.read()
                print("Data loaded successfully.")
                # Process your data here
        except IOError as e:
            print(f"Error reading file: {e}")
    else:
        print(f"File not found or is not a regular file: {data_file}")

    return data