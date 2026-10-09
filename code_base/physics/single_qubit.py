import numpy as np

def single_qubit_model_physics(times, params)-> np.ndarray:
	    
    # safe guard: times may be None
    if times is None:
        raise ValueError("times must be provided for 'auto'/'legacy' mode.")
      
    omega_r = params["omega_r"]
    detuning = params["detuning"]
    offset = params["offset"] 
    amp = params["amp"] 
    gamma = params["gamma"]

    omega_eff = np.sqrt(omega_r**2 + detuning**2)
    model_spectrum =  offset + amp * np.exp(-gamma * times) * np.cos(omega_eff * times)

    return model_spectrum



