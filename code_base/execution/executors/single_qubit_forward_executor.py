#execution\executors\single_qubit_forward_executor.py


from core.contracts.execution_result import ExecutionResult
from core.contracts.scientific_intent import ScientificIntent
from physics.single_qubit import single_qubit_model_physics

import numpy as np

class SingleQubitForwardExecutor:

    def handle(self, intent: ScientificIntent) -> ExecutionResult:        
        params = {}

        params["omega_r"] = intent.parameters["omega_r"]
        params["detuning"] = intent.parameters["detuning"]
        params["offset"]  = intent.parameters["offset"]
        params["amp"] = intent.parameters["amplitude"] 
        params["gamma"] = intent.parameters["gamma"]
        
        tmax = intent.parameters["tmax"]
        times = np.linspace(0.0, tmax, 1001)
        single_qubit_signal = single_qubit_model_physics(times, params)
        
        # import matplotlib.pyplot as plt
        # plt.plot(times, single_qubit_signal)
        # plt.show()
        single_qubit_plot_spec = {
            "x": times,
            "y": single_qubit_signal,
            "label": "",
            "style": ".",
            "xlabel": "Time",
            "ylabel": "Measured Signal",
            "title": "Single Qubit Time Evolution",
            "multiplicator_x": 1.0,
             "multiplicator_y": 0.25,
             "xlim_low": 0,
             "xlim_high": tmax+1,
        }
        from stats_visuals.plotting import plot_curves
        plot_curves([single_qubit_plot_spec], show=True)

        return ExecutionResult(
            status="success",
            payload={
                "evolution": single_qubit_signal
            },
            metadata={
                "domain": "single_qubit",
                "action": "forward",
                "model": "damped_rabi",
            }
        )
        
       
