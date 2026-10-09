#execution\executors\multi_qubit_forward_executor.py

from core.contracts.execution_result import ExecutionResult
from core.contracts.scientific_intent import ScientificIntent
from physics.multi_qubit import multi_qubit_model_physics
from physics.multi_qubit_builder.multi_qubit_initial_params_builder import build_initial_state, build_topology, build_observable
from default_inits.default_theta_params import build_default_theta_full
from stats_visuals.plotting import plot_curves
import numpy as np

class MultiQubitForwardExecutor:

    def handle(self, intent: ScientificIntent) -> ExecutionResult:
        
        print("Multi Qubit forward executor called")
        theta = build_default_theta_full(which = "multi_qubit", nlines= 3, nqubits=2)

        theta[14] = 0.5
        spectrum_mode = False

        config_params = {
            "theta": theta,
            "qubit_params_override": None,

            "n_qubits": 2,

            "sigma_instr": intent.parameters["sigma_instr"],
            "background": intent.parameters["background"],

            "spectrum_mode": spectrum_mode,#spectrum_mode,

            "couplings": None,

            "fast": True,
            "debug_compare": False,

            "h_range": 1.0,
            "amp_range": 1.0,
        }
       
        
        times = np.linspace(0, 100, 5000)

        if (spectrum_mode):
            freq_grid, spectrum = multi_qubit_model_physics(
                times,
                config_params
            )

            # plt.plot(freq_grid, spectrum)
            # plt.xlabel("transition frequency")
            # plt.ylabel("intensity")
            # plt.show()
            
            multi_qubit_frequency_spectrum_spec = {
                "x": freq_grid,
                "y": spectrum,
                #"label": "Synthetic Hydrogen",
                "style": ".",
                "xlabel": "Transition Energy",
                "ylabel": "Relative Spectral Amplitude",
                "title": " Multi-Qubit Energy Transition Spectrum",
                 "multiplicator_x": 1.0,
                 "multiplicator_y": 0.25,
            }

            plot_curves([multi_qubit_frequency_spectrum_spec],show = True)

            return ExecutionResult(
                status="success",
                payload={
                    "x": freq_grid,
                    "y": spectrum

                    },
                metadata={
                    "mode": "spectrum"
                }
            )

        else:

            y = multi_qubit_model_physics(
               times,
               config_params
            )

            plot_specs = []

            for q in range(y.shape[1]):
                plot_specs.append(
                    {
                        "x": times,
                        "y": y[:, q],
                        "label": f"Qubit {q}",
                    }
                )

            plot_specs[0]["title"] = "Multi-Qubit Evolution"
            plot_specs[0]["xlabel"] = "Time"
            plot_specs[0]["ylabel"] = "Expectation Value"

            plot_curves(plot_specs)            

            return ExecutionResult(
                status="success",
                payload={
                    "x": times,
                    "y": y

                    },
                metadata={
                    "mode": "evolution"
                }
            )

    
        #params = intent.parameters
        # psi0 = build_initial_state(
        #     params["initial_state_family"],
        #     params["n_qubits"]
        # )

        # couplings = build_topology(
        #     params["coupling_topology"],
        #     params["n_qubits"],
        #     params["coupling_strength"]
        # )

        # observables_spec = build_observable(
        #     params["observable_type"],
        #     params["n_qubits"]
        # )
        # if initial_state_family == "bell":
        #     n_qubits = 2
        # local_fields = build_local_fields(
        #     omega_r,
        #     detuning,
        #     n_qubits
        # )
        # def build_theta_from_semantic_slots(
        #     slots: dict,
        #     rng=None
        # ):

        # Step 1

        # Call:

        # build_default_theta_full()

        # to get a stable baseline.

        # Step 2

        # Overwrite selected semantic dimensions.

        # Example:
        # theta[base + 0] ← omega_r

        # theta[coupling_idx] ← coupling_strength

        # theta[noise_idx] ← noise_std
        