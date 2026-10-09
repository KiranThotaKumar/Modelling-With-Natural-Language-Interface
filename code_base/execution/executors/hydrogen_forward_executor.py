#execution/executors/hydrogen_forward_executor.py

from core.contracts.execution_result import ExecutionResult
from core.contracts.scientific_intent import ScientificIntent
from execution.domain_executor import DomainExecutor
from core.domains.hydrogen.hydrogen_physics import hydrogen_forward_spectrum
from execution.domain.defaults import DEFAULT_HYDROGEN_OBSERVATION

import numpy as np

class HydrogenForwardExecutor:

    def handle(self, intent: ScientificIntent) -> ExecutionResult:

        # ------------------------------
        # 1. Extract + Validate
        # ------------------------------

        transitions = intent.parameters.get("transitions")
        series = intent.parameters.get("series")

        if not transitions:
            return ExecutionResult(
                status="error",
                payload={},
                metadata={"reason": "No transitions specified"}
            )

        # ------------------------------
        # 2. Build Energy Grid (executor responsibility)
        # ------------------------------

        emin = intent.parameters.get("emin")
        emax = intent.parameters.get("emax")
        nbins = intent.parameters.get("nbins")

        energies = np.linspace(emin, emax, nbins)

        # ------------------------------
        # 3. Extract Observation Parameters
        # ------------------------------
        sigma_instr = intent.parameters.get("sigma_instr")
        if sigma_instr is None or sigma_instr <= 0:
            sigma_instr = DEFAULT_HYDROGEN_OBSERVATION["sigma_instr"]

        background = intent.parameters.get("background")
        if background is None or background <= 0:
            background = DEFAULT_HYDROGEN_OBSERVATION["background"]

        amplitudes = np.array(
            intent.parameters.get("amplitudes", [1.0] * len(transitions)),
            dtype=float
        )

        # ------------------------------
        # 4. Forward Physics
        # ------------------------------
        
        spectrum = hydrogen_forward_spectrum(
            energies,
            transitions,
            amplitudes,
            sigma_instr,
            background
        )

        series_name = (
            str(series).capitalize()
            if series is not None
            else "Hydrogen"
        )

        hydrogen_plot_spec = {
            "x": energies,
            "y": spectrum,
            "label": f"{series} Spectrum",
            "xlabel": "Energy",
            "ylabel": "Counts",
            "title": f"Hydrogen {series} Spectrum (Forward Model)",
        }
        from stats_visuals.plotting import plot_curves
        plot_curves([hydrogen_plot_spec])


        return ExecutionResult(
            status="success",
            payload={
                "energies": energies,
                "spectrum": spectrum
            },
            metadata={
                "domain": "hydrogen",
                "series": series
            }
        )