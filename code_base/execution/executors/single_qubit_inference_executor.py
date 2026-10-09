#execution\executors\single_qubit_inference_executor.py


from core.contracts.scientific_intent import ScientificIntent
from core.contracts.execution_result import ExecutionResult
import numpy as np
from inference.runners import run_singlequbit_inference

class SingleQubitInferenceExecutor:

    def handle(self, intent: ScientificIntent) -> ExecutionResult:        
        return self._run_mcmc(intent)

    def _run_mcmc(self, intent: ScientificIntent) -> ExecutionResult:        
              
        flat_samples, log_prob, metadata = run_singlequbit_inference(intent)
        
        return ExecutionResult(
            status="success",
            payload={
                "flat_samples": flat_samples,
                "log_prob": log_prob
            },
            metadata = metadata     
        )

