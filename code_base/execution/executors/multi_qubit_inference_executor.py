#execution\executors\multi_qubit_inference_executor.py

from core.contracts.scientific_intent import ScientificIntent
from core.contracts.execution_result import ExecutionResult
import numpy as np
from inference.runners import run_multiqubit_inference

class MultiQubitInferenceExecutor:

    def handle(self, intent: ScientificIntent) -> ExecutionResult:        
        return self._run_mcmc(intent)

    def _run_mcmc(self, intent: ScientificIntent) -> ExecutionResult:        
        print("Multi Qubit Inference Executor called")
        print(intent.parameters["file_name"])
        
        flat_samples, log_prob, metadata = run_multiqubit_inference(intent)

        return ExecutionResult(
        status="success",
        payload={
            "flat_samples": flat_samples,
            "log_prob": log_prob
        },
        metadata = metadata     
    )

