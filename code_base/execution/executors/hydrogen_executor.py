#execution.executors.hydrogen_executor.py

from execution.domain_executor import DomainExecutor
from core.contracts.scientific_intent import ScientificIntent
from core.contracts.execution_result import ExecutionResult
from core.domains.hydrogen.hydrogen_physics import hydrogen_forward_spectrum
from execution.domain.defaults import DEFAULT_HYDROGEN_OBSERVATION

from models.hydrogen_model import (
    HydrogenModel,
    HydrogenModelConfig,
    HydrogenData
)
import numpy as np


from execution.executors.hydrogen_forward_executor import HydrogenForwardExecutor
from execution.executors.hydrogen_inference_executor import HydrogenInferenceExecutor
from core.contracts.execution_result import ExecutionResult
from core.contracts.scientific_intent import ScientificIntent


class HydrogenDomainExecutor(DomainExecutor):

    def __init__(self):
        self.forward_executor = HydrogenForwardExecutor()
        self.inference_executor = HydrogenInferenceExecutor()

    def handle(self, intent: ScientificIntent) -> ExecutionResult:

        if intent.action == "forward":
            return self.forward_executor.handle(intent)

        elif intent.action == "infer_parameters":
            return self.inference_executor.handle(intent)

        return ExecutionResult(
            status="error",
            payload={},
            metadata={"reason": f"Unsupported action: {intent.action}"}
        )
        