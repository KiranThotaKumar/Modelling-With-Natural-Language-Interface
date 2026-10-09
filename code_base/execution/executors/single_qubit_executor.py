#execution\executors\single_qubit_executor.py

from execution.domain_executor import DomainExecutor
from core.contracts.scientific_intent import ScientificIntent
from core.contracts.execution_result import ExecutionResult
from execution.executors.single_qubit_forward_executor import SingleQubitForwardExecutor
from execution.executors.single_qubit_inference_executor import SingleQubitInferenceExecutor


class SingleQubitDomainExecutor(DomainExecutor):

    def __init__(self):
        self.forward_executor = SingleQubitForwardExecutor()
        self.inference_executor = SingleQubitInferenceExecutor()

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
        
