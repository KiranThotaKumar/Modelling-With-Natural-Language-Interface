#execution\executors\multi_qubit_executor.py

from execution.domain_executor import DomainExecutor
from core.contracts.scientific_intent import ScientificIntent
from core.contracts.execution_result import ExecutionResult
from execution.executors.multi_qubit_inference_executor import MultiQubitInferenceExecutor
from execution.executors.multi_qubit_forward_executor import MultiQubitForwardExecutor


class MultiQubitDomainExecutor(DomainExecutor):

    def __init__(self):
        self.forward_executor = MultiQubitForwardExecutor()
        self.inference_executor = MultiQubitInferenceExecutor()

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
        

