
# execution/intent_execution_bridge.py

from core.contracts.scientific_intent import ScientificIntent
from registrys.executor_registry import ExecutorRegistry
from core.contracts.execution_result import ExecutionResult

class IntentExecutionBridge:
    """
    Responsible for routing ScientificIntent to the correct DomainExecutor.
    """

    def __init__(self, registry: ExecutorRegistry):
        self._registry = registry

    def execute(self, intent: ScientificIntent) -> ExecutionResult:
        executor = self._registry.get(intent.domain)
        return executor.handle(intent)
