
# execution/domain_executor.py

from abc import ABC, abstractmethod
from core.contracts.scientific_intent import ScientificIntent
from core.contracts.execution_result import ExecutionResult


class DomainExecutor(ABC):
    """
    Execution contract for all domain-specific executors.

    Implementations must:
    - Accept a ScientificIntent
    - Perform domain-specific validation and execution
    - Return a structured result (dict for now)
    """

    @abstractmethod
    def handle(self, intent: ScientificIntent) -> ExecutionResult:
        pass