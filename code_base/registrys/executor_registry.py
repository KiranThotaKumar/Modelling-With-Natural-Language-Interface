#registrys\executor_registry.py

from typing import Dict
from execution.domain_executor import DomainExecutor
from execution.executors.single_qubit_executor import SingleQubitDomainExecutor
from execution.executors.multi_qubit_executor import MultiQubitDomainExecutor



class ExecutorRegistry:

    def __init__(self):
        self._executors: Dict[str, DomainExecutor] = {}

    def register(self, domain: str, executor: DomainExecutor):
        if not isinstance(domain, str) or not domain:
            raise ValueError("Domain must be a non-empty string")

        if domain in self._executors:
            raise ValueError(f"Executor already registered for domain: {domain}")

        self._executors[domain] = executor

    def get(self, domain: str) -> DomainExecutor:
        if domain not in self._executors:
            raise ValueError(f"No executor registered for domain: {domain}")

        return self._executors[domain]

    def list_domains(self):
        return list(self._executors.keys())


def build_registry():

    registry = ExecutorRegistry()
    
    from execution.executors.hydrogen_executor import HydrogenDomainExecutor

    registry.register(
        "hydrogen",
        HydrogenDomainExecutor()
    )

    registry.register(
        "single_qubit",
        SingleQubitDomainExecutor()
    )

    registry.register(
        "multi_qubit",
        MultiQubitDomainExecutor()
    )

    return registry