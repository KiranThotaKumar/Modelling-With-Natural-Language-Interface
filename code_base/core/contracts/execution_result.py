# core/contracts/execution_result.py

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass(frozen=True)
class ExecutionResult:
    """
    Declarative representation of execution output.

    This object contains no routing or execution logic and serves
    as the authoritative result contract across execution,
    serialization, and output layers.
    """

    status: str
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
