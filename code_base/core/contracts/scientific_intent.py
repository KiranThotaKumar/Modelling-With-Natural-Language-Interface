
# core/contracts/scientific_intent.py

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass(frozen=True)
class ScientificIntent:
    """
    Declarative, domain-agnostic representation of user intent.

    This object contains no execution logic and serves as the
    authoritative contract across NLP, execution, registry,
    and serialization layers.
    """

    domain: str
    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    intent_type: str = "scientific"
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

