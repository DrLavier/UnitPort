"""Safety policy editor placeholder."""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class SafetyPolicyDraft:
    rules: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def to_runtime_policy(self) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        return dict(self.rules)
