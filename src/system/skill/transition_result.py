"""Transition validation result dataclasses."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import List, Optional


class TransitionStatus(enum.Enum):
    """Outcome of checking one adjacent skill pair."""
    COMPATIBLE = "compatible"
    NEEDS_TRANSITION = "needs_transition"
    INCOMPATIBLE = "incompatible"


class TransitionWarningKind(enum.Enum):
    """Non-blocking issues that deserve attention."""
    FREQUENCY_RATIO = "frequency_ratio"
    ACTION_SPACE_MISMATCH = "action_space_mismatch"


@dataclass(frozen=True)
class TransitionWarning:
    """A non-fatal warning attached to a transition check."""
    kind: TransitionWarningKind
    message: str


@dataclass(frozen=True)
class TransitionCheckResult:
    """Result of checking a single (skill_A → skill_B) transition.

    Attributes:
        from_skill_id: ID of the upstream skill.
        to_skill_id:   ID of the downstream skill.
        status:        Overall verdict.
        reason:        Human-readable explanation when not COMPATIBLE.
        transition_id: If *status* is NEEDS_TRANSITION, the ID of the
                       transition behaviour that can bridge the gap.
                       ``None`` when COMPATIBLE or INCOMPATIBLE.
        warnings:      Non-blocking issues (e.g. large frequency ratio).
    """
    from_skill_id: str
    to_skill_id: str
    status: TransitionStatus
    reason: str = ""
    transition_id: Optional[str] = None
    warnings: List[TransitionWarning] = field(default_factory=list)


@dataclass(frozen=True)
class TransitionValidationReport:
    """Aggregated results for an entire DAG skill chain.

    Attributes:
        results:    One :class:`TransitionCheckResult` per adjacent pair.
        is_valid:   ``True`` only if no INCOMPATIBLE results remain.
        insertions: Skill IDs of transition behaviours to insert (ordered).
    """
    results: List[TransitionCheckResult] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return all(
            r.status != TransitionStatus.INCOMPATIBLE for r in self.results
        )

    @property
    def insertions(self) -> List[str]:
        return [
            r.transition_id
            for r in self.results
            if r.status == TransitionStatus.NEEDS_TRANSITION and r.transition_id
        ]

    @property
    def warnings(self) -> List[TransitionWarning]:
        out: List[TransitionWarning] = []
        for r in self.results:
            out.extend(r.warnings)
        return out
