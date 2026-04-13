"""Skill module — SkillManifest contract and utilities.

Central normalisation layer that makes heterogeneous behaviour policies
(SB3, HuggingFace, VLA, manual scripts) interoperable in the Mission layer.
"""

from src.system.skill.skill_manifest import (
    ActionSpaceType,
    CommandField,
    CommandInterface,
    InferenceBackend,
    Postcondition,
    Precondition,
    SkillManifest,
    SourceType,
)
from src.system.skill.manifest_loader import load_skill_manifest
from src.system.skill.manifest_inferrer import infer_skill_manifest_fields
from src.system.skill.transition_result import (
    TransitionCheckResult,
    TransitionStatus,
    TransitionValidationReport,
    TransitionWarning,
    TransitionWarningKind,
)
from src.system.skill.transition_validator import validate_skill_chain
from src.system.skill.transition_library import (
    lookup_transition,
    register_transition,
    list_transitions,
)

__all__ = [
    "ActionSpaceType",
    "CommandField",
    "CommandInterface",
    "infer_skill_manifest_fields",
    "InferenceBackend",
    "list_transitions",
    "load_skill_manifest",
    "lookup_transition",
    "Postcondition",
    "Precondition",
    "register_transition",
    "SkillManifest",
    "SourceType",
    "TransitionCheckResult",
    "TransitionStatus",
    "TransitionValidationReport",
    "TransitionWarning",
    "TransitionWarningKind",
    "validate_skill_chain",
]
