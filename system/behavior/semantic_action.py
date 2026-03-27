#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Semantic action descriptor — Circle 1 STEP 1.1.

Defines the canonical semantic action contract used by:
  - Behavior authoring  (intent_id as the durable saved key)
  - Model-switch compat  (semantic comparison, not raw string comparison)
  - Runtime resolution   (semantic intent -> raw backend action at exec time)

Public API
----------
SemanticActionDescriptor   — immutable descriptor dataclass
ActionAvailability         — "available" | "degraded" | "unsupported" constants
is_valid_intent_id(s)      — validate intent ID format
descriptor_errors(d)       -> List[str]  — validation error list
validate_descriptor(d)     -> bool       — True iff no errors
find_duplicate_intents(ds) -> Dict[str, List[int]]  — duplicate key report

Design constraints:
  - Pure Python; no Qt imports; no adapter-specific logic.
  - SemanticActionDescriptor is frozen (no field reassignment after creation).
  - intent_id is the durable authoring key; raw_action is execution metadata.
  - ASCII-only intent IDs: lowercase letters, digits, underscores, dots.
  - from_dict never raises; missing optional fields fall back to defaults.
"""

from __future__ import annotations

import re
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Tuple


# ---------------------------------------------------------------------------
# Availability constants
# ---------------------------------------------------------------------------

class ActionAvailability:
    """Legal values for ``SemanticActionDescriptor.availability``.

    ``AVAILABLE``   — action can be fully executed on the target model.
    ``DEGRADED``    — action executes but with reduced functionality or
                      missing parameters compared to the authoring descriptor.
    ``UNSUPPORTED`` — no executable mapping exists; execution must fail fast.
    """

    AVAILABLE = "available"
    DEGRADED = "degraded"
    UNSUPPORTED = "unsupported"

    _ALL: frozenset = frozenset({"available", "degraded", "unsupported"})


# ---------------------------------------------------------------------------
# Intent ID format validation
# ---------------------------------------------------------------------------

# Intent IDs must be ASCII lowercase, dot-separated, at least two segments.
# Each segment: one or more of [a-z0-9_].  No leading/trailing dots.
# Examples: "posture.stand", "locomotion.forward", "gesture.lift_right_leg"
_INTENT_ID_RE = re.compile(r"^[a-z0-9_]+(\.[a-z0-9_]+)+$")
_CATEGORY_RE = re.compile(r"^[a-z0-9_]+$")


def is_valid_intent_id(intent_id: str) -> bool:
    """Return True iff *intent_id* is a valid ASCII dot-separated identifier.

    Rules:
    - Must be a non-empty ``str``.
    - Lowercase letters, digits, and underscores only (no uppercase, no hyphen).
    - At least two dot-separated segments (e.g. ``"posture.stand"``).
    - No leading or trailing dots; no consecutive dots.
    """
    if not isinstance(intent_id, str):
        return False
    return bool(_INTENT_ID_RE.match(intent_id))


# ---------------------------------------------------------------------------
# Deep freeze / thaw helpers
# ---------------------------------------------------------------------------

def _deep_freeze(obj: Any) -> Any:
    """Recursively wrap every nested dict in ``MappingProxyType``.

    This produces a fully read-only tree: neither the outer mapping nor any
    nested dict can be mutated after the call returns.  Non-dict values are
    returned unchanged (scalars, strings, numbers, tuples, etc.).

    Used in ``SemanticActionDescriptor.__post_init__`` to satisfy the deeply-
    immutable parameters contract.
    """
    if isinstance(obj, (dict, types.MappingProxyType)):
        return types.MappingProxyType({k: _deep_freeze(v) for k, v in obj.items()})
    return obj


def _deep_thaw(obj: Any) -> Any:
    """Inverse of ``_deep_freeze``: recursively convert ``MappingProxyType``
    back to plain mutable dicts.

    Used in ``SemanticActionDescriptor.to_dict`` so the serialized output
    contains only standard JSON-compatible types.
    """
    if isinstance(obj, (dict, types.MappingProxyType)):
        return {k: _deep_thaw(v) for k, v in obj.items()}
    return obj


# ---------------------------------------------------------------------------
# Immutable descriptor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SemanticActionDescriptor:
    """Immutable semantic action descriptor.

    Fields
    ------
    intent_id    : Durable authoring key (e.g. ``"posture.stand"``).  ASCII
                   lowercase, dot-separated.  This is the stable identifier
                   stored in behavior drafts and used for cross-model compat.
    display_name : Human-readable label shown in UI surfaces.
    category     : High-level semantic category (e.g. ``"posture"``,
                   ``"locomotion"``, ``"gesture"``).  Matches the first
                   segment of intent_id by convention.
    raw_action   : Backend action key for execution (e.g. ``"stand"`` for the
                   Unitree SDK2 adapter).  This is execution metadata, NOT the
                   authoring identity.
    brand        : Robot brand identifier (``"unitree"``, ``"bostiondynamics"``,
                   ``"xiaomi"``).  Empty string means brand-independent.
    robot_type   : Specific model within the brand (e.g. ``"go2"``, ``"spot"``).
                   Empty string means applicable to all models of the brand.
    parameters   : Read-only mapping of parameter name → metadata dict.  Each
                   metadata entry must contain at minimum a ``"type"`` key.
                   Example::

                       {"speed": {"type": "float", "default": 0.5, "min": 0.0}}

                   Pass a plain ``dict`` at construction; it is wrapped in
                   ``MappingProxyType`` automatically so callers cannot mutate
                   descriptor contents after creation.
    availability : ``"available"`` | ``"degraded"`` | ``"unsupported"``.
                   Indicates whether this action can execute on the target.
    tags         : Tuple of string tags for filtering (e.g. ``("safe", "slow")``).
    """

    intent_id: str
    display_name: str
    category: str
    raw_action: str
    brand: str
    robot_type: str
    parameters: Mapping[str, Any] = field(default_factory=dict, hash=False)
    availability: str = ActionAvailability.AVAILABLE
    tags: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # Deep-freeze parameters so neither the outer mapping nor any nested
        # metadata dict can be mutated after construction.  _deep_freeze copies
        # each dict level into a new MappingProxyType, breaking aliasing to the
        # caller's source dicts as well.
        object.__setattr__(self, "parameters", _deep_freeze(self.parameters))

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-safe dict (all values are primitive types)."""
        return {
            "intent_id": self.intent_id,
            "display_name": self.display_name,
            "category": self.category,
            "raw_action": self.raw_action,
            "brand": self.brand,
            "robot_type": self.robot_type,
            "parameters": _deep_thaw(self.parameters),
            "availability": self.availability,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticActionDescriptor":
        """Deserialize from a dict.  Missing optional fields use defaults.

        Never raises — malformed input produces a descriptor that will fail
        ``validate_descriptor()``.
        """
        if not isinstance(data, dict):
            data = {}
        return cls(
            intent_id=data.get("intent_id") or "",
            display_name=data.get("display_name") or "",
            category=data.get("category") or "",
            raw_action=data.get("raw_action") or "",
            brand=data.get("brand") or "",
            robot_type=data.get("robot_type") or "",
            parameters=dict(data["parameters"]) if isinstance(data.get("parameters"), dict) else {},
            availability=data.get("availability") or ActionAvailability.AVAILABLE,
            tags=tuple(data["tags"]) if isinstance(data.get("tags"), (list, tuple)) else (),
        )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def descriptor_errors(descriptor: Any) -> List[str]:
    """Return a list of validation error strings for *descriptor*.

    An empty list means the descriptor is fully valid.

    Checks:
    1. Input must be a ``SemanticActionDescriptor`` instance.
    2. ``intent_id`` must be non-empty and match the ASCII dot-separated format.
    3. ``display_name`` must be non-empty.
    4. ``category`` must be non-empty ASCII lowercase.
    5. ``availability`` must be one of the ``ActionAvailability`` constants.
    6. ``parameters`` must be a dict.
    7. Each element of ``tags`` must be a str.
    """
    if not isinstance(descriptor, SemanticActionDescriptor):
        return [
            f"expected SemanticActionDescriptor, got {type(descriptor).__name__!r}"
        ]

    errors: List[str] = []

    # intent_id
    if not descriptor.intent_id:
        errors.append("intent_id must not be empty")
    elif not is_valid_intent_id(descriptor.intent_id):
        errors.append(
            f"intent_id {descriptor.intent_id!r} must be ASCII lowercase "
            "dot-separated with at least two segments (e.g. 'posture.stand')"
        )

    # display_name
    if not descriptor.display_name:
        errors.append("display_name must not be empty")

    # category — must be non-empty ASCII lowercase and match the first
    # segment of intent_id (the two are canonical by contract).
    if not descriptor.category:
        errors.append("category must not be empty")
    elif not _CATEGORY_RE.match(descriptor.category):
        errors.append(
            f"category {descriptor.category!r} must be ASCII lowercase "
            "(letters, digits, underscores only)"
        )
    elif is_valid_intent_id(descriptor.intent_id):
        first_segment = descriptor.intent_id.split(".")[0]
        if descriptor.category != first_segment:
            errors.append(
                f"category {descriptor.category!r} does not match the first "
                f"segment of intent_id {descriptor.intent_id!r} "
                f"(expected {first_segment!r})"
            )

    # availability
    if descriptor.availability not in ActionAvailability._ALL:
        errors.append(
            f"availability {descriptor.availability!r} must be one of "
            f"{sorted(ActionAvailability._ALL)}"
        )

    # parameters — after __post_init__ this is always a MappingProxyType;
    # a plain Mapping is also accepted so the check covers both paths.
    if not isinstance(descriptor.parameters, (dict, types.MappingProxyType)):
        errors.append(
            f"parameters must be a mapping, got {type(descriptor.parameters).__name__!r}"
        )

    # tags
    if not isinstance(descriptor.tags, tuple):
        errors.append(
            f"tags must be a tuple, got {type(descriptor.tags).__name__!r}"
        )
    else:
        for i, tag in enumerate(descriptor.tags):
            if not isinstance(tag, str):
                errors.append(
                    f"tags[{i}] must be str, got {type(tag).__name__!r} ({tag!r})"
                )

    return errors


def validate_descriptor(descriptor: Any) -> bool:
    """Return True iff *descriptor* passes all validation checks."""
    return len(descriptor_errors(descriptor)) == 0


def find_duplicate_intents(
    descriptors: List[SemanticActionDescriptor],
) -> Dict[str, List[int]]:
    """Detect duplicate intent declarations in *descriptors*.

    Two descriptors are considered duplicates when they share the same
    ``(intent_id, brand, robot_type)`` combination.

    Args:
        descriptors: List of ``SemanticActionDescriptor`` instances (non-descriptor
                     items are silently skipped).

    Returns:
        Dict mapping the duplicate composite key to a list of indices in
        *descriptors* where it appears.  Only keys that appear more than once
        are included; an empty dict means no duplicates.

    Example::

        >>> find_duplicate_intents([a, b, c])
        {"posture.stand|unitree|go2": [0, 2]}
    """
    seen: Dict[str, List[int]] = {}
    for i, desc in enumerate(descriptors):
        if not isinstance(desc, SemanticActionDescriptor):
            continue
        key = f"{desc.intent_id}|{desc.brand}|{desc.robot_type}"
        seen.setdefault(key, []).append(i)
    return {k: v for k, v in seen.items() if len(v) > 1}
