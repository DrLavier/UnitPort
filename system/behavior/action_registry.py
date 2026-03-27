#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Central ActionRegistry for Behavior IR actions.

Single source of truth for semantic action lookup and raw-action resolution.
All three public functions are pure: same inputs always produce the same
output; no global mutable state; no Qt imports; no adapter instances required.

Public API
----------
get_semantic_actions(brand, robot_type) -> List[SemanticActionDescriptor]
    Return all semantic descriptors registered in the IR action layer,
    filtered by brand and robot_type when provided.

resolve_intent(raw_action, brand, robot_type) -> Optional[SemanticActionDescriptor]
    Reverse lookup: raw backend action string → semantic descriptor.
    Returns None when no mapping exists.

resolve_raw_action(intent_id, brand, robot_type) -> Optional[str]
    Forward lookup: semantic intent ID → executable raw action string.
    Returns None when no mapping exists for the given brand + model.

Design constraints:
    - Authoring/UI surfaces source declarations from the IR action layer,
      not from brand adapter capabilities.
    - brand / robot_type filtering added in Circle 4 Step 4.1 per DEVELOP_TODO.txt.
    - resolve_intent picks the first AVAILABLE match; UNSUPPORTED entries are
      deprioritised but returned as a fallback if no AVAILABLE match exists.
    - Callers must not assume list ordering beyond the adapter's declaration order.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from system.behavior.semantic_action import ActionAvailability, SemanticActionDescriptor
from system.behavior.ir_action_registry import list_ir_actions
from system.model_registry import canonical_brand_ids, get_ir_action_binding, normalize_brand_id


def _filter_by_brand_robot(
    descriptors: List[SemanticActionDescriptor],
    brand: str,
    robot_type: str,
) -> List[SemanticActionDescriptor]:
    """Overlay model-specific support on top of the canonical IR catalog."""
    brand_lc = normalize_brand_id(brand) if brand else ""
    robot_type_lc = robot_type.lower().strip() if robot_type else ""

    result: List[SemanticActionDescriptor] = []
    for descriptor in descriptors:
        binding = (
            get_ir_action_binding(brand_lc, robot_type_lc, descriptor.intent_id)
            if brand_lc and robot_type_lc
            else {}
        )
        result.append(
            SemanticActionDescriptor(
                intent_id=descriptor.intent_id,
                display_name=descriptor.display_name,
                category=descriptor.category,
                raw_action=str(binding.get("raw_action") or descriptor.raw_action),
                brand=brand_lc,
                robot_type=robot_type_lc,
                parameters=dict(binding.get("parameters") or descriptor.parameters),
                availability=str(binding.get("availability") or descriptor.availability),
                tags=tuple(binding.get("tags") or descriptor.tags),
            )
        )
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_semantic_actions(brand: str, robot_type: str) -> List[SemanticActionDescriptor]:
    """Return all semantic descriptors registered in the IR action layer.

    Args:
        brand:      Robot brand identifier (e.g. "unitree"). Empty for all brands.
        robot_type: Model identifier within the brand (e.g. "go2"). Empty for all models.

    Returns:
        List of ``SemanticActionDescriptor`` instances filtered by brand/robot_type.
        When brand and robot_type are both empty, returns all actions (backward compat).
    """
    all_descriptors = list_ir_actions()
    return _filter_by_brand_robot(all_descriptors, brand, robot_type)


def resolve_intent(
    raw_action: str,
    brand: str,
    robot_type: str,
) -> Optional[SemanticActionDescriptor]:
    """Reverse lookup: map a raw backend action string to a semantic descriptor.

    When multiple descriptors share the same ``raw_action`` (e.g. because a
    model is brand-agnostic), the first AVAILABLE match is returned.  An
    UNSUPPORTED entry is only returned when no AVAILABLE or DEGRADED match
    exists.

    Args:
        raw_action:  The raw backend action key (e.g. ``"walk"``).
        brand:       Robot brand identifier.
        robot_type:  Model identifier within the brand.

    Returns:
        The matching ``SemanticActionDescriptor``, or ``None`` when no entry
        with that ``raw_action`` exists for the given brand + model.
    """
    if not raw_action:
        return None

    best: Optional[SemanticActionDescriptor] = None
    for d in get_semantic_actions(brand, robot_type):
        resolved_raw = resolve_raw_action(d.intent_id, brand, robot_type)
        if resolved_raw != raw_action:
            continue
        if d.availability == ActionAvailability.AVAILABLE:
            return d
        if best is None:
            best = d
    return best


def resolve_raw_action(
    intent_id: str,
    brand: str,
    robot_type: str,
) -> Optional[str]:
    """Forward lookup: resolve a semantic intent to an executable raw action.

    Args:
        intent_id:   Semantic intent identifier (e.g. ``"locomotion.forward"``).
        brand:       Robot brand identifier.
        robot_type:  Model identifier within the brand.

    Returns:
        The raw backend action string (e.g. ``"walk"``), or ``None`` when no
        descriptor for *intent_id* exists or when the descriptor is marked
        UNSUPPORTED.  Returns the raw_action even for DEGRADED entries so
        runtime can attempt execution with degraded capability.
    """
    if not intent_id:
        return None

    binding = get_ir_action_binding(brand, robot_type, intent_id)
    raw_action = str(binding.get("raw_action") or "")
    availability = str(binding.get("availability") or "")
    if availability == ActionAvailability.UNSUPPORTED or not raw_action:
        return None
    return raw_action


def list_known_brands() -> List[str]:
    """Return the list of brand identifiers known to the ActionRegistry API.

    The IR registry is intentionally brand-neutral, so this returns an empty
    list while preserving the public function for compatibility.
    """
    return list(canonical_brand_ids())
