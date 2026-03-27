#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared builder for adapter capability semantic actions."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from system.behavior.intent_catalog import (
    GESTURE_LIFT_RIGHT_LEG,
    LOCOMOTION_FORWARD,
    LOCOMOTION_HALT,
    POSTURE_SIT,
    POSTURE_STAND,
    get_base_catalog_by_intent,
)
from system.behavior.semantic_action import ActionAvailability, SemanticActionDescriptor
from system.model_registry import get_ir_action_binding


_BASE_BY_INTENT = get_base_catalog_by_intent()


def _clone_descriptor(
    intent_id: str,
    *,
    brand: str,
    robot_type: str,
    raw_action: Optional[str] = None,
    availability: Optional[str] = None,
    parameters: Optional[Mapping[str, object]] = None,
    tags: Optional[Sequence[str]] = None,
    display_name: Optional[str] = None,
) -> SemanticActionDescriptor:
    base = _BASE_BY_INTENT[intent_id]
    return SemanticActionDescriptor(
        intent_id=base.intent_id,
        display_name=display_name or base.display_name,
        category=base.category,
        raw_action=raw_action or base.raw_action,
        brand=brand,
        robot_type=robot_type,
        parameters=dict(parameters or base.parameters),
        availability=availability or base.availability,
        tags=tuple(tags or base.tags),
    )


def build_common_adapter_actions(
    *,
    brand: str,
    robot_type: str,
    supported_intents: Optional[Iterable[str]] = None,
    raw_action_map: Optional[Mapping[str, str]] = None,
    gesture_availability: str = ActionAvailability.UNSUPPORTED,
) -> List[SemanticActionDescriptor]:
    """Build the standard adapter capability declaration set.

    Uses the canonical IR catalog for labels/categories and overlays model-
    specific binding metadata for intents that are declared in the unified
    model registry.
    """
    raw_action_map = dict(raw_action_map or {})
    supported = set(supported_intents or ())
    actions: List[SemanticActionDescriptor] = []

    for intent_id in (POSTURE_STAND, POSTURE_SIT):
        if supported and intent_id not in supported:
            continue
        actions.append(
            _clone_descriptor(
                intent_id,
                brand=brand,
                robot_type=robot_type,
                raw_action=raw_action_map.get(intent_id, _BASE_BY_INTENT[intent_id].raw_action),
            )
        )

    if not supported or LOCOMOTION_FORWARD in supported:
        binding = get_ir_action_binding(brand, robot_type, LOCOMOTION_FORWARD)
        actions.append(
            _clone_descriptor(
                LOCOMOTION_FORWARD,
                brand=brand,
                robot_type=robot_type,
                raw_action=raw_action_map.get(
                    LOCOMOTION_FORWARD,
                    str(binding.get("raw_action") or _BASE_BY_INTENT[LOCOMOTION_FORWARD].raw_action),
                ),
                availability=str(
                    binding.get("availability") or ActionAvailability.UNSUPPORTED
                ),
                parameters=dict(binding.get("parameters") or _BASE_BY_INTENT[LOCOMOTION_FORWARD].parameters),
                tags=tuple(binding.get("tags") or _BASE_BY_INTENT[LOCOMOTION_FORWARD].tags),
            )
        )

    if not supported or LOCOMOTION_HALT in supported:
        actions.append(
            _clone_descriptor(
                LOCOMOTION_HALT,
                brand=brand,
                robot_type=robot_type,
                raw_action=raw_action_map.get(LOCOMOTION_HALT, _BASE_BY_INTENT[LOCOMOTION_HALT].raw_action),
            )
        )

    if not supported or GESTURE_LIFT_RIGHT_LEG in supported:
        actions.append(
            _clone_descriptor(
                GESTURE_LIFT_RIGHT_LEG,
                brand=brand,
                robot_type=robot_type,
                raw_action=raw_action_map.get(
                    GESTURE_LIFT_RIGHT_LEG,
                    _BASE_BY_INTENT[GESTURE_LIFT_RIGHT_LEG].raw_action,
                ),
                availability=gesture_availability,
            )
        )

    return actions
