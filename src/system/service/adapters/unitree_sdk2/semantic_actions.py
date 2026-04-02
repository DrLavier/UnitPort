#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unitree adapter semantic action declarations built from the shared builder."""

from __future__ import annotations

from typing import List

from src.system.behavior.semantic_action import ActionAvailability, SemanticActionDescriptor
from src.system.models.model_registry import iter_model_specs
from src.system.service.semantic_action_builder import build_common_adapter_actions

_QUADRUPED_TYPES = frozenset(
    spec.model_id
    for spec in iter_model_specs(brand="unitree")
    if spec.forward is not None
    and spec.forward.adapter_available
    and (spec.forward.robot_category or "") == "quadruped"
)
_HUMANOID_TYPES = frozenset(
    spec.model_id
    for spec in iter_model_specs(brand="unitree")
    if spec.forward is not None
    and spec.forward.adapter_available
    and (spec.forward.robot_category or "") == "humanoid"
)
_ALL_VERIFIED_TYPES = _QUADRUPED_TYPES | _HUMANOID_TYPES

_BRAND = "unitree"


def get_unitree_semantic_actions(robot_type: str) -> List[SemanticActionDescriptor]:
    rt = (robot_type or "").strip().lower()
    if rt not in _ALL_VERIFIED_TYPES:
        return []

    gesture_availability = (
        ActionAvailability.AVAILABLE if rt == "go2" else ActionAvailability.UNSUPPORTED
    )
    return build_common_adapter_actions(
        brand=_BRAND,
        robot_type=rt,
        gesture_availability=gesture_availability,
    )
