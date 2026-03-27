#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Spot adapter semantic action declarations built from the shared builder."""

from __future__ import annotations

from typing import List

from system.behavior.semantic_action import ActionAvailability, SemanticActionDescriptor
from system.service.semantic_action_builder import build_common_adapter_actions

_BRAND = "bostiondynamics"


def get_spot_semantic_actions(robot_type: str = "spot") -> List[SemanticActionDescriptor]:
    rt = (robot_type or "spot").strip().lower() or "spot"
    return build_common_adapter_actions(
        brand=_BRAND,
        robot_type=rt,
        gesture_availability=ActionAvailability.UNSUPPORTED,
    )
