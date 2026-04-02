#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CyberDog adapter semantic action declarations built from the shared builder."""

from __future__ import annotations

from typing import List

from src.system.behavior.semantic_action import ActionAvailability, SemanticActionDescriptor
from src.system.service.semantic_action_builder import build_common_adapter_actions

_BRAND = "xiaomi"


def get_cyberdog_semantic_actions(robot_type: str = "cyberdog") -> List[SemanticActionDescriptor]:
    rt = (robot_type or "cyberdog").strip().lower() or "cyberdog"
    return build_common_adapter_actions(
        brand=_BRAND,
        robot_type=rt,
        gesture_availability=ActionAvailability.UNSUPPORTED,
    )
