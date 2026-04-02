#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical IR action registry for Behavior authoring.

This module exposes the single brand/model-independent IR action catalog.
Model-specific support and backend raw-action mapping live elsewhere.
"""

from __future__ import annotations

from typing import Dict, List

from src.system.behavior.intent_catalog import LOCOMOTION_FORWARD, get_base_catalog
from src.system.behavior.semantic_action import (
    SemanticActionDescriptor,
    find_duplicate_intents,
    validate_descriptor,
)


_REGISTERED_ACTIONS: List[SemanticActionDescriptor] = [
    descriptor
    for descriptor in get_base_catalog()
    if descriptor.intent_id == LOCOMOTION_FORWARD
]


def list_ir_actions() -> List[SemanticActionDescriptor]:
    """Return the ordered list of canonical IR actions."""
    return list(_REGISTERED_ACTIONS)


def list_ir_actions_by_intent() -> Dict[str, SemanticActionDescriptor]:
    """Return the canonical IR action registry indexed by intent_id."""
    return {desc.intent_id: desc for desc in _REGISTERED_ACTIONS}


def validate_ir_registry() -> List[str]:
    """Return a flat list of canonical-registry validation errors."""
    errors: List[str] = []
    for index, desc in enumerate(_REGISTERED_ACTIONS):
        if not validate_descriptor(desc):
            errors.append(f"descriptor[{index}] invalid: {desc.intent_id}")
    duplicates = find_duplicate_intents(_REGISTERED_ACTIONS)
    for key, indices in duplicates.items():
        errors.append(f"duplicate descriptor {key}: {indices}")
    return errors
