"""Capability schema validation — Phase 5 STAGE-03 / Circle 2 STEP 2.1.

Defines the canonical schema for the dict returned by adapter.capabilities()
and provides two public validation functions:

    validate_capability(cap)         -> bool
    capability_schema_errors(cap)    -> List[str]

Circle 2 additive change — ``semantic_actions``:
    Adapters may now include a ``"semantic_actions"`` key in their capability
    dict.  When present it must be a list of dicts that each conform to the
    SemanticActionDescriptor contract (validated via
    ``semantic_action_item_errors``).  The key is optional so old adapters
    that omit it remain fully conformant.

Design constraints:
    - Pure Python; no external validation libraries.
    - No file I/O.
    - Extra keys beyond the required set are tolerated (forward compatibility).
    - List element types for actions/sensors/required_settings are also checked.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.system.behavior.semantic_action import (
    ActionAvailability,
    is_valid_intent_id,
)


# ── Schema definition ──────────────────────────────────────────────────────
#
# Maps each required top-level key to its expected Python type.
# Sourced from README §8.1 (Phase 5 schema contract).

CAPABILITY_SCHEMA: Dict[str, type] = {
    "brand":             str,   # e.g. "unitree" | "boston_dynamics" | "xiaomi"
    "adapter":           str,   # e.g. "unitree_sdk2" | "spot_sdk" | "cyberdog_sdk"
    "actions":           list,  # List[str] — canonical action names
    "sensors":           list,  # List[str] — sensor data keys
    "flags":             dict,  # Dict[str, Any] — feature flags
    "required_settings": list,  # List[str] — settings keys needed before execution
}

# Keys whose list elements must all be str
_LIST_STR_KEYS = ("actions", "sensors", "required_settings")

# Required string fields in each semantic_actions item
_SEMANTIC_ITEM_STR_FIELDS = ("intent_id", "display_name", "category", "raw_action",
                              "brand", "robot_type", "availability")


# ── Semantic action item validation ────────────────────────────────────────

def semantic_action_item_errors(item: Any, index: int) -> List[str]:
    """Return validation errors for one entry inside ``semantic_actions``.

    Each item must be a dict with all ``SemanticActionDescriptor`` string
    fields present and non-empty (except ``brand``/``robot_type`` which may
    be empty for brand-independent entries).  ``availability`` must be one of
    the ``ActionAvailability`` constants and ``intent_id`` must pass the
    dot-separated format check.

    Args:
        item:  A single element from the ``semantic_actions`` list.
        index: Position within the list (used in error messages only).

    Returns:
        List of human-readable error strings; empty on success.
    """
    prefix = f"semantic_actions[{index}]"
    if not isinstance(item, dict):
        return [f"{prefix} must be a dict, got {type(item).__name__!r}"]

    errors: List[str] = []

    # All declared string fields must be present and be str
    for field in _SEMANTIC_ITEM_STR_FIELDS:
        if field not in item:
            errors.append(f"{prefix} missing required field {field!r}")
        elif not isinstance(item[field], str):
            errors.append(
                f"{prefix}.{field} must be str, got {type(item[field]).__name__!r}"
            )

    if errors:
        return errors  # field-level errors make deeper checks unreliable

    # intent_id format
    if not is_valid_intent_id(item["intent_id"]):
        errors.append(
            f"{prefix}.intent_id {item['intent_id']!r} is not a valid "
            "dot-separated ASCII intent ID"
        )

    # Non-empty fields (brand/robot_type may be empty)
    for field in ("display_name", "category"):
        if not item[field]:
            errors.append(f"{prefix}.{field} must not be empty")

    # availability must be a known constant
    if item["availability"] not in ActionAvailability._ALL:
        errors.append(
            f"{prefix}.availability {item['availability']!r} must be one of "
            f"{sorted(ActionAvailability._ALL)}"
        )

    # parameters must be a dict when present
    if "parameters" in item and not isinstance(item["parameters"], dict):
        errors.append(
            f"{prefix}.parameters must be a dict, "
            f"got {type(item['parameters']).__name__!r}"
        )

    # tags must be a list of str when present
    if "tags" in item:
        if not isinstance(item["tags"], list):
            errors.append(
                f"{prefix}.tags must be a list, got {type(item['tags']).__name__!r}"
            )
        else:
            for ti, tag in enumerate(item["tags"]):
                if not isinstance(tag, str):
                    errors.append(f"{prefix}.tags[{ti}] must be str")

    return errors


# ── Public API ─────────────────────────────────────────────────────────────


def capability_schema_errors(cap: Any) -> List[str]:
    """Return a list of schema violation messages for a capabilities() dict.

    An empty list means the dict is fully conformant.  Extra keys beyond the
    required set are silently ignored (non-breaking forward compatibility).

    Checks performed:
        1. Input must be a dict.
        2. Every key in CAPABILITY_SCHEMA must be present.
        3. Each present key must have the correct Python type.
        4. Elements of list-typed keys (actions / sensors / required_settings)
           must all be str.
        5. When ``"semantic_actions"`` is present it must be a list of
           schema-valid semantic descriptor dicts (optional key).

    Args:
        cap: The value returned by adapter.capabilities() to validate.

    Returns:
        List of human-readable error strings; empty on success.
    """
    if not isinstance(cap, dict):
        return [
            f"capabilities() must return a dict, got {type(cap).__name__!r}"
        ]

    errors: List[str] = []

    # ── Required keys + top-level type checks ─────────────────────────────
    for key, expected_type in CAPABILITY_SCHEMA.items():
        if key not in cap:
            errors.append(f"missing required key: {key!r}")
            continue
        if not isinstance(cap[key], expected_type):
            actual = type(cap[key]).__name__
            errors.append(
                f"{key!r} must be {expected_type.__name__}, got {actual!r}"
            )

    # ── List element type checks ───────────────────────────────────────────
    for list_key in _LIST_STR_KEYS:
        if list_key not in cap:
            continue  # already reported as missing above
        if not isinstance(cap[list_key], list):
            continue  # already reported as wrong type above
        for i, item in enumerate(cap[list_key]):
            if not isinstance(item, str):
                errors.append(
                    f"{list_key!r}[{i}] must be str, "
                    f"got {type(item).__name__!r} ({item!r})"
                )

    # ── Optional semantic_actions validation ──────────────────────────────
    if "semantic_actions" in cap:
        sa = cap["semantic_actions"]
        if not isinstance(sa, list):
            errors.append(
                f"'semantic_actions' must be a list, got {type(sa).__name__!r}"
            )
        else:
            for i, item in enumerate(sa):
                errors.extend(semantic_action_item_errors(item, i))

    return errors


def validate_capability(cap: Any) -> bool:
    """Return True iff the capabilities() dict conforms to the Phase 5 schema.

    Equivalent to ``len(capability_schema_errors(cap)) == 0``.

    Args:
        cap: The value returned by adapter.capabilities() to validate.

    Returns:
        True if conformant, False otherwise.
    """
    return len(capability_schema_errors(cap)) == 0
