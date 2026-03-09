"""Capability schema validation — Phase 5 STAGE-03.

Defines the canonical schema for the dict returned by adapter.capabilities()
and provides two public validation functions:

    validate_capability(cap)         -> bool
    capability_schema_errors(cap)    -> List[str]

Design constraints:
    - Pure Python; no external validation libraries.
    - No file I/O.
    - Extra keys beyond the required set are tolerated (forward compatibility).
    - List element types for actions/sensors/required_settings are also checked.
"""

from __future__ import annotations

from typing import Any, Dict, List


# ── Schema definition ──────────────────────────────────────────────────────
#
# Maps each required top-level key to its expected Python type.
# Sourced from README §8.1 (Phase 5 schema contract).

CAPABILITY_SCHEMA: Dict[str, type] = {
    "brand":             str,   # e.g. "unitree" | "bostiondynamics" | "xiaomi"
    "adapter":           str,   # e.g. "unitree_sdk2" | "spot_sdk" | "cyberdog_sdk"
    "actions":           list,  # List[str] — canonical action names
    "sensors":           list,  # List[str] — sensor data keys
    "flags":             dict,  # Dict[str, Any] — feature flags
    "required_settings": list,  # List[str] — settings keys needed before execution
}

# Keys whose list elements must all be str
_LIST_STR_KEYS = ("actions", "sensors", "required_settings")


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
