"""Settings form descriptor layer — Cycle 2 STAGE-02.

Maps raw settings schema dicts (from ``system.service.settings_schema``) to
deterministic, UI-facing form descriptors.

Public API:

    WIDGET_TYPES                frozenset of valid widget_type strings
    descriptor_for_field(field_id, entry, order) -> dict
        Convert a single schema entry to a form descriptor.
    schema_to_form_descriptors(brand) -> List[dict]
        Return an ordered list of form descriptors for the given brand.
    compute_missing_settings(brand, current_config, required_settings) -> List[str]
        Return required field_ids that are absent or empty in current_config.

Each descriptor has exactly these keys:

    {
        "field_id":         str,              # original schema key
        "label":            str,              # human-readable display label
        "widget_type":      str,              # see WIDGET_TYPES
        "required":         bool,
        "default":          Any,              # None when required=True
        "choices":          Optional[List],   # None = unrestricted
        "description":      str,
        "order":            int,              # 0-based insertion order
        "group":            str,              # "required" | "optional"
        "validation_hints": str,              # inline operator hint; "" = unrestricted
    }

Field order policy
------------------
1. Global fields appear first in the fixed canonical order defined by
   ``_GLOBAL_FIELD_ORDER`` (brand, robot_type, connection_mode, timeout,
   stop_policy, safety_enabled).
2. Brand-specific fields follow in their schema definition order.
3. Order is stable across additive schema evolution — new fields appended to
   either list will always appear after existing fields.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from system.service.settings_schema import KNOWN_BRANDS, get_settings_schema


# ── Widget type registry ─────────────────────────────────────────────────────

#: All valid widget_type values produced by this module.
WIDGET_TYPES: frozenset = frozenset(
    {"text", "integer", "float", "bool", "choice", "list"}
)

_TYPE_TO_WIDGET: Dict[type, str] = {
    str:   "text",
    int:   "integer",
    float: "float",
    bool:  "bool",
    list:  "list",
}


# ── Label overrides for known abbreviations ──────────────────────────────────

_LABEL_OVERRIDES: Dict[str, str] = {
    "dds_domain_id":     "DDS Domain ID",
    "ros_domain_id":     "ROS Domain ID",
    "rmw_impl":          "RMW Implementation",
    "auth_token":        "Auth Token",
    "timesync_required": "Timesync Required",
    "safety_enabled":    "Safety Enabled",
}


# ── Global field canonical order ─────────────────────────────────────────────

#: Fixed order for the six global settings keys.  Brand-specific keys follow
#: after the last global key.
_GLOBAL_FIELD_ORDER: List[str] = [
    "brand",
    "robot_type",
    "connection_mode",
    "timeout",
    "stop_policy",
    "safety_enabled",
]


# ── Internal helpers ─────────────────────────────────────────────────────────

def _validation_hints(entry: Dict[str, Any]) -> str:
    """Derive an inline validation hint string from a schema entry.

    Rules (in priority order):
    - ``choices`` set  → ``"One of: a, b, c"``
    - ``type`` is int  → ``"Integer"``
    - ``type`` is float → ``"Decimal number"``
    - ``type`` is bool  → ``"true / false"``
    - ``type`` is list  → ``"List of values"``
    - ``type`` is str (no choices) → ``""``  (unrestricted)
    """
    choices = entry.get("choices")
    if choices:
        return "One of: " + ", ".join(str(c) for c in choices)
    typ = entry.get("type", str)
    return {
        int:   "Integer",
        float: "Decimal number",
        bool:  "true / false",
        list:  "List of values",
    }.get(typ, "")


def _field_label(field_id: str) -> str:
    """Return the display label for a field_id."""
    if field_id in _LABEL_OVERRIDES:
        return _LABEL_OVERRIDES[field_id]
    return field_id.replace("_", " ").title()


def _widget_type(entry: Dict[str, Any]) -> str:
    """Derive widget type string from a schema entry dict."""
    typ = entry.get("type", str)
    choices = entry.get("choices")
    if typ is str and choices:
        return "choice"
    return _TYPE_TO_WIDGET.get(typ, "text")


# ── Public API ───────────────────────────────────────────────────────────────

def descriptor_for_field(
    field_id: str,
    entry: Dict[str, Any],
    order: int = 0,
) -> Dict[str, Any]:
    """Build a single form descriptor from a schema entry.

    Args:
        field_id: The schema key name (e.g. ``"timeout"``).
        entry:    The schema entry dict with keys ``type``, ``required``,
                  ``default``, ``description``, ``choices``.
        order:    0-based position in the final descriptor list.

    Returns:
        Form descriptor dict with keys ``field_id``, ``label``,
        ``widget_type``, ``required``, ``default``, ``choices``,
        ``description``, ``order``, ``group``, ``validation_hints``.
    """
    is_required = bool(entry.get("required", False))
    return {
        "field_id":         field_id,
        "label":            _field_label(field_id),
        "widget_type":      _widget_type(entry),
        "required":         is_required,
        "default":          entry.get("default"),
        "choices":          entry.get("choices"),
        "description":      entry.get("description", ""),
        "order":            order,
        "group":            "required" if is_required else "optional",
        "validation_hints": _validation_hints(entry),
    }


def compute_missing_settings(
    brand: str,
    current_config: Dict[str, Any],
    required_settings: List[str],
) -> List[str]:
    """Return field_ids from *required_settings* that are not yet configured.

    A required setting is "missing" when:

    - The key is absent from *current_config*, **or**
    - It is a ``"text"`` / ``"choice"`` field whose value is an empty string, **or**
    - It is a ``"list"`` field whose value is an empty list.

    ``"integer"``, ``"float"``, and ``"bool"`` fields with zero / ``False``
    values are **not** considered missing — those are valid configured values.

    Args:
        brand:             Brand key used to look up field widget types.
        current_config:    Dict from ``SettingsPanel.get_current_config()``.
        required_settings: Field-ids to check (from ``adapter.capabilities()``).

    Returns:
        Ordered subset of *required_settings* that are missing/empty.
    """
    if not required_settings:
        return []

    # Build widget-type lookup; tolerate unknown brands gracefully.
    try:
        descriptors = schema_to_form_descriptors(brand)
        wt_map: Dict[str, str] = {d["field_id"]: d["widget_type"] for d in descriptors}
    except Exception:
        wt_map = {}

    missing: List[str] = []
    for fid in required_settings:
        if fid not in current_config:
            missing.append(fid)
            continue
        val = current_config[fid]
        wt  = wt_map.get(fid, "text")
        if wt in ("text", "choice") and val == "":
            missing.append(fid)
        elif wt == "list" and val == []:
            missing.append(fid)
        # "integer", "float", "bool" → zero / False are valid; never missing
    return missing


def schema_to_form_descriptors(brand: str) -> List[Dict[str, Any]]:
    """Return an ordered list of form descriptors for the given brand.

    Order policy:
    1. Global fields in ``_GLOBAL_FIELD_ORDER`` (positions 0–5).
    2. Brand-specific fields in schema definition order (positions 6+).

    Args:
        brand: One of ``KNOWN_BRANDS``
               (``"unitree"``, ``"bostiondynamics"``, ``"xiaomi"``).

    Returns:
        List of form descriptor dicts; see module docstring for key contract.

    Raises:
        ValueError: If ``brand`` is not a known brand.
    """
    schema = get_settings_schema(brand)  # raises ValueError for unknown brand

    descriptors: List[Dict[str, Any]] = []
    order = 0

    # Pass 1 — global fields in canonical order
    for fid in _GLOBAL_FIELD_ORDER:
        if fid in schema:
            descriptors.append(descriptor_for_field(fid, schema[fid], order))
            order += 1

    # Pass 2 — brand-specific fields not already emitted
    for fid, entry in schema.items():
        if fid not in _GLOBAL_FIELD_ORDER:
            descriptors.append(descriptor_for_field(fid, entry, order))
            order += 1

    return descriptors
