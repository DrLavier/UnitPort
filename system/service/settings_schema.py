"""Settings schema data model — Phase 5 STAGE-04.

Defines per-brand settings schemas as plain Python dicts.  No external files,
no INI dependency, no third-party libraries.

Public API:

    KNOWN_BRANDS        tuple of valid brand key strings
    get_settings_schema(brand) -> Dict[str, dict]
        Returns the merged global + brand-specific schema for the given brand.
        Raises ValueError for unknown brands.

Each returned schema is a ``Dict[setting_key, entry_dict]`` where every
entry_dict has exactly these fields (from README §8.2):

    {
        "type":        type,           # Python type (str / int / float / bool / list)
        "required":    bool,           # True if key must be present (no usable default)
        "default":     Any,            # None when required=True and no default exists
        "description": str,            # human-readable explanation
        "choices":     Optional[List], # allowed values; None = unrestricted
    }
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from models.brand_registry import BrandRegistry


# ── Known brands ────────────────────────────────────────────────────────────

KNOWN_BRANDS: Tuple[str, ...] = ("unitree", "bostiondynamics", "xiaomi")


# ── Entry builder ───────────────────────────────────────────────────────────

def _entry(
    typ: type,
    required: bool,
    default: Any,
    description: str,
    choices: Optional[List] = None,
) -> Dict[str, Any]:
    """Build a single settings schema entry dict."""
    return {
        "type":        typ,
        "required":    required,
        "default":     default,
        "description": description,
        "choices":     choices,
    }


# ── Global settings (all brands) ────────────────────────────────────────────
#
# Sourced from README §8.3.

_GLOBAL_SCHEMA: Dict[str, Dict[str, Any]] = {
    "brand": _entry(
        str, True, None,
        "Robot brand identifier.",
        choices=["unitree", "bostiondynamics", "xiaomi"],
    ),
    "robot_type": _entry(
        str, True, None,
        "Robot model type within the brand (e.g. 'go2', 'spot', 'cyberdog').",
        choices=[],
    ),
    "connection_mode": _entry(
        str, True, None,
        "Execution target: simulation environment or real hardware.",
        choices=["simulation", "hardware"],
    ),
    "timeout": _entry(
        int, True, None,
        "Maximum allowed execution time in seconds (positive integer).",
    ),
    "stop_policy": _entry(
        str, True, None,
        "How to stop the robot on abort or session teardown.",
        choices=["immediate", "graceful"],
    ),
    "safety_enabled": _entry(
        bool, True, None,
        "Whether safety interception checks are enforced during execution.",
    ),
}


# ── Unitree-specific settings ────────────────────────────────────────────────
#
# Sourced from README §8.4 (brand = "unitree").

_UNITREE_SCHEMA: Dict[str, Dict[str, Any]] = {
    "dds_domain_id": _entry(
        int, True, None,
        "DDS domain ID for Unitree SDK2 communication channel (0–232).",
    ),
    "network_interface": _entry(
        str, True, None,
        "Network interface name for SDK2 communication (e.g. 'eth0', 'wlan0').",
    ),
    "control_level": _entry(
        str, True, None,
        "SDK control level: high-level sport client or low-level motor control.",
        choices=["high", "low"],
    ),
    "mode_switch_policy": _entry(
        str, True, None,
        "How to handle control-level switching during an active session.",
        choices=["immediate", "graceful"],
    ),
}


# ── Spot-specific settings ───────────────────────────────────────────────────
#
# Sourced from README §8.4 (brand = "bostiondynamics").

_SPOT_SCHEMA: Dict[str, Dict[str, Any]] = {
    "hostname": _entry(
        str, True, None,
        "Spot robot IP address or resolvable hostname.",
    ),
    "auth_token": _entry(
        str, True, None,
        "Authentication token for Spot SDK access (bosdyn-client credential).",
    ),
    "timesync_required": _entry(
        bool, False, True,
        "Whether time synchronisation with the robot must complete before execution.",
    ),
    "lease_required": _entry(
        bool, False, True,
        "Whether a Spot body lease must be acquired and held before execution.",
    ),
    "safety_channel": _entry(
        str, True, None,
        "E-stop / safety channel identifier used for emergency-stop checks.",
    ),
    "power_policy": _entry(
        str, True, None,
        "Motor power-off policy applied on session teardown.",
        choices=["safe_power_off", "emergency_cut"],
    ),
}


# ── CyberDog-specific settings ───────────────────────────────────────────────
#
# Sourced from README §8.4 (brand = "xiaomi").

_CYBERDOG_SCHEMA: Dict[str, Dict[str, Any]] = {
    "ros_domain_id": _entry(
        int, True, None,
        "ROS2 DDS domain ID for CyberDog topic discovery (0–232).",
    ),
    "rmw_impl": _entry(
        str, True, None,
        "ROS2 middleware implementation to use (e.g. 'rmw_fastrtps_cpp').",
    ),
    "namespace": _entry(
        str, False, "",
        "ROS2 topic namespace prefix for CyberDog nodes (empty = root namespace).",
    ),
    "action_endpoints": _entry(
        list, True, None,
        "List of ROS2 action endpoint topic names the adapter connects to.",
    ),
    "timestamp_policy": _entry(
        str, True, None,
        "Strictness of ROS2 message timestamp validation.",
        choices=["strict", "relaxed"],
    ),
}


# ── Brand → schema registry ──────────────────────────────────────────────────

_BRAND_SCHEMAS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "unitree":         _UNITREE_SCHEMA,
    "bostiondynamics": _SPOT_SCHEMA,
    "xiaomi":          _CYBERDOG_SCHEMA,
}


_ROBOT_TYPE_FALLBACKS: Dict[str, List[str]] = {
    "unitree": ["go2", "a1", "b1", "b2", "h1"],
    "bostiondynamics": ["spot"],
    "xiaomi": ["cyberdog", "cyberdog2"],
}


def _dynamic_robot_type_choices(brand: str) -> List[str]:
    """Load robot_type choices for *brand* from BrandRegistry with fallback."""
    try:
        models = BrandRegistry().get_models(brand)
    except Exception:
        models = []

    cleaned = [m for m in models if m and m != "(no models)"]
    if cleaned:
        return cleaned
    return list(_ROBOT_TYPE_FALLBACKS.get(brand, []))


# ── Public API ───────────────────────────────────────────────────────────────


def get_settings_schema(brand: str) -> Dict[str, Dict[str, Any]]:
    """Return the merged settings schema for the given brand.

    Merges the global required settings with the brand-specific settings.
    Global keys always appear first; brand-specific keys follow.

    Args:
        brand: Brand key string — one of ``KNOWN_BRANDS``
               (``"unitree"``, ``"bostiondynamics"``, ``"xiaomi"``).

    Returns:
        Dict mapping setting key names to entry dicts.  Each entry has keys:
        ``type``, ``required``, ``default``, ``description``, ``choices``.

    Raises:
        ValueError: If ``brand`` is not a known brand key.
    """
    if brand not in _BRAND_SCHEMAS:
        raise ValueError(
            f"Unknown brand {brand!r}. "
            f"Known brands: {sorted(_BRAND_SCHEMAS)}"
        )
    # Return a fresh merged dict with per-brand dynamic robot_type choices.
    merged = {**_GLOBAL_SCHEMA, **_BRAND_SCHEMAS[brand]}
    merged = {k: dict(v) for k, v in merged.items()}
    merged["robot_type"]["choices"] = _dynamic_robot_type_choices(brand)
    return merged
