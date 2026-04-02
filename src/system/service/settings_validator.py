"""Settings validator — Phase 5 STAGE-05.

Validates a configuration dict against the per-brand settings schema defined
in ``system.service.settings_schema``.

Public API:

    validate_settings(brand, config) -> Dict[str, Any]

Returns a ``ValidationResult`` dict (README §8.5):

    {
        "status":  "ok" | "error",   # "ok" only when all three lists are empty
        "brand":   str,
        "errors":  List[str],        # human-readable summaries of all violations
        "missing": List[str],        # truly-required keys absent from config
        "invalid": List[str],        # keys with wrong type or disallowed choice
    }

Design notes:
    - Pure Python; no external libraries.
    - No file I/O.
    - bool-for-int guard: ``True``/``False`` are rejected for ``type=int`` entries
      because ``bool`` is a subclass of ``int`` in Python and would otherwise
      pass the ``isinstance`` check silently.
    - Optional entries (``required=False``) with a usable ``default`` are not
      flagged as missing when absent; they *are* flagged as ``invalid`` if
      present with the wrong type or a disallowed choice value.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .settings_schema import get_settings_schema


def validate_settings(brand: str, config: Any) -> Dict[str, Any]:
    """Validate *config* against the settings schema for *brand*.

    Args:
        brand:  Brand key (one of ``KNOWN_BRANDS``).
        config: Configuration dict to validate (typically
                ``LifecyclePolicy.session_config``).

    Returns:
        ``ValidationResult`` dict with keys
        ``status``, ``brand``, ``errors``, ``missing``, ``invalid``.
    """
    missing: List[str] = []
    invalid: List[str] = []
    errors:  List[str] = []

    # ── Guard: config must be a dict ──────────────────────────────────────
    if not isinstance(config, dict):
        errors.append(
            f"config must be a dict, got {type(config).__name__!r}"
        )
        return _result(brand, errors, missing, invalid)

    # ── Guard: unknown brand ───────────────────────────────────────────────
    try:
        schema = get_settings_schema(brand)
    except ValueError as exc:
        errors.append(str(exc))
        return _result(brand, errors, missing, invalid)

    # ── Per-key validation ─────────────────────────────────────────────────
    for key, entry in schema.items():
        required = entry["required"]
        default  = entry["default"]
        typ      = entry["type"]
        choices  = entry["choices"]

        if key not in config:
            # Truly required (no usable default) → missing
            if required and default is None:
                missing.append(key)
                errors.append(f"required setting missing: {key!r}")
            # Optional with default → absent is fine
            continue

        value = config[key]

        # ── Type check ─────────────────────────────────────────────────────
        # Special case: bool is a subclass of int in Python — reject bool
        # values for entries whose expected type is int.
        if typ is int and isinstance(value, bool):
            invalid.append(key)
            errors.append(
                f"{key!r} must be int (not bool), "
                f"got {type(value).__name__!r} value {value!r}"
            )
            continue

        if not isinstance(value, typ):
            invalid.append(key)
            errors.append(
                f"{key!r} must be {typ.__name__}, "
                f"got {type(value).__name__!r} value {value!r}"
            )
            continue

        # ── Choices check ──────────────────────────────────────────────────
        # Backward compatibility: legacy mission files may carry robot_type
        # values that are no longer present in current brand catalogs.
        if key == "robot_type" and choices is not None and value not in choices:
            continue

        if choices is not None and value not in choices:
            invalid.append(key)
            errors.append(
                f"{key!r} value {value!r} is not in allowed choices: {choices!r}"
            )

    return _result(brand, errors, missing, invalid)


# ── Internal helper ────────────────────────────────────────────────────────


def _result(
    brand:   str,
    errors:  List[str],
    missing: List[str],
    invalid: List[str],
) -> Dict[str, Any]:
    """Build a ValidationResult dict."""
    return {
        "status":  "ok" if not errors else "error",
        "brand":   brand,
        "errors":  errors,
        "missing": missing,
        "invalid": invalid,
    }
