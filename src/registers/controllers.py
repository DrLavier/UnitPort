# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""registers.controllers -- input controller catalog (keyboard / gamepad).

Single source of truth for controller TYPES and each type's available inputs
(buttons / axes). Replaces the scattered hardcoded control constants
(``controller_panel._GAMEPAD_BUTTON_HW_ROWS``, ``gamepad_source.AXIS_INDEX``,
``controller_panel._KEYBOARD_ROWS``) so the binding UI renders each controller
type from real data -- a picker over a closed set -- instead of a freeform text
box the user has to type ``gamepad.button_a`` into.

``binding_mode`` drives how the UI binds an input on that controller type:
  * ``"pick"``    -- a fixed, closed input set; the UI shows a picker (gamepad).
  * ``"capture"`` -- an open input set; the UI captures whatever key the user
                     presses (keyboard).

Two types ship today (keyboard, gamepad); the gamepad catalog is the generic
SDL layout that ``gamepad_source`` polls. Specific pad brands (Xbox / PS /
Switch) are a future overlay -- add a type to the catalog, no consumer change.

Data layout
-----------
* Factory catalog: ``registers/data/controllers_catalog.json`` (read-only).
* User overlay: ``<USER_CONFIG_DIR>/registers/controllers_custom.json``
  (merged at load via :func:`merge_user_extensions`; user wins on id clash).

Fail-loud -- no silent fallbacks (CLAUDE.md §8)
----------------------------------------------
Malformed catalog / unknown ``binding_mode`` / duplicate input id / bad
``kind`` RAISE at load. Querying an unknown controller type RAISES.

API:
    load() -> int
    list_controllers() -> list[str]
    has_controller(type_id) -> bool
    get_controller(type_id) -> dict | None
    binding_mode(type_id) -> str
    list_inputs(type_id, kind=None) -> list[dict]   # {id,label,kind}
    input_label(type_id, input_id) -> str
    has_input(type_id, input_id) -> bool
    merge_user_extensions(catalog) -> int
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from unitport_sdk import Paths, log_warning, read_data

_DATA_DIR = Paths.REGISTERS_DIR / "data"
_DEFAULTS_PATH = _DATA_DIR / "controllers_catalog.json"

#: ``push_data`` / read path for the user overlay.
_USER_OVERLAY_REL = "registers/controllers_custom.json"


def _user_overlay_path():
    return Paths.USER_CONFIG_DIR / "registers" / "controllers_custom.json"


_VALID_KINDS = ("button", "axis")
_VALID_MODES = ("pick", "capture")

_catalog: Dict[str, Dict[str, Any]] = {}
_loaded = False


# =============================================================================
# Validation
# =============================================================================
def _validate_type(tid: str, spec: Any) -> Dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError(f"controllers[{tid!r}] must be an object, got {type(spec).__name__}")
    mode = str(spec.get("binding_mode", "")).strip().lower()
    if mode not in _VALID_MODES:
        raise ValueError(
            f"controllers[{tid!r}].binding_mode must be one of {list(_VALID_MODES)}, "
            f"got {mode!r}"
        )
    inputs: List[Dict[str, str]] = []
    seen = set()
    for raw in spec.get("inputs", []) or []:
        if not isinstance(raw, dict):
            raise ValueError(f"controllers[{tid!r}] has a non-object input entry")
        iid = str(raw.get("id", "")).strip()
        if not iid:
            raise ValueError(f"controllers[{tid!r}] has an input with no id")
        if iid in seen:
            raise ValueError(f"controllers[{tid!r}] duplicate input id {iid!r}")
        seen.add(iid)
        kind = str(raw.get("kind", "")).strip().lower()
        if kind not in _VALID_KINDS:
            raise ValueError(
                f"controllers[{tid!r}].inputs[{iid!r}].kind must be one of "
                f"{list(_VALID_KINDS)}, got {kind!r}"
            )
        inputs.append({"id": iid, "label": str(raw.get("label", iid)), "kind": kind})
    return {"name": str(spec.get("name", tid)), "binding_mode": mode, "inputs": inputs}


def _validate(catalog: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(tid): _validate_type(str(tid), spec) for tid, spec in catalog.items()}


# =============================================================================
# Load
# =============================================================================
def load() -> int:
    """Load the factory catalog + user overlay. Returns the controller-type count."""
    global _catalog, _loaded
    raw = read_data(_DEFAULTS_PATH)
    if not isinstance(raw, dict):
        raise ValueError(f"controllers catalog {_DEFAULTS_PATH} is not a JSON object")
    catalog = raw.get("controllers")
    if not isinstance(catalog, dict) or not catalog:
        raise ValueError(
            f"controllers catalog {_DEFAULTS_PATH} missing a non-empty 'controllers' object"
        )
    resolved = _validate(catalog)
    merge_user_extensions(resolved)   # user overlay (best-effort, id-clash = user wins)
    _catalog = resolved
    _loaded = True
    return len(_catalog)


def _ensure() -> None:
    if not _loaded:
        load()


def merge_user_extensions(catalog: Dict[str, Dict[str, Any]]) -> int:
    """Merge the optional user overlay into ``catalog`` in place (user wins on a
    type-id clash). Returns the number of types added/overridden. Absent overlay
    is a no-op; a malformed overlay WARNs and is skipped (it is user-authored, so
    a typo must not brick the app — but the factory catalog is never dropped)."""
    path = _user_overlay_path()
    try:
        if not path.exists():
            return 0
        raw = read_data(path)
    except Exception as exc:                                   # pragma: no cover
        log_warning(f"[controllers] user overlay read failed: {exc}")
        return 0
    over = raw.get("controllers") if isinstance(raw, dict) else None
    if not isinstance(over, dict):
        return 0
    n = 0
    for tid, spec in over.items():
        try:
            catalog[str(tid)] = _validate_type(str(tid), spec)
            n += 1
        except ValueError as exc:
            log_warning(f"[controllers] user overlay type {tid!r} rejected: {exc}")
    return n


# =============================================================================
# Query API
# =============================================================================
def list_controllers() -> List[str]:
    _ensure()
    return list(_catalog.keys())


def has_controller(type_id: str) -> bool:
    _ensure()
    return str(type_id) in _catalog


def get_controller(type_id: str) -> Optional[Dict[str, Any]]:
    _ensure()
    spec = _catalog.get(str(type_id))
    return {**spec, "inputs": [dict(i) for i in spec["inputs"]]} if spec else None


def binding_mode(type_id: str) -> str:
    _ensure()
    spec = _catalog.get(str(type_id))
    if spec is None:
        raise KeyError(f"unknown controller type {type_id!r} (known: {list(_catalog)})")
    return str(spec["binding_mode"])


def list_inputs(type_id: str, kind: Optional[str] = None) -> List[Dict[str, str]]:
    _ensure()
    spec = _catalog.get(str(type_id))
    if spec is None:
        raise KeyError(f"unknown controller type {type_id!r} (known: {list(_catalog)})")
    return [
        dict(i) for i in spec["inputs"]
        if kind is None or i["kind"] == kind
    ]


def has_input(type_id: str, input_id: str) -> bool:
    return any(i["id"] == str(input_id) for i in list_inputs(type_id))


def input_label(type_id: str, input_id: str) -> str:
    for i in list_inputs(type_id):
        if i["id"] == str(input_id):
            return i["label"]
    return str(input_id)
