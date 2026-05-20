"""registers.commands — 训练指令模板注册表 / Training-command (velocity-template) registry.

Holds the catalog of trainable behavior patterns (Stand / Walk / Turn /
Pace + 7 directional defaults) that the unified Training Motion node
picks from. Each entry is a ``TrainingItem`` carrying:

* a canonical motion tag (see ``scripts.training_motion.labels``);
* a command-channel template — ``{channel: (low, high)}`` ranges at
  ``speed = 1`` that the UI scales by the Speed slider;
* a dominant speed axis for the simplified Speed slider;
* optional zeroed channels and a gait hint.

Data layout / Sources:

* Factory defaults: ``data/commands_defaults.json``  (read-only)
* User overlay   : ``<USER_CONFIG_DIR>/registers/commands_custom.json``
                   (merged on load via ``merge_user_extensions``;
                   written by ``save_user_item`` / ``delete_user_item``
                   through ``Storage.push_data`` — never into the
                   project tree).

API:
    load() -> int
    get_item(item_id) -> TrainingItem | None
    list_items() -> list[TrainingItem]                # builtin + user, user wins on id clash
    list_ids() -> list[str]
    builtin_items() -> dict[str, TrainingItem]
    load_user_items() -> dict[str, TrainingItem]
    save_user_item(item) -> bool
    delete_user_item(item_id) -> bool
    merge_user_extensions(items) -> int
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from unitport_sdk import Paths, log_warning, push_data, read_data


_DATA_DIR = Paths.REGISTERS_DIR / "data"
_DEFAULTS_PATH = _DATA_DIR / "commands_defaults.json"

#: Path passed to ``push_data`` when the user creates / edits / deletes
#: a custom command. Resolves to ``USER_CONFIG_DIR/registers/commands_custom.json``.
_USER_OVERLAY_REL = "registers/commands_custom.json"


def _user_overlay_path():
    """Lazy resolver for the user-overlay path (matches other registers)."""
    return Paths.USER_CONFIG_DIR / "registers" / "commands_custom.json"


# =============================================================================
# Dataclass
# =============================================================================


@dataclass(frozen=False)  # mutable because user items can be edited
class TrainingItem:
    """A single trainable behavior pattern entry.

    Attributes
    ----------
    id:
        Stable slug (e.g. ``"walk"``). Unique within the registry.
    display_name:
        Human-readable label shown in UI.
    default_motion_tag:
        Canonical motion tag that maps into ``scripts.training_motion.labels``.
    command_template:
        Mapping of command-channel name → ``(low, high)`` range at
        ``speed = 1``. The UI scales these by the Speed slider.
    speed_channel:
        Dominant axis for the simplified Speed slider (one of the
        keys in ``command_template``).
    zero_channels:
        Channels that are force-zeroed regardless of the Speed slider
        (e.g. Stand zeros all velocity channels).
    gait_hint:
        Optional gait metadata (frequency, type) consumed by the
        training-spec compiler.
    description:
        Long-form description shown in the inspector panel.
    builtin:
        ``True`` for factory items, ``False`` for user-defined items.
    """

    id: str
    display_name: str
    default_motion_tag: str
    command_template: Dict[str, Tuple[float, float]]
    speed_channel: str
    zero_channels: List[str] = field(default_factory=list)
    gait_hint: Optional[Dict[str, Any]] = None
    description: str = ""
    builtin: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (JSON / YAML safe).

        Tuples in ``command_template`` are stored as 2-element lists.
        """
        return {
            "id": self.id,
            "display_name": self.display_name,
            "default_motion_tag": self.default_motion_tag,
            "command_template": {
                ch: [float(lo), float(hi)]
                for ch, (lo, hi) in self.command_template.items()
            },
            "speed_channel": self.speed_channel,
            "zero_channels": list(self.zero_channels),
            "gait_hint": dict(self.gait_hint) if self.gait_hint else None,
            "description": self.description,
            "builtin": bool(self.builtin),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingItem":
        """Deserialize from a plain dict.

        Accepts either list- or tuple-valued ranges in
        ``command_template``. Missing optional fields fall back to
        dataclass defaults.
        """
        raw_template = d.get("command_template", {}) or {}
        command_template: Dict[str, Tuple[float, float]] = {}
        for ch, rng in raw_template.items():
            if rng is None:
                continue
            try:
                lo, hi = rng
                command_template[str(ch)] = (float(lo), float(hi))
            except (TypeError, ValueError):
                # Skip malformed entries rather than hard-failing.
                continue

        gait_hint_raw = d.get("gait_hint", None)
        gait_hint: Optional[Dict[str, Any]]
        if isinstance(gait_hint_raw, dict):
            gait_hint = dict(gait_hint_raw)
        else:
            gait_hint = None

        return cls(
            id=str(d["id"]),
            display_name=str(d.get("display_name", d["id"])),
            default_motion_tag=str(d.get("default_motion_tag", "")),
            command_template=command_template,
            speed_channel=str(d.get("speed_channel", "")),
            zero_channels=list(d.get("zero_channels", []) or []),
            gait_hint=gait_hint,
            description=str(d.get("description", "")),
            builtin=bool(d.get("builtin", False)),
        )


# =============================================================================
# Module state
# =============================================================================

_state: Dict[str, Any] = {
    "loaded": False,
    "builtin": {},   # id -> TrainingItem
    "user": {},      # id -> TrainingItem (always builtin=False after coercion)
}


# =============================================================================
# Load / merge
# =============================================================================


def load() -> int:
    """Read factory defaults + merge user overlay. Returns total entry count."""
    if _state["loaded"]:
        return len(_state["builtin"]) + len(_state["user"])

    _state["builtin"] = _read_items_from(_DEFAULTS_PATH, force_builtin=True)
    _state["user"] = _read_items_from(_user_overlay_path(), force_builtin=False)

    _state["loaded"] = True
    return len(_state["builtin"]) + len(_state["user"])


def reload() -> int:
    _state["loaded"] = False
    _state["builtin"] = {}
    _state["user"] = {}
    return load()


def merge_user_extensions(items: List[Dict[str, Any]]) -> int:
    """Merge an in-memory list of user-item dicts into the registry.

    Used by RegistryHub-style user overlay loading from a parsed JSON
    payload (in case a caller already read the file). Returns the number
    of items successfully merged.
    """
    if not _state["loaded"]:
        load()
    added = 0
    for entry in items or []:
        if not isinstance(entry, dict):
            continue
        try:
            item = TrainingItem.from_dict(entry)
        except Exception as exc:
            log_warning(
                f"[registers.commands] skip invalid user item "
                f"{entry.get('id', '?')!r}: {exc}"
            )
            continue
        item.builtin = False
        _state["user"][item.id] = item
        added += 1
    return added


# =============================================================================
# Query API
# =============================================================================


def get_item(item_id: str) -> Optional[TrainingItem]:
    """User overlay wins on id clash."""
    if not _state["loaded"]:
        load()
    return _state["user"].get(item_id) or _state["builtin"].get(item_id)


def list_items() -> List[TrainingItem]:
    """Return builtin + user items merged. User overrides on id clash."""
    if not _state["loaded"]:
        load()
    merged: Dict[str, TrainingItem] = dict(_state["builtin"])
    merged.update(_state["user"])
    return list(merged.values())


def list_ids() -> List[str]:
    if not _state["loaded"]:
        load()
    merged_ids = set(_state["builtin"].keys()) | set(_state["user"].keys())
    return sorted(merged_ids)


def builtin_items() -> Dict[str, TrainingItem]:
    """Return a fresh copy of the builtin sub-registry (safe to mutate)."""
    if not _state["loaded"]:
        load()
    return {
        item_id: TrainingItem.from_dict({**item.to_dict(), "builtin": True})
        for item_id, item in _state["builtin"].items()
    }


def load_user_items() -> Dict[str, TrainingItem]:
    """Return a fresh copy of the user overlay (always ``builtin=False``)."""
    if not _state["loaded"]:
        load()
    return {
        item_id: TrainingItem.from_dict({**item.to_dict(), "builtin": False})
        for item_id, item in _state["user"].items()
    }


# =============================================================================
# User-overlay persistence (push_data → Paths.USER_CONFIG_DIR)
# =============================================================================


def save_user_item(item: TrainingItem) -> bool:
    """Append or replace a user item by id; persist via ``push_data``.

    ``item.builtin`` is forced to ``False`` before writing. The whole
    user overlay is rewritten as a list payload.
    """
    if not _state["loaded"]:
        load()
    item.builtin = False
    _state["user"][item.id] = item
    return _flush_user_overlay()


def delete_user_item(item_id: str) -> bool:
    """Remove a user item by id and persist. Returns True iff something was removed."""
    if not _state["loaded"]:
        load()
    if item_id not in _state["user"]:
        return False
    del _state["user"][item_id]
    return _flush_user_overlay()


def _flush_user_overlay() -> bool:
    """Rewrite the user overlay as ``{ "items": [...] }``."""
    payload = {
        "schema": "commands/user",
        "items": [it.to_dict() for it in _state["user"].values()],
    }
    try:
        return bool(push_data(_USER_OVERLAY_REL, payload, channel="local"))
    except Exception as exc:
        log_warning(f"[registers.commands] failed to save user overlay: {exc}")
        return False


# =============================================================================
# Internal: read items from a JSON path, tolerating shape variants
# =============================================================================


def _read_items_from(path, *, force_builtin: bool) -> Dict[str, TrainingItem]:
    """Read either ``{ "items": [...] }`` or a bare list ``[...]``.

    Returns id -> TrainingItem. Missing files / malformed payloads
    yield ``{}`` (never raises).
    """
    if not path.exists():
        return {}
    try:
        data = read_data(path)
    except Exception as exc:
        log_warning(f"[registers.commands] failed to read {path}: {exc}")
        return {}

    if isinstance(data, dict):
        items = data.get("items", []) or []
    elif isinstance(data, list):
        items = data
    else:
        log_warning(
            f"[registers.commands] unexpected payload shape in {path}: "
            f"{type(data).__name__}"
        )
        return {}

    out: Dict[str, TrainingItem] = {}
    for entry in items:
        if not isinstance(entry, dict):
            continue
        try:
            item = TrainingItem.from_dict(entry)
        except Exception as exc:
            log_warning(
                f"[registers.commands] skip invalid item "
                f"{entry.get('id', '?')!r} from {path.name}: {exc}"
            )
            continue
        item.builtin = bool(force_builtin)
        out[item.id] = item
    return out


def to_node_entry(item: TrainingItem) -> Dict[str, Any]:
    """Convert a registry :class:`TrainingItem` into the per-item dict shape
    consumed by ``TrainingMotionNode.training_items`` / ``MotionConfig.training_items``.

    The node-shape carries ``{enabled, speed, clip, advanced}``;
    ``advanced`` holds the full registry envelope so downstream consumers
    (compiler, SB3 env command sampler, AMP launcher) have the canonical
    multi-channel ranges + speed_channel + zero_channels + gait_hint
    without re-resolving the registry.

    The ``speed`` field is the registry's ``speed_channel`` range — that's
    what the canvas Speed slider scales — so the node-side legacy "speed"
    list and the registry stay in lock-step.
    """
    spch = item.speed_channel or next(iter(item.command_template), "")
    speed_lo, speed_hi = item.command_template.get(spch, (0.0, 1.0))
    return {
        "enabled": True,
        "speed": [float(speed_lo), float(speed_hi)],
        "clip": None,
        "advanced": {
            "default_motion_tag": item.default_motion_tag,
            "command_template": {
                ch: [float(lo), float(hi)]
                for ch, (lo, hi) in item.command_template.items()
            },
            "speed_channel": item.speed_channel,
            "zero_channels": list(item.zero_channels),
            "gait_hint": dict(item.gait_hint) if item.gait_hint else None,
            "display_name": item.display_name,
            "description": item.description,
            "source": "registry",
        },
    }


def default_items_for_families(families: List[str]) -> Dict[str, Dict[str, Any]]:
    """Return the canonical default-item dict (id → node-shape entry) for a
    robot's families.

    Currently the registry's 11 builtin items are all locomotion templates
    (lin_vel_x / lin_vel_y / ang_vel_z) so they apply uniformly to every
    legged family; this helper still takes ``families`` so a future
    family-specific filter (manipulator commands etc.) can land without
    callers needing to change. Falls back to ``stand`` + ``walk`` when the
    registry is empty so the compile never produces a zero-item dict.
    """
    if not _state["loaded"]:
        load()
    items = list_items()
    if not items:
        log_warning(
            "[registers.commands] default_items_for_families: registry is empty; "
            "this should not happen if commands_defaults.json is shipped"
        )
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for it in items:
        out[it.id] = to_node_entry(it)
    return out


def enrich_user_item(item_id: str, user_value: Any) -> Dict[str, Any]:
    """Enrich a user-authored ``training_items[item_id]`` value with registry
    metadata. The returned dict is always node-shape; missing fields are
    populated from the registry entry for ``item_id`` when available.

    This is the H4-A canonical pattern: the canvas saves a thin user dict
    (typically ``{enabled, speed, clip}``) and the compiler enriches it
    with the registry's command_template / speed_channel / zero_channels
    / gait_hint at compile time. Unknown ``item_id`` (no registry entry)
    passes through unchanged so user-defined items still work.
    """
    base: Dict[str, Any] = (
        dict(user_value) if isinstance(user_value, dict) else {"enabled": True}
    )
    reg = get_item(item_id)
    if reg is None:
        # User-defined item not in registry — pass through with safe defaults.
        base.setdefault("enabled", True)
        base.setdefault("speed", [0.0, 1.0])
        base.setdefault("clip", None)
        base.setdefault("advanced", {})
        return base
    skeleton = to_node_entry(reg)
    out = dict(skeleton)
    # User overrides win for the top-level node fields.
    for key in ("enabled", "speed", "clip"):
        if key in base:
            out[key] = base[key]
    user_advanced = base.get("advanced") if isinstance(base.get("advanced"), dict) else {}
    if user_advanced:
        merged_adv = dict(skeleton["advanced"])
        merged_adv.update(user_advanced)
        out["advanced"] = merged_adv
    return out


__all__ = [
    "TrainingItem",
    "load",
    "reload",
    "merge_user_extensions",
    "get_item",
    "list_items",
    "list_ids",
    "builtin_items",
    "load_user_items",
    "save_user_item",
    "delete_user_item",
    "to_node_entry",
    "default_items_for_families",
    "enrich_user_item",
]
