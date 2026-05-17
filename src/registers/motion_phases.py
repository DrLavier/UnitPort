"""registers.motion_phases — Motion phase registry / 动作 phase 注册表.

Concept
-------
A *motion phase* is a coarse behavioral grouping over the canonical
``motion_tag`` set used by :mod:`registers.commands`. Three factory phases
ship by default: ``static`` (stand), ``locomotion`` (walk/turn/backward/
strafe/pace), and ``agile`` (run/canter/jump). Phases serve two purposes:

1.  **Reward isolation.** Reward terms in the canvas ``rewards`` node may
    declare ``applies_to: [phase_id, ...]``. The training-spec compiler
    converts that list into a runtime mask keyed off the active task
    item's ``motion_tag``. Reward terms with ``applies_to == []`` apply
    unconditionally (every phase).
2.  **Coverage Badge.** Each phase carries a ``polarity_required`` hint
    (``negative`` / ``positive`` / ``both`` / ``any``) consumed by the
    Reward Coverage Evaluator to flag under-constrained phases — e.g. a
    ``static`` phase with no negative-weight reward triggers a red badge.

Data layout
-----------
* Factory defaults: ``data/motion_phases.json``  (read-only).
* User overlay   : ``~/UnitPort/registers/motion_phases_custom.json``
                   (merged on load via :func:`merge_user_extensions`;
                   never written into the project tree).

API
---
* :func:`load` — read defaults + overlay; returns phase count.
* :func:`reload` — drop caches and reload.
* :func:`list_phases` — every phase, user overrides win on ``id`` clash.
* :func:`get_phase` — fetch a single phase by id.
* :func:`resolve_phase` — map a ``motion_tag`` to its first matching phase id.
* :func:`phase_ids_for_motion_tags` — phase id set covering a tag iterable.
* :func:`fallback_phase_id` — the catch-all phase id for unmapped tags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from unitport_sdk import Paths, log_warning, read_data


_DATA_DIR = Paths.REGISTERS_DIR / "data"
_DEFAULTS_PATH = _DATA_DIR / "motion_phases.json"


def _user_overlay_path():
    """Lazy resolver for the user-overlay path."""
    return Paths.USER_CONFIG_DIR / "registers" / "motion_phases_custom.json"


# =============================================================================
# Dataclass
# =============================================================================


@dataclass
class MotionPhase:
    """A single motion-phase entry.

    Attributes
    ----------
    id:
        Stable slug (``static`` / ``locomotion`` / ``agile``). Unique
        within the registry.
    display_name_key:
        i18n key used by the canvas (lane header, Coverage Inspector).
    display_name_default:
        Fallback string when the i18n key is unresolved.
    motion_tags:
        Set of canonical motion tags this phase covers. A tag may appear
        in at most one phase; the first match wins on lookup.
    theme_slot:
        ``system.ini[Theme]`` slot name driving lane header / chip color.
        Never embed hex literals at the consumer side — always read via
        ``Config.get_color(phase.theme_slot)``.
    polarity_required:
        Coverage hint. One of ``"negative"``, ``"positive"``, ``"both"``,
        or ``"any"``. The Reward Coverage Evaluator uses this to decide
        Badge color when a phase is missing a polarity.
    description_key / description_default:
        i18n key + fallback for the Coverage Inspector long-form text.
    builtin:
        ``True`` for factory items, ``False`` for user-defined items.
    """

    id: str
    display_name_key: str
    display_name_default: str
    motion_tags: List[str]
    theme_slot: str
    polarity_required: str = "any"
    description_key: str = ""
    description_default: str = ""
    builtin: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "display_name_key": self.display_name_key,
            "display_name_default": self.display_name_default,
            "motion_tags": list(self.motion_tags),
            "theme_slot": self.theme_slot,
            "polarity_required": self.polarity_required,
            "description_key": self.description_key,
            "description_default": self.description_default,
            "builtin": bool(self.builtin),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MotionPhase":
        return cls(
            id=str(d["id"]),
            display_name_key=str(d.get("display_name_key", "")),
            display_name_default=str(d.get("display_name_default", d["id"])),
            motion_tags=[str(t) for t in (d.get("motion_tags", []) or []) if t],
            theme_slot=str(d.get("theme_slot", "main_c2")),
            polarity_required=str(d.get("polarity_required", "any")),
            description_key=str(d.get("description_key", "")),
            description_default=str(d.get("description_default", "")),
            builtin=bool(d.get("builtin", False)),
        )


# =============================================================================
# Module state
# =============================================================================


_state: Dict[str, Any] = {
    "loaded": False,
    "builtin": {},          # id -> MotionPhase (ordered insertion = priority)
    "user": {},             # id -> MotionPhase (always builtin=False)
    "order": [],            # canonical id ordering (builtin first, then user-only)
    "fallback": "",         # fallback phase id (catch-all bucket for unmapped tags)
}


# =============================================================================
# Load / merge
# =============================================================================


def load() -> int:
    """Read factory defaults + merge user overlay. Returns total phase count."""
    if _state["loaded"]:
        return len(_state["builtin"]) + len([k for k in _state["user"] if k not in _state["builtin"]])

    builtin, fallback = _read_phases_from(_DEFAULTS_PATH, force_builtin=True)
    user, user_fallback = _read_phases_from(_user_overlay_path(), force_builtin=False)
    _state["builtin"] = builtin
    _state["user"] = user
    _state["fallback"] = user_fallback or fallback or ""

    order: List[str] = list(builtin.keys())
    for pid in user.keys():
        if pid not in order:
            order.append(pid)
    _state["order"] = order

    _state["loaded"] = True
    return len(order)


def reload() -> int:
    _state["loaded"] = False
    _state["builtin"] = {}
    _state["user"] = {}
    _state["order"] = []
    _state["fallback"] = ""
    return load()


def merge_user_extensions(payload: Dict[str, Any]) -> int:
    """Merge an in-memory payload (already parsed JSON) into the registry.

    Used by ``RegistryHub._merge_user_overlays`` when the caller already
    has the parsed user JSON. Returns the number of phases merged in.
    """
    if not _state["loaded"]:
        load()
    if not isinstance(payload, dict):
        return 0
    phases = payload.get("phases", []) or []
    added = 0
    for entry in phases:
        if not isinstance(entry, dict):
            continue
        try:
            phase = MotionPhase.from_dict(entry)
        except Exception as exc:
            log_warning(
                f"[registers.motion_phases] skip invalid user phase "
                f"{entry.get('id', '?')!r}: {exc}"
            )
            continue
        phase.builtin = False
        _state["user"][phase.id] = phase
        if phase.id not in _state["order"]:
            _state["order"].append(phase.id)
        added += 1
    user_fallback = str(payload.get("fallback_phase", "") or "").strip()
    if user_fallback:
        _state["fallback"] = user_fallback
    return added


# =============================================================================
# Query API
# =============================================================================


def get_phase(phase_id: str) -> Optional[MotionPhase]:
    """User overlay wins on id clash."""
    if not _state["loaded"]:
        load()
    return _state["user"].get(phase_id) or _state["builtin"].get(phase_id)


def list_phases() -> List[MotionPhase]:
    """Return builtin + user phases in canonical order (user overrides win)."""
    if not _state["loaded"]:
        load()
    out: List[MotionPhase] = []
    for pid in _state["order"]:
        ph = _state["user"].get(pid) or _state["builtin"].get(pid)
        if ph is not None:
            out.append(ph)
    return out


def list_ids() -> List[str]:
    if not _state["loaded"]:
        load()
    return list(_state["order"])


def resolve_phase(motion_tag: str) -> Optional[str]:
    """Return the phase id whose ``motion_tags`` contains ``motion_tag``.

    Lookup is ordered: builtin phases first (insertion order), then any
    user-only phases. A tag missing from every phase returns ``None``
    (callers may fall back to :func:`fallback_phase_id`).
    """
    if not _state["loaded"]:
        load()
    if not motion_tag:
        return None
    needle = str(motion_tag).strip()
    if not needle:
        return None
    for pid in _state["order"]:
        ph = _state["user"].get(pid) or _state["builtin"].get(pid)
        if ph is None:
            continue
        if needle in ph.motion_tags:
            return ph.id
    return None


def phase_ids_for_motion_tags(motion_tags: Iterable[str]) -> List[str]:
    """Resolve every tag to its phase id, deduplicated, in canonical order.

    Unmapped tags are skipped (they do not silently land in the fallback;
    the fallback bucket is reserved for explicit runtime use).
    """
    if not _state["loaded"]:
        load()
    seen: Dict[str, None] = {}
    for tag in motion_tags or []:
        pid = resolve_phase(tag)
        if pid and pid not in seen:
            seen[pid] = None
    # Preserve the canonical phase ordering.
    return [pid for pid in _state["order"] if pid in seen]


def fallback_phase_id() -> str:
    """The catch-all phase id used when a tag fails to resolve.

    Returns ``""`` when no fallback is configured — callers should treat
    this as 'do not auto-bucket; surface the unmapped tag in the UI'.
    """
    if not _state["loaded"]:
        load()
    return _state["fallback"]


# =============================================================================
# Internal: read phases from a JSON path
# =============================================================================


def _read_phases_from(path, *, force_builtin: bool):
    """Read ``{"phases": [...], "fallback_phase": "..."}`` payload shape.

    Returns ``(dict[id -> MotionPhase], fallback_phase_id)``. Missing
    files or malformed payloads yield ``({}, "")`` (never raises).
    """
    if not path.exists():
        return {}, ""
    try:
        data = read_data(path)
    except Exception as exc:
        log_warning(f"[registers.motion_phases] failed to read {path}: {exc}")
        return {}, ""
    if isinstance(data, dict):
        phases = data.get("phases", []) or []
        fallback = str(data.get("fallback_phase", "") or "").strip()
    elif isinstance(data, list):
        phases = data
        fallback = ""
    else:
        log_warning(
            f"[registers.motion_phases] unexpected payload shape in {path}: "
            f"{type(data).__name__}"
        )
        return {}, ""
    out: Dict[str, MotionPhase] = {}
    for entry in phases:
        if not isinstance(entry, dict):
            continue
        try:
            phase = MotionPhase.from_dict(entry)
        except Exception as exc:
            log_warning(
                f"[registers.motion_phases] skip invalid phase "
                f"{entry.get('id', '?')!r} from {path.name}: {exc}"
            )
            continue
        phase.builtin = bool(force_builtin)
        out[phase.id] = phase
    return out, fallback


__all__ = [
    "MotionPhase",
    "load",
    "reload",
    "merge_user_extensions",
    "get_phase",
    "list_phases",
    "list_ids",
    "resolve_phase",
    "phase_ids_for_motion_tags",
    "fallback_phase_id",
]
