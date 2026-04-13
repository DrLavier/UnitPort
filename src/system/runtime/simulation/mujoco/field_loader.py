"""field_loader — load MjField specs from project-local / shared YAML files.

Per knowledge_base/mission_design.yaml §7:

    project_local: projects/<slug>/scenarios/fields/<field_id>.yaml
    shared_global: shared/fields/<field_id>.yaml

Resolution rule (mirrors src/system/core/asset_resolver.py):
    1. project-local first (a project can override a shared field)
    2. shared global as fallback
    3. KeyError if neither has the field_id

The loader is intentionally schema-loose: it accepts any YAML that
parses to a dict and forwards the keys to :class:`MjField`. Unknown
top-level keys are preserved on the dataclass via the existing
``offbody_sensor_specs`` / ``randomization`` slots; richer fields (P5+)
are expected to add their own keys here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .mj_field import MjField


# In-tree shared field root. Project-local roots are passed in by the
# caller (typically through ProjectStore).
_SHARED_ROOT = Path(__file__).resolve().parents[5] / "shared" / "fields"


class FieldNotFoundError(KeyError):
    """Raised when a field_id cannot be resolved in any scope."""


def shared_root() -> Path:
    """Return the in-tree shared/fields/ root."""
    return _SHARED_ROOT


def list_shared_fields() -> List[str]:
    """Return field_ids of every shipped default."""
    if not _SHARED_ROOT.is_dir():
        return []
    return sorted(p.stem for p in _SHARED_ROOT.glob("*.yaml"))


def list_project_fields(project_root: Optional[Path]) -> List[str]:
    """Return field_ids declared inside a project's scenarios/fields/."""
    if project_root is None:
        return []
    project_dir = Path(project_root) / "scenarios" / "fields"
    if not project_dir.is_dir():
        return []
    return sorted(p.stem for p in project_dir.glob("*.yaml"))


def list_all_fields(project_root: Optional[Path] = None) -> List[str]:
    """Merged listing — project-local first, shared as fallback. Names
    that exist in both scopes appear once (project-local wins, with
    a ``(GLOB)`` suffix on the shared one for UI hint use).
    """
    local = set(list_project_fields(project_root))
    shared = set(list_shared_fields())
    return sorted(local | shared)


def resolve_field_path(
    field_id: str,
    project_root: Optional[Path] = None,
) -> Path:
    """Resolve a field_id to its YAML path.

    Project-local first, shared global second. Raises
    :class:`FieldNotFoundError` if neither scope has it.
    """
    if not field_id:
        raise FieldNotFoundError("field_id is empty")

    if project_root is not None:
        candidate = Path(project_root) / "scenarios" / "fields" / f"{field_id}.yaml"
        if candidate.is_file():
            return candidate

    candidate = _SHARED_ROOT / f"{field_id}.yaml"
    if candidate.is_file():
        return candidate

    raise FieldNotFoundError(
        f"field_id '{field_id}' not found in project-local "
        f"({project_root}) or shared ({_SHARED_ROOT})"
    )


def load_field(
    field_id: str,
    project_root: Optional[Path] = None,
) -> MjField:
    """Load + parse a field YAML into an :class:`MjField`.

    Schema (loose; all keys optional except ``field_id``):
        field_id          : str
        base_terrain      : str  ("flat" | "stairs" | "obstacles" | ...)
        props             : list[dict]   geom specs (type/name/pos/size/rgba)
        lighting          : dict
        offbody_sensors   : list[dict]
        randomization     : dict
    """
    import yaml

    path = resolve_field_path(field_id, project_root=project_root)
    with path.open("r", encoding="utf-8") as fh:
        data: Dict[str, Any] = yaml.safe_load(fh) or {}

    if not isinstance(data, dict):
        raise ValueError(
            f"field yaml '{path}' must parse to a dict, got {type(data).__name__}"
        )

    # Pull keys with defaults that mirror MjField's dataclass.
    return MjField(
        field_id=str(data.get("field_id", field_id)),
        base_terrain=str(data.get("base_terrain", "flat")),
        props=list(data.get("props", []) or []),
        lighting=dict(data.get("lighting", {}) or {}),
        offbody_sensor_specs=list(data.get("offbody_sensors", []) or []),
        randomization=data.get("randomization"),
        heightfield=data.get("heightfield"),
    )
