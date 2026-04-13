"""RobotAsset dataclass + in-process registry.

Per AMP_design.yaml §3.custom_robot_assets:

- Receive user-supplied .usd / .xml (MJCF) / .urdf files. **No conversion.**
- Parse metadata (joint_names, dof_count, link tree, …) via the parsers/
  package and validate consistency via validator.py.
- Hand training (Isaac Lab) and sim2sim (MuJoCo) the same metadata view.
- ``joint_names`` / ``dof_count`` are the **single source of truth** for
  downstream code; never duplicated.

This module is intentionally process-local (a module-level dict). Projects
load their assets at startup; cross-process sharing (e.g. between the main
venv and the Isaac Sim subprocess) is handled by re-running registration
in the child process from the same project files, NOT by IPC.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RobotAssetValidationError(ValueError):
    """Raised when a candidate RobotAsset fails validation.

    Carries a human-readable diff (set by ``validator.py``) so the Canvas
    layer can show it directly to the user.
    """


# ---------------------------------------------------------------------------
# RobotAsset
# ---------------------------------------------------------------------------


@dataclass
class RobotAsset:
    """User-supplied robot asset + parsed metadata.

    Invariants (enforced by ``validator.py``, NOT by this dataclass):

    - At least one of ``usd_path`` / ``mjcf_path`` is set (otherwise neither
      training nor sim2sim has anything to load).
    - If both ``usd_path`` and ``mjcf_path`` are set, their joint name
      orderings must match (case- and order-sensitive). Mismatches are
      surfaced as a ``RobotAssetValidationError`` with a diff.
    - ``joint_names`` is parsed from the primary metadata source (mjcf
      preferred, urdf fallback; see ``validator.py``).
    """

    asset_id: str
    family: str  # quadruped | biped | wheeled | manipulator
    usd_path: Optional[Path] = None
    #: Isaac Lab Nucleus URL (e.g. ``isaac_lab_assets://robots/unitree/go2/go2.usd``).
    #: Stored as a plain string because ``pathlib.Path`` misparses
    #: ``scheme://`` on Windows. ``resolve_for_training`` prefers
    #: ``usd_path`` when set, otherwise falls back to ``usd_url``.
    usd_url: str = ""
    mjcf_path: Optional[Path] = None
    urdf_path: Optional[Path] = None

    # Parsed metadata (populated by parsers/ + validator)
    joint_names: List[str] = field(default_factory=list)
    dof_count: int = 0
    foot_link_names: List[str] = field(default_factory=list)
    default_joint_pos: Dict[str, float] = field(default_factory=dict)
    base_link: str = ""
    mass: float = 0.0

    # Bookkeeping
    source_meta: Dict[str, Any] = field(default_factory=dict)
    """Which file was used as the primary metadata source. Keys:
    - ``primary``: 'mjcf' | 'urdf' | 'usd' | 'none'
    - ``warnings``: List[str] (e.g. usd-only assets, deferred validation)
    """

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def has_training_asset(self) -> bool:
        """True iff either a usd file or a Nucleus URL is present
        (training side requirement)."""
        return self.usd_path is not None or bool(self.usd_url)

    def has_mujoco_asset(self) -> bool:
        """True iff a mjcf file is present (sim2sim side requirement)."""
        return self.mjcf_path is not None

    def to_dict(self) -> Dict[str, Any]:
        """Plain-dict representation for serialization into TrainingSpec / canvas."""
        return {
            "asset_id": self.asset_id,
            "family": self.family,
            "usd_path": str(self.usd_path) if self.usd_path else "",
            "usd_url": self.usd_url,
            "mjcf_path": str(self.mjcf_path) if self.mjcf_path else "",
            "urdf_path": str(self.urdf_path) if self.urdf_path else "",
            "joint_names": list(self.joint_names),
            "dof_count": int(self.dof_count),
            "foot_link_names": list(self.foot_link_names),
            "default_joint_pos": dict(self.default_joint_pos),
            "base_link": self.base_link,
            "mass": float(self.mass),
            "source_meta": dict(self.source_meta),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RobotAsset":
        def _path(v: Any) -> Optional[Path]:
            return Path(str(v)) if v else None

        return cls(
            asset_id=str(d.get("asset_id", "")),
            family=str(d.get("family", "quadruped")),
            usd_path=_path(d.get("usd_path")),
            usd_url=str(d.get("usd_url", "") or ""),
            mjcf_path=_path(d.get("mjcf_path")),
            urdf_path=_path(d.get("urdf_path")),
            joint_names=list(d.get("joint_names") or []),
            dof_count=int(d.get("dof_count", 0) or 0),
            foot_link_names=list(d.get("foot_link_names") or []),
            default_joint_pos=dict(d.get("default_joint_pos") or {}),
            base_link=str(d.get("base_link", "")),
            mass=float(d.get("mass", 0.0) or 0.0),
            source_meta=dict(d.get("source_meta") or {}),
        )


# ---------------------------------------------------------------------------
# Process-local registry
# ---------------------------------------------------------------------------


_REGISTRY: Dict[str, RobotAsset] = {}


def register_asset(asset: RobotAsset) -> None:
    """Validate then register a RobotAsset.

    Validation runs through ``validator.py`` (which calls the parsers as
    needed). On failure raises ``RobotAssetValidationError`` with a
    human-readable diff in the message.

    Re-registering the same ``asset_id`` overwrites the prior entry — this
    is intentional so that hot-reload from the Canvas just works.
    """
    # Local import to avoid a circular at module load time. registry.py is
    # imported by validator.py for the dataclass + error type.
    from src.system.training.robot_assets.validator import validate_asset

    validate_asset(asset)
    _REGISTRY[asset.asset_id] = asset


def get_asset(asset_id: str) -> RobotAsset:
    """Look up a registered asset. Raises KeyError if absent."""
    if asset_id not in _REGISTRY:
        raise KeyError(
            f"RobotAsset '{asset_id}' is not registered. "
            f"Known assets: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[asset_id]


def has_asset(asset_id: str) -> bool:
    return asset_id in _REGISTRY


def list_assets(family: Optional[str] = None) -> List[RobotAsset]:
    """Return all registered assets, optionally filtered by family."""
    items = list(_REGISTRY.values())
    if family:
        items = [a for a in items if a.family == family]
    return sorted(items, key=lambda a: a.asset_id)


def resolve_for_training(asset_id: str) -> str:
    """Return the USD reference for the Isaac Lab training side.

    Prefers an on-disk ``usd_path`` when set, otherwise falls back to a
    Nucleus URL (``usd_url``) which Isaac Sim resolves at runtime. Both
    are returned as plain strings so downstream code can substitute into
    ``UsdFileCfg(usd_path="...")`` without worrying about the distinction.

    Raises ``RobotAssetValidationError`` if the asset has neither.
    """
    asset = get_asset(asset_id)
    if asset.usd_path is not None:
        return str(asset.usd_path)
    if asset.usd_url:
        return asset.usd_url
    raise RobotAssetValidationError(
        f"RobotAsset '{asset_id}' has no usd_path or usd_url. "
        f"Add a .usd file, or point the asset at an "
        f"isaac_lab_assets:// Nucleus URL."
    )


def resolve_for_mujoco(asset_id: str) -> Path:
    """Return the .xml (MJCF) path for the sim2sim side."""
    asset = get_asset(asset_id)
    if asset.mjcf_path is None:
        raise RobotAssetValidationError(
            f"RobotAsset '{asset_id}' has no mjcf_path. "
            f"Upload a .xml MJCF file in the RobotAssetNode to enable MuJoCo sim2sim."
        )
    return asset.mjcf_path


def clear_registry() -> None:
    """Drop all registered assets. Test-only."""
    _REGISTRY.clear()


def rescan() -> "DiscoveryReport":
    """Run the discovery scanner on the default locations.

    Convenience wrapper over
    ``src.system.training.robot_assets.discovery.walk_and_register()``.
    Idempotent — safe to call from UI widget render callbacks to pick
    up files the user dropped into ``custom_mods/archives/`` without
    restarting the app.

    Returns
    -------
    DiscoveryReport
        Lists of registered + rejected candidates. The registry is
        mutated in place.
    """
    from src.system.training.robot_assets.discovery import (
        walk_and_register, DiscoveryReport,
    )
    return walk_and_register()
