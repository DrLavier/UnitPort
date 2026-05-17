"""MjActor — MJCF embodiment loader for Mission runtime.

Owns the robot side of an MjSimEnv: an MJCF file path, the loaded
``mujoco.MjModel`` (and a fresh ``mujoco.MjData``), the joint name
ordering, and helpers for SkillManifest-driven action remapping.

This module is intentionally thin: it loads an MJCF and exposes
the basic introspection PolicyRunner needs (joint_names, mj_model,
mj_data, ctrl write surface). Full embodiment-matching logic
(``remap_to_manifest``) lands in P4 alongside ``obs_adapter.py``.

Resolution rules
----------------
``MjActor.from_sku(sku)`` delegates to
:class:`application.service.robot_assets.service.RobotAssetService` for
MJCF path resolution — this module owns no brand-to-folder mapping.

``MjActor.from_path(mjcf_path)`` accepts an arbitrary MJCF file (project
or shared asset).

Both routes yield an actor that can stand on its own.

Ported from DEMO/src/system/runtime/simulation/mujoco/mj_actor.py with
``from_menagerie``/``from_asset`` (DEMO RobotAsset registry-driven)
collapsed into a single :meth:`from_sku` against RELEASE's
``RobotAssetService``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional


@dataclass
class MjActor:
    """An MJCF embodiment ready to be placed inside an MjSimEnv.

    Attributes
    ----------
    robot_id:
        Human-readable identifier (``"go2"``, ``"h1"``, …) or canonical SKU.
        Used for diagnostics and SkillManifest matching.
    mjcf_path:
        Absolute path of the MJCF file the actor was loaded from.
    mj_model:
        ``mujoco.MjModel`` instance. Typed ``Any`` to avoid hard-importing
        mujoco at module load time.
    mj_data:
        Fresh ``mujoco.MjData`` paired with ``mj_model``.
    joint_names:
        Joint names in **qpos order** (i.e. as the model itself reports
        them). PolicyRunner permutes between bundle order and this order
        via ``JointSpace`` so we don't need to second-guess the layout.
    """

    robot_id: str
    mjcf_path: Path
    mj_model: Any
    mj_data: Any
    joint_names: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_path(cls, mjcf_path: Path | str, robot_id: Optional[str] = None) -> "MjActor":
        """Load an actor from an explicit MJCF path."""
        import mujoco

        path = Path(mjcf_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"MjActor: MJCF not found at {path}")

        mj_model = mujoco.MjModel.from_xml_path(str(path))
        mj_data = mujoco.MjData(mj_model)
        joint_names = _extract_joint_names(mj_model)

        return cls(
            robot_id=robot_id or path.parent.name,
            mjcf_path=path,
            mj_model=mj_model,
            mj_data=mj_data,
            joint_names=joint_names,
        )

    @classmethod
    def from_sku(cls, sku: str) -> "MjActor":
        """Load an actor from a robot SKU via RELEASE's RobotAssetService.

        Resolves ``sku`` to a ``RobotAsset`` and uses its ``mjcf_path``.
        Raises ``KeyError`` if the SKU is unknown to ``registers.robots``,
        ``FileNotFoundError`` if no MJCF is registered for it.
        """
        from application.service.robot_assets.service import (
            get_robot_asset_service,
        )

        svc = get_robot_asset_service()
        asset = svc.resolve(sku)
        if asset is None:
            raise KeyError(
                f"MjActor.from_sku: SKU {sku!r} unknown to registers.robots"
            )
        if asset.mjcf_path is None:
            raise FileNotFoundError(
                f"MjActor.from_sku: SKU {sku!r} has no MJCF registered "
                f"(asset_status={asset.asset_status('MJCF').name})"
            )
        return cls.from_path(asset.mjcf_path, robot_id=sku)

    # ------------------------------------------------------------------
    # Introspection helpers used by MjSimEnv / PolicyRunner
    # ------------------------------------------------------------------

    @property
    def n_qpos(self) -> int:
        return int(self.mj_model.nq)

    @property
    def n_ctrl(self) -> int:
        return int(self.mj_model.nu)

    def reset_state(self) -> None:
        """Reset mj_data to its keyframe / zero pose."""
        import mujoco

        mujoco.mj_resetData(self.mj_model, self.mj_data)


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _extract_joint_names(mj_model: Any) -> List[str]:
    """Return joint names in the order MuJoCo iterates them.

    Mirrors :func:`application.service.runtime.policy.joint_space.joint_spaces_from_mj_model`
    name extraction so the orderings stay aligned (Phase 3 delivers that module).
    """
    import mujoco

    names: List[str] = []
    for j in range(int(mj_model.njnt)):
        try:
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j)
        except Exception:
            name = None
        names.append(name or f"joint_{j}")
    return names
