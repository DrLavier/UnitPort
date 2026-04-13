"""MjActor — MJCF embodiment loader for Mission runtime.

Owns the robot side of an MjSimEnv: an MJCF file path, the loaded
``mujoco.MjModel`` (and a fresh ``mujoco.MjData``), the joint name
ordering, and helpers for SkillManifest-driven action remapping.

This module is intentionally thin in P1: it loads an MJCF and exposes
the basic introspection PolicyRunner needs (joint_names, mj_model,
mj_data, ctrl write surface). Full embodiment-matching logic
(``remap_to_manifest``) lands in P4 alongside ``obs_adapter.py``.

Resolution rules
----------------
``MjActor.from_menagerie(robot_id)`` looks up
``src/system/runtime/simulation/mujoco/menagerie/<robot_id>/scene.xml``.
``MjActor.from_path(mjcf_path)`` accepts an arbitrary MJCF file (project
or shared asset).

Both routes yield an actor that can stand on its own (the menagerie
``scene.xml`` files already include a flat ground), so a fresh Mission
canvas with no FieldNode still constructs a usable world.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# Default MJCF lookup root: in-tree menagerie mirror.
_MENAGERIE_ROOT = (
    Path(__file__).resolve().parent / "menagerie"
)

# Known robot_id → mjcf relative path. We prefer ``scene.xml`` (which
# wraps the bare robot xml in a flat-ground scene with lighting) over the
# bare ``<robot>.xml``, because Mission's StartNode default expects an
# immediately-usable world.
_MENAGERIE_SCENE_OVERRIDES: Dict[str, str] = {
    "go2": "unitree_go2/scene.xml",
    "g1": "unitree_g1/scene.xml",
    "h1": "unitree_h1/scene.xml",
    "a1": "unitree_a1/scene.xml",
}


@dataclass
class MjActor:
    """An MJCF embodiment ready to be placed inside an MjSimEnv.

    Attributes
    ----------
    robot_id:
        Human-readable identifier (``"go2"``, ``"h1"``, …). Used for
        diagnostics and SkillManifest matching.
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
    def from_menagerie(cls, robot_id: str) -> "MjActor":
        """Load an actor from the in-tree menagerie mirror.

        ``robot_id`` is the short canonical key (``"go2"``, ``"h1"``…),
        not the full menagerie folder name.
        """
        rel = _MENAGERIE_SCENE_OVERRIDES.get(robot_id)
        if rel is None:
            raise KeyError(
                f"MjActor: robot_id '{robot_id}' is not registered for the "
                f"menagerie loader. Known ids: {sorted(_MENAGERIE_SCENE_OVERRIDES)}"
            )
        return cls.from_path(_MENAGERIE_ROOT / rel, robot_id=robot_id)

    @classmethod
    def from_asset(cls, asset_id: str) -> "MjActor":
        """Load an actor from a registered ``RobotAsset`` (phase_1 path).

        Looks up the asset in
        :mod:`src.system.training.robot_assets.registry` and resolves its
        ``mjcf_path``. Raises ``RobotAssetValidationError`` if the asset
        exists but has no mjcf file (the user needs to upload one in the
        RobotAssetNode).

        This is the third construction route, sitting alongside
        ``from_menagerie`` (legacy menagerie ids) and ``from_path``
        (arbitrary file). All three return the same shape of actor.
        """
        from src.system.training.robot_assets import resolve_for_mujoco

        mjcf_path = resolve_for_mujoco(asset_id)
        return cls.from_path(mjcf_path, robot_id=asset_id)

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

    Mirrors :func:`src.system.policy.joint_space.joint_spaces_from_mj_model`
    name extraction so the orderings stay aligned.
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
