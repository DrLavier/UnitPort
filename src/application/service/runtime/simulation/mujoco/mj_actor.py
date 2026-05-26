# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

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

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

log = logging.getLogger(__name__)


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
        # Strip any built-in MJCF PD actuators (position / general affine)
        # down to plain motor (ctrl = direct torque). UnitPort's
        # PDController owns the PD math (mass-weighted Kp/Kd from
        # mujoco_gain_solver); double-layered PD against MJCF's stock
        # kp/kv was the root cause of the Spot "robot flies away" sim2sim
        # regression. See _neutralize_mjcf_actuators below for details.
        _neutralize_mjcf_actuators(mj_model, mjcf_label=str(path))
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


def _neutralize_mjcf_actuators(mj_model: Any, mjcf_label: str = "") -> None:
    """Convert all non-motor actuators on ``mj_model`` to plain motor form.

    Why this exists
    ---------------
    UnitPort's sim2sim PD framework (CLAUDE.md §1.10) routes ALL PD math
    through :class:`PDController`: mass-weighted Kp/Kd are derived from
    the canonical ``(omega_n, zeta)`` via ``mujoco_gain_solver`` and
    written into the bundle as ``mujoco_pd_gains`` / ``mujoco_pd_damping``.
    At runtime, ``PDController.compute()`` produces per-joint torques
    that get written to ``mj_data.ctrl[:]``.

    For ``<motor>`` actuators MuJoCo interprets ``ctrl[i]`` as a direct
    force/torque and applies it unchanged — this is the contract
    UnitPort assumes everywhere (Go2's MJCF is purely motor-based).

    For ``<position>`` / ``<general>`` actuators the MJCF carries its
    own ``kp`` / ``kv`` and MuJoCo interprets ``ctrl[i]`` as a
    **position target**, computing internally
    ``tau = kp*(ctrl - q) + kv*(0 - qvel)``. Boston Dynamics Spot's
    menagerie MJCF ships with ``<position kp="500" kv="40">``, which
    means:
      - UnitPort's PDController-computed torques (e.g. -13.86 N·m for
        a calf joint) get treated as a position target in radians;
      - MuJoCo's internal PD with stock kp=500 then drives the joint
        towards that "target", producing huge spurious forces;
      - the joint range clamp keeps the effective error bounded, but
        the resulting torques still launch the robot off the floor on
        step 0 ("robot flies away" sim2sim regression).

    This function neutralizes that path: every actuator with a
    non-NONE bias or non-identity gain is rewritten in-place to a
    motor (gain=1, bias=0, ctrllimited off because the original
    ctrlrange was angular, not torque-bounded). UnitPort's
    PDController is then the single PD authority — the only place
    Kp / Kd math happens.

    Idempotency
    -----------
    Re-running this is a no-op: motor-shape actuators are detected
    and skipped. The transform writes only to the model's actuator
    arrays; ``mj_data`` is unaffected (callers may build a fresh
    ``mujoco.MjData(mj_model)`` after this returns).
    """
    import mujoco

    nu = int(mj_model.nu)
    if nu == 0:
        return

    converted: List[str] = []
    sample_kp = sample_kv = 0.0

    motor_gain = int(mujoco.mjtGain.mjGAIN_FIXED)
    motor_bias = int(mujoco.mjtBias.mjBIAS_NONE)

    for i in range(nu):
        gaintype = int(mj_model.actuator_gaintype[i])
        biastype = int(mj_model.actuator_biastype[i])
        gainprm0 = float(mj_model.actuator_gainprm[i, 0])

        # Already a pure motor (gain=FIXED with gainprm[0]=1 + bias=NONE):
        # leave it alone so we don't perturb correctly-configured MJCFs.
        if (
            gaintype == motor_gain
            and biastype == motor_bias
            and abs(gainprm0 - 1.0) < 1e-9
        ):
            continue

        # Record the original kp/kv (only meaningful for position-style
        # actuators where biasprm == [0, -kp, -kv]) for the diagnostic
        # log line. Best-effort — non-standard biasprm layouts just
        # leave the sample values at zero.
        if biastype == int(mujoco.mjtBias.mjBIAS_AFFINE):
            orig_kp = -float(mj_model.actuator_biasprm[i, 1])
            orig_kv = -float(mj_model.actuator_biasprm[i, 2])
            if abs(orig_kp) > abs(sample_kp):
                sample_kp = orig_kp
            if abs(orig_kv) > abs(sample_kv):
                sample_kv = orig_kv

        # Capture actuator name for the log (best-effort).
        try:
            name = mujoco.mj_id2name(
                mj_model, int(mujoco.mjtObj.mjOBJ_ACTUATOR), i
            ) or f"actuator_{i}"
        except Exception:
            name = f"actuator_{i}"
        converted.append(str(name))

        # Rewrite to motor: gain=FIXED with unit scale, bias=NONE.
        mj_model.actuator_gaintype[i] = motor_gain
        mj_model.actuator_biastype[i] = motor_bias
        mj_model.actuator_gainprm[i, :] = 0.0
        mj_model.actuator_gainprm[i, 0] = 1.0
        mj_model.actuator_biasprm[i, :] = 0.0
        # The MJCF's ctrlrange was the joint angular range
        # (``inheritrange="1"``); for a motor it would be interpreted
        # as a torque range, which is meaningless. Disable the limit
        # — UnitPort's PDController applies its own ``effort_limit``.
        mj_model.actuator_ctrllimited[i] = 0

    if converted:
        preview = ", ".join(converted[:6])
        if len(converted) > 6:
            preview += f", … (+{len(converted) - 6} more)"
        log.info(
            "MjActor: neutralized %d MJCF actuator(s) → motor form "
            "(stripped built-in PD; UnitPort PDController owns PD math). "
            "Source MJCF: %s. Original kp_max≈%.1f kv_max≈%.1f. Affected: %s",
            len(converted),
            mjcf_label or "?",
            sample_kp,
            sample_kv,
            preview,
        )


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
