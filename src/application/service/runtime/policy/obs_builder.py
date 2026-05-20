"""ObsBuilder — runtime observation vector builder (verbatim port from DEMO).

Phase 3 RELEASE port. Two adjustments to the file content:

1. The ``from src.system.training.obs_contracts import ...`` block is
   replaced with a tiny inline stub returning None. ``obs_contracts``
   carries DEMO's brand-specific legacy preset tables (community Go2 SAC
   34D, Isaac Lab velocity 48D); new IsaacLab/SB3 bundles ship a
   ``deploy_contract`` instead, which takes precedence and is the only
   path Phase 3 / Phase 4 exporters need. Legacy-preset bundles fall
   through to ``DEFAULT_COMPONENT_ORDER`` — the same final fallback as
   DEMO would have hit anyway.

2. ``manifest_schema`` lives at ``application.training.manifest_schema``
   in RELEASE, so the import path is updated.

Everything else — every regex, every contract-mode insertion-order
guarantee, the per-term scale/clip/history pipeline, the Isaac-Lab
``base_lin_vel_b = R^T base_lin_vel_w`` body-frame projection, the
height_scan ray-cast (R1/R3 invariants), the qpos→bundle JointSpace
permutation — is byte-for-byte identical to DEMO.
"""
from __future__ import annotations

import logging
import warnings
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from application.training.manifest_schema import CheckpointBundle

from .sim_env_context import SimEnvContext

if TYPE_CHECKING:
    from .deploy_contract import DeployContract
    from .joint_space import JointSpace

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# obs_contracts stub (legacy preset tables not ported; new bundles use
# deploy_contract instead — see module docstring).
# ---------------------------------------------------------------------------

PRESET_COMMUNITY_GO2_SAC_34D = "community_go2_sac_34d"
PRESET_ISAAC_LAB_GO2_VELOCITY_48D = "isaac_lab_go2_velocity_48d"


def get_obs_contract(preset_name: str) -> Optional[Dict[str, Any]]:  # noqa: ARG001
    """Stub — RELEASE does not ship legacy obs_contract presets.

    New bundles carry a deploy_contract; ObsBuilder takes that path
    and never reaches this fallback. Returns None so the legacy
    component-order resolution falls through to
    ``DEFAULT_COMPONENT_ORDER``.
    """
    return None


@dataclass
class ObsComponentSpec:
    name: str
    dim: int


def _quat_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to a 3x3 rotation matrix."""
    w, x, y, z = quat.astype(float)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def _project_gravity(quat: np.ndarray) -> np.ndarray:
    """Project world gravity into the robot base frame."""
    rotation = _quat_to_rotation_matrix(quat)
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return (rotation.T @ gravity_world).astype(np.float32)


class ObsBuilder:
    """Builds the flat runtime observation vector for a policy bundle."""

    DEFAULT_COMPONENT_ORDER = (
        "base_angular_velocity",
        "projected_gravity",
        "joint_positions",
        "joint_velocities",
        "last_action",
        "velocity_command",
    )

    _DIM_ALIASES = {
        "base_angular_velocity": 3,
        "base_ang_vel": 3,
        "base_linear_velocity": 3,
        "base_lin_vel": 3,
        "projected_gravity": 3,
        "gravity_vec": 3,
        "joint_positions": "num_joints",
        "joint_pos": "num_joints",
        "joint_pos_rel": "num_joints",
        "joint_velocities": "num_joints",
        "joint_vel": "num_joints",
        "joint_vel_rel": "num_joints",
        "last_action": "action_dim",
        "previous_action": "action_dim",
        "velocity_command": 3,
        "velocity_commands": 3,
        "command": 4,
        "imu": 6,
        "phase_sin_cos": 2,
        "phase": 2,
        "actions": "action_dim",
        "reference_joint_positions": "num_joints",
        "ref_joint_pos": "num_joints",
        "reference_joint_velocities": "num_joints",
        "ref_joint_vel": "num_joints",
        "height_scan": 187,
        "heightfield_scan": 187,
    }

    def __init__(
        self,
        bundle: CheckpointBundle,
        component_order: Optional[Sequence[str]] = None,
        *,
        bundle_space: Optional["JointSpace"] = None,
        qpos_space: Optional["JointSpace"] = None,
        robot_sku: str = "",
    ):
        self._bundle = bundle
        # Caller-supplied canonical SKU (PolicyRunner.load plumbs it).
        # Used by _resolve_default_component_order to pick the
        # community Go2 34D obs preset by SKU rather than by reading
        # arbitrary brand/type strings from the manifest.
        self._robot_sku = str(robot_sku or "").strip()

        # Strict-mode contract (CLAUDE.md §1.8): if the bundle ships a
        # deploy_contract section, it MUST parse cleanly. Catching
        # ValueError and falling back to a legacy preset silently
        # produces a malformed obs vector when the contract was present
        # but invalid — the policy then receives the wrong layout and
        # behaves "almost right but actually wrong", which is the
        # worst possible failure mode. If parse fails, the bundle is
        # broken — re-export.
        contract: Optional["DeployContract"] = bundle.deploy_contract
        self._contract: Optional["DeployContract"] = contract
        self._use_contract: bool = contract is not None

        self._per_term_history: Dict[str, Deque[np.ndarray]] = {}
        self._per_term_initialized: Dict[str, bool] = {}
        if self._use_contract and contract is not None:
            for term_name, spec in contract.observations.items():
                if spec.history_length > 1:
                    self._per_term_history[term_name] = deque(
                        maxlen=spec.history_length
                    )
                    self._per_term_initialized[term_name] = False

        if self._use_contract and contract is not None:
            self._component_order = tuple(contract.observations.keys())
        else:
            self._component_order = (
                tuple(component_order)
                if component_order is not None
                else self._resolve_default_component_order(
                    bundle, robot_sku=self._robot_sku,
                )
            )

        raw_obs = (bundle.raw_manifest or {}).get("observation_space") or {}
        self._frame_stack = (
            1 if self._use_contract
            else max(1, int(raw_obs.get("frame_stack") or 1))
        )
        self._history: Deque[np.ndarray] = deque(maxlen=self._frame_stack)

        raw_convention = str(
            (bundle.raw_manifest or {}).get("inference_convention", "")
        ).strip().lower()
        if raw_convention:
            self._convention = raw_convention
        else:
            # Legacy-bundle fallback. RELEASE Phase 5 bundles ship
            # ``inference_convention`` explicitly via BundleExporter; bundles
            # exported before that miss the field. Without a body-frame
            # rotation IL policies see world-frame base velocities at deploy
            # time and produce the classic "drifts with no command + joints
            # twitch" sim2sim failure. Infer ``isaac_lab`` when the
            # deploy_contract carries non-zero stiffness — that's an
            # RSL-RL / ImplicitActuator signature SB3 never emits.
            inferred = "sb3"
            try:
                stiffness = getattr(contract, "stiffness", None) if contract is not None else None
                if stiffness and any(float(s) > 0.0 for s in stiffness):
                    inferred = "isaac_lab"
            except Exception:
                pass
            if inferred == "isaac_lab" and not getattr(
                ObsBuilder, "_warned_legacy_il_convention", False
            ):
                log.warning(
                    "ObsBuilder: bundle missing top-level "
                    "`inference_convention`; inferring 'isaac_lab' from "
                    "deploy_contract.stiffness. Re-export the bundle with "
                    "the current RELEASE BundleExporter to silence this "
                    "warning and guarantee future correctness."
                )
                ObsBuilder._warned_legacy_il_convention = True
            self._convention = inferred

        if self._use_contract and contract is not None and contract.default_joint_pos:
            try:
                self._default_joint_pos = np.asarray(
                    contract.default_joint_pos, dtype=np.float32
                )
            except Exception:
                self._default_joint_pos = np.zeros(0, dtype=np.float32)
        else:
            raw_default = (
                (bundle.raw_manifest or {}).get("robot", {}) or {}
            ).get("default_joint_pos") or []
            try:
                self._default_joint_pos = np.asarray(raw_default, dtype=np.float32)
            except Exception:
                self._default_joint_pos = np.zeros(0, dtype=np.float32)

        self._bundle_space = bundle_space
        self._qpos_space = qpos_space

        # Phase 5 sanity check: bundle joint order vs Isaac Lab USD
        # articulation order for this SKU. When BundleFinalizer ships a
        # correctly-ordered contract this is a no-op; legacy IL bundles
        # exported before the order fix trip the error and tell the user
        # to re-export. Only fires for IL deploy_contract bundles where
        # we have a known SKU and a registered USD order to compare to.
        if (
            self._use_contract
            and self._convention == "isaac_lab"
            and self._robot_sku
            and bundle_space is not None
        ):
            from registers.robots import get_robot_spec

            rs = get_robot_spec(self._robot_sku)
            expected: list = []
            if rs is not None:
                try:
                    expected = list(rs.joint_ir_roles_for("USD"))
                except Exception:
                    expected = []
            actual = list(
                getattr(bundle_space, "__dict__", {}).get("ir_labels") or ()
            )
            if expected and actual and expected != actual:
                # CLAUDE.md §1.8: previously logged an ERROR and continued —
                # the policy then fed obs/action through the wrong joint
                # slots, producing the "electric shock" twitching the user
                # reported. A wrong-ordered bundle is unusable; refuse
                # to load it so the user re-exports instead of trusting
                # a silently-broken viewer/deploy.
                raise RuntimeError(
                    f"[ObsBuilder] bundle joint order does NOT match "
                    f"Isaac Lab USD articulation order for SKU "
                    f"{self._robot_sku!r} — sim2sim with this bundle "
                    f"would produce twitching policy outputs.\n"
                    f"  expected (USD order): {expected}\n"
                    f"  bundle has         : {actual}\n"
                    f"Re-export the bundle with the current RELEASE "
                    f"BundleFinalizer (the registry must have "
                    f"joints_per_format['USD'] populated for the SKU — "
                    f"use the Robot Asset card's \"Dump USD\" button). "
                    f"Old bundle path: re-train this canvas with the "
                    f"updated registry overlay, then re-launch review."
                )

        if self._use_contract and contract is not None:
            for _term_name in ("height_scan", "heightfield_scan"):
                spec = contract.observations.get(_term_name)
                if spec is None or spec.grid_params is None:
                    continue
                gp = spec.grid_params
                try:
                    self._HEIGHT_SCAN_NX = int(round(float(gp["nx"])))
                    self._HEIGHT_SCAN_NY = int(round(float(gp["ny"])))
                    self._HEIGHT_SCAN_RES = float(gp["resolution"])
                    self._HEIGHT_SCAN_OFFSET_Z = float(
                        gp.get("offset_z", type(self)._HEIGHT_SCAN_OFFSET_Z)
                    )
                    self._HEIGHT_SCAN_TARGET_OFFSET = float(
                        gp.get("target_offset", type(self)._HEIGHT_SCAN_TARGET_OFFSET)
                    )
                    log.info(
                        "ObsBuilder: height_scan grid overridden from "
                        "deploy_contract — nx=%d ny=%d res=%.3f "
                        "offset_z=%.3f target_offset=%.3f (total rays=%d)",
                        self._HEIGHT_SCAN_NX,
                        self._HEIGHT_SCAN_NY,
                        self._HEIGHT_SCAN_RES,
                        self._HEIGHT_SCAN_OFFSET_Z,
                        self._HEIGHT_SCAN_TARGET_OFFSET,
                        self._HEIGHT_SCAN_NX * self._HEIGHT_SCAN_NY,
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    log.warning(
                        "ObsBuilder: invalid height_scan grid_params "
                        "in deploy_contract (%s); falling back to class "
                        "defaults (17×11).",
                        exc,
                    )
                break

    def get_component_specs(self, env: SimEnvContext) -> List[ObsComponentSpec]:  # noqa: ARG002
        specs = []
        for name in self._component_order:
            dim = self._known_dim(name)
            if dim is not None:
                specs.append(ObsComponentSpec(name=name, dim=dim))
        return specs

    def expected_dim(self, env: SimEnvContext) -> int:  # noqa: ARG002
        total = 0
        for name in self._component_order:
            dim = self._known_dim(name)
            if dim is not None:
                total += dim
        return total * self._frame_stack

    def build(
        self,
        env: SimEnvContext,
        last_action: Optional[np.ndarray] = None,
        command: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        if self._use_contract and self._contract is not None:
            return self._build_with_contract(env, last_action, command)
        return self._build_legacy(env, last_action, command)

    def _build_with_contract(
        self,
        env: SimEnvContext,
        last_action: Optional[np.ndarray],
        command: Optional[Sequence[float]],
    ) -> np.ndarray:
        """Contract path: per-term scale + clip + history ring buffer.

        Iterates ``contract.observations`` in insertion order (== policy
        training input order). For each term:
            raw = _build_component(term_name, ...)
            raw = raw * spec.scale
            if spec.clip is not None: raw = clip(raw, *spec.clip)
            if spec.history_length > 1:
                push into per-term ring buffer (initialized with first obs)
                raw = concat(buffer)
        """
        contract = self._contract
        assert contract is not None  # narrowing for type checker
        parts: List[np.ndarray] = []
        dump_first_call = not getattr(
            type(self), "_logged_first_obs_dump", False
        )
        term_dump: List[Tuple[str, np.ndarray, np.ndarray]] = []
        for term_name, spec in contract.observations.items():
            raw_pre = self._build_component(term_name, env, last_action, command)
            raw_pre = np.asarray(raw_pre, dtype=np.float32)
            raw_pre_copy = raw_pre.copy() if dump_first_call else raw_pre

            if isinstance(spec.scale, (list, tuple)):
                scale_arr = np.asarray(spec.scale, dtype=np.float32)
                raw = raw_pre * scale_arr
            else:
                raw = raw_pre * np.float32(spec.scale)

            if spec.clip is not None:
                raw = np.clip(raw, np.float32(spec.clip[0]), np.float32(spec.clip[1]))

            if dump_first_call:
                term_dump.append(
                    (term_name, raw_pre_copy, np.asarray(raw, dtype=np.float32).copy())
                )

            if spec.history_length > 1:
                buf = self._per_term_history.get(term_name)
                if buf is None:
                    buf = deque(maxlen=spec.history_length)
                    self._per_term_history[term_name] = buf
                if not self._per_term_initialized.get(term_name, False):
                    for _ in range(spec.history_length):
                        buf.append(raw.copy())
                    self._per_term_initialized[term_name] = True
                else:
                    buf.append(raw.copy())
                stacked = np.concatenate(list(buf)).astype(np.float32)
                parts.append(stacked)
            else:
                parts.append(raw.astype(np.float32))

        obs = (
            np.concatenate(parts).astype(np.float32)
            if parts
            else np.zeros(0, dtype=np.float32)
        )

        if dump_first_call:
            try:
                rows = []
                for name, pre, post in term_dump:
                    pre_flat = pre.flatten()
                    post_flat = post.flatten()
                    pre_repr = (
                        "[" + ",".join("%+.3f" % v for v in pre_flat.tolist())
                        + "]"
                        if pre_flat.size <= 16
                        else "min=%.3f max=%.3f mean=%.3f (n=%d)" % (
                            float(pre_flat.min()), float(pre_flat.max()),
                            float(pre_flat.mean()), int(pre_flat.size),
                        )
                    )
                    post_repr = (
                        "[" + ",".join("%+.3f" % v for v in post_flat.tolist())
                        + "]"
                        if post_flat.size <= 16
                        else "min=%.3f max=%.3f mean=%.3f (n=%d)" % (
                            float(post_flat.min()), float(post_flat.max()),
                            float(post_flat.mean()), int(post_flat.size),
                        )
                    )
                    rows.append(
                        f"  {name:32s} raw={pre_repr}\n"
                        f"  {'':32s} scaled+clipped={post_repr}"
                    )
                log.info(
                    "ObsBuilder first-call term dump "
                    "(obs_dim=%d, command=%s, last_action_max=%s):\n%s",
                    int(obs.shape[0]),
                    "None" if command is None
                    else ["%.3f" % v for v in np.asarray(command).flatten().tolist()],
                    "None" if last_action is None
                    else "%.4f" % float(np.max(np.abs(np.asarray(last_action)))),
                    "\n".join(rows),
                )
                type(self)._logged_first_obs_dump = True
            except Exception:
                log.exception("ObsBuilder first-call term dump failed")

        if obs.shape[0] != self._bundle.obs_dim:
            raise ValueError(
                f"ObsBuilder (contract mode): built obs has dim "
                f"{obs.shape[0]} but bundle.obs_dim={self._bundle.obs_dim}. "
                f"The contract sum-of-dims must equal bundle.obs_dim — "
                f"the bundle's manifest is internally inconsistent. "
                f"Re-export the bundle."
            )
        return obs

    def _build_legacy(
        self,
        env: SimEnvContext,
        last_action: Optional[np.ndarray],
        command: Optional[Sequence[float]],
    ) -> np.ndarray:
        """Legacy preset path — flat frame_stack, no per-term scale/clip."""
        parts = [
            self._build_component(name, env, last_action, command)
            for name in self._component_order
        ]
        frame = np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, dtype=np.float32)
        while len(self._history) < self._frame_stack:
            self._history.append(np.zeros_like(frame))
        self._history.append(frame)
        obs = np.concatenate(list(self._history)).astype(np.float32) if self._history else frame

        if obs.shape[0] != self._bundle.obs_dim:
            raise ValueError(
                f"ObsBuilder (legacy preset mode): built obs has dim "
                f"{obs.shape[0]} but bundle.obs_dim={self._bundle.obs_dim}. "
                f"Component order: {self._component_order}. "
                f"Re-export the bundle or fix the preset definition — "
                f"silent pad/truncate corrupts the policy input."
            )
        return obs

    def reset(self) -> None:
        self._history.clear()
        for buf in self._per_term_history.values():
            buf.clear()
        for term_name in self._per_term_initialized:
            self._per_term_initialized[term_name] = False

    @classmethod
    def _resolve_default_component_order(
        cls,
        bundle: CheckpointBundle,
        *,
        robot_sku: str = "",
    ) -> tuple[str, ...]:
        raw_obs = (bundle.raw_manifest or {}).get("observation_space") or {}

        components = raw_obs.get("components")
        if isinstance(components, list) and components:
            return tuple(str(component) for component in components if str(component).strip())

        # Isaac Lab exporter currently writes the per-term order under the
        # top-level ``observation_space_keys`` field (see
        # ``application.training.isaac_lab.bundle_finalizer._skill_manifest_to_v1_dict``).
        # Honor it as the source of truth for component order so legacy-path
        # bundles without a deploy_contract still build the obs vector in
        # the order the policy was trained on. Names like ``base_lin_vel`` /
        # ``velocity_command`` / ``joint_pos`` / ``last_action`` are already
        # aliased in ``_DIM_ALIASES`` and dispatched in ``_build_component``.
        raw_keys = (bundle.raw_manifest or {}).get("observation_space_keys")
        if isinstance(raw_keys, list) and raw_keys:
            return tuple(str(k) for k in raw_keys if str(k).strip())

        preset_name = str(
            raw_obs.get("contract_preset") or raw_obs.get("preset") or ""
        ).strip()
        contract = get_obs_contract(preset_name) if preset_name else None
        if contract and contract.get("obs_components"):
            return tuple(contract["obs_components"])

        # Community Go2 SAC 34D preset auto-selection: keyed by canonical
        # SKU comparison, NOT by reading manifest.robot.type literals.
        # The SKU plumbed in by PolicyRunner.load is the source of truth.
        if (
            robot_sku
            and bundle.obs_dim == 34
            and bundle.action_dim == 12
            and bundle.action_type == "torque"
        ):
            try:
                from registers.robots import resolve_id
                go2_sku = resolve_id("Unitree.go2")
                if go2_sku and (resolve_id(robot_sku) or robot_sku) == go2_sku:
                    contract = get_obs_contract(PRESET_COMMUNITY_GO2_SAC_34D) or {}
                    if contract.get("obs_components"):
                        return tuple(contract["obs_components"])
            except Exception:
                pass

        return cls.DEFAULT_COMPONENT_ORDER

    def _known_dim(self, name: str) -> Optional[int]:
        dim = self._DIM_ALIASES.get(name)
        if dim == "num_joints":
            return self._bundle.num_joints
        if dim == "action_dim":
            return self._bundle.action_dim
        return dim

    def _build_component(
        self,
        name: str,
        env: SimEnvContext,
        last_action: Optional[np.ndarray],
        command: Optional[Sequence[float]],
    ) -> np.ndarray:
        if name in ("base_angular_velocity", "base_ang_vel"):
            return self._get_base_angular_velocity(env)
        if name in ("base_linear_velocity", "base_lin_vel"):
            return self._get_base_linear_velocity(env)
        if name == "imu":
            return np.concatenate([
                self._get_base_angular_velocity(env),
                self._get_projected_gravity(env),
            ]).astype(np.float32)
        if name in ("projected_gravity", "gravity_vec"):
            return self._get_projected_gravity(env)
        if name in ("joint_positions", "joint_pos", "joint_pos_rel"):
            return self._get_joint_positions(env)
        if name in ("joint_velocities", "joint_vel", "joint_vel_rel"):
            return self._get_joint_velocities(env)
        if name in ("last_action", "previous_action", "actions"):
            return self._get_action_history(last_action)
        if name in ("velocity_command", "velocity_commands"):
            return self._get_command(command, dim=3)
        if name == "command":
            return self._get_command(command, dim=4)

        if name in ("height_scan", "heightfield_scan"):
            return self._get_height_scan(env)

        if name in ("reference_joint_positions", "ref_joint_pos"):
            return self._get_ref_obs(env, "_get_ref_joint_positions")
        if name in ("reference_joint_velocities", "ref_joint_vel"):
            return self._get_ref_obs(env, "_get_ref_joint_velocities")
        if name in ("phase_sin_cos", "phase"):
            return self._get_phase_obs(env)

        # Unknown component — degrade to a zero vector at the right
        # absolute index. Pre-v1.4 raised ValueError; the new behavior
        # matches DEMO's P4 obs_remap path so Isaac-Lab bundles on flat
        # MuJoCo (height_scan unknown) don't crash.
        unknown_names = [
            n for n in self._component_order if self._known_dim(n) is None
        ]
        known_sum = sum(
            (self._known_dim(n) or 0)
            for n in self._component_order
            if self._known_dim(n) is not None
        )
        gap = max(0, int(self._bundle.obs_dim) - int(known_sum))
        n_unknown = max(1, len(unknown_names))
        per_unknown_base = gap // n_unknown
        remainder = gap - per_unknown_base * n_unknown
        try:
            idx = unknown_names.index(name)
        except ValueError:
            idx = 0
        size = per_unknown_base + (remainder if idx == 0 else 0)
        raise KeyError(
            f"ObsBuilder: unknown obs component '{name}' (would have "
            f"silently zero-padded {size} dims into the policy input). "
            f"Bundle obs_dim={self._bundle.obs_dim}, known_sum={known_sum}, "
            f"unknown_components={unknown_names}. Either re-export the "
            f"bundle with this component, or add it to the supported "
            f"component list in ObsBuilder._build_component."
        )

    def _world_to_body(self, env: SimEnvContext, world_vec: np.ndarray) -> np.ndarray:
        """Project a 3-vector from world frame into the robot base frame."""
        try:
            qpos = np.asarray(env.mj_data.qpos, dtype=np.float32).flatten()
            quat = qpos[3:7]
            norm = np.linalg.norm(quat)
            if norm < 1e-8:
                return world_vec.astype(np.float32)
            quat = quat / norm
            R = _quat_to_rotation_matrix(quat)
            return (R.T @ world_vec.astype(np.float32)).astype(np.float32)
        except Exception:
            return world_vec.astype(np.float32)

    def _get_base_angular_velocity(self, env: SimEnvContext) -> np.ndarray:
        """Return base angular velocity in the **body frame**, matching
        Isaac Lab's ``root_ang_vel_b`` observation term.

        MuJoCo freejoint convention (per mujoco.readthedocs.io
        /programming/simulation.html#degrees-of-freedom):

          * ``qvel[0:3]`` — linear velocity in **world** frame.
          * ``qvel[3:6]`` — angular velocity **already in body frame**.

        So ``qvel[3:6]`` is returned unrotated. Applying ``_world_to_body``
        here (as earlier versions did) rotates a body-frame vector twice
        and silently corrupts the obs the moment base orientation drifts
        off identity — produces asymmetric standing-pose failures
        (one leg lifts, body leans). SB3 also expects body-frame ω here.
        """
        try:
            qvel = np.asarray(env.mj_data.qvel, dtype=np.float32).flatten()
            return qvel[3:6].astype(np.float32)
        except Exception as exc:
            raise RuntimeError(
                f"ObsBuilder: could not read base_angular_velocity from "
                f"mj_data.qvel ({type(exc).__name__}: {exc}). Zeroing this "
                f"obs term would silently corrupt the policy input; the "
                f"env adapter is required to provide a usable qvel array."
            ) from exc

    def _get_base_linear_velocity(self, env: SimEnvContext) -> np.ndarray:
        try:
            qvel = np.asarray(env.mj_data.qvel, dtype=np.float32).flatten()
            world = qvel[0:3]
            if self._convention == "isaac_lab":
                return self._world_to_body(env, world)
            return world
        except Exception as exc:
            raise RuntimeError(
                f"ObsBuilder: could not read base_linear_velocity from "
                f"mj_data.qvel ({type(exc).__name__}: {exc}). Zeroing this "
                f"obs term would silently corrupt the policy input."
            ) from exc

    # ------------------------------------------------------------------
    # Height scan — Isaac Lab RayCaster equivalent (verbatim from DEMO)
    # ------------------------------------------------------------------

    _HEIGHT_SCAN_NX = 17
    _HEIGHT_SCAN_NY = 11
    _HEIGHT_SCAN_RES = 0.1
    _HEIGHT_SCAN_OFFSET_Z = 20.0
    _HEIGHT_SCAN_TARGET_OFFSET = 0.5

    def _get_height_scan(self, env: SimEnvContext) -> np.ndarray:
        """Return a 17×11=187-dim height_scan vector via ray casting.

        Implements the Isaac Lab RayCaster contract:
            - Grid of (nx × ny) sample points centered above the robot
              base, yaw-aligned to the base orientation
            - Each ray fires straight down from base_z + offset
            - Output value = ray_distance - target_offset
              (equivalent to ``sensor_z - hit_z - target_offset``)
        """
        import mujoco
        import math

        m = env.mj_model
        d = env.mj_data
        try:
            if float(d.geom_xmat[0, 0]) == 0.0 and float(d.geom_xmat[0, 4]) == 0.0:
                mujoco.mj_forward(m, d)

            qpos = np.asarray(d.qpos, dtype=np.float64).flatten()
            base_x = float(qpos[0])
            base_y = float(qpos[1])
            base_z = float(qpos[2])
            quat = qpos[3:7]
            qn = float(np.linalg.norm(quat))
            if qn > 1e-8:
                quat = quat / qn
                w, qx, qy, qz = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
                yaw = math.atan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            else:
                yaw = 0.0
            cy, sy = math.cos(yaw), math.sin(yaw)

            geomgroup = np.zeros(6, dtype=np.uint8)
            geomgroup[0] = 1
            root_body = -1

            planes = self._get_plane_geom_cache(m)

            nx = int(self._HEIGHT_SCAN_NX)
            ny = int(self._HEIGHT_SCAN_NY)
            res = float(self._HEIGHT_SCAN_RES)
            sensor_z = base_z + float(self._HEIGHT_SCAN_OFFSET_Z)
            target_offset = float(self._HEIGHT_SCAN_TARGET_OFFSET)

            n_rays = nx * ny
            out = np.zeros(n_rays, dtype=np.float32)
            vec = np.array([0.0, 0.0, -1.0], dtype=np.float64)
            geomid = np.zeros(1, dtype=np.int32)

            for i in range(nx):
                lx = (i - (nx - 1) * 0.5) * res
                for j in range(ny):
                    ly = (j - (ny - 1) * 0.5) * res
                    dx = cy * lx - sy * ly
                    dy_ = sy * lx + cy * ly
                    pnt = np.array(
                        [base_x + dx, base_y + dy_, sensor_z],
                        dtype=np.float64,
                    )
                    dist = mujoco.mj_ray(m, d, pnt, vec, geomgroup, 1, root_body, geomid)
                    if dist < 0:
                        dist = float("inf")
                    plane_dist = self._cast_against_planes(d, planes, pnt, vec)
                    if plane_dist < dist:
                        dist = plane_dist
                    if dist == float("inf"):
                        out[i * ny + j] = float(base_z - target_offset)
                    else:
                        # Isaac Lab canonical formula:
                        #   obs = base_body_world_z - target_offset - hit_z
                        # = dist - offset_z - target_offset
                        hit_z = sensor_z - dist
                        out[i * ny + j] = float(base_z - target_offset - hit_z)
            return out
        except Exception as exc:
            raise RuntimeError(
                f"ObsBuilder._get_height_scan: ray-cast failed "
                f"({type(exc).__name__}: {exc}). Zeroing the height-scan "
                f"obs would silently feed flat-terrain data to a policy "
                f"that learned on rough terrain — the robot would step "
                f"into holes it can't see. Fix the ray-cast setup or "
                f"use a bundle without height_scan obs term."
            ) from exc

    def _get_plane_geom_cache(self, mj_model: Any) -> List[int]:
        """Return + cache the list of plane-geom indices in *mj_model*."""
        import mujoco

        cache_key = id(mj_model)
        cached = getattr(self, "_plane_cache", None)
        if cached is not None and cached[0] == cache_key:
            return cached[1]
        plane_type = int(mujoco.mjtGeom.mjGEOM_PLANE)
        planes = [
            i for i in range(int(mj_model.ngeom))
            if int(mj_model.geom_type[i]) == plane_type
        ]
        self._plane_cache = (cache_key, planes)
        return planes

    @staticmethod
    def _cast_against_planes(
        mj_data: Any,
        plane_ids: List[int],
        pnt: np.ndarray,
        vec: np.ndarray,
    ) -> float:
        """Analytical ray-vs-plane intersection."""
        if not plane_ids:
            return float("inf")
        best = float("inf")
        for gid in plane_ids:
            xmat = np.asarray(mj_data.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
            normal = xmat[:, 2]
            denom = float(np.dot(vec, normal))
            if abs(denom) < 1e-9:
                continue
            plane_pos = np.asarray(mj_data.geom_xpos[gid], dtype=np.float64)
            t = float(np.dot(plane_pos - pnt, normal) / denom)
            if t <= 0.0 or t >= best:
                continue
            best = t
        return best

    def _get_projected_gravity(self, env: SimEnvContext) -> np.ndarray:
        try:
            qpos = np.asarray(env.mj_data.qpos, dtype=np.float32).flatten()
            if qpos.shape[0] < 7:
                raise ValueError("qpos too short for quaternion")
            quat = qpos[3:7]
            norm = np.linalg.norm(quat)
            if norm < 1e-8:
                raise ValueError("Quaternion norm is near-zero")
            quat = quat / norm
            return _project_gravity(quat)
        except Exception as exc:
            raise RuntimeError(
                f"ObsBuilder: projected_gravity computation failed "
                f"({type(exc).__name__}: {exc}). Falling back to "
                f"[0, 0, -1] would silently feed wrong orientation to "
                f"an orientation-sensitive policy — that has caused the "
                f"robot to launch sideways at startup. Fix the base quat "
                f"source on the env adapter."
            ) from exc

    def _read_joint_vector_in_bundle_order(
        self,
        raw_qpos_aligned: np.ndarray,
    ) -> np.ndarray:
        """Take a per-joint vector in MuJoCo qpos order and return it in
        bundle order. When the JointSpace pair isn't configured (legacy
        callers), assume the input is already in bundle order — same
        contract the historical code had.
        """
        n_bundle = self._bundle.num_joints
        if self._qpos_space is None or self._bundle_space is None:
            arr = raw_qpos_aligned[:n_bundle]
            if arr.shape[0] < n_bundle:
                arr = np.pad(arr, (0, n_bundle - arr.shape[0]))
            return arr.astype(np.float32)
        n_qpos = len(self._qpos_space)
        src = raw_qpos_aligned
        if src.shape[0] < n_qpos:
            src = np.pad(src, (0, n_qpos - src.shape[0]))
        elif src.shape[0] > n_qpos:
            src = src[:n_qpos]
        return self._qpos_space.permute(src, self._bundle_space).astype(np.float32)

    def _get_joint_positions(self, env: SimEnvContext) -> np.ndarray:
        try:
            qpos = np.asarray(env.mj_data.qpos, dtype=np.float32).flatten()
            joint_block = qpos[7:]
            in_bundle_order = self._read_joint_vector_in_bundle_order(joint_block)
            n = self._bundle.num_joints
            wants_relative = self._use_contract or self._convention == "isaac_lab"
            if (
                wants_relative
                and self._default_joint_pos.shape[0] == n
                and in_bundle_order.shape[0] == n
            ):
                in_bundle_order = in_bundle_order - self._default_joint_pos
            return in_bundle_order.astype(np.float32)
        except Exception as exc:
            raise RuntimeError(
                f"ObsBuilder._get_joint_positions: failed reading "
                f"mj_data.qpos joints ({type(exc).__name__}: {exc}). "
                f"Zeroing this obs would silently feed the policy a "
                f"static stand pose regardless of actual joint state."
            ) from exc

    def _get_joint_velocities(self, env: SimEnvContext) -> np.ndarray:
        try:
            qvel = np.asarray(env.mj_data.qvel, dtype=np.float32).flatten()
            joint_block = qvel[6:]
            return self._read_joint_vector_in_bundle_order(joint_block)
        except Exception as exc:
            raise RuntimeError(
                f"ObsBuilder._get_joint_velocities: failed reading "
                f"mj_data.qvel joints ({type(exc).__name__}: {exc}). "
                f"Zeroing this obs would silently tell the policy the "
                f"robot is perfectly still while it may be falling."
            ) from exc

    def _get_action_history(self, last_action: Optional[np.ndarray]) -> np.ndarray:
        if last_action is not None:
            arr = np.asarray(last_action, dtype=np.float32).flatten()
            if arr.shape[0] < self._bundle.action_dim:
                arr = np.pad(arr, (0, self._bundle.action_dim - arr.shape[0]))
            return arr[: self._bundle.action_dim]
        # WHY KEPT (Rule 1.c — genuine "first frame" sentinel):
        # ``last_action`` is None on the very first step before any
        # action has been emitted by the policy. Zeros are the
        # documented in-distribution initial value the policy was
        # trained against (all trainers initialise last_action=0 at
        # episode start). This is NOT a silent error-swallow — it's
        # the only legal value on step 0.
        return np.zeros(self._bundle.action_dim, dtype=np.float32)

    @staticmethod
    def _get_command(command: Optional[Sequence[float]], dim: int) -> np.ndarray:
        if command is None:
            raise ValueError(
                f"ObsBuilder._get_command: command is None — callers must "
                f"pass an explicit command vector of length {dim} "
                f"(e.g. ``np.zeros({dim})`` for stand-still, or a [vx, vy, "
                f"vyaw] triple for locomotion). Implicit-zero fallback "
                f"is forbidden — it has masked bugs where the user "
                f"expected motion but the robot stood still."
            )
        arr = np.asarray(command, dtype=np.float32).flatten()
        if arr.shape[0] < dim:
            arr = np.pad(arr, (0, dim - arr.shape[0]))
        return arr[:dim].astype(np.float32)

    def _get_ref_obs(self, env: SimEnvContext, method_name: str) -> np.ndarray:
        """Read a reference motion observation from the env adapter."""
        n = self._bundle.num_joints
        adapter = getattr(env, "adapter", None)
        if adapter is None or not hasattr(adapter, method_name):
            raise RuntimeError(
                f"ObsBuilder._get_ref_obs: env adapter "
                f"{type(adapter).__name__ if adapter is not None else 'None'} "
                f"does not expose {method_name!r}. AMP / reference-motion "
                f"obs terms require an env adapter that streams the "
                f"reference data — zero-filling silently shifts the "
                f"policy off-distribution. Use a non-AMP bundle or "
                f"connect a reference-motion-aware env adapter."
            )
        val = getattr(adapter, method_name)()
        arr = np.asarray(val, dtype=np.float32).flatten()
        if arr.shape[0] >= n:
            return arr[:n]
        return np.pad(arr, (0, n - arr.shape[0])).astype(np.float32)

    def _get_phase_obs(self, env: SimEnvContext) -> np.ndarray:
        """Read phase_sin_cos from the env adapter."""
        adapter = getattr(env, "adapter", None)
        if adapter is None:
            raise RuntimeError(
                "ObsBuilder._get_phase_obs: env has no adapter — "
                "phase obs requires a reference-motion-aware adapter. "
                "Zero-filling phase would freeze the policy in the "
                "first reference frame."
            )
        T = float(getattr(adapter, "_ref_num_frames", 0) or 0)
        if T <= 0:
            T = 1.0
        phase_f = float(getattr(adapter, "_ref_phase_f", 0.0) or 0.0)
        angle = 2.0 * np.pi * (phase_f / T)
        return np.array([np.sin(angle), np.cos(angle)], dtype=np.float32)
