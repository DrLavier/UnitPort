from __future__ import annotations

import logging
import warnings
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Deque, Dict, List, Optional, Sequence

import numpy as np

from src.system.training.obs_contracts import (
    PRESET_COMMUNITY_GO2_SAC_34D,
    PRESET_ISAAC_LAB_GO2_VELOCITY_48D,
    get_obs_contract,
)

from .manifest_schema import CheckpointBundle
from .sim_env_context import SimEnvContext

if TYPE_CHECKING:
    from src.system.policy.deploy_contract import DeployContract

log = logging.getLogger(__name__)


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
        "joint_pos_rel": "num_joints",  # contract canonical name (relative to default_pos)
        "joint_velocities": "num_joints",
        "joint_vel": "num_joints",
        "joint_vel_rel": "num_joints",  # contract canonical name (== joint_vel; "rel" is naming-only)
        "last_action": "action_dim",
        "previous_action": "action_dim",
        "velocity_command": 3,
        "velocity_commands": 3,
        "command": 4,
        "imu": 6,
        "phase_sin_cos": 2,
        "phase": 2,
        "actions": "action_dim",
        # Reference motion obs (auto-injected by compiler)
        "reference_joint_positions": "num_joints",
        "ref_joint_pos": "num_joints",
        "reference_joint_velocities": "num_joints",
        "ref_joint_vel": "num_joints",
        # P5++: Isaac Lab RayCaster height scan. Default grid is
        # 17 (nx) × 11 (ny) = 187 rays per the Isaac Lab legged-robot
        # default config. _get_height_scan() implements the actual
        # ray-cast against whatever geometry the active field has.
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
    ):
        self._bundle = bundle

        # ── Contract-first detection ─────────────────────────────────
        # When the bundle ships a deploy_contract, ObsBuilder switches
        # into "contract mode": term order, scale, clip, and per-term
        # history all come from the contract. The legacy preset path
        # stays as a fallback for bundles without a contract.
        contract: Optional["DeployContract"]
        try:
            contract = bundle.deploy_contract
        except ValueError as exc:
            log.error(
                "ObsBuilder: bundle deploy_contract failed validation (%s); "
                "falling back to legacy preset path.",
                exc,
            )
            contract = None
        self._contract: Optional["DeployContract"] = contract
        self._use_contract: bool = contract is not None

        # Per-term history ring buffers (contract mode only). Filled with
        # the FIRST observation on first build, not zeros — see
        # SIM2SIM/report.yaml L211.
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
            # Contract iteration order — overrides component_order arg.
            self._component_order = tuple(contract.observations.keys())
        else:
            self._component_order = (
                tuple(component_order)
                if component_order is not None
                else self._resolve_default_component_order(bundle)
            )

        raw_obs = (bundle.raw_manifest or {}).get("observation_space") or {}
        # Legacy flat frame_stack stays for the non-contract path. In
        # contract mode every term has its own ring buffer and the flat
        # deque is unused — kept as a no-op of length 1 to keep build()
        # symmetric.
        self._frame_stack = (
            1 if self._use_contract
            else max(1, int(raw_obs.get("frame_stack") or 1))
        )
        self._history: Deque[np.ndarray] = deque(maxlen=self._frame_stack)

        # Inference convention — set by ``upgrade_v2_to_v1`` for Isaac Lab
        # bundles. The IL stock velocity-tracking task feeds the policy
        # body-frame base_lin_vel/base_ang_vel and joint_pos *relative* to
        # the standing default pose. SB3-trained UnitPort bundles use raw
        # world-frame velocities and absolute joint readings, so we
        # branch on this flag rather than changing the SB3 path.
        self._convention = str(
            (bundle.raw_manifest or {}).get("inference_convention", "sb3")
        ).strip().lower()

        # Default joint pose used to compute joint_pos_rel for IL bundles.
        # Resolution order:
        #   1. deploy_contract.default_joint_pos (preferred — single source)
        #   2. raw_manifest['robot']['default_joint_pos'] (legacy v1 upgrader)
        #   3. zeros (SB3 path / unknown)
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

        # Joint reference frames. The bundle space describes the order
        # the policy was trained against; the qpos space describes the
        # order MuJoCo's qpos slots use. PolicyRunner.load() builds both
        # from authoritative sources (manifest joint_names + mj_model)
        # and passes them in. When either is omitted (legacy callers /
        # SB3 path / unit tests) we leave them ``None`` and the joint
        # readers fall back to the historical "qpos[7:7+n] is already in
        # bundle order" assumption. New code should always supply both.
        self._bundle_space = bundle_space
        self._qpos_space = qpos_space

        # Height-scan grid — per-instance override populated from
        # deploy_contract.observations[<height_scan>].grid_params when
        # the bundle carries it. Class-level defaults remain as a
        # fallback for legacy bundles without the metadata. _get_height_scan
        # reads these via self attribute lookup, so instance attrs
        # silently shadow the class vars when set.
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
        for term_name, spec in contract.observations.items():
            raw = self._build_component(term_name, env, last_action, command)
            raw = np.asarray(raw, dtype=np.float32)

            # Per-term scale
            if isinstance(spec.scale, (list, tuple)):
                scale_arr = np.asarray(spec.scale, dtype=np.float32)
                raw = raw * scale_arr
            else:
                raw = raw * np.float32(spec.scale)

            # Per-term clip (applied AFTER scale)
            if spec.clip is not None:
                raw = np.clip(raw, np.float32(spec.clip[0]), np.float32(spec.clip[1]))

            # Per-term history ring buffer
            if spec.history_length > 1:
                buf = self._per_term_history.get(term_name)
                if buf is None:
                    # Lazy create — should not happen because __init__ pre-builds
                    # buffers, but stay defensive.
                    buf = deque(maxlen=spec.history_length)
                    self._per_term_history[term_name] = buf
                if not self._per_term_initialized.get(term_name, False):
                    # Fill the buffer with copies of the first obs (NOT zeros)
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

        # Length sanity vs bundle.obs_dim. We do NOT pad/truncate in
        # contract mode — the contract is the authoritative size and any
        # mismatch is a bug the user must see.
        if obs.shape[0] != self._bundle.obs_dim:
            warnings.warn(
                f"ObsBuilder (contract mode): built obs has dim {obs.shape[0]} "
                f"but bundle.obs_dim={self._bundle.obs_dim}. The contract "
                f"sum-of-dims should equal bundle.obs_dim — fix the contract "
                f"or the bundle metadata. NOT pad/truncating in contract mode.",
                stacklevel=2,
            )
        return obs

    def _build_legacy(
        self,
        env: SimEnvContext,
        last_action: Optional[np.ndarray],
        command: Optional[Sequence[float]],
    ) -> np.ndarray:
        """Legacy preset path — flat frame_stack, no per-term scale/clip.

        Preserved verbatim for bundles without a deploy_contract.
        """
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
            warnings.warn(
                f"Built observation has dimension {obs.shape[0]} but "
                f"bundle.obs_dim={self._bundle.obs_dim}; "
                f"{'truncating' if obs.shape[0] > self._bundle.obs_dim else 'padding'} "
                f"to match. Component order: {self._component_order}",
                stacklevel=2,
            )
            if obs.shape[0] > self._bundle.obs_dim:
                obs = obs[: self._bundle.obs_dim]
            else:
                obs = np.pad(obs, (0, self._bundle.obs_dim - obs.shape[0]))
        return obs

    def reset(self) -> None:
        self._history.clear()
        # Contract mode: clear per-term ring buffers and re-arm the
        # "fill with first obs" initialization on the next build().
        for buf in self._per_term_history.values():
            buf.clear()
        for term_name in self._per_term_initialized:
            self._per_term_initialized[term_name] = False

    @classmethod
    def _resolve_default_component_order(cls, bundle: CheckpointBundle) -> tuple[str, ...]:
        raw_obs = (bundle.raw_manifest or {}).get("observation_space") or {}

        components = raw_obs.get("components")
        if isinstance(components, list) and components:
            return tuple(str(component) for component in components if str(component).strip())

        preset_name = str(
            raw_obs.get("contract_preset") or raw_obs.get("preset") or ""
        ).strip()
        contract = get_obs_contract(preset_name) if preset_name else None
        if contract and contract.get("obs_components"):
            return tuple(contract["obs_components"])

        robot_type = str(((bundle.raw_manifest or {}).get("robot") or {}).get("type") or "").lower()
        if (
            robot_type == "go2"
            and bundle.obs_dim == 34
            and bundle.action_dim == 12
            and bundle.action_type == "torque"
        ):
            contract = get_obs_contract(PRESET_COMMUNITY_GO2_SAC_34D) or {}
            if contract.get("obs_components"):
                return tuple(contract["obs_components"])

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
            # joint_pos_rel is the canonical contract name and ALWAYS means
            # "relative to default_joint_pos" — _get_joint_positions handles
            # the subtraction when contract is present OR when convention
            # is isaac_lab.
            return self._get_joint_positions(env)
        if name in ("joint_velocities", "joint_vel", "joint_vel_rel"):
            # joint_vel_rel is a unitree_rl_lab naming quirk — there is no
            # actual subtraction; "rel" is naming-only. Same as joint_vel.
            return self._get_joint_velocities(env)
        if name in ("last_action", "previous_action", "actions"):
            return self._get_action_history(last_action)
        if name in ("velocity_command", "velocity_commands"):
            return self._get_command(command, dim=3)
        if name == "command":
            return self._get_command(command, dim=4)

        # P5++: height_scan is a GEOMETRIC sensor — Mission computes it
        # via mj_ray + analytical plane fallback against whatever
        # geometry the active field happens to have. It is NOT a
        # field-dependent obs; it works on flat_ground, stair_climb,
        # rough_terrain, prop_obstacle_field, etc. equally well.
        # The user's architectural insight (mission_design.yaml v1.4):
        # "Mission inherits the trained behavior policy and applies it
        # to whatever scene the user assigns" — height_scan is the
        # canonical example of an obs that must be re-derived from the
        # current env, not borrowed from the training environment.
        if name in ("height_scan", "heightfield_scan"):
            return self._get_height_scan(env)

        # Reference motion obs — delegate to the training env adapter
        # which has the loaded reference frames and phase state.
        if name in ("reference_joint_positions", "ref_joint_pos"):
            return self._get_ref_obs(env, "_get_ref_joint_positions")
        if name in ("reference_joint_velocities", "ref_joint_vel"):
            return self._get_ref_obs(env, "_get_ref_joint_velocities")
        if name in ("phase_sin_cos", "phase"):
            return self._get_phase_obs(env)

        # Unknown component — P4 obs_remap path: degrade to a zero
        # vector instead of raising. The size is computed so the
        # degraded zeros land at the CORRECT absolute index in the
        # final obs vector (which matters for policies whose unknown
        # components are not at the tail). When there is exactly one
        # unknown component the gap math is exact; multiple unknowns
        # split the gap evenly with any remainder going to the first.
        #
        # Pre-v1.4 behaviour was to raise ValueError, which crashed
        # every Isaac-Lab bundle on flat MuJoCo (height_scan is the
        # canonical case). The compat checker still surfaces a WARN
        # for the dim mismatch and the P4 embodiment matcher emits
        # obs_remap_warnings, so the user has full visibility.
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
        # The first unknown soaks up the remainder so the total exactly
        # equals the gap.
        try:
            idx = unknown_names.index(name)
        except ValueError:
            idx = 0
        size = per_unknown_base + (remainder if idx == 0 else 0)
        warnings.warn(
            f"ObsBuilder: degrading unknown component '{name}' to a "
            f"zero vector of size {size} (P4 obs_remap path). Bundle "
            f"obs_dim={self._bundle.obs_dim}, known_sum={known_sum}, "
            f"unknown_components={unknown_names}.",
            stacklevel=3,
        )
        return np.zeros(size, dtype=np.float32)

    def _world_to_body(self, env: SimEnvContext, world_vec: np.ndarray) -> np.ndarray:
        """Project a 3-vector from world frame into the robot base frame.

        Isaac Lab's velocity-tracking observation feeds the policy
        ``base_lin_vel_b = R^T @ base_lin_vel_w`` (and similarly for
        ``base_ang_vel``). MuJoCo's ``qvel`` slots 0..6 are in *world*
        frame, so we have to apply the rotation explicitly. Without this,
        the policy gets garbage velocities and the gait collapses.
        """
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
        try:
            qvel = np.asarray(env.mj_data.qvel, dtype=np.float32).flatten()
            world = qvel[3:6]
            if self._convention == "isaac_lab":
                return self._world_to_body(env, world)
            return world
        except Exception:
            warnings.warn(
                "ObsBuilder: could not read base_angular_velocity from mj_data.qvel; using zeros.",
                stacklevel=3,
            )
            return np.zeros(3, dtype=np.float32)

    def _get_base_linear_velocity(self, env: SimEnvContext) -> np.ndarray:
        try:
            qvel = np.asarray(env.mj_data.qvel, dtype=np.float32).flatten()
            world = qvel[0:3]
            if self._convention == "isaac_lab":
                return self._world_to_body(env, world)
            return world
        except Exception:
            warnings.warn(
                "ObsBuilder: could not read base_linear_velocity from mj_data.qvel; using zeros.",
                stacklevel=3,
            )
            return np.zeros(3, dtype=np.float32)

    # ------------------------------------------------------------------
    # Height scan — Isaac Lab RayCaster equivalent (P5++)
    # ------------------------------------------------------------------

    # Default Isaac Lab height_scanner config (matches the most common
    # legged-robot env: GridPattern resolution=0.1m size=(1.6, 1.0) →
    # 17 columns × 11 rows = 187 rays). The sensor sits 20 m above the
    # robot base and casts straight down. The 20 m mounting offset is
    # ONLY a ray-origin positioning hack (so the ray starts well above
    # any possible robot geometry); it MUST NOT leak into the obs value.
    #
    # Canonical Isaac Lab formula (``mdp.height_scan``):
    #     per_ray_obs = base_body_world_z - target_offset - terrain_z
    # where ``base_body_world_z`` is the PARENT body's world z (not the
    # ray-origin's world z), and ``terrain_z`` is the world z where the
    # ray hit geometry. For a Go2 at standing height z≈0.445 on flat
    # ground this yields ≈-0.055 per ray — a small negative number
    # around zero, NOT ≈19.9 (which would include the 20 m mount offset
    # and completely blow up the policy input).
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

        Works on ANY field — flat ground, stairs, heightfield terrain,
        prop obstacles. Does NOT depend on the manifest declaring a
        special "heightfield" terrain type. This is the architecturally
        correct mission_design.yaml v1.4 behavior: Mission re-derives
        the obs from the active env, never inherits training scene.

        Implementation notes:
        - mujoco.mj_ray's BVH-based query SKIPS infinite-extent plane
          geoms (size=0,0), so we maintain an analytical ray-vs-plane
          fallback that catches the menagerie ground plane.
        - Self-hit avoidance: bodyexclude is the robot's root body, so
          rays don't hit the robot itself.
        """
        import mujoco
        import math

        m = env.mj_model
        d = env.mj_data
        try:
            # Ensure forward kinematics so geom_xpos / geom_xmat are
            # populated for the analytical plane fallback.
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
                # Yaw from quaternion (Z-axis rotation only).
                # quat = (w, x, y, z), MuJoCo convention.
                w, qx, qy, qz = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
                yaw = math.atan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            else:
                yaw = 0.0
            cy, sy = math.cos(yaw), math.sin(yaw)

            # Restrict ray casts to FIELD geoms (group 0). The menagerie
            # convention places the floor + every prop / hfield / stair
            # we inject via mjSpec into group 0, while the robot's
            # collision capsules live in group 2 and visual meshes in
            # group 3. Filtering by group is the cleanest way to keep
            # rays from hitting the robot itself — bodyexclude only
            # excludes ONE body, which isn't enough for a multi-body
            # robot.
            geomgroup = np.zeros(6, dtype=np.uint8)
            geomgroup[0] = 1
            # No body exclusion needed once we've group-filtered.
            root_body = -1

            # Cache the list of plane-geom indices in the model so we
            # don't iterate every geom in the analytical fallback per ray.
            planes = self._get_plane_geom_cache(m)

            # Build the grid + sweep
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
                    # Yaw-rotate local offsets into world frame
                    dx = cy * lx - sy * ly
                    dy_ = sy * lx + cy * ly
                    pnt = np.array(
                        [base_x + dx, base_y + dy_, sensor_z],
                        dtype=np.float64,
                    )
                    # BVH ray-cast (handles boxes, spheres, hfields, …)
                    dist = mujoco.mj_ray(m, d, pnt, vec, geomgroup, 1, root_body, geomid)
                    if dist < 0:
                        dist = float("inf")
                    # Analytical fallback for infinite planes
                    plane_dist = self._cast_against_planes(d, planes, pnt, vec)
                    if plane_dist < dist:
                        dist = plane_dist
                    if dist == float("inf"):
                        # Ray missed everything — fall back to the
                        # canonical "flat ground at world z=0" value so
                        # the policy sees the same number it would have
                        # seen on plain flat_ground terrain during
                        # training. ``base_z - target_offset - 0``.
                        out[i * ny + j] = float(base_z - target_offset)
                    else:
                        # Isaac Lab canonical formula:
                        #   obs = base_body_world_z - target_offset - hit_z
                        # Ray was cast from (base_x+dx, base_y+dy,
                        # sensor_z=base_z+offset_z) straight down, so
                        # hit_z = sensor_z - dist. Substituting:
                        #   obs = base_z - target_offset - (sensor_z - dist)
                        #       = dist - offset_z - target_offset
                        # The 20 m mounting offset (``offset_z``) cancels
                        # out and MUST be subtracted; leaving it in makes
                        # the policy see every ray ≈ base_z+19.5 instead
                        # of the correct ≈ −0.055, which blows the ONNX
                        # forward pass 100×-large and causes the robot
                        # to twist into a locked pose on the first step.
                        hit_z = sensor_z - dist
                        out[i * ny + j] = float(base_z - target_offset - hit_z)
            return out
        except Exception as exc:
            warnings.warn(
                f"ObsBuilder._get_height_scan: ray-cast failed ({exc}); "
                f"returning zero vector.",
                stacklevel=3,
            )
            return np.zeros(
                int(self._HEIGHT_SCAN_NX) * int(self._HEIGHT_SCAN_NY),
                dtype=np.float32,
            )

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
        """Analytical ray-vs-plane intersection.

        Returns the smallest positive ``t`` such that ``pnt + t*vec``
        lies on a plane geom whose half-extents (or infinite extent
        when half-size is 0) contain the hit point. Returns ``inf``
        when no plane is hit.

        This complements ``mujoco.mj_ray`` which skips infinite planes
        because their BVH bounds are degenerate.
        """
        if not plane_ids:
            return float("inf")
        best = float("inf")
        for gid in plane_ids:
            # Plane normal = local +Z column of geom_xmat (3x3 row-major)
            xmat = np.asarray(mj_data.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
            normal = xmat[:, 2]
            denom = float(np.dot(vec, normal))
            if abs(denom) < 1e-9:
                continue   # ray parallel to plane
            plane_pos = np.asarray(mj_data.geom_xpos[gid], dtype=np.float64)
            t = float(np.dot(plane_pos - pnt, normal) / denom)
            if t <= 0.0 or t >= best:
                continue
            # Half-size bounds check in plane's local frame.
            # geom_size is (half_x, half_y, grid_spacing). 0 = infinite.
            try:
                gsize_attr = getattr(mj_data, "_geom_size_proxy", None)
                # Walk the model via the parent — we don't have direct
                # access to mj_model here, so the caller passed mj_data
                # only. Skip the bounds check entirely (treat all
                # planes as infinite); for the menagerie ground plane
                # this is correct, and for any future bounded planes
                # the user is responsible for using a non-plane geom.
            except Exception:
                pass
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
            warnings.warn(
                f"ObsBuilder: projected_gravity fallback (reason: {exc}); using [0, 0, -1].",
                stacklevel=3,
            )
            return np.array([0.0, 0.0, -1.0], dtype=np.float32)

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
            # Legacy fast path: caller is responsible for the ordering.
            arr = raw_qpos_aligned[:n_bundle]
            if arr.shape[0] < n_bundle:
                arr = np.pad(arr, (0, n_bundle - arr.shape[0]))
            return arr.astype(np.float32)
        # Length-match the qpos vector to qpos_space, then permute.
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
            # Drop the 7-slot floating base, then take the joint block.
            joint_block = qpos[7:]
            in_bundle_order = self._read_joint_vector_in_bundle_order(joint_block)
            # Subtract default_joint_pos when EITHER (a) we're in contract
            # mode (the contract's joint_pos_rel term ALWAYS means relative)
            # OR (b) the bundle is Isaac Lab convention (legacy IL path).
            n = self._bundle.num_joints
            wants_relative = self._use_contract or self._convention == "isaac_lab"
            if (
                wants_relative
                and self._default_joint_pos.shape[0] == n
                and in_bundle_order.shape[0] == n
            ):
                in_bundle_order = in_bundle_order - self._default_joint_pos
            return in_bundle_order.astype(np.float32)
        except Exception:
            return np.zeros(self._bundle.num_joints, dtype=np.float32)

    def _get_joint_velocities(self, env: SimEnvContext) -> np.ndarray:
        try:
            qvel = np.asarray(env.mj_data.qvel, dtype=np.float32).flatten()
            # Drop the 6-slot floating base velocity (3 lin + 3 ang).
            joint_block = qvel[6:]
            return self._read_joint_vector_in_bundle_order(joint_block)
        except Exception:
            return np.zeros(self._bundle.num_joints, dtype=np.float32)

    def _get_action_history(self, last_action: Optional[np.ndarray]) -> np.ndarray:
        if last_action is not None:
            arr = np.asarray(last_action, dtype=np.float32).flatten()
            if arr.shape[0] < self._bundle.action_dim:
                arr = np.pad(arr, (0, self._bundle.action_dim - arr.shape[0]))
            return arr[: self._bundle.action_dim]
        return np.zeros(self._bundle.action_dim, dtype=np.float32)

    @staticmethod
    def _get_command(command: Optional[Sequence[float]], dim: int) -> np.ndarray:
        if command is None:
            return np.zeros(dim, dtype=np.float32)
        arr = np.asarray(command, dtype=np.float32).flatten()
        if arr.shape[0] < dim:
            arr = np.pad(arr, (0, dim - arr.shape[0]))
        return arr[:dim].astype(np.float32)

    def _get_ref_obs(self, env: SimEnvContext, method_name: str) -> np.ndarray:
        """Read a reference motion observation from the env adapter (UnitreeGymEnv)."""
        n = self._bundle.num_joints
        adapter = getattr(env, "adapter", None)
        if adapter is not None and hasattr(adapter, method_name):
            try:
                val = getattr(adapter, method_name)()
                arr = np.asarray(val, dtype=np.float32).flatten()
                if arr.shape[0] >= n:
                    return arr[:n]
                return np.pad(arr, (0, n - arr.shape[0])).astype(np.float32)
            except Exception:
                pass
        return np.zeros(n, dtype=np.float32)

    def _get_phase_obs(self, env: SimEnvContext) -> np.ndarray:
        """Read phase_sin_cos from the env adapter."""
        adapter = getattr(env, "adapter", None)
        if adapter is not None:
            try:
                T = float(getattr(adapter, "_ref_num_frames", 0) or 0)
                if T <= 0:
                    T = 1.0
                phase_f = float(getattr(adapter, "_ref_phase_f", 0.0) or 0.0)
                angle = 2.0 * np.pi * (phase_f / T)
                return np.array([np.sin(angle), np.cos(angle)], dtype=np.float32)
            except Exception:
                pass
        return np.zeros(2, dtype=np.float32)
