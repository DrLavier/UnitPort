from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Sequence

import numpy as np

from system.training.obs_contracts import PRESET_COMMUNITY_GO2_SAC_34D, get_obs_contract

from .manifest_schema import CheckpointBundle
from .sim_env_context import SimEnvContext


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
        "joint_velocities": "num_joints",
        "joint_vel": "num_joints",
        "last_action": "action_dim",
        "previous_action": "action_dim",
        "velocity_command": 3,
        "command": 4,
        "imu": 6,
        "phase_sin_cos": 2,
        "phase": 2,
    }

    def __init__(
        self,
        bundle: CheckpointBundle,
        component_order: Optional[Sequence[str]] = None,
    ):
        self._bundle = bundle
        self._component_order: tuple[str, ...] = (
            tuple(component_order)
            if component_order is not None
            else self._resolve_default_component_order(bundle)
        )
        raw_obs = (bundle.raw_manifest or {}).get("observation_space") or {}
        self._frame_stack = max(1, int(raw_obs.get("frame_stack") or 1))
        self._history: Deque[np.ndarray] = deque(maxlen=self._frame_stack)

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
                f"Built observation has dimension {obs.shape[0]} but "
                f"bundle.obs_dim={self._bundle.obs_dim}. "
                f"Component order: {self._component_order}"
            )
        return obs

    def reset(self) -> None:
        self._history.clear()

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
        if name in ("joint_positions", "joint_pos"):
            return self._get_joint_positions(env)
        if name in ("joint_velocities", "joint_vel"):
            return self._get_joint_velocities(env)
        if name in ("last_action", "previous_action"):
            return self._get_action_history(last_action)
        if name == "velocity_command":
            return self._get_command(command, dim=3)
        if name == "command":
            return self._get_command(command, dim=4)

        # BUG-2 fix: returning zeros(0) caused a silent dim mismatch further up
        # the stack with an opaque error message.  Raise clearly now so the
        # problem is diagnosed at observation-build time rather than after
        # concatenation.
        known_dim = self._known_dim(name)
        raise ValueError(
            f"ObsBuilder: observation component '{name}' is not implemented. "
            f"Expected dimension from manifest: {known_dim}. "
            f"Component order: {self._component_order}"
        )

    def _get_base_angular_velocity(self, env: SimEnvContext) -> np.ndarray:
        try:
            qvel = np.asarray(env.mj_data.qvel, dtype=np.float32).flatten()
            return qvel[3:6]
        except Exception:
            warnings.warn(
                "ObsBuilder: could not read base_angular_velocity from mj_data.qvel; using zeros.",
                stacklevel=3,
            )
            return np.zeros(3, dtype=np.float32)

    def _get_base_linear_velocity(self, env: SimEnvContext) -> np.ndarray:
        try:
            qvel = np.asarray(env.mj_data.qvel, dtype=np.float32).flatten()
            return qvel[0:3]
        except Exception:
            warnings.warn(
                "ObsBuilder: could not read base_linear_velocity from mj_data.qvel; using zeros.",
                stacklevel=3,
            )
            return np.zeros(3, dtype=np.float32)

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

    def _get_joint_positions(self, env: SimEnvContext) -> np.ndarray:
        try:
            qpos = np.asarray(env.mj_data.qpos, dtype=np.float32).flatten()
            joint_pos = qpos[7:7 + self._bundle.num_joints]
            if joint_pos.shape[0] < self._bundle.num_joints:
                joint_pos = np.pad(joint_pos, (0, self._bundle.num_joints - joint_pos.shape[0]))
            return joint_pos.astype(np.float32)
        except Exception:
            return np.zeros(self._bundle.num_joints, dtype=np.float32)

    def _get_joint_velocities(self, env: SimEnvContext) -> np.ndarray:
        try:
            qvel = np.asarray(env.mj_data.qvel, dtype=np.float32).flatten()
            joint_vel = qvel[6:6 + self._bundle.num_joints]
            if joint_vel.shape[0] < self._bundle.num_joints:
                joint_vel = np.pad(joint_vel, (0, self._bundle.num_joints - joint_vel.shape[0]))
            return joint_vel.astype(np.float32)
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
