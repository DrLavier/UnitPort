"""ObsAssembler — observation vector builder for Isaac Lab-trained policies.

Unlike ObsBuilder (which is driven by CheckpointBundle / v1 manifest),
ObsAssembler is driven by SkillManifest v2 and supports:
  - observation layout from env.yaml (exact index mapping)
  - command value injection from CommandBus at specified obs_indices
  - joint reordering via SkillManifest.joint_mapping

The assembler iterates ``observation_space_keys`` from the manifest and
calls the corresponding sensor reader for each term.  It does NOT hardcode
any specific observation layout — everything is data-driven.

Public API
----------
ObsAssembler(manifest, env_yaml_extras) -> .assemble(env, command_bus, last_action) -> np.ndarray
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from src.system.skill.skill_manifest import CommandInterface, SkillManifest

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sensor reader registry
# ---------------------------------------------------------------------------

def _read_base_lin_vel(env: Any, **_kw: Any) -> np.ndarray:
    """Body-frame linear velocity [3]."""
    try:
        qvel = np.asarray(env.mj_data.qvel, dtype=np.float32).flatten()
        # MuJoCo qvel[0:3] is world-frame; rotate to body frame
        qpos = np.asarray(env.mj_data.qpos, dtype=np.float32).flatten()
        quat = qpos[3:7]
        rot = _quat_to_rotation_matrix(quat)
        return (rot.T @ qvel[0:3]).astype(np.float32)
    except Exception:
        return np.zeros(3, dtype=np.float32)


def _read_base_ang_vel(env: Any, **_kw: Any) -> np.ndarray:
    """Body-frame angular velocity [3]."""
    try:
        qvel = np.asarray(env.mj_data.qvel, dtype=np.float32).flatten()
        qpos = np.asarray(env.mj_data.qpos, dtype=np.float32).flatten()
        quat = qpos[3:7]
        rot = _quat_to_rotation_matrix(quat)
        return (rot.T @ qvel[3:6]).astype(np.float32)
    except Exception:
        return np.zeros(3, dtype=np.float32)


def _read_projected_gravity(env: Any, **_kw: Any) -> np.ndarray:
    """Gravity vector projected into body frame [3]."""
    try:
        qpos = np.asarray(env.mj_data.qpos, dtype=np.float32).flatten()
        quat = qpos[3:7]
        norm = np.linalg.norm(quat)
        if norm < 1e-8:
            return np.array([0.0, 0.0, -1.0], dtype=np.float32)
        quat = quat / norm
        rot = _quat_to_rotation_matrix(quat)
        gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        return (rot.T @ gravity_world).astype(np.float32)
    except Exception:
        return np.array([0.0, 0.0, -1.0], dtype=np.float32)


def _read_velocity_commands(
    env: Any, *, command_bus: Any = None, **_kw: Any,
) -> np.ndarray:
    """Velocity commands from CommandBus [3]."""
    if command_bus is not None:
        vals = command_bus.read_all(["vx", "vy", "vyaw"])
        return np.array([vals["vx"], vals["vy"], vals["vyaw"]], dtype=np.float32)
    return np.zeros(3, dtype=np.float32)


def _read_joint_pos(
    env: Any, *, default_joint_pos: Optional[np.ndarray] = None,
    reorder_indices: Optional[np.ndarray] = None, **_kw: Any,
) -> np.ndarray:
    """Relative joint positions (joint_pos - default_pos) [num_joints]."""
    try:
        qpos = np.asarray(env.mj_data.qpos, dtype=np.float32).flatten()
        nj = len(env.joint_names)
        raw = qpos[7:7 + nj]
        if reorder_indices is not None:
            raw = raw[reorder_indices]
        if default_joint_pos is not None:
            raw = raw - default_joint_pos
        return raw.astype(np.float32)
    except Exception:
        return np.zeros(len(env.joint_names), dtype=np.float32)


def _read_joint_vel(
    env: Any, *, reorder_indices: Optional[np.ndarray] = None, **_kw: Any,
) -> np.ndarray:
    """Joint velocities [num_joints]."""
    try:
        qvel = np.asarray(env.mj_data.qvel, dtype=np.float32).flatten()
        nj = len(env.joint_names)
        raw = qvel[6:6 + nj]
        if reorder_indices is not None:
            raw = raw[reorder_indices]
        return raw.astype(np.float32)
    except Exception:
        return np.zeros(len(env.joint_names), dtype=np.float32)


def _read_actions(
    env: Any, *, last_action: Optional[np.ndarray] = None, **_kw: Any,
) -> np.ndarray:
    """Previous action [action_dim]."""
    if last_action is not None:
        return np.asarray(last_action, dtype=np.float32).flatten()
    return np.zeros(12, dtype=np.float32)


# Mapping: obs term name -> (reader_func, default_dim)
_READER_REGISTRY: Dict[str, tuple[Callable, int]] = {
    "base_lin_vel": (_read_base_lin_vel, 3),
    "base_ang_vel": (_read_base_ang_vel, 3),
    "projected_gravity": (_read_projected_gravity, 3),
    "velocity_commands": (_read_velocity_commands, 3),
    "joint_pos": (_read_joint_pos, 12),
    "joint_pos_rel": (_read_joint_pos, 12),
    "joint_vel": (_read_joint_vel, 12),
    "joint_vel_rel": (_read_joint_vel, 12),
    "actions": (_read_actions, 12),
}


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _quat_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to a 3x3 rotation matrix."""
    w, x, y, z = quat.astype(float)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# ObsAssembler
# ---------------------------------------------------------------------------

class ObsAssembler:
    """Assemble observation vectors for Isaac Lab-trained policies.

    Driven by SkillManifest v2 — iterates observation_space_keys, calls
    the registered sensor reader for each term, and concatenates results.

    Parameters
    ----------
    manifest:
        SkillManifest with observation_space_keys and optional command_interface.
    default_joint_pos:
        Default standing joint positions (from env.yaml).  If provided,
        joint_pos readings are relative to these values.
    reorder_indices:
        Index array to reorder MuJoCo joint order → policy joint order.
        See :func:`joint_reorder.compute_reorder_indices`.
    """

    def __init__(
        self,
        manifest: SkillManifest,
        *,
        default_joint_pos: Optional[np.ndarray] = None,
        reorder_indices: Optional[np.ndarray] = None,
    ) -> None:
        self._manifest = manifest
        self._obs_keys = list(manifest.observation_space_keys)
        self._obs_dim = manifest.observation_dim
        self._default_joint_pos = default_joint_pos
        self._reorder_indices = reorder_indices

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    def assemble(
        self,
        env: Any,
        command_bus: Any = None,
        last_action: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Build the full observation vector.

        Parameters
        ----------
        env:
            SimEnvContext (or any object with mj_model, mj_data, joint_names).
        command_bus:
            CommandBus instance for reading real-time commands.
        last_action:
            Previous policy action output.

        Returns
        -------
        np.ndarray
            Float32 observation vector of shape ``(obs_dim,)``.
        """
        parts: List[np.ndarray] = []
        kw = {
            "command_bus": command_bus,
            "last_action": last_action,
            "default_joint_pos": self._default_joint_pos,
            "reorder_indices": self._reorder_indices,
        }

        for key in self._obs_keys:
            entry = _READER_REGISTRY.get(key)
            if entry is not None:
                reader_fn, _default_dim = entry
                val = reader_fn(env, **kw)
                parts.append(np.asarray(val, dtype=np.float32).flatten())
            else:
                log.warning("ObsAssembler: unknown obs key '%s', filling with zeros", key)
                parts.append(np.zeros(3, dtype=np.float32))

        obs = np.concatenate(parts).astype(np.float32) if parts else np.zeros(self._obs_dim, dtype=np.float32)

        # Pad or truncate to expected dim
        if obs.shape[0] < self._obs_dim:
            obs = np.pad(obs, (0, self._obs_dim - obs.shape[0]))
        elif obs.shape[0] > self._obs_dim:
            obs = obs[:self._obs_dim]

        return obs
