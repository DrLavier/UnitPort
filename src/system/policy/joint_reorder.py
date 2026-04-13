"""Joint reorder utilities for Isaac Lab ↔ MuJoCo joint order mapping.

Isaac Lab and MuJoCo may enumerate a robot's joints in different order.
This module computes an index array to reorder one to match the other.

Typical Go2 Isaac Lab order (from URDF/USD import):
    FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
    RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf

MuJoCo menagerie order may differ depending on the MJCF definition.

Usage::

    indices = compute_reorder_indices(mujoco_joints, policy_joints)
    reordered = mujoco_values[indices]  # now in policy order
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)


def canonicalize(name: str) -> str:
    """Normalize joint name for fuzzy matching."""
    return name.lower().replace("-", "_").replace(" ", "_").strip()


def compute_reorder_indices(
    source_joints: Sequence[str],
    target_joints: Sequence[str],
) -> Optional[np.ndarray]:
    """Compute index array to reorder *source_joints* to match *target_joints*.

    Returns an int array ``idx`` such that ``source_values[idx]`` gives values
    in *target_joints* order.  Returns ``None`` if orders already match or
    if mapping cannot be established.

    Parameters
    ----------
    source_joints:
        Joint names in the source order (e.g. MuJoCo MJCF order).
    target_joints:
        Joint names in the target order (e.g. Isaac Lab policy order).
    """
    if len(source_joints) != len(target_joints):
        log.warning(
            "Joint count mismatch: source=%d, target=%d",
            len(source_joints), len(target_joints),
        )
        return None

    src_canon = [canonicalize(n) for n in source_joints]
    tgt_canon = [canonicalize(n) for n in target_joints]

    # Already in same order?
    if src_canon == tgt_canon:
        return None

    # Build source index map
    src_map: Dict[str, int] = {}
    for i, name in enumerate(src_canon):
        src_map[name] = i

    indices: List[int] = []
    missing: List[str] = []
    for tgt_name in tgt_canon:
        idx = src_map.get(tgt_name)
        if idx is None:
            missing.append(tgt_name)
        else:
            indices.append(idx)

    if missing:
        log.warning(
            "Cannot compute joint reorder: %d target joints not found in source: %s",
            len(missing), missing[:5],
        )
        return None

    return np.array(indices, dtype=np.int32)


def reorder_array(
    values: np.ndarray,
    indices: Optional[np.ndarray],
) -> np.ndarray:
    """Apply reorder indices to a 1D array. No-op if indices is None."""
    if indices is None:
        return values
    return values[indices]


def build_reorder_from_mapping(
    env_joint_names: Sequence[str],
    joint_mapping: Optional[Dict[str, str]],
) -> Optional[np.ndarray]:
    """Build reorder indices from SkillManifest.joint_mapping.

    ``joint_mapping`` maps policy joint names → robot joint names.
    We need: for each policy joint (in order), find its index in env_joint_names.

    Returns None if mapping is None or orders already match.
    """
    if not joint_mapping:
        return None

    env_canon_map: Dict[str, int] = {
        canonicalize(n): i for i, n in enumerate(env_joint_names)
    }

    indices: List[int] = []
    for _policy_name, robot_name in joint_mapping.items():
        idx = env_canon_map.get(canonicalize(robot_name))
        if idx is None:
            log.warning("Joint '%s' not found in env joints", robot_name)
            return None
        indices.append(idx)

    result = np.array(indices, dtype=np.int32)

    # Check if it's identity (no reorder needed)
    if np.array_equal(result, np.arange(len(indices))):
        return None

    return result
