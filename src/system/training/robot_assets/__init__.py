"""Robot asset registry — phase_1 of AMP_design.yaml.

Receive + validate user-supplied URDF / USD / MJCF files. **No conversion.**
Single source of truth for joint_names / dof_count / link tree, shared by
training (Isaac Lab) and sim2sim (MuJoCo) sides.

See ``knowledge_base/AMP_design.yaml`` §3.custom_robot_assets for the
contract this package implements.
"""
from __future__ import annotations

from src.system.training.robot_assets.registry import (
    RobotAsset,
    RobotAssetValidationError,
    register_asset,
    get_asset,
    list_assets,
    resolve_for_training,
    resolve_for_mujoco,
    has_asset,
    clear_registry,
    rescan,
)
from src.system.training.robot_assets.discovery import (
    DiscoveryReport,
    RejectedCandidate,
    walk_and_register,
)

__all__ = [
    # registry
    "RobotAsset",
    "RobotAssetValidationError",
    "register_asset",
    "get_asset",
    "list_assets",
    "resolve_for_training",
    "resolve_for_mujoco",
    "has_asset",
    "clear_registry",
    "rescan",
    # discovery
    "DiscoveryReport",
    "RejectedCandidate",
    "walk_and_register",
]
