#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 5-C — Obs/Action contract presets.

Named obs-space layouts used by ObsActionConfigNode and the compat checker.
Each preset pins obs_components, obs_dim, action_type, and target robot_type so
the compatibility checker can compute expected obs_dim from the preset rather
than from the free-form component string.

No Qt.  No SB3.  No file I/O.
"""

from __future__ import annotations

from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Preset name constants
# ---------------------------------------------------------------------------

PRESET_CUSTOM              = "custom"
PRESET_UNITPORT_GO2_V1     = "unitport_go2_v1"
PRESET_COMMUNITY_GO2_SAC_34D = "community_go2_sac_34d"
PRESET_ISAAC_LAB_GO2_VELOCITY_48D = "isaac_lab_go2_velocity_48d"

ALL_PRESET_NAMES: List[str] = [
    PRESET_CUSTOM,
    PRESET_UNITPORT_GO2_V1,
    PRESET_COMMUNITY_GO2_SAC_34D,
    PRESET_ISAAC_LAB_GO2_VELOCITY_48D,
]

# ---------------------------------------------------------------------------
# Contract definitions
# ---------------------------------------------------------------------------
#
# obs_dim is the authoritative expected input size for the policy network.
# Component math for reference (go2 action_dim = 12):
#   unitport_go2_v1      : joint_pos(12) + joint_vel(12) + imu(6)
#                          + velocity_command(3) + previous_action(12) = 45
#   community_go2_sac_34d: joint_pos(12) + joint_vel(12) + imu(6) + command(4) = 34
#
# ---------------------------------------------------------------------------

_CONTRACTS: Dict[str, dict] = {
    PRESET_UNITPORT_GO2_V1: {
        "obs_components": [
            "joint_pos", "joint_vel", "imu", "velocity_command", "previous_action",
        ],
        "obs_dim":     45,
        "action_type": "joint_position",
        "action_dim":  12,
        "robot_type":  "go2",
        "description": "UnitPort Go2 default — 45-d (joint_pos×12 + joint_vel×12 "
                       "+ imu×6 + velocity_command×3 + previous_action×12)",
    },
    PRESET_COMMUNITY_GO2_SAC_34D: {
        "obs_components": [
            "joint_pos", "joint_vel", "imu", "command",
        ],
        "obs_dim":     34,
        "action_type": "torque",
        "action_dim":  12,
        "robot_type":  "go2",
        "description": "Community Go2 SAC — 34-d (joint_pos×12 + joint_vel×12 "
                       "+ imu×6 + command×4); matches sac_unitree_go2_mujoco",
    },
    PRESET_ISAAC_LAB_GO2_VELOCITY_48D: {
        "obs_components": [
            "base_lin_vel", "base_ang_vel", "projected_gravity",
            "velocity_commands", "joint_pos", "joint_vel", "actions",
        ],
        "obs_dim":     48,
        "action_type": "joint_position",
        "action_dim":  12,
        "robot_type":  "go2",
        "description": "Isaac Lab Go2 velocity-tracking — 48-d "
                       "(base_lin_vel×3 + base_ang_vel×3 + projected_gravity×3 "
                       "+ velocity_commands×3 + joint_pos×12 + joint_vel×12 + actions×12)",
        "command_obs_indices": [9, 10, 11],
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_obs_contract(preset_name: str) -> Optional[dict]:
    """
    Return the contract dict for *preset_name*, or ``None`` for
    ``"custom"`` / unknown presets.

    The returned dict has keys: ``obs_components``, ``obs_dim``,
    ``action_type``, ``action_dim``, ``robot_type``, ``description``.
    """
    return _CONTRACTS.get(preset_name)


def list_preset_names() -> List[str]:
    """Return all valid preset names including ``"custom"``."""
    return list(ALL_PRESET_NAMES)


def apply_preset_to_params(preset_name: str, params: dict) -> dict:
    """
    Return a copy of *params* with obs_components and action_type overridden
    by the contract.  No-op when preset_name is ``"custom"`` or unknown.
    """
    contract = get_obs_contract(preset_name)
    if contract is None:
        return params
    out = dict(params)
    out["obs_components"] = " ".join(contract["obs_components"])
    out["action_type"]    = contract["action_type"]
    return out
