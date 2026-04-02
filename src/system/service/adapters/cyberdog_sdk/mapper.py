"""Mapping between canonical actions and CyberDog ROS2 motion commands."""

from __future__ import annotations

from typing import Dict


# Canonical → internal command key mapping
ACTION_MAP: Dict[str, str] = {
    "stand": "stand",
    "sit":   "sit",
    "walk":  "walk",
    "stop":  "stop",
}

# CyberDog motion_result_cmd action IDs (cyberdog_ros2 protocol)
# Reference: cyberdog_ros2/cyberdog_decision/decision_maker
MOTION_ID: Dict[str, int] = {
    "stand": 101,   # Trot in place / stand up
    "sit":   111,   # Sit / back-sit
    "stop":  0,     # Stop all motion
}

# Default ROS2 topic names (relative to namespace)
TOPIC_MOTION_CMD = "motion_result_cmd"
TOPIC_CMD_VEL    = "cmd_vel"


def map_action(action: str) -> str:
    """Map canonical action name to CyberDog adapter command key."""
    if not action:
        return ""
    return ACTION_MAP.get(action, action)
