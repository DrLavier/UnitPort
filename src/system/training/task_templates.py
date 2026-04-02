from __future__ import annotations

from typing import Any, Dict


TaskTemplate = Dict[str, Any]


def _template(
    *,
    reward_terms: Dict[str, float] | None = None,
    termination_conditions: Dict[str, float] | None = None,
    precondition_posture: str = "any",
    postcondition_posture: str = "standing",
) -> TaskTemplate:
    return {
        "reward_terms": dict(reward_terms or {}),
        "termination_conditions": dict(termination_conditions or {}),
        "precondition_posture": precondition_posture,
        "postcondition_posture": postcondition_posture,
    }


TASK_TEMPLATE_FAMILY: Dict[str, TaskTemplate] = {
    "generic_locomotion": _template(
        reward_terms={
            "velocity_tracking": 1.0,
            "alive": 0.5,
            "energy": -0.01,
        },
        termination_conditions={
            "fall_threshold_roll": 1.2,
            "min_height": 0.15,
        },
        precondition_posture="standing",
        postcondition_posture="standing",
    ),
    "quadruped": _template(
        reward_terms={
            "velocity_tracking": 1.0,
            "alive": 0.5,
            "energy": -0.01,
            "action_smoothness": -0.02,
        },
        termination_conditions={
            "fall_threshold_roll": 1.2,
            "fall_threshold_pitch": 1.2,
            "min_height": 0.18,
        },
        precondition_posture="standing",
        postcondition_posture="standing",
    ),
    "biped": _template(
        reward_terms={
            "velocity_tracking": 1.0,
            "yaw_tracking": 0.5,
            "alive": 1.0,
            "upright": 0.8,
            "energy": -0.01,
        },
        termination_conditions={
            "fall_threshold_roll": 0.8,
            "fall_threshold_pitch": 0.8,
            "min_height": 0.35,
        },
        precondition_posture="standing",
        postcondition_posture="standing",
    ),
    "wheeled": _template(
        reward_terms={
            "velocity_tracking": 1.0,
            "yaw_tracking": 0.5,
            "energy": -0.01,
            "lateral_penalty": -0.1,
        },
        termination_conditions={
            "min_height": 0.18,
            "max_contact_impulse": 250.0,
        },
        precondition_posture="any",
        postcondition_posture="any",
    ),
    "manipulator": _template(
        reward_terms={
            "goal_distance": 1.0,
            "energy": -0.01,
            "action_smoothness": -0.02,
            "collision_penalty": -0.2,
        },
        termination_conditions={
            "timeout": 1000.0,
            "joint_limit_violation": 1.0,
            "self_collision": 1.0,
        },
        precondition_posture="any",
        postcondition_posture="any",
    ),
}


TASK_TEMPLATE_ROBOT: Dict[str, TaskTemplate] = {
    "go2": _template(
        reward_terms={
            "velocity_tracking": 2.0,
            "alive": 0.8,
        },
        termination_conditions={
            "min_height": 0.18,
        },
    ),
    "go2w": _template(
        reward_terms={
            "yaw_tracking": 0.8,
            "lateral_penalty": -0.2,
        },
        termination_conditions={
            "min_height": 0.20,
        },
    ),
    "h1": _template(
        reward_terms={
            "upright": 1.2,
        },
        termination_conditions={
            "min_height": 0.45,
        },
    ),
    "g1": _template(
        reward_terms={
            "upright": 1.0,
            "yaw_tracking": 0.6,
        },
        termination_conditions={
            "min_height": 0.40,
        },
    ),
}


TASK_TEMPLATE_ASSET: Dict[str, TaskTemplate] = {
    # Asset-level overrides can now be supplied by
    # training_assets/<asset_id>/task_template.json.
}
