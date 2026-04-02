from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet


@dataclass(frozen=True)
class TaskModuleItem:
    key: str
    kind: str
    polarity: str
    title: str
    desc: str
    default: float
    min_value: float
    max_value: float
    step: float
    applicable_families: FrozenSet[str] = field(default_factory=frozenset)


_LOCOMOTION_FAMILIES = frozenset({"quadruped", "biped", "wheeled", "generic_locomotion"})
_LEGGED_FAMILIES = frozenset({"quadruped", "biped", "generic_locomotion"})
_ALL_FAMILIES = frozenset(
    {"quadruped", "biped", "wheeled", "manipulator", "generic_locomotion"}
)


def _reward_item(
    *,
    key: str,
    polarity: str,
    title: str,
    desc: str,
    default: float,
    min_value: float,
    max_value: float,
    step: float,
    applicable_families: FrozenSet[str] = _ALL_FAMILIES,
) -> TaskModuleItem:
    return TaskModuleItem(
        key=key,
        kind="reward",
        polarity=polarity,
        title=title,
        desc=desc,
        default=default,
        min_value=min_value,
        max_value=max_value,
        step=step,
        applicable_families=applicable_families,
    )


def _termination_item(
    *,
    key: str,
    title: str,
    desc: str,
    default: float,
    min_value: float,
    max_value: float,
    step: float,
    applicable_families: FrozenSet[str] = _ALL_FAMILIES,
) -> TaskModuleItem:
    return TaskModuleItem(
        key=key,
        kind="termination",
        polarity="",
        title=title,
        desc=desc,
        default=default,
        min_value=min_value,
        max_value=max_value,
        step=step,
        applicable_families=applicable_families,
    )


REWARD_REGISTRY: Dict[str, TaskModuleItem] = {
    "velocity_tracking": _reward_item(
        key="velocity_tracking",
        polarity="reward",
        title="Velocity Tracking",
        desc="Reward matching commanded forward and lateral velocity.",
        default=1.0,
        min_value=0.0,
        max_value=10.0,
        step=0.05,
        applicable_families=_LOCOMOTION_FAMILIES,
    ),
    "alive": _reward_item(
        key="alive",
        polarity="reward",
        title="Alive Bonus",
        desc="Small survival bonus that encourages staying upright.",
        default=0.5,
        min_value=0.0,
        max_value=10.0,
        step=0.05,
        applicable_families=_LEGGED_FAMILIES,
    ),
    "energy": _reward_item(
        key="energy",
        polarity="penalty",
        title="Energy Penalty",
        desc="Penalty on large actions to reduce wasteful actuation.",
        default=-0.01,
        min_value=-10.0,
        max_value=0.0,
        step=0.01,
    ),
    "yaw_tracking": _reward_item(
        key="yaw_tracking",
        polarity="reward",
        title="Yaw Tracking",
        desc="Reward matching the target yaw-rate command.",
        default=0.5,
        min_value=0.0,
        max_value=10.0,
        step=0.05,
        applicable_families=_LOCOMOTION_FAMILIES,
    ),
    "action_smoothness": _reward_item(
        key="action_smoothness",
        polarity="penalty",
        title="Action Smoothness",
        desc="Penalty for abrupt changes between consecutive actions.",
        default=-0.02,
        min_value=-10.0,
        max_value=0.0,
        step=0.01,
    ),
    "upright": _reward_item(
        key="upright",
        polarity="reward",
        title="Upright Bonus",
        desc="Reward maintaining an upright torso orientation.",
        default=0.3,
        min_value=0.0,
        max_value=10.0,
        step=0.05,
        applicable_families=_LEGGED_FAMILIES,
    ),
    "lateral_penalty": _reward_item(
        key="lateral_penalty",
        polarity="penalty",
        title="Lateral Penalty",
        desc="Penalty for sideways drift when forward tracking is desired.",
        default=-0.1,
        min_value=-10.0,
        max_value=0.0,
        step=0.05,
        applicable_families=_LOCOMOTION_FAMILIES,
    ),
    "foot_clearance": _reward_item(
        key="foot_clearance",
        polarity="reward",
        title="Foot Clearance",
        desc="Reward for lifting feet enough to avoid stumbling during swing.",
        default=0.2,
        min_value=0.0,
        max_value=10.0,
        step=0.05,
        applicable_families=frozenset({"quadruped", "biped"}),
    ),
    "slip_penalty": _reward_item(
        key="slip_penalty",
        polarity="penalty",
        title="Slip Penalty",
        desc="Penalty for excessive foot or wheel slip against the ground.",
        default=-0.1,
        min_value=-10.0,
        max_value=0.0,
        step=0.05,
        applicable_families=_LOCOMOTION_FAMILIES,
    ),
    "collision_penalty": _reward_item(
        key="collision_penalty",
        polarity="penalty",
        title="Collision Penalty",
        desc="Penalty for unwanted collisions with the environment or self.",
        default=-0.2,
        min_value=-10.0,
        max_value=0.0,
        step=0.05,
    ),
    "goal_distance": _reward_item(
        key="goal_distance",
        polarity="reward",
        title="Goal Distance",
        desc="Reward reducing distance to a target pose or position.",
        default=1.0,
        min_value=-5.0,
        max_value=10.0,
        step=0.1,
        applicable_families=frozenset({"manipulator"}),
    ),
    "grasp_success": _reward_item(
        key="grasp_success",
        polarity="reward",
        title="Grasp Success",
        desc="Sparse success bonus for stable grasp or task completion.",
        default=5.0,
        min_value=0.0,
        max_value=20.0,
        step=0.5,
        applicable_families=frozenset({"manipulator"}),
    ),
    "reference_tracking": _reward_item(
        key="reference_tracking",
        polarity="reward",
        title="Reference Tracking",
        desc="Reward for matching a reference motion trajectory keyframe via Gaussian similarity.",
        default=1.0,
        min_value=0.0,
        max_value=10.0,
        step=0.1,
        applicable_families=_LEGGED_FAMILIES,
    ),
    "base_height_penalty": _reward_item(
        key="base_height_penalty",
        polarity="penalty",
        title="Base Height Penalty",
        desc="Penalty for base height deviating from the nominal standing height (~0.32 m). "
             "Discourages crouching or over-extension during locomotion.",
        default=-1.0,
        min_value=-10.0,
        max_value=0.0,
        step=0.1,
        applicable_families=_LOCOMOTION_FAMILIES,
    ),
    "angular_rate_penalty": _reward_item(
        key="angular_rate_penalty",
        polarity="penalty",
        title="Angular Rate Penalty",
        desc="Penalty on squared roll and pitch angular rates. "
             "Reduces trunk wobble and promotes smooth, stable locomotion.",
        default=-0.05,
        min_value=-5.0,
        max_value=0.0,
        step=0.01,
        applicable_families=_LEGGED_FAMILIES,
    ),
    "foot_air_time": _reward_item(
        key="foot_air_time",
        polarity="reward",
        title="Foot Air Time",
        desc="Bonus at each foot touchdown proportional to how long the foot was airborne "
             "(capped at 0.5 s). Encourages regular, symmetric gait cycles.",
        default=0.5,
        min_value=0.0,
        max_value=5.0,
        step=0.05,
        applicable_families=frozenset({"quadruped", "biped"}),
    ),
    "joint_pose_tracking": _reward_item(
        key="joint_pose_tracking",
        polarity="reward",
        title="Joint Pose Tracking",
        desc="Gaussian reward for joint positions matching the reference frame. "
             "Falls back to GO2 default standing pose when no reference motion is loaded.",
        default=1.0,
        min_value=0.0,
        max_value=10.0,
        step=0.1,
        applicable_families=_LEGGED_FAMILIES,
    ),
    "joint_vel_tracking": _reward_item(
        key="joint_vel_tracking",
        polarity="reward",
        title="Joint Velocity Tracking",
        desc="Gaussian reward for joint velocities matching the reference (finite-diff of "
             "reference frames). Falls back to tracking zero velocity when no reference is loaded.",
        default=0.5,
        min_value=0.0,
        max_value=10.0,
        step=0.05,
        applicable_families=_LEGGED_FAMILIES,
    ),
    "foot_pos_tracking": _reward_item(
        key="foot_pos_tracking",
        polarity="reward",
        title="Foot Position Tracking",
        desc="Gaussian reward for foot Cartesian positions matching reference FK positions. "
             "Falls back to nominal standing foot positions when no reference motion is loaded.",
        default=0.5,
        min_value=0.0,
        max_value=10.0,
        step=0.05,
        applicable_families=frozenset({"quadruped", "biped"}),
    ),
}


TERMINATION_REGISTRY: Dict[str, TaskModuleItem] = {
    "fall_threshold_roll": _termination_item(
        key="fall_threshold_roll",
        title="Roll Threshold",
        desc="Terminate when roll exceeds the allowed upright margin.",
        default=1.2,
        min_value=0.2,
        max_value=3.14,
        step=0.05,
        applicable_families=_LEGGED_FAMILIES,
    ),
    "fall_threshold_pitch": _termination_item(
        key="fall_threshold_pitch",
        title="Pitch Threshold",
        desc="Terminate when pitch exceeds the allowed upright margin.",
        default=1.2,
        min_value=0.2,
        max_value=3.14,
        step=0.05,
        applicable_families=_LEGGED_FAMILIES,
    ),
    "min_height": _termination_item(
        key="min_height",
        title="Minimum Height",
        desc="Terminate when the robot base drops below this height.",
        default=0.15,
        min_value=0.05,
        max_value=0.5,
        step=0.01,
        applicable_families=_LOCOMOTION_FAMILIES,
    ),
    "max_contact_impulse": _termination_item(
        key="max_contact_impulse",
        title="Contact Impulse",
        desc="Terminate on excessive impact indicating unstable collapse.",
        default=250.0,
        min_value=50.0,
        max_value=500.0,
        step=5.0,
        applicable_families=_LOCOMOTION_FAMILIES,
    ),
    "timeout": _termination_item(
        key="timeout",
        title="Timeout",
        desc="Terminate when the task exceeds its allowed duration.",
        default=1000.0,
        min_value=10.0,
        max_value=100000.0,
        step=10.0,
    ),
    "joint_limit_violation": _termination_item(
        key="joint_limit_violation",
        title="Joint Limit",
        desc="Terminate when joints exceed safe configured limits. "
             "Threshold is a count: 3 = terminate when 3 or more joints exceed their range.",
        default=3.0,
        min_value=1.0,
        max_value=12.0,
        step=1.0,
        applicable_families=_ALL_FAMILIES,
    ),
    "self_collision": _termination_item(
        key="self_collision",
        title="Self Collision",
        desc="Terminate when forbidden self-collision is detected.",
        default=1.0,
        min_value=0.0,
        max_value=5.0,
        step=0.1,
        applicable_families=frozenset({"manipulator"}),
    ),
}


def reward_registry() -> Dict[str, TaskModuleItem]:
    return dict(REWARD_REGISTRY)


def termination_registry() -> Dict[str, TaskModuleItem]:
    return dict(TERMINATION_REGISTRY)


def default_reward_terms() -> Dict[str, float]:
    return {
        key: REWARD_REGISTRY[key].default
        for key in ("velocity_tracking", "alive", "energy")
    }


def default_termination_conditions() -> Dict[str, float]:
    return {
        key: TERMINATION_REGISTRY[key].default
        for key in (
            "fall_threshold_roll",
            "fall_threshold_pitch",
            "min_height",
            "max_contact_impulse",
            "joint_limit_violation",
        )
    }
