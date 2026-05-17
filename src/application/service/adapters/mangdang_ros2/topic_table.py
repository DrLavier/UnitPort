"""Mangdang Mini Pupper v2 — declared topic table.

Every topic the host-side adapter subscribes to or publishes is declared
here, then registered with :mod:`topic_registry` at package import time.
Adding/removing/renaming a topic is a single edit in this file; the
adapter and the connector pick up the change without further modifications.

Topic names match the UnitPort-deployed Mini Pupper v2 / CHAMP runtime
(brownfield assumption — robot already runs ``unitport-minipupper.service``,
servos owned by CHAMP, /cmd_vel listener live).
"""

from __future__ import annotations

from typing import List

from application.service.adapters.topic_registry import TopicSpec


# Core observation telemetry — subscribed by BaseROS2Adapter._subscribe_declared
# at session open. P8 confirmation watches these for steady-state samples.
OBS_SPECS: List[TopicSpec] = [
    TopicSpec(
        topic="/joint_states",
        msg_type="sensor_msgs/msg/JointState",
        qos_profile="best_effort",
        role="obs",
    ),
    TopicSpec(
        topic="/imu/data",
        msg_type="sensor_msgs/msg/Imu",
        qos_profile="best_effort",
        role="obs",
    ),
    TopicSpec(
        topic="/battery_state",
        msg_type="sensor_msgs/msg/BatteryState",
        qos_profile="best_effort",
        role="obs",
    ),
]


# Outbound command surface — referenced by capabilities() (TeleopCapability
# carries cmd_topic / cmd_msg_type / cmd_qos directly, but listed here so a
# Phase 4 inspector can verify these subscribers exist on the robot side).
CMD_SPECS: List[TopicSpec] = [
    TopicSpec(
        topic="/cmd_vel",
        msg_type="geometry_msgs/msg/Twist",
        qos_profile="reliable",   # CHAMP rejects best_effort; see qos_profiles note
        role="cmd",
    ),
    # /body_pose for static stand pose — declared so future inspector verifies
    # the subscriber, but Phase 3 controller does not publish it (Phase 6+).
    TopicSpec(
        topic="/body_pose",
        msg_type="geometry_msgs/msg/Pose",
        qos_profile="reliable",
        role="cmd",
    ),
]


# Status / heartbeat surface. Heartbeat is published by HeartbeatPublisher
# (not the adapter directly) but listed so verify_topic_hash sees it as part
# of the brand's declared topic table.
STATUS_SPECS: List[TopicSpec] = [
    TopicSpec(
        topic="/unitport/heartbeat",
        msg_type="unitport_msgs/msg/Heartbeat",
        qos_profile="reliable",
        role="status",
    ),
]


# Aggregate every spec in registration order — consumed by __init__.py to
# call topic_registry.register_brand once.
ALL_SPECS: List[TopicSpec] = OBS_SPECS + CMD_SPECS + STATUS_SPECS


__all__ = [
    "OBS_SPECS",
    "CMD_SPECS",
    "STATUS_SPECS",
    "ALL_SPECS",
]
