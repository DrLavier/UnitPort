# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""IdlStruct definitions for ROS2 ``rosgraph_msgs`` (Clock)."""

from dataclasses import dataclass, field

from cyclonedds.idl import IdlStruct

from application.service.runtime.ros2.idl_messages.builtin_interfaces import Time


@dataclass
class Clock(IdlStruct, typename="rosgraph_msgs::msg::dds_::Clock_"):
    clock: Time = field(default_factory=Time)
