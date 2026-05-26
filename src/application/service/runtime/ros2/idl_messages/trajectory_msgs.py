# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""IdlStruct definitions for ROS2 ``trajectory_msgs`` used by champ controllers."""

from dataclasses import dataclass, field

from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import sequence, float64

from application.service.runtime.ros2.idl_messages.std_msgs import Header
from application.service.runtime.ros2.idl_messages.builtin_interfaces import Duration


@dataclass
class JointTrajectoryPoint(
    IdlStruct, typename="trajectory_msgs::msg::dds_::JointTrajectoryPoint_"
):
    positions: sequence[float64] = field(default_factory=list)
    velocities: sequence[float64] = field(default_factory=list)
    accelerations: sequence[float64] = field(default_factory=list)
    effort: sequence[float64] = field(default_factory=list)
    time_from_start: Duration = field(default_factory=Duration)


@dataclass
class JointTrajectory(IdlStruct, typename="trajectory_msgs::msg::dds_::JointTrajectory_"):
    header: Header = field(default_factory=Header)
    joint_names: sequence[str] = field(default_factory=list)
    points: sequence[JointTrajectoryPoint] = field(default_factory=list)
