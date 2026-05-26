# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""IdlStruct definitions for ROS2 ``nav_msgs`` messages used by UnitPort."""

from dataclasses import dataclass, field

from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import array, float64

from application.service.runtime.ros2.idl_messages.std_msgs import Header
from application.service.runtime.ros2.idl_messages.geometry_msgs import Pose, Twist


@dataclass
class PoseWithCovariance(IdlStruct, typename="geometry_msgs::msg::dds_::PoseWithCovariance_"):
    pose: Pose = field(default_factory=Pose)
    covariance: array[float64, 36] = field(default_factory=lambda: [0.0] * 36)


@dataclass
class TwistWithCovariance(IdlStruct, typename="geometry_msgs::msg::dds_::TwistWithCovariance_"):
    twist: Twist = field(default_factory=Twist)
    covariance: array[float64, 36] = field(default_factory=lambda: [0.0] * 36)


@dataclass
class Odometry(IdlStruct, typename="nav_msgs::msg::dds_::Odometry_"):
    header: Header = field(default_factory=Header)
    child_frame_id: str = ""
    pose: PoseWithCovariance = field(default_factory=PoseWithCovariance)
    twist: TwistWithCovariance = field(default_factory=TwistWithCovariance)
