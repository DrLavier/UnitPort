"""IdlStruct definitions for ROS2 ``geometry_msgs`` messages used by UnitPort."""

from dataclasses import dataclass, field

from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import float64

from application.service.runtime.ros2.idl_messages.std_msgs import Header


@dataclass
class Vector3(IdlStruct, typename="geometry_msgs::msg::dds_::Vector3_"):
    x: float64 = 0.0
    y: float64 = 0.0
    z: float64 = 0.0


@dataclass
class Point(IdlStruct, typename="geometry_msgs::msg::dds_::Point_"):
    x: float64 = 0.0
    y: float64 = 0.0
    z: float64 = 0.0


@dataclass
class Quaternion(IdlStruct, typename="geometry_msgs::msg::dds_::Quaternion_"):
    x: float64 = 0.0
    y: float64 = 0.0
    z: float64 = 0.0
    w: float64 = 1.0


@dataclass
class Twist(IdlStruct, typename="geometry_msgs::msg::dds_::Twist_"):
    linear: Vector3 = field(default_factory=Vector3)
    angular: Vector3 = field(default_factory=Vector3)


@dataclass
class TwistStamped(IdlStruct, typename="geometry_msgs::msg::dds_::TwistStamped_"):
    header: Header = field(default_factory=Header)
    twist: Twist = field(default_factory=Twist)


@dataclass
class Pose(IdlStruct, typename="geometry_msgs::msg::dds_::Pose_"):
    position: Point = field(default_factory=Point)
    orientation: Quaternion = field(default_factory=Quaternion)


@dataclass
class PoseStamped(IdlStruct, typename="geometry_msgs::msg::dds_::PoseStamped_"):
    header: Header = field(default_factory=Header)
    pose: Pose = field(default_factory=Pose)
