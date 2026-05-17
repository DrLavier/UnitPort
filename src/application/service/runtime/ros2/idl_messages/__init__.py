"""UnitPort core ROS2 message set for NativeDDSBridge.

Hand-written and reviewed by the UnitPort team. Custom user messages are not
supported here — the v0.2 packaged ``compile-msgs`` tool is the path forward.

This module populates :data:`CORE_REGISTRY` with every type it exports, keyed
by the ROS2 msg_type string (e.g. ``geometry_msgs/msg/Twist``).
"""

from typing import Dict, Tuple, Type, Any

from application.service.runtime.ros2.idl_messages.builtin_interfaces import Duration, Time
from application.service.runtime.ros2.idl_messages.geometry_msgs import (
    Point,
    Pose,
    PoseStamped,
    Quaternion,
    Twist,
    TwistStamped,
    Vector3,
)
from application.service.runtime.ros2.idl_messages.nav_msgs import (
    Odometry,
    PoseWithCovariance,
    TwistWithCovariance,
)
from application.service.runtime.ros2.idl_messages.rosgraph_msgs import Clock
from application.service.runtime.ros2.idl_messages.sensor_msgs import (
    BatteryState,
    CameraInfo,
    Imu,
    JointState,
    LaserScan,
    RegionOfInterest,
)
from application.service.runtime.ros2.idl_messages.std_msgs import (
    Bool,
    Empty,
    Float64MultiArray,
    Header,
    MultiArrayDimension,
    MultiArrayLayout,
    String,
)
from application.service.runtime.ros2.idl_messages.trajectory_msgs import (
    JointTrajectory,
    JointTrajectoryPoint,
)
from application.service.runtime.ros2.idl_messages.unitport_msgs import DeploymentStep, Heartbeat


def _entry(cls: Type[Any]) -> Tuple[str, Type[Any]]:
    """Convert CDR typename ``pkg::msg::dds_::MsgName_`` to ROS2 form
    ``pkg/msg/MsgName`` so callers keep using the familiar ROS2 string.
    """
    cdr = cls.__idl_typename__
    ros = cdr.replace("::dds_::", "::")
    if ros.endswith("_"):
        ros = ros[:-1]
    return ros.replace("::", "/"), cls


CORE_REGISTRY: Dict[str, Type[Any]] = dict(
    [
        _entry(Time),
        _entry(Duration),
        _entry(Header),
        _entry(String),
        _entry(Bool),
        _entry(Empty),
        _entry(MultiArrayDimension),
        _entry(MultiArrayLayout),
        _entry(Float64MultiArray),
        _entry(Vector3),
        _entry(Point),
        _entry(Quaternion),
        _entry(Twist),
        _entry(TwistStamped),
        _entry(Pose),
        _entry(PoseStamped),
        _entry(PoseWithCovariance),
        _entry(TwistWithCovariance),
        _entry(Odometry),
        _entry(JointState),
        _entry(Imu),
        _entry(BatteryState),
        _entry(LaserScan),
        _entry(RegionOfInterest),
        _entry(CameraInfo),
        _entry(Clock),
        _entry(JointTrajectory),
        _entry(JointTrajectoryPoint),
        _entry(DeploymentStep),
        _entry(Heartbeat),
    ]
)


__all__ = ["CORE_REGISTRY"]
