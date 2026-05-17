"""IdlStruct definitions for UnitPort's own ROS2 message set."""

from dataclasses import dataclass, field

from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import float32

from application.service.runtime.ros2.idl_messages.builtin_interfaces import Time


@dataclass
class DeploymentStep(
    IdlStruct, typename="unitport_msgs::msg::dds_::DeploymentStep_"
):
    step_id: str = ""
    status: str = ""
    progress_pct: float32 = 0.0
    detail: str = ""
    timestamp: Time = field(default_factory=Time)


@dataclass
class Heartbeat(
    IdlStruct, typename="unitport_msgs::msg::dds_::Heartbeat_"
):
    timestamp_monotonic: Time = field(default_factory=Time)
    robot_state: str = ""
    active_bundle_name: str = ""
    inference_rate_measured_hz: float32 = 0.0
    systemd_unit_state: str = ""
