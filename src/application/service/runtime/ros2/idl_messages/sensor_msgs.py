"""IdlStruct definitions for ROS2 ``sensor_msgs`` messages used by UnitPort."""

from dataclasses import dataclass, field

from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import array, sequence, float32, float64, uint8, uint32

from application.service.runtime.ros2.idl_messages.std_msgs import Header


@dataclass
class JointState(IdlStruct, typename="sensor_msgs::msg::dds_::JointState_"):
    header: Header = field(default_factory=Header)
    name: sequence[str] = field(default_factory=list)
    position: sequence[float64] = field(default_factory=list)
    velocity: sequence[float64] = field(default_factory=list)
    effort: sequence[float64] = field(default_factory=list)


@dataclass
class Imu(IdlStruct, typename="sensor_msgs::msg::dds_::Imu_"):
    header: Header = field(default_factory=Header)
    orientation_x: float64 = 0.0
    orientation_y: float64 = 0.0
    orientation_z: float64 = 0.0
    orientation_w: float64 = 1.0
    orientation_covariance: array[float64, 9] = field(
        default_factory=lambda: [0.0] * 9
    )
    angular_velocity_x: float64 = 0.0
    angular_velocity_y: float64 = 0.0
    angular_velocity_z: float64 = 0.0
    angular_velocity_covariance: array[float64, 9] = field(
        default_factory=lambda: [0.0] * 9
    )
    linear_acceleration_x: float64 = 0.0
    linear_acceleration_y: float64 = 0.0
    linear_acceleration_z: float64 = 0.0
    linear_acceleration_covariance: array[float64, 9] = field(
        default_factory=lambda: [0.0] * 9
    )


@dataclass
class BatteryState(IdlStruct, typename="sensor_msgs::msg::dds_::BatteryState_"):
    header: Header = field(default_factory=Header)
    voltage: float32 = 0.0
    temperature: float32 = 0.0
    current: float32 = 0.0
    charge: float32 = 0.0
    capacity: float32 = 0.0
    design_capacity: float32 = 0.0
    percentage: float32 = 0.0
    power_supply_status: uint8 = 0
    power_supply_health: uint8 = 0
    power_supply_technology: uint8 = 0
    present: bool = True
    cell_voltage: sequence[float32] = field(default_factory=list)
    cell_temperature: sequence[float32] = field(default_factory=list)
    location: str = ""
    serial_number: str = ""


@dataclass
class LaserScan(IdlStruct, typename="sensor_msgs::msg::dds_::LaserScan_"):
    header: Header = field(default_factory=Header)
    angle_min: float32 = 0.0
    angle_max: float32 = 0.0
    angle_increment: float32 = 0.0
    time_increment: float32 = 0.0
    scan_time: float32 = 0.0
    range_min: float32 = 0.0
    range_max: float32 = 0.0
    ranges: sequence[float32] = field(default_factory=list)
    intensities: sequence[float32] = field(default_factory=list)


@dataclass
class RegionOfInterest(IdlStruct, typename="sensor_msgs::msg::dds_::RegionOfInterest_"):
    x_offset: uint32 = 0
    y_offset: uint32 = 0
    height: uint32 = 0
    width: uint32 = 0
    do_rectify: bool = False


@dataclass
class CameraInfo(IdlStruct, typename="sensor_msgs::msg::dds_::CameraInfo_"):
    header: Header = field(default_factory=Header)
    height: uint32 = 0
    width: uint32 = 0
    distortion_model: str = ""
    d: sequence[float64] = field(default_factory=list)
    k: array[float64, 9] = field(default_factory=lambda: [0.0] * 9)
    r: array[float64, 9] = field(default_factory=lambda: [0.0] * 9)
    p: array[float64, 12] = field(default_factory=lambda: [0.0] * 12)
    binning_x: uint32 = 0
    binning_y: uint32 = 0
    roi: RegionOfInterest = field(default_factory=RegionOfInterest)
