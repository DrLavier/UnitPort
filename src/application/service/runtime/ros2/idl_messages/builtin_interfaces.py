"""IdlStruct definitions for ROS2 ``builtin_interfaces`` messages.

Time and Duration are structurally identical (int32 sec + uint32 nanosec) per
the upstream ROS2 Humble IDL.
"""

from dataclasses import dataclass

from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import int32, uint32


@dataclass
class Time(IdlStruct, typename="builtin_interfaces::msg::dds_::Time_"):
    sec: int32 = 0
    nanosec: uint32 = 0


@dataclass
class Duration(IdlStruct, typename="builtin_interfaces::msg::dds_::Duration_"):
    sec: int32 = 0
    nanosec: uint32 = 0
