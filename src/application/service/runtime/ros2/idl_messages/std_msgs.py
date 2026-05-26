# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""IdlStruct definitions for ROS2 ``std_msgs`` core messages."""

from dataclasses import dataclass, field

from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import sequence, float64, uint32

from application.service.runtime.ros2.idl_messages.builtin_interfaces import Time


@dataclass
class Header(IdlStruct, typename="std_msgs::msg::dds_::Header_"):
    stamp: Time = field(default_factory=Time)
    frame_id: str = ""


@dataclass
class String(IdlStruct, typename="std_msgs::msg::dds_::String_"):
    data: str = ""


@dataclass
class Bool(IdlStruct, typename="std_msgs::msg::dds_::Bool_"):
    data: bool = False


@dataclass
class Empty(IdlStruct, typename="std_msgs::msg::dds_::Empty_"):
    structure_needs_at_least_one_member: bool = False


@dataclass
class MultiArrayDimension(IdlStruct, typename="std_msgs::msg::dds_::MultiArrayDimension_"):
    label: str = ""
    size: uint32 = 0
    stride: uint32 = 0


@dataclass
class MultiArrayLayout(IdlStruct, typename="std_msgs::msg::dds_::MultiArrayLayout_"):
    dim: sequence[MultiArrayDimension] = field(default_factory=list)
    data_offset: uint32 = 0


@dataclass
class Float64MultiArray(IdlStruct, typename="std_msgs::msg::dds_::Float64MultiArray_"):
    layout: MultiArrayLayout = field(default_factory=MultiArrayLayout)
    data: sequence[float64] = field(default_factory=list)
