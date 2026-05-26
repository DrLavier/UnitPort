# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""generic_state_pub_node — republish the robot's scattered sensor topics
as a unified RobotStateGeneric on /unitport/robot_state.

Subscribes to the topics listed in /etc/unitport/robot_profile.json.sensors
and publishes at the mapping's deploy_freq_hz. Unknown sensors are passed
through via the ``extra_sensors`` array with JSON-serialised payloads.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import BatteryState, Imu, JointState
from std_msgs.msg import Header
from unitport_msgs.msg import RobotStateGeneric, SensorEntry


PROFILE_PATH = Path("/etc/unitport/robot_profile.json")
MAPPING_PATH = Path("/etc/unitport/robot_mapping.json")

DEFAULT_FREQ_HZ = 50.0


class GenericStatePubNode(Node):

    def __init__(self) -> None:
        super().__init__("unitport_generic_state_pub")
        self._session_id = uuid.uuid4().hex[:12]
        self._profile = _read_json(PROFILE_PATH) or {}
        self._mapping = _read_json(MAPPING_PATH) or {}

        self._pub = self.create_publisher(RobotStateGeneric, "/unitport/robot_state", 10)

        self._latest: Dict[str, Any] = {}
        self._extras: Dict[str, SensorEntry] = {}

        self._subscribe_sensors()

        freq = self._resolve_publish_freq()
        self.create_timer(1.0 / max(freq, 1.0), self._publish_state)
        self.get_logger().info(
            f"generic_state_pub_node up — freq={freq}Hz session={self._session_id} "
            f"sensors={[s.get('label') for s in (self._profile.get('sensors') or [])]}"
        )

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    def _subscribe_sensors(self) -> None:
        for s in self._profile.get("sensors") or []:
            label = str(s.get("label", ""))
            topic = str(s.get("topic", ""))
            ros_type = str(s.get("ros_type", ""))
            if not topic or not ros_type:
                continue
            handler = self._handler_for(label, ros_type)
            if handler is None:
                continue
            try:
                msg_cls = _resolve_msg_class(ros_type)
            except Exception as exc:
                self.get_logger().warn(f"state_pub: cannot resolve {ros_type!r}: {exc}")
                continue
            self.create_subscription(msg_cls, topic, handler, 10)

    def _handler_for(self, label: str, ros_type: str) -> Optional[Callable]:
        if label == "joint_state":
            return self._on_joint_state
        if label == "imu":
            return self._on_imu
        if label == "battery":
            return self._on_battery
        return lambda msg, label=label, ros_type=ros_type: self._on_extra(label, ros_type, msg)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _on_joint_state(self, msg: JointState) -> None:
        self._latest["joint_state"] = msg

    def _on_imu(self, msg: Imu) -> None:
        self._latest["imu"] = msg

    def _on_battery(self, msg: BatteryState) -> None:
        self._latest["battery"] = msg

    def _on_extra(self, label: str, ros_type: str, msg: Any) -> None:
        try:
            data_json = _msg_to_json(msg)
        except Exception:
            return
        entry = SensorEntry()
        entry.label = label
        entry.ros_type = ros_type
        entry.data_json = data_json
        self._extras[label] = entry

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------

    def _publish_state(self) -> None:
        msg = RobotStateGeneric()
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "unitport"
        msg.header = header
        msg.joint_state = self._latest.get("joint_state") or JointState()
        msg.imu = self._latest.get("imu") or Imu()
        msg.battery = self._latest.get("battery") or BatteryState()
        msg.extra_sensors = list(self._extras.values())
        msg.kinematic_class = str(
            self._profile.get("kinematics", {}).get("kinematic_class", "")
        )
        msg.controller_type = str(self._mapping.get("controller_type", "") or
                                  self._profile.get("controllers", {}).get("detected_type", ""))
        msg.transport_mode = str(self._mapping.get("transport", "wifi"))
        msg.session_id = self._session_id
        try:
            self._pub.publish(msg)
        except Exception as exc:
            self.get_logger().warn(f"state_pub publish failed: {exc}")

    def _resolve_publish_freq(self) -> float:
        deploy_freq = self._mapping.get("deploy_freq_hz") or {}
        transport = str(self._mapping.get("transport", "wifi"))
        if isinstance(deploy_freq, dict) and transport in deploy_freq:
            try:
                return float(deploy_freq[transport])
            except (TypeError, ValueError):
                pass
        return DEFAULT_FREQ_HZ


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_json(path: Path):
    try:
        return json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _resolve_msg_class(msg_type: str):
    parts = msg_type.split("/")
    if len(parts) == 3 and parts[1] == "msg":
        pkg, _, cls = parts
    elif len(parts) == 2:
        pkg, cls = parts
    else:
        raise ValueError(f"bad msg_type: {msg_type!r}")
    module = __import__(f"{pkg}.msg", fromlist=[cls])
    return getattr(module, cls)


def _msg_to_json(msg) -> str:
    """Shallow introspection → JSON. Works for simple POD fields; complex
    nested arrays degrade to empty strings."""
    out: Dict[str, Any] = {}
    for slot in getattr(msg, "__slots__", []) or []:
        name = slot.lstrip("_")
        try:
            v = getattr(msg, name)
        except Exception:
            continue
        out[name] = _scalar_or_repr(v)
    try:
        return json.dumps(out, ensure_ascii=False)
    except TypeError:
        return "{}"


def _scalar_or_repr(v):
    if isinstance(v, (int, float, bool, str)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_scalar_or_repr(x) for x in v]
    # ROS time stamps, nested messages — fall back to repr.
    return repr(v)


def main(args=None):
    rclpy.init(args=args)
    node = GenericStatePubNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
