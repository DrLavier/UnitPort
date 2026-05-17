"""Single entry launch file for the UnitPort robot-side stack.

Semantics:
  * ``unitport:=false`` (default) returns an empty LaunchDescription.
  * ``unitport:=true`` spawns estop → bridge → state_pub → param_srv.

The systemd unit always passes ``unitport:=true``; the ``--unitport`` CLI
convention is the *target* isolation (systemctl isolate unitport.target),
not the launch arg. See knowledge_base/ROS2_onboarding.yaml Section 4.
"""

from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _spawn_if_enabled(context, *_args, **_kwargs):
    arg = LaunchConfiguration("unitport").perform(context)
    if arg.lower() not in ("true", "1", "yes", "on"):
        return []
    return [
        Node(package="unitport_estop",
             executable="estop_node",
             name="unitport_estop",
             output="screen"),
        Node(package="unitport_bridge",
             executable="generic_bridge_node",
             name="unitport_generic_bridge",
             output="screen"),
        Node(package="unitport_state_pub",
             executable="generic_state_pub_node",
             name="unitport_generic_state_pub",
             output="screen"),
        Node(package="unitport_param_srv",
             executable="param_srv_node",
             name="unitport_param_srv",
             output="screen"),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "unitport", default_value="false",
            description="Enable UnitPort mode (all nodes up). "
                        "systemd units always pass true; manual invocation "
                        "defaults to false so nothing surprising happens.",
        ),
        OpaqueFunction(function=_spawn_if_enabled),
    ])
