#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensor Input Nodes

These nodes read sensor data from robots. They automatically adapt to different
robot brands/models by using the global RobotContext.

Design:
    - Sensor nodes don't directly depend on any specific robot SDK
    - They use RobotContext to get the appropriate robot model
    - The robot model handles the actual sensor reading
"""

from typing import Dict, Any
from .base_node import BaseNode


class SensorInputNode(BaseNode):
    """
    Sensor input node - reads sensor data from robot.

    This node automatically adapts to the currently selected robot type
    by using RobotContext. The actual sensor reading is delegated
    to the appropriate brand-specific model.
    """

    def __init__(self, node_id: str):
        super().__init__(node_id, "sensor_input")
        self.inputs = {}
        self.outputs = {'out': None}
        self.parameters = {
            'sensor_type': 'imu',  # imu, camera, ultrasonic, infrared, odometry
            'robot_model': None
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Read sensor data from the robot.

        The robot model is obtained in this priority:
        1. From parameters['robot_model'] if explicitly set
        2. From global RobotContext (based on UI robot selection)
        """
        sensor_type = self.get_parameter('sensor_type', 'imu')
        robot_model = self.get_parameter('robot_model')

        # If no robot model provided, try to get from RobotContext
        if robot_model is None:
            try:
                from bin.core.robot_context import RobotContext
                robot_model = RobotContext.get_robot_model()
            except ImportError:
                pass

        if robot_model is None:
            return {'out': {'status': 'error', 'message': 'Robot model not available'}}

        try:
            sensor_data = robot_model.get_sensor_data()
            return {
                'out': {
                    'status': 'success',
                    'sensor_type': sensor_type,
                    'data': sensor_data,
                    'robot_type': getattr(robot_model, 'robot_type', 'unknown')
                }
            }
        except Exception as e:
            return {'out': {'status': 'error', 'message': str(e)}}

    def get_display_name(self) -> str:
        return "Sensor Input"

    def get_description(self) -> str:
        return "Read robot sensor data (IMU, camera, etc.). Adapts to selected robot type."

    def to_code(self) -> str:
        """
        Generate code that uses RobotContext for robot-agnostic sensor reading.
        """
        sensor_type = self.get_parameter('sensor_type', 'imu')
        return (
            f"# Sensor read: {sensor_type}\n"
            f"# Uses RobotContext to automatically select correct robot model\n"
            f"from bin.core.robot_context import RobotContext\n"
            f"sensor_data = RobotContext.get_sensor_data()\n"
        )
