#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensor Input Nodes
"""

from typing import Dict, Any
from .base_node import BaseNode


class SensorInputNode(BaseNode):
    """Sensor input node"""

    def __init__(self, node_id: str):
        super().__init__(node_id, "sensor_input")
        self.inputs = {}
        self.outputs = {'out': None}
        self.parameters = {
            'sensor_type': 'imu',  # imu, camera, ultrasonic, infrared, odometry
            'robot_model': None
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Read sensor data"""
        sensor_type = self.get_parameter('sensor_type', 'imu')
        robot_model = self.get_parameter('robot_model')

        if robot_model is None:
            return {'out': {'status': 'error', 'message': 'Robot model not set'}}

        try:
            sensor_data = robot_model.get_sensor_data()
            return {
                'out': {
                    'status': 'success',
                    'sensor_type': sensor_type,
                    'data': sensor_data
                }
            }
        except Exception as e:
            return {'out': {'status': 'error', 'message': str(e)}}

    def get_display_name(self) -> str:
        return "Sensor Input"

    def get_description(self) -> str:
        return "Read robot sensor data"

    def to_code(self) -> str:
        sensor_type = self.get_parameter('sensor_type', 'imu')
        return f"# Sensor input: {sensor_type}\nsensor_data = robot.get_sensor_data()\n"
