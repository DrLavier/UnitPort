#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Protocol Emit Node

Compiles upstream sensor data or a named signal into a motor-weight protocol
payload and emits it downstream (e.g. to a BehaviorCall condition port).

Inputs
------
sensor_data : dict  (optional) — output from an upstream SensorInputNode

Params
------
motor_key   : str  — motor identifier (e.g. "FL_hip")
param_key   : str  — parameter key   (e.g. "stiffness")
address     : str  — protocol address (e.g. "FL_hip.stiffness")
dtype       : str  — value type hint  ("float" by default)
max_age_ms  : int  — payload freshness window in ms (default 1000)
signal      : str  — signal name (SIGNAL_ENABLE / SIGNAL_DISABLE /
                     SIGNAL_SWITCH); empty → sensor path
sensor_field: str  — field in sensor_data["data"] to read (default "x")

Outputs
-------
protocol_payload : dict — validated envelope ready for BehaviorSubgraphInvoker
condition        : dict — alias for protocol_payload (for condition ports)
"""

from typing import Dict, Any

from .base_node import BaseNode


class ProtocolEmitNode(BaseNode):
    """
    Emit a motor-weight protocol payload from sensor data or a named signal.

    Never crashes: bad inputs produce a safe empty payload with status=error.
    """

    def __init__(self, node_id: str):
        super().__init__(node_id, "protocol_emit")
        self.inputs = {"sensor_data": None}
        self.outputs = {"protocol_payload": None, "condition": None}
        self.parameters = {
            "motor_key":    "FL_hip",
            "param_key":    "stiffness",
            "address":      "FL_hip.stiffness",
            "dtype":        "float",
            "max_age_ms":   1000,
            "signal":       "",
            "sensor_field": "x",
        }

    # ------------------------------------------------------------------
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        motor_key    = self.get_parameter("motor_key", "FL_hip")
        param_key    = self.get_parameter("param_key", "stiffness")
        address      = self.get_parameter("address", "FL_hip.stiffness")
        max_age_ms   = int(self.get_parameter("max_age_ms", 1000))
        signal       = self.get_parameter("signal", "")
        sensor_field = self.get_parameter("sensor_field", "x")

        try:
            from compiler.lowering.protocol_compiler import (
                ProtocolPayloadCompiler,
                SIGNAL_SWITCH,
            )

            if signal:
                # Signal convention path
                switch_value = None
                if signal == SIGNAL_SWITCH:
                    switch_value = inputs.get("switch_value", 0.0)
                payload = ProtocolPayloadCompiler.for_signal(
                    signal=signal,
                    motor_key=motor_key,
                    address=address,
                    switch_value=switch_value,
                    max_age_ms=max_age_ms,
                )
            else:
                # Sensor path
                sensor_out = inputs.get("sensor_data") or {}
                if isinstance(sensor_out, dict) and "data" in sensor_out:
                    sensor_data = sensor_out["data"]
                else:
                    sensor_data = sensor_out
                payload = ProtocolPayloadCompiler.from_sensor_imu(
                    sensor_data=sensor_data,
                    motor_key=motor_key,
                    param_key=param_key,
                    sensor_field=sensor_field,
                    address=address,
                    max_age_ms=max_age_ms,
                )

            return {
                "protocol_payload": payload,
                "condition":        payload,
                "status":           "success",
            }

        except Exception as exc:
            _safe = self._safe_empty_payload(motor_key, address, max_age_ms)
            _safe["_error"] = str(exc)
            return {
                "protocol_payload": _safe,
                "condition":        _safe,
                "status":           "error",
                "reason":           str(exc),
            }

    # ------------------------------------------------------------------
    def _safe_empty_payload(self, motor_key: str, address: str,
                            max_age_ms: int) -> Dict[str, Any]:
        """Return a structurally valid but empty payload for error cases."""
        import time, uuid
        return {
            "protocol_version": "1.0",
            "trace_id":         str(uuid.uuid4())[:8],
            "timestamp_ms":     int(time.time() * 1000),
            "mode":             "constant",
            "max_age_ms":       max_age_ms,
            "targets":          [],
        }

    # ------------------------------------------------------------------
    def get_display_name(self) -> str:
        return "Protocol Emit"

    def get_description(self) -> str:
        return (
            "Compile sensor data or a signal into a motor-weight protocol "
            "payload for downstream behavior invocation."
        )
