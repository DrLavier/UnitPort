"""application.service.runtime.ros2 — Native DDS bridge + IDL infra.

Phase 2 port from DEMO ``src/system/runtime/ros2/``. Hosts:

- ``native_dds_bridge``   in-process cyclonedds-python participant
- ``bridge_protocol``     runtime_checkable Protocol both bridges implement
- ``qos_profiles``        canonical QoS descriptor presets (sensor_data, ...)
- ``idl_registry``        msg_type → IdlStruct dataclass resolver
- ``idl_messages/``       hand-written ROS2 message dataclasses (core set)
- ``native_dds_errors``   structured error taxonomy with stable ``code`` strings
- ``bridge_metrics``      per-topic rate/payload cache
- ``diagnostics``         TopicHealth evaluator
- ``bridge_bringup``      thin factory: ``ConnectionProfile`` + ``Transport``
                          → :class:`NativeDDSBridge`
"""
