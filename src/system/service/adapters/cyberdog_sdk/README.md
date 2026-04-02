# CyberDogAdapter — XiaoMi CyberDog MVP

Phase 4 minimum viable adapter for XiaoMi CyberDog via ROS2 under the
UnitPort service layer lifecycle contract.

## Package Layout

```
cyberdog_sdk/
  __init__.py   — exports CyberDogAdapter, ROS2_AVAILABLE, map_action
  adapter.py    — CyberDogAdapter(BaseAdapter) with full lifecycle overrides
  mapper.py     — canonical → CyberDog command key / motion ID mapping
  README.md     — this file
```

## Lifecycle Sequence

```
open_session(config)
    1. ROS2 availability guard    → SESSION_OPEN_FAILED / cyberdog_ros2_unavailable
    2. ROS2 context bootstrap     → SESSION_OPEN_FAILED / cyberdog_node_init_failed
       rclpy.init (if not ok) → rclpy.create_node(node_name, namespace=namespace)
    3. Endpoint availability      → advisory (non-blocking in MVP)
       node.count_publishers on motion_result_cmd + cmd_vel
    4. Publisher creation         → non-fatal if fails (session opens anyway)
       cmd_vel (geometry_msgs/Twist) + motion_result_cmd (std_msgs/Int32)

preflight(context)
    1. Session guard              → PREFLIGHT_FAILED / cyberdog_no_session
    2. Timestamp freshness        → advisory log (non-blocking in MVP)
    3. Namespace check            → PREFLIGHT_FAILED / cyberdog_namespace_mismatch

close_session()
    1. Zero velocity publish      (cmd_vel reset)
    2. node.destroy_node()
    3. Clear all session state
```

## SDK Unavailability Rule

When `rclpy` is not installed or ROS2 is not sourced, `open_session` returns
`SESSION_OPEN_FAILED` with `ros2_available: False` and `reason:
cyberdog_ros2_unavailable`.  No exception propagates to the caller.

Install ROS2 and source the environment:
```bash
# Ubuntu/Debian (ROS2 Humble example)
sudo apt install ros-humble-desktop
source /opt/ros/humble/setup.bash
pip install rclpy
```

## Supported Actions (canonical)

| Canonical | ROS2 dispatch | Motion ID |
|-----------|---------------|-----------|
| `stand`   | `motion_result_cmd` Int32 | 101 |
| `sit`     | `motion_result_cmd` Int32 | 111 |
| `walk`    | `cmd_vel` Twist (vx, vy, wz) | — |
| `stop`    | zero `cmd_vel` + motion ID 0 | 0 |

## Topic Names

| Topic (relative to namespace) | Message type | Purpose |
|-------------------------------|-------------|---------|
| `/{namespace}/cmd_vel`            | geometry_msgs/Twist | Velocity control |
| `/{namespace}/motion_result_cmd`  | std_msgs/Int32      | Named motion commands |

## Connection Config (open_session / connect)

| Key | Default | Description |
|-----|---------|-------------|
| `namespace` | `"cyberdog"` | ROS2 namespace for topics |
| `node_name` | `"unitport_cyberdog"` | ROS2 node name |
| `skip_endpoint_check` | `False` | Skip topic publisher count check |

## Phase 4 Reason Codes

| Code | Stage | Meaning |
|------|-------|---------|
| `cyberdog_ros2_unavailable`       | open_session | rclpy not installed / ROS2 not sourced |
| `cyberdog_node_init_failed`       | open_session | rclpy.init or create_node failed |
| `cyberdog_endpoint_unavailable`   | open_session | Motion topics not found (reserved) |
| `cyberdog_mode_precondition_failed` | open_session | Robot in uncontrollable mode (reserved) |
| `cyberdog_no_session`             | preflight    | open_session not called |
| `cyberdog_stale_timestamp`        | preflight    | Clock freshness check failed (reserved) |
| `cyberdog_namespace_mismatch`     | preflight    | Node namespace ≠ configured namespace |
| `cyberdog_action_failed`          | execute      | Topic publish error |
