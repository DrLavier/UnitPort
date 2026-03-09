# Xiaomi CyberDog ROS 2 SDK Notes (for UnitPort Integration)

## 1. What is included in this folder

`models/xiaomi/cyberdog_ros2` is an upstream ROS 2 workspace for CyberDog.

Top-level functional domains:

- `cyberdog_bringup`: startup/launch orchestration.
- `cyberdog_decision`: decision managers (automation, perception, interaction, motion).
- `cyberdog_interfaces`: ROS interface contracts (`msg`, `srv`, `action`).
- `cyberdog_ception`: perception modules (BMS, obstacle, scene, body state, light sensor).
- `cyberdog_interaction`: audio/camera/LED/touch/wireless.
- `cyberdog_common`: shared utilities and vendors (Lifecycle, grpc/lcm vendors, BT libs).

## 2. Platform and runtime model

From project docs:

- ROS 2 distribution: Galactic.
- DDS middleware: Cyclone DDS.
- Target environment: CyberDog device (Jetson/Ubuntu 18.04 based flow in docs).

Unlike Spot SDK's direct gRPC client model, CyberDog control is ROS-native:

- action/service/topic interfaces at ROS 2 layer.
- lower-level motion exchange through LCM channels in `decision_maker/motion_manager`.

## 3. Control interface model (important for Service adapter)

The key motion contracts are in `cyberdog_interfaces/motion_msgs`.

### Actions

- `ChangeMode.action`
- `ChangeGait.action`
- `ExtMonOrder.action`

### Messages

- command: `SE3VelocityCMD`, `Mode`, `Gait`, `MonOrder`, `Parameters`
- state: `ControlState`, `Safety`, `Scene`, `ErrorFlag`, `SE3Pose`, `SE3Velocity`

### Default motion-manager ROS endpoints (from source)

- action servers:
  - `checkout_mode`
  - `checkout_gait`
  - `exe_monorder`
- subscribers:
  - `body_cmd` (remote velocity cmd input)
  - `ObstacleDetection`
  - `para_change`
  - `nav_status`
  - `status_out`
  - `safe_guard`
- publishers:
  - `cmd_out`
  - `gait_out`
  - `status_out`
  - `odom_out`
- service clients:
  - `nav_mode`
  - `obstacle_detection`

## 4. Execution path in motion stack

Observed in `motion_manager.cpp`:

1. Receive ROS motion intents (`body_cmd`, actions, parameter changes).
2. Validate timestamp, mode/gait constraints, frame/source ids.
3. Apply safety/limit checks (velocity/angular/body/gait constraints, low battery scale).
4. Publish ROS outputs (`cmd_out`, `gait_out`, `status_out`, `odom_out`).
5. Bridge to LCM:
   - publish `exec_request` (motion commands),
   - consume `exec_response` and `state_estimator` for feedback/odom.

This means CyberDog adapter needs both ROS 2 integration and LCM-aware assumptions in lower layers.

## 5. Bringup and lifecycle orchestration

`cyberdog_bringup` controls system startup using YAML-driven launch composition:

- `launch_nodes.yaml`: base_nodes vs optional nodes.
- `launch_groups.yaml`: runtime profile selection.
- `default_param.yaml`: node parameters.

`decision_maker` modules use lifecycle patterns (`cascade_manager`) to activate/deactivate dependent node groups.

## 6. Build/deploy summary

Typical workflow in docs:

1. clone workspace,
2. `colcon build --merge-install` (or minimal build up to bringup),
3. deploy to target install path,
4. launch via `ros2 launch cyberdog_bringup lc_bringup_launch.py` or service restart.

## 7. Integration implications for UnitPort IR

For cross-vendor IR design, CyberDog suggests modeling:

- **intent channel type**: action/service/topic (not only RPC).
- **strong typed command/state envelopes**:
  - mode/gait/order are asynchronous actions with feedback + result codes.
  - velocity is stream-style command topic with source/frame constraints.
- **lifecycle orchestration**:
  - mode switching may activate/deactivate node groups.
- **timestamp monotonicity**:
  - many requests are accepted only if newer than last accepted timestamp.
- **hybrid transport**:
  - ROS API outward, LCM bridge inward.

Recommended split:

- **IR-Env**: middleware profile (ROS 2 domain/QoS/lifecycle), safety and launch context.
- **IR-SDK**: typed motion/interaction/perception intents mapped to `msg/srv/action` contracts.

## 8. Risks and caveats

- Current interfaces are highly customized (many redefined messages, reduced generic ROS compatibility).
- Motion manager enforces strict timestamp and mode/gait constraints; adapters must preserve ordering semantics.
- Full behavior may depend on closed-source or device-specific packages not fully present in open workspace.

