# Unitree SDK2 Python Real-World Execution Path (Knowledge Base)

This document summarizes the **actual end-to-end execution path** of `unitree_sdk2_python` inside this repository, and maps it to product/framework design requirements for UnitPort.

Scope:
- Source of truth: `models/unitree/unitree_sdk2_python/` (local vendored SDK)
- Focus: deployment, planning, execution, runtime constraints, and required configuration
- Goal: provide a stable reference for both human developers and AI agents

---

## 1. Evidence Base (Local Sources)

Primary references used in this document:

- `models/unitree/unitree_sdk2_python/README.md`
- `models/unitree/unitree_sdk2_python/unitree_sdk2py/core/channel.py`
- `models/unitree/unitree_sdk2_python/unitree_sdk2py/core/channel_config.py`
- `models/unitree/unitree_sdk2_python/unitree_sdk2py/core/channel_name.py`
- `models/unitree/unitree_sdk2_python/unitree_sdk2py/rpc/client.py`
- `models/unitree/unitree_sdk2_python/unitree_sdk2py/rpc/client_base.py`
- `models/unitree/unitree_sdk2_python/unitree_sdk2py/rpc/client_stub.py`
- `models/unitree/unitree_sdk2_python/unitree_sdk2py/rpc/internal.py`
- `models/unitree/unitree_sdk2_python/unitree_sdk2py/rpc/lease_client.py`
- `models/unitree/unitree_sdk2_python/unitree_sdk2py/comm/motion_switcher/*`
- `models/unitree/unitree_sdk2_python/unitree_sdk2py/go2/sport/*`
- `models/unitree/unitree_sdk2_python/unitree_sdk2py/go2/robot_state/*`
- `models/unitree/unitree_sdk2_python/unitree_sdk2py/go2/obstacles_avoid/*`
- `models/unitree/unitree_sdk2_python/unitree_sdk2py/go2/video/*`
- `models/unitree/unitree_sdk2_python/unitree_sdk2py/go2/vui/*`
- `models/unitree/unitree_sdk2_python/example/go2/high_level/go2_sport_client.py`
- `models/unitree/unitree_sdk2_python/example/go2/low_level/go2_stand_example.py`
- `models/unitree/unitree_sdk2_python/example/obstacles_avoid/*`
- `models/unitree/unitree_sdk2_python/example/motionSwitcher/motion_switcher_example.py`
- `models/unitree/unitree_sdk2_python/example/wireless_controller/wireless_controller.py`
- `models/unitree/unitree_sdk2_python/example/g1/readme.md`

---

## 2. What the SDK Really Is

`unitree_sdk2_python` is a communication/control SDK built on top of **CycloneDDS**:

- Topic pub/sub for state/command streams
- RPC request/response for service-style APIs
- Optional lease mechanism for service arbitration

It is not a simple "action API". It is a **multi-layer runtime system** with:
- network binding
- middleware initialization
- service/version compatibility
- mode/service conflict management
- high-level vs low-level control mutual exclusion

---

## 3. Deployment and Initialization Path

### 3.1 Dependency and environment

From SDK README:
- Python >= 3.8
- `cyclonedds==0.10.2`
- `numpy`
- `opencv-python`

If CycloneDDS is not found, `CYCLONEDDS_HOME` or `CMAKE_PREFIX_PATH` must be configured.

### 3.2 Runtime middleware init

All practical scripts first call:

```python
ChannelFactoryInitialize(domain_id, network_interface)
```

This does:
- Create DDS Domain with XML config
- Create DomainParticipant
- Store singleton factory state for channel creation

Network interface handling:
- explicit interface: XML uses `<NetworkInterface name="...">`
- auto mode: XML uses `autodetermine="true"`

### 3.3 Critical implication

If interface is wrong or unreachable, all downstream APIs (RPC/topic) may fail or timeout.
Therefore, "network interface selection + connectivity validation" is a first-class setup step, not an optional field.

---

## 4. Communication Model (SDK Internal)

### 4.1 Topic channels

`ChannelPublisher` / `ChannelSubscriber` are wrappers over DDS `DataWriter` / `DataReader`.

Typical real-time channels in examples:
- `rt/lowcmd`
- `rt/lowstate`
- `rt/lf/lowstate` (for some robot families / examples)
- `rt/utlidar/switch`

### 4.2 RPC channels

RPC client stub uses service-named channels:
- request: `rt/api/{service}/request`
- response: `rt/api/{service}/response`

Request includes:
- request identity (id/api_id)
- lease info
- policy (priority/noReply)

Timeout is controlled via `ClientBase.SetTimeout(...)`.

### 4.3 Lease (optional)

When a client is created with `enableLease=True`, SDK spins a lease thread:
- apply lease
- renew periodically
- include lease id in requests

Note: `go2/sport/SportClient` supports `enableLease` in constructor, but most examples use `False`.

---

## 5. Service Surface and Capabilities

### 5.1 Motion switcher service (`motion_switcher`)

Purpose: mode ownership and control-path arbitration.

Key APIs:
- `CheckMode()`
- `SelectMode(nameOrAlias)` (example aliases: `ai`, `normal`, `advanced`, `ai-w`)
- `ReleaseMode()`

### 5.2 Robot state service (`robot_state`)

Purpose: enumerate and switch robot-side services.

Key APIs:
- `ServiceList()`
- `ServiceSwitch(name, on/off)`
- `SetReportFreq(...)`

`ServiceSwitch` parses status/protect flags and returns specific errors when protected/switch fails.

### 5.3 Sport service (`sport`)

Purpose: high-level locomotion and posture actions.

Representative APIs:
- `StandUp`, `StandDown`, `RecoveryStand`, `Sit`, `RiseSit`
- `Move(vx, vy, vyaw)` (no-reply call)
- `StopMove`
- behavior toggles / special motions (`FreeWalk`, `FreeAvoid`, etc.)

### 5.4 Obstacles avoid service (`obstacles_avoid`)

Purpose: obstacle-avoid mode and motion entry under avoid subsystem.

Key APIs:
- `SwitchSet(on/off)`
- `SwitchGet()`
- `UseRemoteCommandFromApi(bool)`
- `Move(...)`, `MoveToAbsolutePosition(...)`, `MoveToIncrementPosition(...)`

### 5.5 Video and VUI

- `videohub` / front/back videohub (model-dependent)
- `vui` for volume/brightness/switch

### 5.6 Robot-family differences

SDK includes multiple families (`go2`, `b2`, `g1`, `h1`) with different service names and versions.
Example note indicates:
- `idl/unitree_go` for Go2/B2/H1/B2w/Go2w
- `idl/unitree_hg` for G1/H1-2

Therefore, robot model is not just a label; it changes IDL/topic/service expectations.

---

## 6. Real Execution Paths (From Examples)

This section is the most important for framework design.

### 6.1 High-level control path (typical)

1. Initialize DDS (`ChannelFactoryInitialize`)
2. Create service client (e.g., `SportClient`)
3. Set timeout and `Init()`
4. Optionally verify server API version
5. Send action/move commands
6. Stop move / hold posture / exit

Characteristics:
- concise
- suitable for task-level motion commands
- depends on robot-side sport service availability and mode

### 6.2 Low-level control path (critical constraints)

In `example/go2/low_level/go2_stand_example.py`, before low-level command stream:

1. Init DDS
2. Build lowcmd publisher + lowstate subscriber
3. Init `SportClient` + `MotionSwitcherClient`
4. Repeatedly:
   - `CheckMode()`
   - if mode exists: `StandDown()` then `ReleaseMode()`
5. Start recurrent high-frequency lowcmd write loop
6. Use `CRC` and motor command fields (`q/dq/kp/kd/tau`)

This explicitly proves:
- high-level service and low-level stream can conflict
- mode release is a hard operational prerequisite for low-level takeover

### 6.3 Obstacles avoid path

`obstacles_avoid_move.py` shows sequence:

1. Ensure avoid switch is ON (`SwitchGet`/`SwitchSet`)
2. `UseRemoteCommandFromApi(True)`
3. Send move command
4. Stop and restore command source (`UseRemoteCommandFromApi(False)`)

This proves command-source arbitration exists beyond simple motion APIs.

### 6.4 Robot service management path

`robot_service_client_example.py` shows:
- query service list
- stop/start `sport_mode` via `ServiceSwitch`

This indicates runtime service graph is dynamic and must be observable/configurable.

---

## 7. Planning/Execution Model You Should Assume

For production robot tasks, execution is not:
- "Node A -> Node B -> Node C" only.

It is:
1. **Preflight layer**
   - network/interface check
   - middleware initialization
   - service/version compatibility check
   - safety gate check
2. **Mode arbitration layer**
   - check current mode owner
   - select/release mode
   - verify service availability
3. **Task execution layer**
   - high-level action sequences OR low-level stream loops
   - state feedback subscription
   - timeout/retry/cancel logic
4. **Recovery/fallback layer**
   - stop motion
   - restore service state
   - release mode/locks
   - produce machine-readable failure report

---

## 8. Required Configuration Matrix

### 8.1 User-facing required config

Minimum fields for real deployment:
- robot model (go2/b2/g1/h1/...)
- network interface name
- domain id
- control level (high-level / low-level)
- mission safety profile (max speed, timeout, emergency behavior)
- service toggles (avoidance, camera, vui, etc.)

Recommended additional fields:
- command source policy (remote/API priority)
- API timeout defaults
- operation mode alias (`ai`, `normal`, etc.)
- preflight checklist switches

### 8.2 Developer-facing required config

- robot-family adapter map:
  - service names
  - API versions
  - IDL package (`unitree_go` vs `unitree_hg`)
  - topic naming variants
- transport policy:
  - timeout, retry, noReply usage
  - lease usage strategy
- runtime policy:
  - mode lock rules
  - service dependency graph
  - conflict matrix (high-level vs low-level)
- observability:
  - structured logs
  - action trace ids
  - command/result telemetry

---

## 9. Framework Implications for UnitPort

Current simple 2D workflow can represent sequence logic, but cannot safely represent full robotics execution constraints unless upgraded with:

1. **Mission graph layer**
- user-visible task graph (canvas/code)

2. **Behavior/state-machine layer**
- each complex node expands to finite-state machine
- explicit precondition, running, verify, recover states

3. **Service adapter layer**
- `sport`, `robot_state`, `motion_switcher`, `obstacles_avoid`, `video`, `vui`
- unified internal API independent of UI nodes

4. **Runtime orchestration layer**
- event-driven execution
- timeout/retry/cancel
- parallel subscriptions + serial command semantics

5. **Safety layer**
- hard guards before command issue
- speed/torque limits
- emergency stop path
- deterministic teardown

Without these layers, workflow remains demo-level, not engineering-level.

---

## 10. Known Consistency Caveat Inside Vendored SDK

`unitree_sdk2py/test/client/sport_client_example.py` calls methods such as `Trigger`, `BodyHeight`, `GetState`, but the inspected `go2/sport/sport_client.py` in this repo revision does not define these methods.

Interpretation:
- likely version drift / stale tests / mixed revisions
- do not treat tests as definitive API surface
- treat `*_client.py` under service directories as current local source of truth

Action recommendation:
- add adapter-level capability detection and clear unsupported-method errors

---

## 11. Practical End-to-End Checklist (Real Robot)

Before execution:
1. Verify dependency/runtime environment
2. Select robot profile (model + family)
3. Confirm network interface and reachability
4. Init DDS factory
5. Init required service clients
6. Verify server API version compatibility
7. Query service list and mode state
8. Acquire/release mode as needed
9. Run preflight safety checks

During execution:
1. Execute commands via selected control layer
2. Subscribe/monitor state feedback
3. Enforce timeout and safety guardrails
4. Handle command-source arbitration (if avoid/remote involved)

After execution:
1. Stop/hold safe posture
2. Restore switched services or command source
3. Release mode/lease if used
4. Persist logs and execution report

---

## 12. Integration Notes for This UnitPort Repository

Existing UnitPort config already has SDK paths:
- `config/system.ini`:
  - `unitree_sdk = ./models/unitree/unitree_sdk2_python`
  - `unitree_mujoco = ./models/unitree/unitree_mujoco`
  - `unitree_robots = ./models/unitree/unitree_mujoco/unitree_robots`

Current model integration (`models/unitree/unitree_model.py`) primarily uses:
- MuJoCo simulation path
- partial real control via `SportClient` for walk

Missing for full real-robot path:
- explicit motion switcher integration in runtime
- robot_state service management
- obstacles_avoid command-source management
- robust preflight + safety state machine

---

## 13. Recommended Next Implementation Step (Concrete)

Implement a minimal but correct "real robot runtime core" with three services first:
- `motion_switcher`
- `robot_state`
- `sport`

Minimum flow:
1. preflight check
2. mode/service arbitration
3. execute sport action
4. verify state
5. deterministic cleanup

Then extend to:
- `obstacles_avoid`
- `video`
- `vui`
- low-level streaming control

This staged path gives the highest safety/complexity payoff with the least initial risk.

