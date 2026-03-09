# Boston Dynamics Spot SDK Notes (for UnitPort Integration)

## 1. What is included in this folder

`models/bostion_dynamics/spot-sdk` is a full upstream mirror of Boston Dynamics Spot SDK, currently at version `5.1.1` (from `spot-sdk/VERSION`).

Main parts:

- `docs/concepts/`: architecture and operational concepts.
- `protos/bosdyn/api/`: protobuf contract for all Spot services.
- `python/bosdyn-client`: main Python client library.
- `python/bosdyn-mission`: mission service client wrappers.
- `python/examples`: runnable reference flows (teleop, graph nav, image, mission, arm, etc).
- `prebuilt/*.whl`: prebuilt wheels for offline installation.

## 2. SDK interaction model

Spot uses a client-server gRPC model over IP networking.

Typical control sequence is:

1. Create SDK and robot handle.
2. Authenticate user (JWT token, robot-bound).
3. Start/verify time synchronization.
4. Verify software safety state (E-Stop / Keepalive policy).
5. Acquire lease ownership.
6. Power on robot.
7. Send robot commands and monitor feedback/state.
8. Power off and return lease.

The `python/examples/hello_spot/hello_spot.py` sample shows this flow end to end.

## 3. Core platform mechanisms (must model in Service layer)

### Authentication

- Service: `auth`.
- Token-based (JWT), required by almost all services.

### Service discovery

- Service: `directory`.
- Dynamic listing of available built-in and payload services.

### Time sync

- Service: `time-sync`.
- Required for command timestamps/expirations.

### Ownership and concurrency control

- Service: `lease`.
- Single-owner semantics for control resources (`body`, sub-resources).
- Retain/refresh lifecycle matters for long-running tasks.

### Software safety

- Service: `estop` (legacy and still available).
- Service: `keepalive` (newer policy-based comms-loss actions; can trigger autostop/autoreturn/power-off/lease stale).

### Motion and state

- Command plane: `robot_command`, `power`.
- State plane: `robot_state`, `image`, `local_grid`, `world_object`.
- Autonomy/navigation: `graph_nav`, recording/map processing.

## 4. API surface breadth

`bosdyn-client` exposes many service clients, including:

- base: `auth`, `directory`, `time_sync`, `lease`, `estop`, `keepalive`.
- robot control: `robot_command`, `power`, `docking`, `spot_check`.
- perception/data: `robot_state`, `image`, `point_cloud`, `world_object`, `data_buffer`, `data_acquisition`.
- autonomy: `graph_nav`, `auto_return`, `autowalk`.
- payload extension: payload registration/update and custom services.

This confirms Spot SDK is not a single "motion API", but a full robot platform API.

## 5. Install options

Two practical options in this repo:

1. Install from prebuilt wheels in `spot-sdk/prebuilt`.
2. Install from python package sources under `spot-sdk/python`.

The SDK depends on gRPC, PyJWT, NumPy (and transitive dependencies).

## 6. Integration implications for UnitPort IR

For IR portability, Spot adapter should treat these capabilities as first-class:

- `session/auth`: robot endpoint + credential/token.
- `clock sync`: offset-aware timestamp conversion.
- `ownership`: lease acquire/keepalive/return.
- `safety policy`: estop or keepalive policy binding.
- `actuation`: stand/sit/body pose/velocity/mobility command.
- `state query`: robot state + health + fault + image/local map/world objects.
- `navigation`: optional graph-nav mission primitives.

Recommended split:

- **IR-Env**: connection, identity, auth, time sync, policy/safety, leases.
- **IR-SDK**: typed command/state RPC intents mapped to `bosdyn.api.*` services.

## 7. Risks and constraints

- SDK license is Boston Dynamics SDK license (not pure MIT for this vendored folder).
- Time sync and lease handling are mandatory for robust command execution.
- E-Stop and Keepalive behavior differs by robot software version; adapter should query capabilities/version and branch safely.

