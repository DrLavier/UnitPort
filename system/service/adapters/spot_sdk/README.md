# SpotAdapter — Boston Dynamics Spot MVP

Phase 4 minimum viable adapter for Boston Dynamics Spot under the
UnitPort service layer lifecycle contract.

## Package Layout

```
spot_sdk/
  __init__.py   — exports SpotAdapter, SPOT_SDK_AVAILABLE, map_action
  adapter.py    — SpotAdapter(BaseAdapter) with full lifecycle overrides
  mapper.py     — canonical → Spot command key mapping
  README.md     — this file
```

## Lifecycle Sequence

```
open_session(config)
    1. SDK availability guard     → SESSION_OPEN_FAILED / spot_sdk_unavailable
    2. Auth bootstrap             → SESSION_OPEN_FAILED / spot_auth_failed
       create_standard_sdk → create_robot → authenticate
    3. Time sync check            → SESSION_OPEN_FAILED / spot_time_sync_failed
       robot.time_sync.wait_for_sync(timeout_sec=10)
    4. Lease acquire              → SESSION_OPEN_FAILED / spot_lease_failed
       lease_client.take()

preflight(context)
    1. Session guard              → PREFLIGHT_FAILED / spot_no_session
    2. E-stop / safety channel    → PREFLIGHT_FAILED / spot_estop_active

close_session()
    1. Stand command (safe posture)
    2. Lease release (lease_client.return_lease)
    3. Clear all session state
```

## SDK Unavailability Rule

When `bosdyn-client` is not installed, `open_session` returns
`SESSION_OPEN_FAILED` with `sdk_available: False` and `reason:
spot_sdk_unavailable`.  No exception propagates to the caller.

Install the SDK:
```bash
pip install bosdyn-client
```

## Supported Actions (canonical)

| Canonical | Spot command |
|-----------|--------------|
| `stand`   | `RobotCommandBuilder.stand_command()` |
| `sit`     | `RobotCommandBuilder.safe_power_off_command()` |
| `walk`    | `RobotCommandBuilder.velocity_command(vx, vy, vr)` |
| `stop`    | `RobotCommandBuilder.stop_command()` |

## Connection Config (open_session / connect)

| Key | Default | Description |
|-----|---------|-------------|
| `hostname` | `""` | Spot robot IP or hostname |
| `username` | `"user"` | Authentication username |
| `password` | `""` | Authentication password |
| `time_sync_timeout` | `10.0` | Seconds to wait for time sync |

## Phase 4 Reason Codes

| Code | Stage | Meaning |
|------|-------|---------|
| `spot_sdk_unavailable`  | open_session | bosdyn-client not installed |
| `spot_auth_failed`      | open_session | Authentication error |
| `spot_time_sync_failed` | open_session | Time sync timeout |
| `spot_lease_failed`     | open_session | Lease acquisition error |
| `spot_estop_active`     | preflight    | E-stop is engaged |
| `spot_no_session`       | preflight    | open_session not called |
| `spot_action_failed`    | execute      | Command dispatch error |
