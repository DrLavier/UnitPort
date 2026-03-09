# Behavior Motor Weight Protocol (v1.0)

## Purpose
- Define a strict, structured payload contract from Mission/Script layer to Behavior execution layer.
- Keep Behavior deterministic and minimal: parse payload, map targets, apply values, enforce safety.

## File
- Schema: `system/behavior/behavior_motor_weight_protocol.schema.json`

## Core Principles
- Mission computes, Behavior executes.
- One target entry = one motor param write intent.
- Payload must be schema-valid before apply.
- Heartbeat reads latest payload snapshot each tick.
- Stale/invalid payload must produce diagnostics and skip apply.

## Payload Semantics
- `protocol_version`: fixed `1.0`.
- `timestamp_ms` + `max_age_ms`: used for stale detection.
- `mode`:
  - `constant`: payload carries constant write values.
  - `external`: payload carries externally computed values from script/sensor flow.
- `targets[]`: resolved write intents for motor params.

## Target Semantics
- `motor_key`: stable logical motor id (e.g. `leg_fl_hip`).
- `param_key`: controlled parameter name (`gain`, `amplitude`, etc.).
- `address`: runtime route key (numeric index or symbolic address).
- `dtype`: `float32`/`float64`.
- `shape`:
  - `[1]` for scalar,
  - `[N]` for vector.
- `source.kind`:
  - `constant`: use `source.constant`.
  - `external`: use `source.external_value` (bound from `external_key`).
- `limits`:
  - optional runtime guard; if `clamp=true`, Behavior clamps to `[min,max]`.

## Minimal Runtime Validation Steps
1. Validate JSON payload against schema.
2. Validate staleness: `now_ms - timestamp_ms <= max_age_ms`.
3. Validate each target:
   - shape matches value,
   - dtype accepted,
   - address resolvable.
4. Apply writes atomically per tick.
5. Emit diagnostics for any skipped or clamped target.

## Example (external mode)
```json
{
  "protocol_version": "1.0",
  "trace_id": "mission-abc-001",
  "timestamp_ms": 1760000000123,
  "mode": "external",
  "max_age_ms": 80,
  "quality": "valid",
  "targets": [
    {
      "motor_key": "leg_fl_hip",
      "param_key": "gain",
      "address": "motors.0.gain",
      "dtype": "float32",
      "shape": [1],
      "source": {
        "kind": "external",
        "external_key": "balance.leg_fl_gain",
        "external_value": 1.12
      },
      "limits": {
        "min": 0.5,
        "max": 1.5,
        "clamp": true
      }
    }
  ]
}
```

## Versioning Rules
- Backward-incompatible changes require new `protocol_version`.
- Composite/package nodes must declare protocol version in metadata.
- Mission and Behavior runtime must fail fast on version mismatch.
