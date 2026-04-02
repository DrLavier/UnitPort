# UnitPort IR Developer Manual (v0.1)

## 1. Scope

This manual is for developers/agents working on UnitPort IR authoring, transpilation, and adapter mapping.

The immediate target is a vendor-neutral IR that can be mapped to:

- Unitree SDK style adapters,
- Boston Dynamics Spot SDK service clients,
- Xiaomi CyberDog ROS2 action/service/topic interfaces,
- MuJoCo simulation runner.

## 2. IR Design Goals

- Single semantic source for Canvas + Compiler.
- Sim-to-real continuity: same intent, different backend adapters.
- Explicit lifecycle semantics: connect, prepare, execute, monitor, stop.
- Observable execution: progress, result, fault, and audit metadata.
- Safety-first: constraints and safety policies are in IR, not hidden in adapters.

## 3. Common Cross-SDK Semantics (baseline)

Across current SDK families, IR must cover:

1. Session and connectivity
- endpoint/transport/profile selection,
- authentication token or credential context,
- optional time synchronization.

2. Ownership and safety gating
- control ownership/lease-like semantics,
- safety stop/keepalive policy semantics,
- pre-execution checks.

3. Command and feedback
- asynchronous command model (`submit -> ack -> progress -> result`),
- cancellation/preemption/timeout behavior,
- typed error categories.

4. State and telemetry
- robot mode/state snapshots,
- faults and warnings,
- runtime metrics/events.

5. Environment target
- simulation target (MuJoCo),
- real device target (SDK/ROS backend).

## 4. IR Document Structure (high level)

Each readable IR file should have these top-level sections:

- `meta`: versioning and traceability.
- `target`: runtime target and backend profile.
- `session`: connectivity/auth/ownership/time-sync policy.
- `capabilities`: expected capability set and feature gates.
- `mission`: orchestration graph and flow.
- `behavior`: node-level behavior logic bindings.
- `commands`: normalized executable intents.
- `safety`: constraints, guardrails, stop/degrade rules.
- `observability`: events, metrics, log/audit settings.
- `execution`: timeout/retry/preemption runtime policy.

## 5. Directory Rules

- Authoring/transpilation code: `system/ir/workbench/code/`
- Readable outputs for review: `system/ir/workbench/readable/`
- Canonical templates: `system/ir/workbench/templates/`
- Temporary files only: `system/ir/workbench/tmp/`

Do not place temporary or experimental IR artifacts into stable runtime modules directly.

## 6. Versioning Rules (initial)

- Readable template version starts from `0.1`.
- Breaking schema changes must increase major (`0.x -> 1.0` when stabilized).
- Non-breaking field additions increase minor (`0.1 -> 0.2`).

## 7. Implementation Strategy (current recommendation)

1. Keep readable IR in YAML for fast iteration and review.
2. Add a normalization step to canonical internal Python structures.
3. Map canonical structures to vendor adapters in Service layer.
4. Add strict validator checks before runtime dispatch.

## 8. Out of Scope for v0.1

- Full formal schema freeze.
- Full multi-language compiler support.
- Complete capability mapping for all robot models.

These will be added incrementally.

