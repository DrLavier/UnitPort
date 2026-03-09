# Behavior Layer

`system/behavior` implements node-internal behavior constructs — subgraphs, state
machines, and policy logic that run _inside_ Mission nodes.

---

## Phase 2 — Behavior Layer Productization (STAGE-01 through STAGE-07)

Phase 2 turns Behavior from UI placeholder / action alias into a real executable
behavior subgraph system, delivering:

```
Canvas / Compiler
       │
       ▼
BehaviorCompilerBridge.compile()
       │  source → BehaviorArtifact (artifact_id, behavior_ir, trace_id, diagnostics)
       ▼
In-memory registry (keyed by behavior_ref)
       │
       ▼
RuntimeEngine.execute(WorkflowIR)
       │  detects schema_id="behavior" node
       ▼
NodeExecutor._execute_behavior_node()
       │  injects mission_trace_id, builds BehaviorInvokeInput
       ▼
BehaviorSubgraphInvoker.invoke()
       │  resolve → load IR → execute subgraph → collect results
       ▼
BehaviorInvokeOutput  →  Mission node result  →  RuntimeResult.diagnostics
```

---

## §1  Behavior Contract (behavior_artifact.py)

### BehaviorErrorCode

Machine-readable reason strings for all failure paths.

| Constant | Value |
|----------|-------|
| `BEHAVIOR_REF_NOT_FOUND` | `"behavior_ref_not_found"` |
| `ARTIFACT_INVALID` | `"artifact_invalid"` |
| `SUBGRAPH_EXECUTION_FAILED` | `"subgraph_execution_failed"` |
| `SUBGRAPH_ABORTED` | `"subgraph_aborted"` |
| `BEHAVIOR_COMPILE_ERROR` | `"behavior_compile_error"` |

### BehaviorDiagnostic

Single diagnostic entry (compile-time or runtime).

| Field | Type | Description |
|-------|------|-------------|
| `level` | `str` | `"error"` / `"warning"` / `"info"` |
| `code` | `str` | Machine-readable identifier |
| `message` | `str` | Human-readable description |
| `location` | `Optional[str]` | Source reference — `file:line` or `node_id` |
| `trace_id` | `str` | UUID; defaults to fresh `uuid4()` |

Factory class methods: `.error()`, `.warning()`, `.info()`.

### BehaviorArtifact

Compiled, executable behavior payload.  Produced by `BehaviorCompilerBridge.compile()`;
consumed by `BehaviorSubgraphInvoker` at runtime.

| Field | Type | Description |
|-------|------|-------------|
| `artifact_id` | `str` | UUID per compile invocation |
| `behavior_ref` | `str` | Registry key — canonical name (e.g. `"stand"`) |
| `behavior_ir` | `WorkflowIR` | Compiled subgraph — executable payload |
| `diagnostics` | `List[BehaviorDiagnostic]` | Compile-time diagnostics |
| `trace_id` | `str` | UUID propagated to all runtime log entries |
| `compiled_at` | `str` | ISO 8601 UTC timestamp |
| `source_hash` | `Optional[str]` | SHA-256 of source string for staleness detection |
| `is_valid` | `bool` (property) | `True` iff no error-level diagnostics |

Factory: `BehaviorArtifact.create(behavior_ref, behavior_ir, ...)`.
Serialisation: `to_dict()` — JSON-safe (IR represented as node/edge counts).

### BehaviorInvokeInput

Request payload from a Mission behavior node to the behavior invoker.

| Field | Type | Description |
|-------|------|-------------|
| `behavior_ref` | `str` | Registry lookup key |
| `artifact_id` | `Optional[str]` | Pre-compiled artifact ID; resolved at runtime if absent |
| `inputs` | `Dict[str, Any]` | Port-mapped inputs from upstream Mission nodes |
| `context` | `Dict[str, Any]` | Execution context (robot_model, scenario keys, mission_trace_id, …) |
| `trace_id` | `str` | Propagated trace ID (= mission_trace_id when set by RuntimeEngine) |

### BehaviorInvokeOutput

Result payload returned from the behavior invoker to the Mission node.
`status` vocabulary aligns with Phase 1 `RuntimeResult.status`.

| Field | Type | Description |
|-------|------|-------------|
| `status` | `str` | `"success"` / `"failed"` / `"blocked"` |
| `reason` | `str` | `BehaviorErrorCode` constant on failure; `""` on success |
| `outputs` | `Dict[str, Any]` | Port-mapped outputs for downstream Mission nodes |
| `trace_id` | `str` | Same as `BehaviorInvokeInput.trace_id` |
| `diagnostics` | `List[BehaviorDiagnostic]` | Runtime diagnostics |
| `node_results` | `Dict[str, Any]` | Per-node results from subgraph (node_id → dict); empty on blocked |

Factory class methods: `.success()`, `.failed()`, `.blocked()`.

### BehaviorResolveResult

Structured resolver return type from `BehaviorCompilerBridge.resolve()`.
Always returned instead of bare `None`.

| Field | Type | Description |
|-------|------|-------------|
| `ok` | `bool` | `True` iff resolution succeeded and artifact is valid |
| `reason` | `str` | `BehaviorErrorCode` constant on failure; `""` on success |
| `artifact` | `Optional[BehaviorArtifact]` | `None` when missing |
| `diagnostics` | `List[BehaviorDiagnostic]` | Resolution + compile diagnostics |
| `trace_id` | `str` | Correlation ID (caller-supplied or auto-generated) |

Factory methods: `.success(artifact, trace_id)`, `.missing(behavior_ref, trace_id)`,
`.invalid(artifact, trace_id)`. Serialises via `to_dict()`.

---

## §2  BehaviorCompilerBridge API (behavior_compiler_bridge.py)

| Method | Signature | Description |
|--------|-----------|-------------|
| `compile` | `(source, behavior_ref, robot_type, trace_id) → BehaviorArtifact` | Compile source → artifact; auto-registers if valid |
| `register` | `(artifact) → None` | Explicit registration; overwrites existing entry |
| `resolve` | `(behavior_ref, trace_id) → BehaviorResolveResult` | **Structured resolution — never returns bare None** |
| `lookup` | `(behavior_ref) → Optional[BehaviorArtifact]` | Legacy API; delegates to `resolve()` |
| `lookup_any` | `(behavior_ref) → Optional[BehaviorArtifact]` | Returns invalid artifacts for diagnostics inspection |
| `list_refs` | `() → List[str]` | All registered keys |
| `clear_registry` | `() → None` | Flush registry (test / hot-reload) |
| `from_source` | `(source, robot_type) → Tuple[WorkflowIR, List[Diagnostic]]` | Legacy API; unchanged signature |

### Reason-code contract (deterministic mapping)

| Condition | `reason` | `ok` | `artifact` |
|-----------|----------|------|-----------|
| `behavior_ref` not in registry | `BEHAVIOR_REF_NOT_FOUND` | `False` | `None` |
| Found, `is_valid=False` | `ARTIFACT_INVALID` | `False` | `<artifact>` |
| Found, `is_valid=True` | `""` | `True` | `<artifact>` |

### Resolution order (Phase 2 — in-memory only)

```
resolve(behavior_ref):
  1. In-memory registry lookup (keyed by behavior_ref)
     a. Not found      → BehaviorResolveResult.missing()   # BEHAVIOR_REF_NOT_FOUND
     b. is_valid=False → BehaviorResolveResult.invalid()   # ARTIFACT_INVALID
     c. is_valid=True  → BehaviorResolveResult.success()   # ok=True
  2. Phase 4+: persisted artifact store (not yet implemented)
```

---

## §3  Subgraph Invocation Flow (behavior_invoker.py)

`BehaviorSubgraphInvoker.invoke()` runs the 6-step flow:

```
invoke(invoke_input, bridge, sub_executor, policy):

  Step 1 — Resolve artifact
    bridge.resolve(behavior_ref, trace_id=trace_id)
    → blocked (BEHAVIOR_REF_NOT_FOUND)  if not found
    → blocked (ARTIFACT_INVALID)         if is_valid=False

  Step 2 — Configure sub-executor from policy
    sub_executor.max_loop_iterations = policy.max_loop_iterations

  Step 3 — Load behavior_ir into sub-executor
    for node in artifact.behavior_ir.nodes: sub_executor.add_node(...)
    for edge in artifact.behavior_ir.edges: sub_executor.add_connection(...)

  Step 4 — Execute subgraph with isolated context
    sub_context = {**invoke_input.context,
                   "trace_id": trace_id,
                   "behavior_inputs": invoke_input.inputs}
    node_results = sub_executor.execute(context=sub_context)

  Step 5 — Inspect outcome
    if sub_executor._abort    → BehaviorInvokeOutput.failed(SUBGRAPH_ABORTED)
    if any node has "error"   → BehaviorInvokeOutput.failed(SUBGRAPH_EXECUTION_FAILED)

  Step 6 — Success
    → BehaviorInvokeOutput.success(trace_id, outputs=node_results, node_results=...)
```

**Circular-import prevention**: `behavior_invoker.py` imports only from
`system.behavior.*`.  The `sub_executor` (a `NodeExecutor`) is received as a
parameter from the caller, not imported.

**Backward compatibility**: when `RuntimeEngine.behavior_bridge` is `None`,
behavior nodes return `{"status": "skipped"}` — not an error — so non-behavior
workflows continue to function with zero configuration.

---

## §4  Trace ID and Diagnostics Semantics

### Compile-time trace

```
BehaviorCompilerBridge.compile(source, behavior_ref, trace_id=None):
    tid = trace_id or uuid4()          ← one trace_id per compile call
    BehaviorArtifact.trace_id = tid
    every BehaviorDiagnostic.trace_id = tid
    source_hash = SHA-256(source)      ← staleness detection
```

### Runtime invocation trace (Phase 2 STAGE-05)

```
RuntimeEngine.execute(mission_ir, scenario):
    mission_trace_id = uuid4()         ← one UUID per execute() call

    executor.execute(context={
        "scenario": scenario,
        "mission_trace_id": mission_trace_id   ← injected here
    })

    NodeExecutor._execute_behavior_node():
        trace_id = context["mission_trace_id"]  ← inherited

        BehaviorInvokeInput.trace_id = trace_id
        → BehaviorSubgraphInvoker.invoke()
            → sub_context["trace_id"] = trace_id
            → BehaviorInvokeOutput.trace_id = trace_id

    result["results"]["b0"]["trace_id"] = trace_id

    _collect_behavior_tracing(results) →
        behavior_trace_ids  = {"b0": trace_id, ...}
        behavior_diagnostics = [...from failed behavior nodes...]

RuntimeResult.diagnostics[MISSION_TRACE_ID]      = mission_trace_id
RuntimeResult.diagnostics[BEHAVIOR_TRACE_IDS]    = {"b0": trace_id, ...}
RuntimeResult.diagnostics[BEHAVIOR_DIAGNOSTICS]  = [BehaviorDiagnostic.to_dict(), ...]
```

**Invariant**: within a single `execute()` call, every behavior node execution
shares the same `mission_trace_id`.  Compile-time and runtime use separate
`trace_id` namespaces; they are linked via `artifact_id + behavior_ref`.

### Diagnostics aggregation rules

| Path | `behavior_diagnostics` content |
|------|--------------------------------|
| All behavior nodes succeed | `[]` |
| One or more behavior nodes blocked/failed | All `BehaviorDiagnostic` dicts from failed node results |
| No behavior nodes in mission | `[]` |

---

## §5  BehaviorPanel Compile Entry (STAGE-04)

`bin/layout/behavior_panel.py` wires the user-facing compile button to the real
`BehaviorCompilerBridge.compile()` pipeline:

```
User clicks "Compile"
  → BehaviorCompileWorker(QThread).start()
      → BehaviorCompilerBridge.compile(source, behavior_ref)
          → BehaviorArtifact (artifact + trace_id + diagnostics)
  → compile_done.emit(artifact)   ← main thread callback
  → _format_compile_output(artifact) → rendered in self._output area
```

**Key design constraints (maintained):**
- Compile runs off the main thread — UI stays responsive.
- `_format_compile_output()` is module-level and Qt-free — unit-testable standalone.
- Artifact truth lives in `self._bridge` registry, not in transient widget state.
- No execution truth coupled to UI text.

---

## §6  Module Reference

| Module | Role |
|--------|------|
| `behavior_artifact.py` | **Canonical contract** — all Phase 2 data types |
| `behavior_compiler_bridge.py` | **Bridge** — compile(), resolve(), registry ops |
| `behavior_state_machine.py` | Lightweight state/context transitions (Phase 2-isolated) |
| `behavior_model.py` | Behavior schema model (Phase 2-isolated) |

Runtime modules (in `system/runtime/`):

| Module | Role |
|--------|------|
| `behavior_invoker.py` | Subgraph execution — `BehaviorSubgraphInvoker.invoke()` |

---

## §7  Known Gaps / Phase 3-Ready Checklist

All Phase 2 gaps from STAGE-01 baseline audit are resolved:

| Gap (from STAGE-01 baseline) | Status |
|------------------------------|--------|
| No real compile-to-artifact pipeline | ✅ Resolved — STAGE-02 |
| No runtime dispatch for behavior nodes | ✅ Resolved — STAGE-03 |
| Compile button calls stub | ✅ Resolved — STAGE-04 |
| No end-to-end trace_id propagation | ✅ Resolved — STAGE-05 |
| behavior_state_machine not integrated | Deferred to Phase 3+ (out of scope) |

**Phase 3 prerequisites (safe to start):**

- [ ] Service adapter session lifecycle (Phase 3 scope)
- [ ] Artifact persistence store — bridge Phase 4+  `lookup()` path already declares the hook
- [ ] behavior_state_machine → BehaviorArtifact integration (when state-machine behaviors are needed)
- [ ] Multi-behavior registry (hot-reload / version management)

---

## §8  Out-of-Scope (Phase 2 — explicitly not implemented)

- Adapter session lifecycle redesign (Phase 3)
- Spot / CyberDog adapter delivery (Phase 4)
- Capability / settings schema standardization (Phase 5)
- Safety / observability hardening beyond behavior-level trace plumbing (Phase 6)
- Persistent artifact store (Phase 4+)
- `behavior_state_machine.py` integration with the artifact / runtime pipeline

---

*Phase 2 completed 2026-02-23.*

---

## §9  Motor Weight Protocol Contract (Steps 1–4)

Introduced in the Behavior Interface & UI Refactor series (Steps 1–4).  Full
schema spec lives at `system/behavior/behavior_motor_weight_protocol.md`.

### Modules

| Module | Role |
|--------|------|
| `motor_weight_protocol.py` | Validation (`validate_protocol_payload`, `validate_protocol_structure`), parsing (`parse_protocol_targets`), diagnostic constants (`ProtocolDiagCode`) |
| `motor_weight_nav_model.py` | Pure-Python navigator data model — `build_navigator_model()` → `List[NavRegion]` |
| `motor_param_source.py` | Source-resolution helpers — `get_param_source()`, `STRUCTURAL_KEYS`, `ParamSourceInfo` |
| `behavior_motor_weight_protocol.schema.json` | JSON Schema for the protocol payload (v1.0) |

### Validation layers

| Layer | Function | When called |
|-------|----------|-------------|
| Design-time structural | `validate_protocol_structure(payload)` → `(bool, List[str])` | Canvas/test; no staleness check |
| Runtime full | `validate_protocol_payload(payload, now_ms)` → `(bool, List[BehaviorDiagnostic])` | Invoker before apply |

### Invoker protocol path (`behavior_invoker.py` Step 2c)

```
BehaviorSubgraphInvoker.invoke():
  if invoke_input.protocol_payload is not None:
    validate_protocol_payload(payload, trace_id=trace_id)
      invalid/stale → blocked(reason="INVALID_PROTOCOL")
      valid         → parse_protocol_targets() → protocol_targets injected into sub_context
  else:
    legacy path — sub_context carries no protocol keys
```

**No silent fallback**: invalid or stale payloads are always blocked with
`INVALID_PROTOCOL` reason and emitted as `BehaviorDiagnostic` entries.

### Canvas sub_dot contract (Step 4)

The Behavior node's `condition` input port (`slot="condition"`) has
`data_type="protocol"` since schema v1.4.  The canvas enforces type-matching
on new connections.

Border color states (managed by `GraphScene`):

| State | Color | Trigger |
|-------|-------|---------|
| `protocol_none` | grey `#6b7280` | No connection / legacy mode |
| `protocol_valid` | green `#22c55e` | Valid protocol type connected (design-time) or runtime confirmed valid |
| `protocol_invalid` | amber `#f59e0b` | Type mismatch, or runtime `protocol_status` in `{invalid, stale}` or `reason=INVALID_PROTOCOL` |

Runtime result takes priority: `apply_behavior_protocol_states_from_run_result(run_result)`
in `GraphScene` is called by `MainWindow._apply_node_execution_statuses()` after every run.

### Trace ID propagation

```
mission_trace_id (per execute() call)
  └─ NodeExecutor._execute_behavior_node()
       └─ BehaviorInvokeInput.trace_id = mission_trace_id
            └─ validate_protocol_payload(trace_id=trace_id)
                 protocol diagnostics carry the same trace_id
            └─ parse_protocol_targets(trace_id=trace_id)
                 clamped/unknown-address diagnostics carry the same trace_id
            └─ BehaviorInvokeOutput.trace_id = trace_id
```

Every `BehaviorDiagnostic` emitted during protocol ingest shares the
`mission_trace_id` of the enclosing `execute()` call.

---

## §10  Migration Guide (schema v1.3 → v1.4)

**What changed in v1.4**

- Behavior node `condition` input port changed `data_type` from `"bool"` to `"protocol"`.
- New connections to the `condition` port require a `protocol`-typed output.

**Backward compatibility**

Old mission files (schema < 1.4) load without error.  The canvas applies a
**migration-compat gate** (`_loading_workflow=True`) that permits legacy `bool`
→ `protocol` connections to restore.  After loading:

- The Behavior node shows a **`protocol_invalid` (amber) border** — signalling
  the user that the connected wire does not carry a validated protocol payload.
- Existing logic is not broken; the legacy path in `BehaviorSubgraphInvoker`
  still runs when `protocol_payload` is absent.

**Migration steps for existing projects**

1. Open the mission file — amber border appears on Behavior nodes with old wiring.
2. Rewire the `condition` input from a node/script that emits a
   `protocol_version="1.0"` payload dict.
3. Save — the file is written with `schema_version="1.4"` and the migration
   warning no longer appears on next load.

**`migrate_mission_payload(data)` helper** (`bin/core/mission_persistence.py`)

Returns a migration-info dict without mutating *data*:

```python
{
    "needs_protocol_upgrade": bool,   # True when schema < 1.4
    "prior_schema_version":   str,    # e.g. "1.3" or "unknown"
    "warnings":               list,   # human-readable notices
}
```

Called automatically by `MainWindow._on_open()` before canvas load; warnings
are surfaced via `log_info`.

---

## §11  Module Reference (updated for Steps 1–5)

| Module | Role |
|--------|------|
| `behavior_artifact.py` | Canonical contract — all Phase 2 data types |
| `behavior_compiler_bridge.py` | Bridge — compile(), resolve(), registry ops |
| `motor_weight_protocol.py` | Protocol validation + parsing |
| `motor_weight_nav_model.py` | Navigator data model |
| `motor_param_source.py` | Parameter source-switch helpers |
| `behavior_state_machine.py` | Lightweight state transitions (deferred) |
| `behavior_model.py` | Behavior schema model |

UI modules:

| Module | Role |
|--------|------|
| `bin/components/motor_weight_navigator.py` | Read-only hierarchical motor-weight inspector widget (Step 2) |
| `bin/layout/behavior_panel.py` | Main behavior panel; hosts navigator + movement settings (Steps 2–3) |
| `bin/components/graph_scene.py` | Canvas; Behavior node port contract + protocol border (Steps 4–5) |

Runtime modules:

| Module | Role |
|--------|------|
| `system/runtime/behavior_invoker.py` | Subgraph execution + protocol ingest |
| `system/runtime/node_executor.py` | Behavior node dispatch; trace_id injection |
| `bin/core/mission_persistence.py` | Mission save/load + migration helpers |
