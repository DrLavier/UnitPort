# Service Layer

`system/service` is the unified routing layer between runtime and vendor SDK adapters.

## Modules

| Module | Role |
|--------|------|
| `service_registry.py` | Adapter registry — name → adapter instance |
| `service_router.py` | Action/sensor routing APIs; Phase 3 lifecycle orchestration |
| `adapters/base_adapter.py` | **Canonical adapter contract** — abstract interface every adapter must implement |
| `adapters/unitree_sdk2/` | Unitree SDK2 adapter + action mapper |
| `protocol/` | Command / event / error contracts |

---

## Phase 3 — Service Session Lifecycle Contract

*(STAGE-01 baseline audit completed 2026-02-23)*

---

### §1  Baseline Audit (STAGE-01)

#### 1.1  Current BaseAdapter interface (pre-Phase 3)

| Method | Signature | Status |
|--------|-----------|--------|
| `connect` | `(**kwargs) → bool` | Existing — implicit session init |
| `run_action` | `(action: str, **params) → Any` | Existing — legacy, preserved |
| `stop` | `() → None` | Existing — legacy, preserved |
| `get_sensor_data` | `() → Dict[str, Any]` | Existing — legacy, preserved |
| `health` | `() → Dict[str, Any]` | Existing — preserved |

#### 1.2  Current ServiceRouter API (pre-Phase 3)

| Method | Signature | Notes |
|--------|-----------|-------|
| `register_adapter` | `(name, adapter) → None` | Delegates to registry |
| `get_adapter` | `(name) → Any` | Raises KeyError on miss |
| `run_action` | `(adapter_name, action, **params) → Any` | Direct passthrough |
| `stop` | `(adapter_name) → None` | Direct passthrough |
| `get_sensor_data` | `(adapter_name) → Dict` | Direct passthrough |
| `health` | `(adapter_name) → Dict` | Direct passthrough |

#### 1.3  Current call path (pre-Phase 3)

```
RobotContext.run_action(action_name)
  → _ensure_adapter(brand, robot_type)
      → ServiceRegistry.get(adapter_name)         [lookup]
      → UnitreeAdapter(robot_type)                [create if missing]
      → ServiceRegistry.register(name, adapter)   [register]
      → adapter.connect(robot_type=...)           [implicit session init]
  → ServiceRouter.run_action(adapter_name, action)
      → adapter.run_action(action)
          → UnitreeModel.run_action(mapped_action)

RobotContext.get_sensor_data()
  → _ensure_adapter(...)                          [same pattern]
  → ServiceRouter.get_sensor_data(adapter_name)
      → adapter.get_sensor_data()

RobotContext.stop()
  → _ensure_adapter(...)
  → ServiceRouter.stop(adapter_name)
      → adapter.stop()
```

**Fallback path (all three callers)**: if router raises, falls back to `robot.run_action()` / `robot.get_sensor_data()` / `robot.stop()` via `get_robot_model()`.

#### 1.4  Legacy compatibility requirements

All three public entry points **must remain callable without change**:

| Caller API | Signature (unchanged) | Return (unchanged) |
|------------|-----------------------|--------------------|
| `RobotContext.run_action(action, **kwargs)` | classmethod | `bool` |
| `RobotContext.get_sensor_data()` | classmethod | `Dict[str, Any]` |
| `RobotContext.stop()` | classmethod | `None` |
| `ServiceRouter.run_action(adapter_name, action, **params)` | instance | passthrough |
| `ServiceRouter.stop(adapter_name)` | instance | passthrough |
| `ServiceRouter.get_sensor_data(adapter_name)` | instance | passthrough |

---

### §2  Phase 3 Lifecycle Contract (canonical definition)

#### 2.1  Lifecycle result schema

All lifecycle methods return a **LifecycleResult dict** with this shape:

```python
{
    "status":      "ok" | "error",   # required
    "reason":      str,               # LifecycleReason constant on error; "" on ok
    "stage":       str,               # which lifecycle stage produced this result
    "diagnostics": Dict[str, Any],    # optional structured context (code/message/context)
}
```

Factory helpers (in `system/service/lifecycle.py` — Phase 3 new file):

```python
LifecycleResult.ok(stage)                          → status="ok", reason="", stage=stage
LifecycleResult.error(stage, reason, diagnostics)  → status="error", reason=..., stage=...
```

#### 2.2  Lifecycle error reason codes

```python
class LifecycleReason:
    SESSION_OPEN_FAILED   = "session_open_failed"
    PREFLIGHT_FAILED      = "preflight_failed"
    SESSION_CLOSE_FAILED  = "session_close_failed"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"
    ADAPTER_NOT_FOUND     = "adapter_not_found"
```

#### 2.3  BaseAdapter lifecycle method signatures (Phase 3 additions)

```python
def open_session(self, config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Establish a service session.

    Default implementation: calls connect(**config or {}) for backward compat.

    Returns LifecycleResult dict:
        ok:    {"status": "ok",    "reason": "", "stage": "open_session", "diagnostics": {}}
        error: {"status": "error", "reason": SESSION_OPEN_FAILED, "stage": "open_session",
                "diagnostics": {"message": str, "context": dict}}
    """

def preflight(self, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Validate adapter readiness before action execution.

    Default implementation: returns ok unconditionally (permissive default).
    Concrete adapters override for real checks (connection alive, safety limits, etc.).

    Returns LifecycleResult dict:
        ok:    {"status": "ok",    "reason": "", "stage": "preflight", "diagnostics": {}}
        error: {"status": "error", "reason": PREFLIGHT_FAILED, "stage": "preflight",
                "diagnostics": {"checks": list, "context": dict}}
    """

def close_session(self) -> Dict[str, Any]:
    """
    Teardown a service session.

    Default implementation: no-op, returns ok (backward compat — no teardown needed for legacy adapters).

    Returns LifecycleResult dict:
        ok:    {"status": "ok",    "reason": "", "stage": "close_session", "diagnostics": {}}
        error: {"status": "error", "reason": SESSION_CLOSE_FAILED, "stage": "close_session",
                "diagnostics": {"message": str}}
    """

def capabilities(self) -> Dict[str, Any]:
    """
    Return adapter capability descriptor.

    Default implementation: empty lists + no flags (safe minimum for legacy adapters).

    Returns:
        {
            "actions":  List[str],        # supported canonical action names
            "sensors":  List[str],        # supported sensor keys
            "flags":    Dict[str, Any],   # feature flags (e.g. {"sim_mode": True})
        }
    """
```

#### 2.4  ServiceRouter session-aware routing (Phase 3 new orchestration)

```
ServiceRouter.execute_with_lifecycle(adapter_name, action, params, policy):

  1. acquire adapter (ServiceRegistry.require)
       → LifecycleReason.ADAPTER_NOT_FOUND on miss

  2. open_session(config=policy.session_config)
       → on error: return LifecycleResult.error(SESSION_OPEN_FAILED)

  3. preflight(context=policy.preflight_context)   [if policy.run_preflight=True]
       → on error: close_session(); return LifecycleResult.error(PREFLIGHT_FAILED)

  4. execute: run_action / get_sensor_data / stop

  5. close_session()                               [if policy.close_after=True]
       → on error: log but do not mask action result

  6. return action result (structured, reason-coded)
```

**Routing policy** (`LifecyclePolicy`):

```python
@dataclass
class LifecyclePolicy:
    run_preflight:    bool = False          # skip preflight by default (backward-compat safe)
    close_after:      bool = False          # keep session open by default (legacy behavior)
    session_config:   Dict[str, Any] = {}  # forwarded to open_session
    preflight_context: Dict[str, Any] = {} # forwarded to preflight
```

Default policy reproduces legacy direct-passthrough behavior.

#### 2.5  RobotContext Phase 3 integration (additive only)

- `_ensure_adapter` continues to exist and call `adapter.connect()` — not removed.
- `run_action`, `get_sensor_data`, `stop` public signatures and return types: **unchanged**.
- New internal helper `_lifecycle_route(op, adapter_name, **kwargs)` uses `execute_with_lifecycle`.
- Brand/model mapping (`BRAND_ADAPTER_MAP`, `_get_robot_brand_map`) stays unchanged.
- Spot/CyberDog extension hooks: reserved via `BRAND_ADAPTER_MAP` extension — no implementation in Phase 3.

#### 2.6  Diagnostics plumbing and boundary logging (STAGE-05)

**Structured diagnostics on every failure path** — each `_route_error` result carries a `diagnostics` dict with at least:

| Failure stage | Diagnostics keys guaranteed |
|---------------|----------------------------|
| `acquire` | `adapter_name` |
| `open_session` | `message` (from `LifecycleResult.diagnostics`) |
| `preflight` | `message` (from `LifecycleResult.diagnostics`) |
| `execute` | `message`, `op` |
| `close_session` (soft) | `close_session_failed`, `close_reason`, `close_diagnostics` |

**Boundary logging** — `ServiceRouter.execute_with_lifecycle` emits stdlib log records (`logging.getLogger("unitport.service.router")`) at each lifecycle step:

| Step | Level | Condition |
|------|-------|-----------|
| acquire (success) | `DEBUG` | adapter found |
| acquire (miss) | `WARNING` | adapter not found |
| open_session | `DEBUG` | before call |
| open_session (fail) | `WARNING` | non-ok status |
| preflight | `DEBUG` | before call (policy-gated) |
| preflight (fail) | `WARNING` | non-ok status |
| execute | `DEBUG` | before dispatch |
| execute (exception) | `WARNING` | any exception |
| close_session | `DEBUG` | before call (policy-gated) |
| close_session (fail) | `WARNING` | non-ok status |
| success | `DEBUG` | result returned ok |

Logger name: `unitport.service.router` — integrate with any stdlib `logging` handler.

---

### §3  Compatibility Guarantee Summary

| Concern | Guarantee |
|---------|-----------|
| `BaseAdapter.run_action/stop/get_sensor_data/health` | Not removed; remain abstract; all existing adapters compile unchanged |
| `BaseAdapter.connect` | Not removed; called by `_ensure_adapter`; lifecycle `open_session` default delegates to it |
| `ServiceRouter.run_action/stop/get_sensor_data/health` | Not removed; remain as direct-passthrough wrappers |
| `RobotContext.run_action/get_sensor_data/stop` | Public classmethod signatures unchanged; return types unchanged |
| `UnitreeAdapter` | Inherits lifecycle defaults from BaseAdapter; zero code changes required in Phase 3 |
| Existing tests (536 passing) | Must remain green after every stage |

---

### §4  Out-of-Scope (Phase 3)

| Item | Phase |
|------|-------|
| Spot / CyberDog minimum adapter delivery | Phase 4 |
| Capability / settings schema standardization | Phase 5 |
| Safety / observability hardening beyond lifecycle diagnostics | Phase 6 |
| Artifact persistence | Phase 4+ |
| Large runtime / behavior refactors | Never in this phase |

---

### §5  Phase 3 Stage Checklist

| Stage | Objective | Status |
|-------|-----------|--------|
| STAGE-01 | Baseline audit + lifecycle contract freeze | ✅ |
| STAGE-02 | Extend BaseAdapter with lifecycle interface | ✅ |
| STAGE-03 | ServiceRouter session-aware routing | ✅ |
| STAGE-04 | RobotContext integration + compat wrappers | ✅ |
| STAGE-05 | Error taxonomy + diagnostics plumbing | ✅ |
| STAGE-06 | Test matrix for Phase 3 | ✅ |
| STAGE-07 | Phase 3 acceptance report | ✅ |

---

*Phase 3 completed 2026-02-23. See `PHASE3_ACCEPTANCE_REPORT.md`.*

---

## Phase 4 — Multi-Brand Minimum Viable Adapters

*(STAGE-01 baseline audit completed 2026-02-23)*

---

### §6  Phase 4 Adapter Onboarding Contract

This section is the canonical reference for implementing Spot and CyberDog
adapters, and for upgrading the Unitree adapter to explicit lifecycle semantics.
All implementation in STAGE-02 through STAGE-05 must conform to this contract.

---

#### 6.1  Adapter package layout

Each adapter lives in its own package under `system/service/adapters/`:

```
system/service/adapters/
  unitree_sdk2/          ← existing (Phase 3 baseline)
    adapter.py           ← UnitreeAdapter(BaseAdapter)
    mapper.py            ← ACTION_MAP + map_action()
    README.md
  spot_sdk/              ← new (Phase 4 STAGE-03)
    __init__.py
    adapter.py           ← SpotAdapter(BaseAdapter)
    mapper.py            ← canonical → Spot action mapping
    README.md
  cyberdog_sdk/          ← new (Phase 4 STAGE-04)
    __init__.py
    adapter.py           ← CyberDogAdapter(BaseAdapter)
    mapper.py            ← canonical → CyberDog action mapping
    README.md
```

No SDK calls are permitted outside these packages.

---

#### 6.2  Required lifecycle method semantics (per brand)

| Method | UnitreeAdapter v2 | SpotAdapter MVP | CyberDogAdapter MVP |
|--------|-------------------|-----------------|---------------------|
| `connect` | Create/refresh UnitreeModel (existing path) | No-op — session handled by `open_session` | No-op — session handled by `open_session` |
| `open_session` | Explicit connect + SDK availability check | Auth bootstrap → time-sync check → lease acquire | ROS2 context init → endpoint discovery |
| `preflight` | High-level vs low-level control conflict guard | Safety channel liveness + power state check | Timestamp freshness + namespace/endpoint check |
| `close_session` | Explicit model teardown / resource release | Lease release + power-down if owned | Gait/mode reset + ROS2 cleanup |
| `capabilities` | Declare `actions`, `sensors`, `flags` | Declare `actions`, `sensors`, `flags` | Declare `actions`, `sensors`, `flags` |

---

#### 6.3  Minimum canonical action set (all brands)

All three adapters must handle the following canonical action names without
raising exceptions.  Unmapped actions return a reason-coded error result.

| Canonical | Unitree v2 | Spot | CyberDog |
|-----------|-----------|------|----------|
| `stand`   | `stand`   | `stand` | `stand` |
| `sit`     | `sit`     | `sit`   | `sit`   |
| `walk`    | `walk`    | `walk`  | `velocity_move` |
| `stop`    | `stop`    | `stop`  | `stop`  |

---

#### 6.4  Failure reason mapping

All adapters reuse Phase 3 `LifecycleReason` codes.  Adapter-specific context
lives in `diagnostics["context"]` — not in new reason codes.

| Failure scenario | Reason code | `diagnostics` keys |
|------------------|-------------|---------------------|
| Session open fails (any brand) | `SESSION_OPEN_FAILED` | `message`, `brand`, `context` |
| Preflight check fails | `PREFLIGHT_FAILED` | `message`, `check`, `brand`, `context` |
| Session close fails | `SESSION_CLOSE_FAILED` | `message`, `brand` |
| Action not supported | `CAPABILITY_UNAVAILABLE` | `action`, `brand` |
| Vendor SDK unavailable | `SESSION_OPEN_FAILED` | `message`, `brand`, `sdk_available: false` |

---

#### 6.5  Vendor SDK unavailability rule

Spot and CyberDog adapters **must not crash** if their vendor SDK is not
installed.  The `open_session` implementation must catch `ImportError` /
`ModuleNotFoundError` and return a reason-coded `SESSION_OPEN_FAILED` result
with `diagnostics["sdk_available"] = False`.

This allows the full adapter to be imported and tested without the SDK present.

---

#### 6.6  Brand key correction (STAGE-05 target)

`BrandRegistry` returns lowercase directory names as brand keys.  The current
`RobotContext.BRAND_ADAPTER_MAP` has stale placeholder keys that do not match.

| Brand directory | BrandRegistry key | Models | Required BRAND_ADAPTER_MAP key |
|-----------------|-------------------|--------|-------------------------------|
| `Unitree/`      | `"unitree"`       | go2, a1, b1, b2, h1 | `"unitree"` ✅ |
| `BostionDynamics/` | `"bostiondynamics"` | spot | `"bostiondynamics"` ✅ |
| `XiaoMi/`       | `"xiaomi"`        | cyberdog, cyberdog2 | `"xiaomi"` ✅ |

Stale keys (`"boston_dynamics"`, `"cyberdog"`) corrected in STAGE-05.
`_FALLBACK_BRAND_MAP` must also be extended with `"spot"` and `"cyberdog"`.

---

#### 6.7  Registration flow (STAGE-05 activation)

```
RobotContext.set_robot_type("spot")
  → BrandRegistry.get_robot_brand_map()   # {"spot": "bostiondynamics", ...}
  → brand = "bostiondynamics"
  → adapter_name = BRAND_ADAPTER_MAP["bostiondynamics"]  # "spot_sdk"
  → _ensure_adapter("bostiondynamics", "spot")
      → ServiceRegistry.get("spot_sdk")   # None on first call
      → SpotAdapter("spot")               # instantiate
      → ServiceRegistry.register("spot_sdk", adapter)
      → adapter.connect(robot_type="spot")
  → ServiceRouter.execute_with_lifecycle("spot_sdk", op, ...)
```

---

### §7  Phase 4 Stage Checklist

| Stage | Objective | Status |
|-------|-----------|--------|
| STAGE-01 | Baseline audit + Phase 4 contract freeze | ✅ |
| STAGE-02 | UnitreeAdapter v2 (lifecycle uplift) | ✅ |
| STAGE-03 | SpotAdapter MVP | ✅ |
| STAGE-04 | CyberDogAdapter MVP | ✅ |
| STAGE-05 | RobotContext multi-brand activation | ✅ |
| STAGE-06 | Error taxonomy + diagnostics consolidation | ✅ |
| STAGE-07 | Phase 4 test matrix | ✅ |
| STAGE-08 | Phase 4 acceptance report + Phase 5 handoff | ✅ |

---

*Phase 4 completed 2026-02-24. See `PHASE4_ACCEPTANCE_REPORT.md`.*

---

## Phase 5 — Capability Schema and Settings Standardization

*(All 7 stages completed 2026-02-24)*

---

### §8  Phase 5 Schema Contract

This section is the canonical, frozen reference for all Phase 5 implementation.
All code in STAGE-02 through STAGE-06 must conform to the contracts defined here.

---

#### 8.1  Capability schema (standardised `capabilities()` return shape)

Every adapter's `capabilities()` must return a dict matching this shape:

```python
{
    "brand":             str,              # "unitree" | "bostiondynamics" | "xiaomi"
    "adapter":           str,              # "unitree_sdk2" | "spot_sdk" | "cyberdog_sdk"
    "actions":           List[str],        # canonical action names the adapter handles
    "sensors":           List[str],        # sensor keys present in get_sensor_data() output
    "flags":             Dict[str, Any],   # feature flags (brand-specific booleans/enums)
    "required_settings": List[str],        # setting keys that must be present before execution
}
```

Required keys: `brand`, `adapter`, `actions`, `sensors`, `flags`, `required_settings`.
Extra keys are tolerated (non-breaking forward compatibility).

---

#### 8.2  Settings schema entry shape (per-brand, Python dict-based — no external files)

Each setting key maps to an entry dict with this shape:

```python
{
    "type":        type,            # e.g. str, int, float, bool
    "required":    bool,            # True = must be present unless a default is given
    "default":     Any,             # None means no default (key is truly required)
    "description": str,             # human-readable explanation
    "choices":     Optional[List],  # allowed enum values; None = unrestricted
}
```

Settings schemas are Python dicts in `system/service/settings_schema.py`.
No JSON or YAML files are created.

---

#### 8.3  Global required settings (all brands)

| Key | Type | Required | Default | Choices |
|-----|------|----------|---------|---------|
| `brand` | `str` | Yes | None | `"unitree"`, `"bostiondynamics"`, `"xiaomi"` |
| `robot_type` | `str` | Yes | None | — (brand-specific) |
| `connection_mode` | `str` | Yes | None | `"simulation"`, `"hardware"` |
| `timeout` | `int` | Yes | None | — (positive integer, seconds) |
| `stop_policy` | `str` | Yes | None | `"immediate"`, `"graceful"` |
| `safety_enabled` | `bool` | Yes | None | `True`, `False` |

---

#### 8.4  Brand-specific required settings

**Unitree** (`brand = "unitree"`):

| Key | Type | Required | Default | Choices |
|-----|------|----------|---------|---------|
| `dds_domain_id` | `int` | Yes | None | — (0–232) |
| `network_interface` | `str` | Yes | None | — (e.g. `"eth0"`) |
| `control_level` | `str` | Yes | None | `"high"`, `"low"` |
| `mode_switch_policy` | `str` | Yes | None | `"immediate"`, `"graceful"` |

**Spot** (`brand = "bostiondynamics"`):

| Key | Type | Required | Default | Choices |
|-----|------|----------|---------|---------|
| `hostname` | `str` | Yes | None | — |
| `auth_token` | `str` | Yes | None | — |
| `timesync_required` | `bool` | No | `True` | `True`, `False` |
| `lease_required` | `bool` | No | `True` | `True`, `False` |
| `safety_channel` | `str` | Yes | None | — |
| `power_policy` | `str` | Yes | None | `"safe_power_off"`, `"emergency_cut"` |

**CyberDog** (`brand = "xiaomi"`):

| Key | Type | Required | Default | Choices |
|-----|------|----------|---------|---------|
| `ros_domain_id` | `int` | Yes | None | — (0–232) |
| `rmw_impl` | `str` | Yes | None | — (e.g. `"rmw_fastrtps_cpp"`) |
| `namespace` | `str` | No | `""` | — |
| `action_endpoints` | `list` | Yes | None | — (list of endpoint strings) |
| `timestamp_policy` | `str` | Yes | None | `"strict"`, `"relaxed"` |

---

#### 8.5  ValidationResult schema

`validate_settings(brand, config)` returns a `ValidationResult` dict:

```python
{
    "status":  "ok" | "error",   # "ok" only if errors/missing/invalid are all empty
    "brand":   str,              # brand key that was validated against
    "errors":  List[str],        # human-readable error descriptions (summary)
    "missing": List[str],        # required keys that were absent from config
    "invalid": List[str],        # keys present but with wrong type or disallowed value
}
```

---

#### 8.6  Validation integration into execution path

`ServiceRouter.execute_with_lifecycle` runs settings validation **before**
`open_session` when `policy.session_config` contains a `"brand"` key:

```
execute_with_lifecycle(adapter_name, op, params, policy):

  0. [NEW] validate_settings(brand, policy.session_config)   [if brand present in config]
       → on validation error: return LifecycleResult.error(PREFLIGHT_FAILED,
             diagnostics={"validation": ValidationResult, "stage": "settings_validation"})

  1. acquire adapter (ServiceRegistry.require)
  2. open_session(config=policy.session_config)
  3. preflight(context=policy.preflight_context)   [if policy.run_preflight=True]
  4. execute: run_action / get_sensor_data / stop
  5. close_session()                               [if policy.close_after=True]
  6. return action result
```

Legacy callers that pass empty `session_config` (no `"brand"` key) skip validation
entirely — backward-compat guaranteed.

---

#### 8.7  New modules (Phase 5)

| Module | Role |
|--------|------|
| `system/service/capability_schema.py` | Validates `capabilities()` return dicts against the frozen schema |
| `system/service/settings_schema.py` | Per-brand settings schemas + `get_settings_schema(brand)` |
| `system/service/settings_validator.py` | `validate_settings(brand, config) → ValidationResult` |

Existing modules changed in Phase 5:

| Module | Change |
|--------|--------|
| `system/service/adapters/unitree_sdk2/adapter.py` | Real `capabilities()` with `required_settings` |
| `system/service/adapters/spot_sdk/adapter.py` | Real `capabilities()` with `required_settings` |
| `system/service/adapters/cyberdog_sdk/adapter.py` | Real `capabilities()` with `required_settings` |
| `system/service/service_router.py` | Settings validation hook in `execute_with_lifecycle` |

---

### §9  Phase 5 Stage Checklist

| Stage | Objective | Status |
|-------|-----------|--------|
| STAGE-01 | Baseline audit + Phase 5 schema contract freeze | ✅ |
| STAGE-02 | Real `capabilities()` for all 3 adapters | ✅ |
| STAGE-03 | Capability schema module (`capability_schema.py`) | ✅ |
| STAGE-04 | Settings schema data model (`settings_schema.py`) | ✅ |
| STAGE-05 | Settings validator + validation integration into execution path | ✅ |
| STAGE-06 | Phase 5 test matrix | ✅ |
| STAGE-07 | Phase 5 acceptance report + Phase 6 handoff | ✅ |

---

*Phase 5 completed 2026-02-24. See `PHASE5_ACCEPTANCE_REPORT.md`.*

---

## Phase 6 — Safety and Observability Hardening

*(STAGE-01 baseline audit completed 2026-02-24)*

---

### §10  Phase 6 Contract

This section is the canonical, frozen reference for all Phase 6 implementation.
All code in STAGE-02 through STAGE-07 must conform to the contracts defined here.

---

#### 10.1  Baseline Audit — Current State and Gaps

| Area | Current state (Phase 5 exit) | Phase 6 gap |
|------|------------------------------|-------------|
| Preflight gate | `run_preflight=False` by default — safety checks **skipped** unless opt-in | Promote to safety-policy-driven; preflight ON by default via `SafetyPolicy` |
| Structured logging | Free-form `_log.debug/warning()` strings — no `trace_id`, no `duration_ms` | Emit `TelemetryEvent` at every lifecycle boundary |
| Retry | No retry logic; no `retryable` field on reason codes | Add `RetryPolicy` + `retryable` classification per reason code |
| Reason codes | 25 codes (Phase 4 taxonomy); no Phase 6 safety/retry codes | Add `SAFETY_GATE_BLOCKED`, `RETRY_BUDGET_EXHAUSTED` to canonical tier |
| Diagnostics keys | `brand`, `message`, `context` mandatory (Phase 4 contract) | Preserved; `telemetry` list added additively to diagnostics |

---

#### 10.2  SafetyPolicy contract

`SafetyPolicy` governs whether and how the preflight gate runs before execution.
It is embedded in `LifecyclePolicy` alongside the existing `run_preflight` flag
(which is superseded by `SafetyPolicy` for Phase 6 paths).

```python
@dataclass
class SafetyPolicy:
    require_preflight: bool = True
    # If True, gate runs even when run_preflight=False on LifecyclePolicy.
    # Default True means preflight is ON for all Phase 6 callers.

    allow_execute_without_preflight: bool = False
    # Explicit override: if True, skip preflight even when require_preflight=True.
    # Only used for trusted internal paths (e.g. STOP emergency command).

    safety_context: Dict[str, Any]  = field(default_factory=dict)
    # Forwarded to adapter.preflight(context=...) when gate runs.
```

Gate outcome:
- `status: "ok"` — all checks passed; execution may proceed.
- `status: "blocked"` — gate rejected; reason = `SAFETY_GATE_BLOCKED`.
- `status: "error"` — gate errored; reason = `PREFLIGHT_FAILED`.

Legacy callers using `LifecyclePolicy()` with no explicit `SafetyPolicy`:
- The `run_preflight` field on `LifecyclePolicy` remains respected.
- `SafetyPolicy` is introduced as an optional field with a default that does **not**
  break callers that do not supply it; existing tests remain green.

---

#### 10.3  TelemetryEvent schema

Every lifecycle boundary emits one or two events (start + end):

```python
{
    "trace_id":     str,          # UUID4 per execute_with_lifecycle call
    "adapter_name": str,          # registry key of the target adapter
    "brand":        str,          # brand from session_config, or ""
    "operation":    str,          # RouteOp constant ("run_action" / "get_sensor_data" / "stop")
    "stage":        str,          # lifecycle stage name (see below)
    "event":        str,          # "start" | "end"
    "timestamp":    float,        # time.monotonic() at emission
    "duration_ms":  float | None, # elapsed since stage start; None for "start" events
    "status":       str,          # "ok" | "error" | "skipped" | "pending"
    "reason":       str,          # LifecycleReason constant or ""
    "retryable":    bool,         # derived from REASON_MAP[reason]["retryable"]
    "attempt":      int,          # attempt index (1-based); 1 when no retry
    "diagnostics":  Dict[str, Any],  # safe subset of adapter diagnostics (no secrets)
}
```

Stage name constants:
`"settings_validation"`, `"acquire"`, `"open_session"`, `"preflight"`,
`"execute"`, `"close_session"`

Privacy rule: `diagnostics` in telemetry must never include `auth_token`,
`hostname`, or any value from `session_config` keys typed as credential.

Accumulated events are stored per-call in `TelemetryCollector` and attached to
the result's `diagnostics["telemetry"]` key on both ok and error paths.

---

#### 10.4  RetryPolicy schema

```python
@dataclass
class RetryPolicy:
    max_attempts: int = 1
    # Total attempts including the first.  max_attempts=1 means no retry.

    retryable_reasons: frozenset = field(
        default_factory=lambda: frozenset({
            LifecycleReason.EXECUTE_FAILED,
            "spot_command_timeout",
            "cyberdog_action_failed",
        })
    )
    # Reason codes that are eligible for retry.

    backoff_ms: int = 0
    # Fixed delay between attempts (milliseconds).  0 = no delay.

    non_retryable_ops: frozenset = field(
        default_factory=lambda: frozenset({"stop"})
    )
    # RouteOp values that never retry regardless of reason.
```

Non-retryable reasons (never retried, regardless of `retryable_reasons`):
- `PREFLIGHT_FAILED`, `SAFETY_GATE_BLOCKED` — safety block
- `SESSION_OPEN_FAILED` — connection-level; retry at connection layer, not here
- `CAPABILITY_UNAVAILABLE`, `ADAPTER_NOT_FOUND` — structural errors
- `RETRY_BUDGET_EXHAUSTED` — terminal; not itself retried
- `settings_validation` stage failures — bad config; retry won't fix

---

#### 10.5  New LifecycleReason constants (Phase 6)

Two new canonical constants are added to `LifecycleReason` and `CANONICAL_REASONS`:

| Constant | String value | Stage | Retryable |
|----------|-------------|-------|-----------|
| `SAFETY_GATE_BLOCKED` | `"safety_gate_blocked"` | `preflight` | No |
| `RETRY_BUDGET_EXHAUSTED` | `"retry_budget_exhausted"` | `execute` | No |

After Phase 6, `CANONICAL_REASONS` has 8 codes total (6 Phase 3/4 + 2 Phase 6).

---

#### 10.6  execute_with_lifecycle updated flow (Phase 6)

```
execute_with_lifecycle(adapter_name, op, params, policy):

  0. [Phase 5] validate_settings(brand, policy.session_config)   [if brand present]
       → on error: return PREFLIGHT_FAILED, stage="settings_validation"

  1. acquire adapter (ServiceRegistry.require)
       → on miss: return ADAPTER_NOT_FOUND, stage="acquire"

  2. open_session(config=policy.session_config)
       → on error: return SESSION_OPEN_FAILED, stage="open_session"

  3. [Phase 6] SafetyPolicy gate                                 [if policy.safety_policy.require_preflight]
       → call adapter.preflight(context=safety_policy.safety_context)
       → on blocked: return SAFETY_GATE_BLOCKED, stage="preflight"
       → on error:   return PREFLIGHT_FAILED, stage="preflight"

  4. execute (with RetryPolicy)                                   [up to policy.retry_policy.max_attempts]
       → on retryable failure + budget remaining: wait backoff_ms, retry
       → on budget exhausted: return RETRY_BUDGET_EXHAUSTED, stage="execute"
       → on non-retryable failure: return EXECUTE_FAILED, stage="execute"

  5. close_session()                                              [if policy.close_after=True]
       → failure surfaced in diagnostics; payload NOT masked

  6. return structured result (with telemetry in diagnostics["telemetry"])
```

Note: Steps 0–2 and 5–6 are unchanged from Phase 5.  Steps 3–4 are Phase 6.
Telemetry events are emitted at start/end of each active step.

---

#### 10.7  New and changed modules (Phase 6)

| Module | Role |
|--------|------|
| `system/service/telemetry.py` | `TelemetryEvent`, `TelemetryCollector`, `emit_event()` |

Existing modules changed in Phase 6:

| Module | Change |
|--------|--------|
| `system/service/lifecycle.py` | Add `SafetyPolicy`, `RetryPolicy`; add 2 new `LifecycleReason` constants |
| `system/service/reason_codes.py` | Add Phase 6 constants; add `retryable: bool` to every `REASON_MAP` entry |
| `system/service/service_router.py` | Wire `SafetyPolicy` gate (Step 3) + `RetryPolicy` (Step 4) + telemetry emission |

---

### §11  Phase 6 Stage Checklist

| Stage | Objective | Status |
|-------|-----------|--------|
| STAGE-01 | Baseline audit + Phase 6 contract freeze | ✅ |
| STAGE-02 | Unified preflight gate (`SafetyPolicy`) | ✅ |
| STAGE-03 | Structured telemetry schema (`TelemetryEvent`) | ✅ |
| STAGE-04 | Error taxonomy hardening (Phase 6 reason codes + retryable) | ✅ |
| STAGE-05 | Retry policy boundaries (`RetryPolicy`) | ✅ |
| STAGE-06 | Runtime integration and compatibility | ✅ |
| STAGE-07 | Phase 6 test matrix | ✅ |
| STAGE-08 | Phase 6 acceptance report + Phase 7 handoff | ✅ |

---

*Phase 6 started 2026-02-24.*

---

## Phase 7 — Migration and Cleanup

*(STAGE-01 baseline audit completed 2026-02-24)*

### §12  Phase 7 Migration Contract

This section is the canonical, frozen reference for all Phase 7 implementation.
All code in STAGE-02 through STAGE-07 must conform to the contracts defined here.

---

#### 12.1  Baseline audit summary

| Area | Current State (Phase 6 exit) | Phase 7 Gap |
|------|------------------------------|-------------|
| Node action dispatch | `action_nodes.py` calls `robot_model.run_action()` directly (bypasses lifecycle) | M1: migrate to `RobotContext.run_action()` |
| Node sensor dispatch | `sensor_nodes.py` calls `robot_model.get_sensor_data()` directly | M2: migrate to `RobotContext.get_sensor_data()` |
| RobotContext model factory | `_create_model_for_brand()` silently falls back to `UnitreeModel` for unknown brands | M3: replace with explicit error return (None) |
| RobotContext action fallback | `run_action()` and `get_sensor_data()` fall back to direct model call on lifecycle exception | M4/M5: remove double-fallback; surface error explicitly |
| Runtime execution path | Flow-aware DFS default; topological sort via `UNITPORT_FLOW_AWARE_EXECUTION=0` | M6: formally document as retained compat gate — no removal |
| Behavior bridge absent path | `node_executor.py` returns `"skipped"` when `_behavior_bridge is None` | M7: formally document as retained compat gate — no removal |
| `bin/core/node_executor.py` | Already a thin re-export: `from system.runtime.node_executor import NodeExecutor` | M8: migration complete — doc only |

---

#### 12.2  Migration map

| ID | Source path | Target unified path | Stage | Blocker |
|----|------------|---------------------|-------|---------|
| M1 | `nodes/sys_nodes/action_nodes.py:60` — `robot_model.run_action()` | `RobotContext.run_action()` (internally lifecycle-routed) | STAGE-02 | None |
| M2 | `nodes/sys_nodes/sensor_nodes.py:60` — `robot_model.get_sensor_data()` | `RobotContext.get_sensor_data()` (internally lifecycle-routed) | STAGE-02 | None |
| M3 | `bin/core/robot_context.py:252-255` — silent UnitreeModel fallback for unknown brand in `_create_model_for_brand()` | Return `None` + explicit log_error (matches `_create_adapter_for_brand()` behaviour) | STAGE-04 | Must audit callers of `get_robot_model()` first |
| M4 | `bin/core/robot_context.py:362-373` — lifecycle-exception fallback to `robot.run_action()` | Remove double-fallback; log error and return `False` | STAGE-04 | Test coverage of exception path required (STAGE-02) |
| M5 | `bin/core/robot_context.py:395-403` — lifecycle-exception fallback in `get_sensor_data()` | Remove double-fallback; log error and return error dict | STAGE-04 | Same as M4 |
| M6 | `system/runtime/migration.py` — `UNITPORT_FLOW_AWARE_EXECUTION=0` topological sort | **Retain** as explicit compat gate; define policy doc | STAGE-05 | — |
| M7 | `system/runtime/node_executor.py:592-597` — bridge-absent → `"skipped"` | **Retain** as explicit compat gate; define policy doc | STAGE-05 | — |
| M8 | `bin/core/node_executor.py` — re-export wrapper | Already migrated; document as complete | STAGE-01 | — |

---

#### 12.3  Default path policy (Phase 7 exit state)

After Phase 7, the canonical execution flow for all node-triggered robot operations is:

```
Node.execute()
  └── RobotContext.run_action() / .get_sensor_data() / .stop()
        └── RobotContext._lifecycle_route()
              └── ServiceRouter.execute_with_lifecycle()
                    ├── Step 0: settings validation (brand-gated)
                    ├── Step 1: acquire adapter
                    ├── Step 2: open_session
                    ├── Step 3: preflight gate (SafetyPolicy / legacy run_preflight)
                    ├── Step 4: execute (bounded retry)
                    └── Step 5: close_session (if close_after)
```

**Rule**: Node execute methods must not call `robot_model.*` directly.
They must call `RobotContext.*` class methods only.
`RobotContext.*` is responsible for all lifecycle routing.

---

#### 12.4  Allowed fallback scope

The following fallbacks are **explicitly retained** after Phase 7 with defined compat policy:

| Fallback | Location | Compat policy | Removal criteria |
|----------|----------|---------------|-----------------|
| Topological sort execution | `system/runtime/migration.py` + `node_executor.py` | Retained; opt-in via `UNITPORT_FLOW_AWARE_EXECUTION=0` env var only; logged as compatibility path | Remove when flow-aware DFS has 3+ months production stability with no regressions |
| Behavior bridge absent → `"skipped"` | `system/runtime/node_executor.py:592-597` | Retained; needed for test environments without a behavior bridge injected; logged as skip | Remove when all callers inject bridge or use explicit null-bridge sentinel |
| Legacy `ServiceRouter` passthrough (`run_action`, `stop`, `get_sensor_data`, `health`) | `system/service/service_router.py` | Retained; Phase 1 guarantee; STAGE-05: all 4 methods now emit `_log.warning("DEPRECATED: …")` on every call | Remove in a future phase when all callers migrate to `execute_with_lifecycle` |

**Removal criteria (general):** A fallback may be removed when:
1. No passing test exercises the fallback branch.
2. The unified path covers the same case.
3. The removal is backed by a new test that confirms the unified path handles the previously-fallback scenario.

---

#### 12.5  Removal criteria for double-fallbacks (M3/M4/M5)

M3, M4, M5 are **silent fallbacks** — they hide misconfigurations from callers.
Unlike the explicitly-gated fallbacks in §12.4, they must be removed because:

- M3: `_create_model_for_brand()` returning a UnitreeModel for an unknown brand silently
  routes all unknown brands to Unitree hardware — this is data-corrupting, not a safe default.
- M4/M5: The lifecycle-exception fallback in `RobotContext.run_action()` suppresses lifecycle
  errors (e.g., adapter not found, preflight blocked) by silently retrying on the model directly,
  making lifecycle hardening from Phase 3–6 ineffective for node-level calls.

---

#### 12.6  New and changed modules (Phase 7)

New modules to be created:
*(none — Phase 7 is migration/cleanup only; no new architectural modules)*

Existing modules changed in Phase 7:

| Module | Change |
|--------|--------|
| `nodes/sys_nodes/action_nodes.py` | STAGE-02 (M1): Replace direct `robot_model.run_action()` with `RobotContext.run_action()` |
| `nodes/sys_nodes/sensor_nodes.py` | STAGE-02 (M2): Replace direct `robot_model.get_sensor_data()` with `RobotContext.get_sensor_data()` |
| `bin/core/robot_context.py` | STAGE-04 (M3/M4/M5): Remove `_create_model_for_brand()` dead code; remove double-fallbacks in `run_action()`, `get_sensor_data()`, `stop()` |
| `system/runtime/migration.py` | STAGE-03: Docstring updated to reflect Phase 7 policy — flow-aware DFS is stable default; topological sort is rollback-only |
| `system/runtime/runtime_engine.py` | STAGE-03: Explicit `log_warning` on compat-path activation; STAGE-05: inject `compat_path`/`compat_reason` into diagnostics when compat path is active |
| `system/runtime/node_executor.py` | STAGE-05 (M7): Bridge-absent skip result now carries `"compat_path": True` marker |
| `system/runtime/contracts.py` | STAGE-05: `DiagnosticsKey.COMPAT_PATH` and `DiagnosticsKey.COMPAT_REASON` constants added |
| `system/service/service_router.py` | STAGE-05: All 4 legacy passthrough methods emit `_log.warning("DEPRECATED: …")` |
| `system/service/README.md` | STAGE-01: §12/§13 added; STAGE-06: §12.4/§12.6 updated, §14 added; stage checklist updated per stage |

---

### §13  Phase 7 Stage Checklist

| Stage | Objective | Status |
|-------|-----------|--------|
| STAGE-01 | Baseline audit + Phase 7 migration contract freeze | ✅ |
| STAGE-02 | Behavior artifact migration (node direct-call → RobotContext) | ✅ |
| STAGE-03 | Runtime default path consolidation | ✅ |
| STAGE-04 | Fallback and duplicate logic cleanup | ✅ |
| STAGE-05 | Compatibility guardrails and deprecation policy | ✅ |
| STAGE-06 | Documentation and developer workflow cleanup | ✅ |
| STAGE-07 | Phase 7 test matrix | ✅ |
| STAGE-08 | Phase 7 final acceptance and framework closure | ✅ |

---

### §14  Phase 7 Developer Notes

#### 14.1  Compat gate reference

Two explicit compat gates are retained after Phase 7.  Both require opt-in; neither
is active by default.

| Gate | Activation | Diagnostic marker | When to use |
|------|-----------|-------------------|-------------|
| Topological sort execution | `UNITPORT_FLOW_AWARE_EXECUTION=0` env var | `RuntimeResult.diagnostics["compat_path"] = True`, `["compat_reason"] = "UNITPORT_FLOW_AWARE_EXECUTION=0"` | Controlled rollback when flow-aware DFS regressions are suspected |
| Behavior bridge absent → `"skipped"` | `NodeExecutor._behavior_bridge = None` (not injected by caller) | `node_result["compat_path"] = True` in the per-node output dict | Test environments that execute workflows without a compiled behavior bridge |

To check whether a result travelled through the compat path:
```python
result = engine.execute(mission_ir, scenario)
if result["diagnostics"].get("compat_path"):
    reason = result["diagnostics"]["compat_reason"]  # e.g. "UNITPORT_FLOW_AWARE_EXECUTION=0"
    # handle or log appropriately
```

#### 14.2  ServiceRouter deprecation warnings

The four Phase 1 legacy passthrough methods now emit `_log.warning("DEPRECATED: …")`
on every call.  Callers should migrate to `execute_with_lifecycle`:

| Deprecated method | Replacement call |
|-------------------|-----------------|
| `router.run_action(adapter, action)` | `router.execute_with_lifecycle(adapter, RouteOp.RUN_ACTION, {"action": action, "params": {}})` |
| `router.stop(adapter)` | `router.execute_with_lifecycle(adapter, RouteOp.STOP)` |
| `router.get_sensor_data(adapter)` | `router.execute_with_lifecycle(adapter, RouteOp.GET_SENSOR_DATA)` |
| `router.health(adapter)` | Direct adapter call (no lifecycle equivalent yet) |

**Note:** Callers that reach these methods through `RobotContext._lifecycle_route()` are
**already on the correct path** — `RobotContext` uses `execute_with_lifecycle` internally
and never calls the legacy passthrough methods.

#### 14.3  Test execution reference

```bash
# Full test suite (Phase 7 state: 1983 tests)
python -m pytest tests/ -q

# Phase 7 stage-specific tests only
python -m pytest \
  tests/unit/test_stage02_node_migration.py \
  tests/unit/test_stage03_runtime_default_path.py \
  tests/unit/test_stage04_fallback_cleanup.py \
  tests/unit/test_stage05_compat_guardrails.py \
  -v

# Verify compat gate is gated (default path asserted)
python -m pytest tests/unit/test_stage03_runtime_default_path.py -v

# Exercise compat path explicitly (topological sort)
UNITPORT_FLOW_AWARE_EXECUTION=0 python -m pytest tests/ -q
```

---

*Phase 7 started 2026-02-24.*

---

## Product Cycle 1 (P0): Mission Editor Completion + Runtime Visibility

*(STAGE-01 baseline audit completed 2026-02-25)*

### §C1-1  Cycle 1 Scope

**Goal:** Deliver a production-usable Mission workflow frontend for daily authoring and execution.

In scope:
1. Mission canvas authoring completeness (node add/remove/duplicate/group, edge connect/disconnect/re-route, branch/loop param editing with inline validation)
2. Runtime visibility in UI (per-node status badges: pending/running/success/failed/skipped; session summary; diagnostics drill-down)
3. Mission persistence (save/load/version snapshots; import/export consistency checks)
4. Error UX (runtime/service error → user-readable text + technical detail view)

Out of scope: Behavior SDK settings panel (Cycle 2), MuJoCo deep customization (Cycle 3), new adapter/brand features.

---

### §C1-2  Baseline Audit Findings (STAGE-01)

#### C1-2.1  Canvas Operations Audit (`bin/components/graph_scene.py`)

| Operation | Status | Location |
|-----------|--------|----------|
| Add node (drag-drop from palette) | ✅ Implemented | `create_node()` |
| Delete node | ✅ Implemented | `_delete_items()` |
| Connect edge (port-to-port) | ✅ Implemented | `_start_connection()` / `_finish_connection()` |
| Disconnect edge | ✅ Implemented | `_detach_connection()` |
| Re-route edge (drag endpoint) | ✅ Implemented | `_start_reconnection()` / `_finish_reconnection()` |
| Inline parameter editing (branch/loop) | ✅ Implemented | `_update_node_params()` |
| Serialize workflow to JSON | ✅ Implemented | `serialize_workflow()` |
| Load workflow from JSON | ✅ Implemented | `load_workflow()` |
| **Duplicate node** | ❌ Missing | Not implemented |
| **Group/ungroup nodes** | ❌ Missing | Not implemented |
| **Undo/redo** | ❌ Missing | No QUndoStack; deferred to out-of-scope for Cycle 1 |

*Undo/redo and group/ungroup are not required for Cycle 1 acceptance; duplicate node IS required.*

#### C1-2.2  Runtime Visibility Audit (`bin/ui.py` + `system/runtime/runtime_engine.py`)

| Item | Status | Notes |
|------|--------|-------|
| `RuntimeEngine.execute()` returns per-node results | ✅ Present | `run_result["results"]` keyed by node_id |
| `diagnostics` dict (failed_nodes, behavior_diagnostics, compat_path) | ✅ Present | `run_result["diagnostics"]` |
| Per-node status badges in canvas | ❌ Missing | Results not fed back to canvas nodes |
| Execution session summary panel | ❌ Missing | Only log line + status bar |
| Async/non-blocking execution (Qt signals from runtime) | ❌ Missing | RuntimeEngine is synchronous; blocks UI thread |
| Diagnostics drill-down panel | ❌ Missing | Not implemented |

#### C1-2.3  Mission Persistence Audit (`bin/layout/main_zone_panel.py`)

| Item | Status | Notes |
|------|--------|-------|
| `serialize_workflow()` / `load_workflow()` in GraphScene | ✅ Present | Backend serialization complete |
| Save mission button in UI | ❌ Missing | No save/load buttons in panel |
| Load mission button in UI | ❌ Missing | No file dialog for open |
| Version snapshot metadata | ❌ Missing | No timestamp/version tag in serialized data |
| Import/export consistency check | ❌ Missing | No validation on load |

#### C1-2.4  Error UX Audit (`bin/ui.py` `_on_run()`)

| Item | Status | Notes |
|------|--------|-------|
| Error reason code handling in `_on_run` | ⚠️ Partial | Only 2 specific codes handled (`simulation_reset_failed`, `simulation_already_running`) |
| User-readable error message mapping | ❌ Missing | No canonical reason_code → display_text table |
| Distinction: validation / preflight / execute / compat failures | ❌ Missing | All collapsed to single warning dialog |
| Technical detail view for engineers | ❌ Missing | Raw diagnostics only in log, not in UI panel |

---

### §C1-3  Frozen Acceptance Checklist (Stages 01–08)

#### STAGE-01 ✅ Baseline Audit
- [x] Canvas operation gap analysis documented
- [x] Runtime visibility gap documented
- [x] Mission persistence gap documented
- [x] Error UX gap documented
- [x] Cycle 1 frozen contract written

#### STAGE-02 ✅ Mission Canvas Authoring Completeness
- [x] `duplicate_selected_nodes()` implemented (Ctrl+D); `_apply_node_data()` extracted for reuse
- [x] `GroupContainerItem` class added; `group_selected_nodes(title)` (Ctrl+G); `ungroup_selected_groups()` (Ctrl+Shift+G)
- [x] Group movement propagates delta to member nodes via `itemChange`
- [x] `serialize_workflow` includes `"groups"` key; `_load_workflow_impl` restores groups (backward compat: missing key is safe)
- [x] `_delete_items` handles group containers (container deleted → members detached, nodes retained)
- [x] `clear_all_nodes` clears group containers and `_groups` dict
- [x] Undo/redo: **deferred to Cycle 4** (out of P0 scope)
- [x] All edge operations confirmed working end-to-end (audit verified)
- [x] Branch/loop inline param editing confirmed for all node types (audit verified)
- [x] Tests: `tests/unit/test_canvas_authoring.py` — 40/40 passed (27 new group/ungroup tests)
- [x] Full regression: 2049/2049 passed

#### STAGE-03 ✅ Runtime Status Surface in UI
- [x] Per-node status badges rendered (pending/running/success/failed/skipped) via `set_node_execution_status()` colored border pen
- [x] Execution session summary panel (`ExecutionSummaryBar`) shown after run in `MainZonePanel`
- [x] **In-flight visibility**: all nodes marked `"pending"` before execute; `node_status_callback(node_id, status)` fires `"running"` then `"success"/"failed"` per node during execution (Cycle 1 compliance fix)
- [x] `node_status_callback` parameter added to `WorkflowRunner.run()`, `NodeExecutor.execute()`, `RuntimeEngine.execute()` — backward-compat: absent/None is a no-op
- [x] `QApplication.processEvents()` called in UI callback so each transition renders before next node starts
- [x] `_apply_node_execution_statuses()` applied post-run to resolve `"skipped"` for nodes not reached by control flow
- [x] Status updates stable after re-run and reset (`reset_execution_statuses()` + `clear_execution_summary()`)
- [x] Tests: `tests/unit/test_runtime_status_surface.py` — 57/57 passed (added `TestWorkflowRunnerCallback` 10, `TestNodeExecutorCallback` 7, `TestPendingAndLifecycle` 12; fixed spy kwarg in `test_stage03_runtime_default_path.py`)
- [x] Full regression: 2152/2152 passed

#### STAGE-04 ✅ Diagnostics Drill-Down UX
- [x] `bin/core/error_ux.py`: `REASON_OPERATOR_TEXT` mapping (27 codes), `format_node_diagnostics()`, `extract_failed_nodes_info()`, deterministic `DISPLAY_KEY_ORDER`
- [x] `bin/components/diagnostics_panel.py`: `DiagnosticsPanel` widget — stage/reason/operator_text/message/adapter_name/retryable/context fields
- [x] Telemetry pointers surfaced in diagnostics (`trace_id`, `mission_trace_id`) with deterministic key ordering and friendly labels in panel view
- [x] Friendly view default; Raw JSON toggle (`QCheckBox`) reveals full payload
- [x] Node selector `QComboBox` for multi-failure navigation (hidden when single failure)
- [x] `navigate_requested(node_id)` signal → `ui.py._on_navigate_to_node()` → `graph_view.centerOn(item)` + `item.setSelected(True)`
- [x] `get_node_item(node_id)` added to `GraphScene` for canvas navigation lookup
- [x] `DiagnosticsPanel` auto-shown after run when failures exist; "Details" toggle button in summary bar for re-open
- [x] Tests: `tests/unit/test_diagnostics_panel.py` — 72/72 passed (includes telemetry pointer extraction + panel rendering suites)
- [x] Done criteria: operator can identify failing node + reason within 2 clicks (auto-show on fail → click "Go to Node")
- [x] Verification rerun: `python -m unittest tests.unit.test_diagnostics_panel -v` passed

#### STAGE-05 ✅ Mission Persistence and Snapshot Flow
- [x] Save mission (file dialog → JSON) in UI — `bin/ui.py._on_save()`
- [x] Load mission (file dialog → JSON → restore canvas) in UI — `bin/ui.py._on_open()`
- [x] Version snapshot metadata (`schema_version`, `unitport_version`, `saved_at`, `source`) via `inject_snapshot_metadata()` — `bin/core/mission_persistence.py`
- [x] Import/export consistency check + mismatch feedback — `validate_mission_schema()` with user-visible `QMessageBox.warning()`
- [x] Backward compatibility for existing mission JSON files — `schema_version`/`source` absence is not an error
- [x] Tests: `tests/integration/test_mission_persistence.py` — 40/40 passed
- [x] Save → load → execute roundtrip tests: `tests/integration/test_save_load_execute.py` — 18/18 passed
- [x] Verification rerun: both integration suites above passed via `python -m unittest ... -v`

#### STAGE-06 ✅ Error UX and Contract Mapping
- [x] Canonical reason_code → operator text mapping table — 27 codes in `REASON_OPERATOR_TEXT` (`bin/core/error_ux.py`)
- [x] Distinction: validation / preflight_safety / runtime / compat_warning — `REASON_CATEGORY`, `BLOCKED_PATH_CATEGORY`, `get_error_category()`, `classify_run_result()` (`bin/core/error_ux.py`)
- [x] `error_category` field in `format_node_diagnostics()` output; rendered in DiagnosticsPanel friendly view with human labels; raw view shows raw constant
- [x] Raw diagnostics preserved and accessible for engineers — Raw toggle in `DiagnosticsPanel` shows unaltered JSON; `error_category` value is raw constant in JSON
- [x] Localization hooks kept clean — all `REASON_OPERATOR_TEXT` values are plain Python strings (no `tr()`); `format_node_diagnostics()` returns locale-independent dicts; display-layer labels localizable independently
- [x] Tests: `tests/unit/test_error_ux_mapping.py` — 74/74 passed

#### STAGE-07 ✅ Cycle 1 Test Matrix
- [x] Unit tests: canvas ops — `tests/unit/test_canvas_authoring.py` (40 tests)
- [x] Unit tests: inline parameter validation — `tests/unit/test_inline_param_validation.py` (27 tests)
- [x] Unit tests: error mapping — `tests/unit/test_error_ux_mapping.py` (74 tests)
- [x] Unit tests: runtime status surface — `tests/unit/test_runtime_status_surface.py` (57 tests)
- [x] Unit tests: diagnostics panel — `tests/unit/test_diagnostics_panel.py` (72 tests)
- [x] Integration tests: authoring → execution → diagnostics — `tests/integration/test_authoring_execution_diagnostics.py` (40 tests)
- [x] Integration tests: save → load → execute roundtrip — `tests/integration/test_save_load_execute.py` (18 tests) + `tests/integration/test_mission_persistence.py` (40 tests)
- [x] Regression: Phase 3-7 critical matrices remain green — 551/551 passed
- [x] All test commands documented and reproducible from repo root (see §C1-5 below)
- [x] Full Cycle 1 suite: 368/368 passed, full suite regression: 551/551 + 852 subtests passed

#### STAGE-08 ✅ Cycle 1 Acceptance Report
- [x] `CYCLE1_ACCEPTANCE_REPORT.md` delivered (repo root)
- [x] Changed files summary complete (new + modified, by stage)
- [x] Test evidence attached — 368/368 Cycle 1 + 551/551 Phase 3-7 regression (commands in §5)
- [x] Known gaps documented — 5 accepted-by-design deferred items listed
- [x] Cycle 2 handoff checklist explicit — 7 C2 entry items + stable contract table

---

### §C1-5  Cycle 1 Test Commands (Documented)

All commands run from the repository root.  Set `QT_QPA_PLATFORM=offscreen`
for headless / CI environments (already set inside each test module via
`os.environ.setdefault`).

```bash
# ── Unit tests ────────────────────────────────────────────────────────────────

# Canvas authoring (duplicate, group/ungroup, serialize roundtrip, shortcuts)
pytest tests/unit/test_canvas_authoring.py -q

# Inline parameter validation (Timer, Loop, Retry, Break, Branch)
pytest tests/unit/test_inline_param_validation.py -q

# Error UX mapping (REASON_CATEGORY, get_error_category, classify_run_result)
pytest tests/unit/test_error_ux_mapping.py -q

# Runtime status surface (ExecutionSummaryBar, node status badges)
pytest tests/unit/test_runtime_status_surface.py -q

# DiagnosticsPanel (friendly view, raw toggle, telemetry pointers, category labels)
pytest tests/unit/test_diagnostics_panel.py -q

# ── Integration tests ─────────────────────────────────────────────────────────

# Authoring → execution → diagnostics end-to-end pipeline
pytest tests/integration/test_authoring_execution_diagnostics.py -q

# Mission persistence (schema validation, snapshot metadata, file roundtrip)
pytest tests/integration/test_mission_persistence.py -q

# Save → load → execute roundtrip (in-memory + file I/O)
pytest tests/integration/test_save_load_execute.py -q

# ── Full Cycle 1 matrix (all 8 suites above, 368 tests) ──────────────────────
pytest tests/unit/test_canvas_authoring.py \
       tests/unit/test_inline_param_validation.py \
       tests/unit/test_error_ux_mapping.py \
       tests/unit/test_runtime_status_surface.py \
       tests/unit/test_diagnostics_panel.py \
       tests/integration/test_authoring_execution_diagnostics.py \
       tests/integration/test_mission_persistence.py \
       tests/integration/test_save_load_execute.py -q

# ── Phase 3-7 critical regression (551 tests + 852 subtests) ─────────────────
pytest tests/integration/test_phase3_matrix.py \
       tests/integration/test_phase4_matrix.py \
       tests/integration/test_phase5_matrix.py \
       tests/integration/test_phase6_matrix.py \
       tests/integration/test_phase7_matrix.py \
       tests/unit/test_stage02_node_migration.py \
       tests/unit/test_stage03_runtime_default_path.py \
       tests/unit/test_stage04_fallback_cleanup.py \
       tests/unit/test_stage05_compat_guardrails.py \
       tests/unit/test_stage06_diagnostics.py -q

# ── Full suite ────────────────────────────────────────────────────────────────
pytest tests/ -q
```

---

### §C1-4  Implementation Guardrails (Cycle 1)

1. Do not break `serialize_workflow()` / `load_workflow()` signatures — canvas and test suites depend on them.
2. Canvas status badges must be additive overlays; do not modify existing node item render logic destructively.
3. Runtime results must be passed to UI via existing `run_result` return dict — do not introduce new synchronous blocking calls.
4. Error mapping table lives in a dedicated module (`bin/core/error_ux.py`) to keep `ui.py` clean.
5. All new files must have corresponding unit or integration tests before STAGE-07.

---

*Cycle 1 started 2026-02-25.*

---

## Product Cycle 2 — Behavior SDK Settings Productization + Execution Control

---

### §C2-1  Baseline Audit Summary

Audit date: 2026-02-25.  All eight touchpoint files inspected.

| Gap | Current state | Cycle 2 target |
|-----|--------------|----------------|
| **G1** Settings UI surface | No dynamic form from `get_settings_schema()`; `MainZonePanel._scenario_settings` contains MuJoCo-only keys | Dedicated settings panel with dynamic form rendering from schema descriptors |
| **G2** Capability inspector | `capabilities()` return value never surfaced in UI; no display widget | Capability inspector widget bound to adapter `capabilities()` dict |
| **G3** Settings → execution handoff | `_on_run()` builds no `session_config` with `brand`; `execute_with_lifecycle` Step-0 validate_settings() never fires from UI | `_on_run()` collects validated settings into `session_config` before calling lifecycle |
| **G4** Async execution / cancel | `_on_run()` is synchronous on main thread; `_on_runtime_abort()` handles only MuJoCo `simulation_thread`, not `RuntimeEngine` | Non-blocking QThread run; explicit cancel path wired to RuntimeEngine |
| **G5** Settings persistence | Settings not saved/loaded with `.unitport` mission files; roundtrip gap | `settings` key in mission JSON; backward-compat for files without it |

Existing services that Cycle 2 builds **on top of** (no changes required):

| Module | Confirmed stable |
|--------|-----------------|
| `system/service/settings_schema.py` — `get_settings_schema(brand)` | ✅ |
| `system/service/settings_validator.py` — `validate_settings(brand, config)` | ✅ |
| `system/service/capability_schema.py` — `validate_capability()` | ✅ |
| All three adapter `capabilities()` methods | ✅ |
| `service_router.execute_with_lifecycle()` Step-0 validation gate | ✅ |

---

### §C2-2  Cycle 2 Frozen Contract

#### Settings domain model

- `settings_schema.get_settings_schema(brand)` is the single source of truth for field descriptors.
- Each field descriptor carries: `{type, required, default, description, choices}`.
- A **form descriptor** layer (`bin/core/settings_form.py`) maps schema dicts → `{field_id, label, type, required, default, choices, description, order}` with deterministic field order.
- Form descriptors are stable across schema evolution (additive only).

#### Settings panel contract

- Entry point: `SettingsPanel(brand, config, parent)` widget.
- Exposes: `get_current_config() → dict` (current form values), `has_unsaved_changes() → bool`.
- Emits: `settings_applied(config: dict)`, `settings_reset()`.
- Inline validation calls `validate_settings(brand, config)` on apply; errors displayed per-field.
- Reset restores last applied (not schema defaults) until first apply.

#### Capability inspector contract

- Entry point: `CapabilityInspector(capabilities_dict, required_settings, parent)` widget.
- Renders: actions list, sensors list, flags list, required_settings list.
- Emits: `focus_setting(field_id: str)` when operator clicks a missing required setting.
- Resilient: renders partial data without crash when adapter returns incomplete capability dict.

#### Runtime settings handoff contract

- `_on_run()` must read `settings_panel.get_current_config()` and include it (with `brand` key) as `session_config` argument to lifecycle.
- `validate_settings()` fires at `execute_with_lifecycle()` Step 0 — no duplicate call in UI layer.
- On validation failure the run is blocked; failure diagnostics are passed to `DiagnosticsPanel`.

#### Async execution contract

- Mission execution runs in a `QThread` subclass (`MissionRunThread`).
- `MissionRunThread` emits: `run_finished(result: dict)`, `node_status(node_id: str, status: str)`.
- `_on_run()` starts thread, disables run button, enables abort button.
- `_on_runtime_abort()` calls `thread.request_cancel()`; thread propagates cancel to `RuntimeEngine`.
- UI status/diagnostics panels update via Qt signal — no direct cross-thread widget access.

#### Persistence contract

- Mission JSON gains an optional top-level `"settings"` key: `{brand, config}`.
- `validate_mission_schema()` treats absence of `"settings"` as valid (backward compat).
- `inject_snapshot_metadata()` does not modify `"settings"` value.
- On load: if `"settings"` present and brand matches active context, pre-populate settings panel.

---

### §C2-3  Cycle 2 Stage Checklist

| Stage | Objective | Status |
|-------|-----------|--------|
| STAGE-01 | Baseline audit + Cycle 2 contract freeze | ✅ |
| STAGE-02 | Settings domain model and form contract | ✅ |
| STAGE-03 | Behavior SDK settings panel (core UX) | ✅ |
| STAGE-04 | Capability inspector, settings-capability linkage, and real runtime wiring in `bin/ui.py` | ✅ |
| STAGE-05 | Validation and error UX integration | ✅ |
| STAGE-06 | Runtime handoff + async execution control | ✅ |
| STAGE-07 | Cycle 2 test matrix | ✅ |
| STAGE-08 | Cycle 2 acceptance report and Cycle 3 handoff | ✅ |

---

### §C2-4  Implementation Guardrails (Cycle 2)

1. Do not bypass `RobotContext` / `execute_with_lifecycle()` — settings must travel through the existing lifecycle gate.
2. No new silent fallback paths: if `brand` is present in session_config and validation fails, abort run with explicit diagnostics.
3. QThread execution: all widget updates from worker thread must use Qt signals — never touch widgets from non-main threads.
4. Settings panel is additive to existing `MainZonePanel` layout — do not remove existing widgets (ExecutionSummaryBar, DiagnosticsPanel).
5. All new modules (`settings_form.py`, `settings_panel.py`, `capability_inspector.py`, `mission_run_thread.py`) must have unit tests before STAGE-07.
6. `validate_mission_schema()` backward-compat guarantee is unconditional — missing `"settings"` key is never an error.

---

*Cycle 2 started 2026-02-25.*

---

## Product Cycle 3 — Settings Persistence, Runtime Policy Scoping, and Deep Runtime Control

*(STAGE-01 baseline audit completed 2026-02-26)*

---

### §C3-1  Cycle 2 Deferred Gap Re-Audit

The following gaps were explicitly deferred from Cycle 2 and form the Cycle 3 implementation scope.

| ID  | Gap | Priority | Source |
|-----|-----|----------|--------|
| G1  | Settings not persisted in mission JSON | P0 | Cycle 2 §5 |
| G2  | Cancel is cooperative-only (node boundary) — adapter mid-action cannot be interrupted | P1 | Cycle 2 §5 |
| G3  | Unsaved-change warning before run/navigation absent | P1 | Cycle 2 §5 |
| G4  | Capability inspector not auto-refreshed on live connect/disconnect events | P2 | Cycle 2 §5 |
| G5  | `RobotContext._lifecycle_policy` is class-level global — non-reentrant under concurrent runs | P2 | Cycle 2 §5 |

Additional item added in Cycle 3 scope:

| ID  | Gap | Priority | Source |
|-----|-----|----------|--------|
| G6  | Advanced MuJoCo settings surface (deferred from Cycle 2 scope) | P3 | DEVELOP_TODO.txt Cycle 3 |

---

### §C3-2  Module Audit (Cycle 3 Baseline)

#### §C3-2.1  `bin/core/mission_persistence.py`

| Item | Current State | Cycle 3 Action |
|------|--------------|----------------|
| `validate_mission_schema()` | Requires only `"nodes"` + `"connections"` keys; tolerates extra keys | Add explicit backward-compat: missing `"settings"` key is never an error (**already true** — no change needed) |
| `inject_snapshot_metadata()` | Stamps schema_version/unitport_version/saved_at/source | **Extend** to accept and stamp optional `settings` payload non-destructively |
| `MISSION_SCHEMA_VERSION` | `"1.0"` | **Bump to `"1.1"`** when `"settings"` key is added to canonical save format |

#### §C3-2.2  `bin/core/robot_context.py`

| Item | Current State | Cycle 3 Action |
|------|--------------|----------------|
| `_lifecycle_policy` | Class-level attribute — mutated by `set_lifecycle_policy()` | **Refactor** `_lifecycle_route()` to accept explicit `policy` arg; keep class-level default as fallback for legacy callers |
| `set_lifecycle_policy()` / `get_lifecycle_policy()` | Public class methods — snapshot/restore used by `bin/ui.py` | Preserve signature; behavior unchanged for legacy callers |
| `_create_adapter_for_brand()` | Factory for temp adapters (capability inspection) | No change required |

#### §C3-2.3  `system/service/adapters/base_adapter.py`

| Item | Current State | Cycle 3 Action |
|------|--------------|----------------|
| `cancel_action()` | **Absent** | **Add** non-abstract default no-op method: `cancel_action() -> Dict[str, Any]` returning `LifecycleResult.ok("cancel_action")` |
| `run_action()` | Abstract | No change |
| `capabilities()` | Non-abstract, returns `{"actions": [], "sensors": [], "flags": {}}` | No change |

#### §C3-2.4  `bin/core/mission_run_thread.py`

| Item | Current State | Cycle 3 Action |
|------|--------------|----------------|
| `request_cancel()` | Delegates to `engine.request_cancel()` — cooperative node-boundary cancel | **Extend** to also call `adapter.cancel_action()` on the active adapter via `RobotContext` cancel path |

#### §C3-2.5  `bin/components/settings_panel.py`

| Item | Current State | Cycle 3 Action |
|------|--------------|----------------|
| `has_unsaved_changes()` | Tracked internally; no guard on run/navigate | **Wire** guard into `bin/ui.py._on_run()` and navigation paths |

#### §C3-2.6  `bin/ui.py`

| Item | Current State | Cycle 3 Action |
|------|--------------|----------------|
| `_on_save_mission()` | Saves canvas graph only | **Extend** to include `"settings": {"brand": ..., "config": {...}}` in payload |
| `_on_load_mission()` | Restores canvas graph only | **Extend** to restore settings when brand is compatible; log + skip gracefully on mismatch |
| `_on_run()` | Validates settings, then executes | **Add** `has_unsaved_changes()` guard before validation |
| `_refresh_capability_inspector()` | Triggered on 4 fixed events | **Add** connection-state-change event trigger |

#### §C3-2.7  `bin/layout/main_zone_panel.py`

| Item | Current State | Cycle 3 Action |
|------|--------------|----------------|
| `SettingsPanel` mount | Settings tab exists; supports apply/reset/unsaved tracking | Keep as Cycle 3 settings source of truth for save/load/run guardrails |
| `CapabilityInspector` mount | Capabilities tab exists; focus-setting jump to Settings tab is wired | No structural change; rely on `bin/ui.py` event wiring for live refresh triggers |
| `_scenario_settings` | MuJoCo baseline fields only (`backend`, `realtime`, `timestep`, `scene_xml`) | Extend bounded advanced MuJoCo fields additively; keep defaults deterministic |
| Existing Mission/Behavior tabs | Stable and in daily use | No redesign; additive-only UX changes for Cycle 3 |

#### §C3-2.8  `system/runtime/runtime_engine.py`

| Item | Current State | Cycle 3 Action |
|------|--------------|----------------|
| `request_cancel()` | Sets cooperative cancel flag consumed by runner/executor | Keep cooperative cancel as fallback path; do not remove |
| Cancel semantics | Effective at node-boundary checks | Integrate adapter-aware cancel plumbing via MissionRunThread + RobotContext path |
| Result schema | Emits `status/reason/stage/diagnostics` compatible payload | Preserve schema; additive diagnostics only for cancel distinction |
| Thread model | Executed in worker thread via MissionRunThread | Maintain Qt signal boundary; no direct UI mutation from runtime thread |

#### §C3-2.9  `system/service/service_router.py`

| Item | Current State | Cycle 3 Action |
|------|--------------|----------------|
| `execute_with_lifecycle()` | Lifecycle routing baseline is stable and already validates settings | Preserve routing contract and validation gate; no bypass from new paths |
| Policy usage | Works with provided policy and legacy fallback behavior | Ensure run-scoped policy pass-through remains deterministic and backward-compatible |
| Diagnostics/telemetry | Existing schema used by UI and tests | Keep keys stable (`status/reason/stage/diagnostics/adapter_name`); additive fields only |
| Legacy passthrough APIs | `run_action` / `stop` / `get_sensor_data` remain for compatibility | Keep callable; avoid regressions while migrating callers to explicit scoped policy |

#### §C3-2.10  Adapter implementations (`unitree_sdk2`, `spot_sdk`, `cyberdog_sdk`)

| Item | Current State | Cycle 3 Action |
|------|--------------|----------------|
| `cancel_action()` support | Not guaranteed across concrete adapters | Add safe adapter-level contract; default no-op via BaseAdapter is acceptable |
| `capabilities()` behavior | Implemented and consumed by capability inspector | Keep contract stable; improve refresh timing via connection event triggers |
| Lifecycle methods | Open/preflight/close contract already in place from prior phases | No semantic regression; cancel integration must not bypass lifecycle routing |
| SDK availability/degradation | Spot/CyberDog already guard missing SDK scenarios | Preserve non-crashing behavior and reason-coded diagnostics |

---

### §C3-3  Frozen Implementation Contract (Cycle 3)

#### Contract C3-A — Mission Settings Persistence (STAGE-02, P0)

1. Mission save payload MUST include a `"settings"` key with structure:
   ```json
   {
     "settings": {
       "brand": "<brand_key>",
       "config": { "<field>": "<value>", ... }
     }
   }
   ```
2. Mission load MUST restore settings when `payload["settings"]["brand"]` matches current or selectable brand.
3. When `"settings"` key is absent (old files), load proceeds silently — no error, no warning beyond debug log.
4. When brand is incompatible (e.g. file saved for `"bostiondynamics"` but current brand is `"unitree"`), settings payload is ignored with an info-level log message; canvas is still loaded.
5. `validate_mission_schema()` backward-compat guarantee: missing `"settings"` is NOT an error — unconditional.
6. `MISSION_SCHEMA_VERSION` bumped from `"1.0"` → `"1.1"` in the save path only; load accepts both versions.

#### Contract C3-B — Runtime Policy Scoping (STAGE-03, P0)

1. `RobotContext._lifecycle_route()` MUST accept an optional `policy` argument (already has it — verify it is used consistently in `run_action`, `get_sensor_data`, `stop`).
2. A `run_scoped_policy(session_config)` context-manager or helper MAY be introduced to produce a per-call `LifecyclePolicy` without mutating class state.
3. The snapshot/restore pattern in `bin/ui.py` (`_pre_run_policy_snapshot`) MUST remain functional for legacy compatibility.
4. Existing `execute_with_lifecycle` orchestration semantics are unchanged.

#### Contract C3-C — Preemptive Cancel (STAGE-04, P1)

1. `BaseAdapter.cancel_action()` MUST be defined as a non-abstract method with safe no-op default.
   ```python
   def cancel_action(self) -> Dict[str, Any]:
       return LifecycleResult.ok("cancel_action").to_dict()
   ```
2. `MissionRunThread.request_cancel()` MUST attempt `adapter.cancel_action()` on the currently active adapter after calling `engine.request_cancel()`.
3. Failure of `cancel_action()` MUST NOT raise — log at warning level and continue cooperative cancel.
4. Run result diagnostics MUST distinguish `"mission_cancelled"` (cooperative) from future adapter-level cancel reason.

#### Contract C3-D — Unsaved-Change Guardrails (STAGE-05, P1)

1. `_on_run()` in `bin/ui.py` MUST check `settings_panel.has_unsaved_changes()` before the validation gate.
2. Dialog options MUST be: **Apply** (apply settings then continue run), **Discard** (ignore pending changes, continue with last applied), **Cancel** (abort run).
3. No duplicate dialog: once operator selects Apply or Discard, the flag is cleared and subsequent run calls within same state do not re-prompt.
4. Navigation/tab-switch guardrail is a best-effort UX improvement — NOT a hard contract requirement for Stage acceptance.

#### Contract C3-E — Live Capability Refresh (STAGE-06, P2)

1. `_refresh_capability_inspector()` MUST be callable on connection-state change events emitted by the robot model.
2. Graceful degradation: if adapter `.capabilities()` throws, last-known state is retained; error is logged at warning level.
3. Advanced MuJoCo settings section (P3) MUST be additive to `SettingsPanel`; new fields MUST go through `settings_schema.py` and `settings_validator.py`.
4. MuJoCo advanced fields MUST be included in mission `"settings"` payload when brand is `"unitree"` (simulation target).

---

### §C3-4  Stage Acceptance Checklist

| Stage | Objective | Acceptance Criteria |
|-------|-----------|---------------------|
| STAGE-01 | Contract freeze + gap audit | ✅ This section complete and committed |
| STAGE-02 | Mission settings persistence | ✅ Save → load roundtrip preserves SDK settings; old files load without error; schema v1.1 |
| STAGE-03 | Runtime policy scoping | ✅ `make_run_policy()` pure factory; thread-local activation; class-level policy never mutated |
| STAGE-04 | Preemptive cancel contract | ✅ `cancel_action()` in base + 3 adapters; `request_cancel()` calls adapter cancel; `ADAPTER_CANCEL_INVOKED` diagnostics key |
| STAGE-05 | Unsaved-change guardrails | ✅ `_handle_unsaved_settings_guard()` gates run/open/close; apply/discard/cancel dialog |
| STAGE-06 | Live capability sync + MuJoCo | ✅ 4 advanced fields (gravity_z, solver_iters, render_w/h); scenario_settings save/load; cap refresh after run/abort; graceful degradation |
| STAGE-07 | Cycle 3 test matrix | ✅ 105 new Cycle 3 tests; 2874 total suite green |
| STAGE-08 | Acceptance report + Cycle 4 handoff | ✅ `CYCLE3_ACCEPTANCE_REPORT.md` delivered 2026-02-26 |

---

### §C3-5  Implementation Guardrails (Cycle 3)

1. Settings and runtime actions MUST traverse `execute_with_lifecycle()` — no bypasses.
2. `validate_mission_schema()` backward-compat is unconditional — `"settings"` absence is never an error.
3. `cancel_action()` default is a no-op — callers MUST handle gracefully; no exception on unimplemented adapters.
4. Qt signal boundary is strict — all widget mutations from worker threads MUST use queued signals.
5. `_lifecycle_policy` class-level mutation is kept only for backward compatibility; new code paths MUST use explicit policy argument.
6. Diagnostics/telemetry schema is additive only — existing keys in `status/reason/stage/diagnostics/adapter_name` are never removed.
7. All new modules and high-risk touched modules MUST have focused unit tests before STAGE-07 closure.

---

*Cycle 3 started 2026-02-26. Cycle 3 closed 2026-02-26.*

---

## Product Cycle 4 — Consolidation and Product Hardening

*(STAGE-01 contract freeze completed 2026-02-26)*

---

### §C4-1  Gap Re-Baselining (Cycle 3 Handoff + TO_FIX Merge)

Cycle 4 scope is formed by merging:
1. Deferred items from `CYCLE3_ACCEPTANCE_REPORT.md` §5
2. Explicit frontend backlog in `TO_FIX.txt`

| ID | Gap | Priority | Source |
|----|-----|----------|--------|
| G1 | Behavior workspace has structural frame but core panes still placeholder-heavy | P0 | `TO_FIX.txt` |
| G2 | Settings `robot_type` is not dynamically populated from `/models` by brand | P0 | `TO_FIX.txt` |
| G3 | No explicit connect/disconnect UI + live connection indicator | P1 | `CYCLE3_ACCEPTANCE_REPORT.md` |
| G4 | MuJoCo settings only applied at simulation start (no live hot-patch path) | P1 | `CYCLE3_ACCEPTANCE_REPORT.md` |
| G5 | Old mission files without `"scenario_settings"` lack migration guidance | P1 | `CYCLE3_ACCEPTANCE_REPORT.md` |
| G6 | `cancel_action()` implementations use coarse `stop()` fallback, not SDK-native interrupt | P1 | `CYCLE3_ACCEPTANCE_REPORT.md` |
| G7 | Unsaved canvas graph changes are not guarded on open/close/navigation | P1 | `CYCLE3_ACCEPTANCE_REPORT.md` |

---

### §C4-2  Module Audit (Cycle 4 Baseline)

#### §C4-2.1  `bin/layout/behavior_panel.py`

| Item | Current State | Cycle 4 Action |
|------|--------------|----------------|
| Pane composition | Header/left/center/right scaffold is complete | Retain layout shell; replace placeholder-only regions with real authoring surfaces |
| Compile entry | Real `BehaviorCompilerBridge.compile()` path via worker thread | Keep non-blocking compile contract and diagnostics output |
| Context binding | Node breadcrumb and node info exist | Improve Mission->Behavior context continuity and operator traceability |

#### §C4-2.2  `bin/components/settings_panel.py`

| Item | Current State | Cycle 4 Action |
|------|--------------|----------------|
| Dynamic form build | Descriptor-driven rendering from settings schema | Keep contract, add dynamic option hydration for model-aware fields |
| Unsaved/apply/reset | Stable and tested | Preserve behavior; avoid new duplicate prompt paths |
| Brand switch handling | `set_brand()` rebuilds form | Ensure `robot_type` options refresh with brand changes |

#### §C4-2.3  `system/service/settings_schema.py`

| Item | Current State | Cycle 4 Action |
|------|--------------|----------------|
| `robot_type` field | Required text field without fixed `choices` | Keep schema backward-compatible while enabling UI-side dynamic options from BrandRegistry |
| Brand schema structure | Stable and consumed by validator/UI | Additive-only changes; no contract-breaking schema rewrite |

#### §C4-2.4  `models/brand_registry.py`

| Item | Current State | Cycle 4 Action |
|------|--------------|----------------|
| Discovery model | AST-based `SUPPORTED_MODELS` discovery from `models/*` | Use as single source for brand→model options in Settings UX |
| Public API | `get_brands()`, `get_models()`, `get_robot_brand_map()` available | Reuse directly; avoid duplicate model catalogs in UI code |

#### §C4-2.5  `bin/layout/main_zone_panel.py`

| Item | Current State | Cycle 4 Action |
|------|--------------|----------------|
| Workspace tabs | Mission/Behavior/Settings/Capabilities present | Keep additive changes only; no disruptive tab redesign |
| Scenario settings | Advanced MuJoCo subset already present | Preserve existing bounds/defaults; focus on workflow coherence |

#### §C4-2.6  `bin/ui.py`

| Item | Current State | Cycle 4 Action |
|------|--------------|----------------|
| Guardrails | Settings unsaved guard for run/open/close is present | Extend coherence across Mission/Behavior/Settings transitions where needed |
| Capability refresh | Triggered on startup/run/abort/settings events | Keep truthful status behavior; add explicit connection UI path in Cycle 4 scope |
| Persistence wiring | Mission settings/scenario settings save/load implemented | Preserve backward compatibility while improving operator clarity |

---

### §C4-3  Frozen Implementation Contract (Cycle 4)

#### Contract C4-A — Behavior Workspace Productization (STAGE-02, P0)

1. Behavior tab must not rely on placeholder-only panes for core operator tasks.
2. Compile flow remains asynchronous; UI thread must not block.
3. Compile output must continue surfacing artifact identity + diagnostics.
4. Mission node context (name/id/ref) must remain visible while editing behavior.

#### Contract C4-B — Dynamic `robot_type` Optioning (STAGE-03, P0)

1. `robot_type` options in Settings must come from `BrandRegistry` model discovery (from `/models`).
2. Option list must refresh when active brand changes.
3. Existing mission files with unknown/legacy `robot_type` values must still load (non-crashing fallback).
4. Settings validation + save/load/run handoff contracts from Cycle 2/3 remain non-regressive.

#### Contract C4-C — Operator Workflow Coherence (STAGE-04, P1)

1. Mission/Behavior/Settings transitions must avoid silent loss of critical edits.
2. Guard dialogs must be deterministic and non-duplicative.
3. Capability/diagnostics views must reflect current context after run/abort/navigation.

#### Contract C4-D — Performance and Stability (STAGE-05, P1)

1. Repeated run/abort/reset cycles must not leak thread state or stale UI indicators.
2. Qt thread boundary remains strict: no widget mutation from worker threads.
3. Compatibility gates remain explicit; no hidden fallback reintroduction.

#### Contract C4-E — Documentation and Release Readiness (STAGE-06/08)

1. Cycle 4 decisions, tests, known gaps, and handoff checklist must be reproducible from repo root.
2. Stage statuses must be updated in this README at cycle close.

---

### §C4-4  Stage Acceptance Checklist

| Stage | Objective | Acceptance Criteria |
|-------|-----------|---------------------|
| STAGE-01 | Contract freeze + gap audit | ✅ This section complete and committed |
| STAGE-02 | Behavior panel de-placeholder | ✅ Placeholder-critical regions replaced by real operator surfaces |
| STAGE-03 | Dynamic robot_type by brand | ✅ Settings model options sourced from BrandRegistry and refresh on brand switch |
| STAGE-04 | UX guardrails coherence | ✅ No silent state loss in key run/open/close/navigation workflows |
| STAGE-05 | Performance/session robustness | ✅ Repeated run/abort/reset remains stable and responsive |
| STAGE-06 | Docs/runbook updates | ✅ Reviewer-facing docs and operation notes updated (2026-02-26) |
| STAGE-07 | Cycle 4 test matrix | ✅ New Cycle 4 tests pass; critical historical regressions remain green (2026-02-26) |
| STAGE-08 | Acceptance + next handoff | ✅ `CYCLE4_ACCEPTANCE_REPORT.md` delivered; deferred list explicit (2026-02-26) |

---

### §C4-5  Implementation Guardrails (Cycle 4)

1. Do not bypass lifecycle routing (`execute_with_lifecycle`) for runtime actions.
2. Preserve diagnostics/telemetry schema stability; additive-only fields unless migration is documented and tested.
3. Keep backward compatibility for mission files and settings payloads.
4. Keep worker-thread UI isolation strict (Qt signal boundary only).
5. All high-risk touched modules must have focused tests before STAGE-07 closure.

---

### §C4-6  Operator Runbook and Troubleshooting (STAGE-06)

Primary reviewer/operator runbook: `CYCLE4_RUNBOOK.md`.

#### C4-6.1  Fast validation workflow (operator)

1. Open app and verify Mission/Behavior/Settings/Capabilities tabs are present.
2. In Settings, switch brand and confirm `robot_type` options refresh by brand.
3. In Behavior, confirm editor/workflow panes are functional (no placeholder-only blocking area).
4. Trigger run, then abort/reset, and ensure status/diagnostics/capability views stay coherent.

#### C4-6.2  Troubleshooting notes

1. If Behavior compile output looks stale, re-run compile and verify diagnostics timestamp/artifact id updates in the Behavior panel output area.
2. If `robot_type` options are empty after brand switch, verify `/models` discovery for that brand and re-open Settings tab.
3. If run/abort/reset appears inconsistent, use Runtime Reset once, then re-run; mission thread stale handles are cleaned on run/abort/reset paths.
4. If capability panel looks stale after execution failure/cancel, trigger Runtime Abort or Reset to force capability refresh.

#### C4-6.3  Known limitations (as of STAGE-06)

1. No explicit connect/disconnect control surface yet; lifecycle connection still occurs on demand.
2. MuJoCo advanced settings are not hot-applied to an already running simulation loop.
3. Adapter `cancel_action()` is still stop-style fallback for some brands, not SDK-native interrupt.
4. Unsaved-canvas graph dirty guardrails are still pending; current guardrails focus on settings/behavior edits.

---

### §C4-7  Release Checklist (STAGE-06)

Pre-demo / pre-review checklist:

1. Run Cycle 4 focused tests (behavior/settings/guardrails/session robustness) from repo root.
2. Run historical critical regression matrix (Phase 3-7 + Cycle 1/2/3 core suites).
3. Verify mission save/load backward compatibility with files missing `settings` or `scenario_settings`.
4. Verify no lifecycle-routing bypass was introduced (`execute_with_lifecycle` remains authoritative path).
5. Confirm acceptance artifacts are up-to-date: runbook, README stage status, and final Cycle 4 acceptance report at cycle close.

---

*Cycle 4 started 2026-02-26. Cycle 4 closed 2026-02-26.*
