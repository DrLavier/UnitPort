# Migration Notes

## Migration Progress

### Day-1: Scaffolding
- Created layered directory skeletons and import-safe interfaces.
- Kept legacy `bin/`, `compiler/`, `models/`, `nodes/` operational.

### Day-2/P0: Service Routing
- Implemented `ServiceRegistry` and `ServiceRouter`.
- Added `UnitreeAdapter` and routed `RobotContext` APIs through service layer.
- Preserved legacy `RobotContext` signatures for compatibility.

### Day-3/P1: Runtime and Frontend Bridge
- Moved node execution and simulation-thread responsibilities into `design/runtime`.
- Turned `bin/core/node_executor.py` and `bin/core/simulation_thread.py` into wrappers.
- Split frontend entry points to `frontend/canvas`, `frontend/compiler`, `frontend/scenario`.
- Updated `bin/ui.py` imports to use `frontend/*` modules.

### Day-4/P2: Runtime Control/Safety Baseline
- Added runtime interception guards (`compile_guard.py`, `execute_guard.py`).
- Added safety subsystem baseline (`safety_policy.py`, `safety_checker.py`,
  `emergency_handler.py`, `audit_logger.py`).
- Moved UI run-flow logic into `design/runtime/workflow_runner.py`.
- Unified UI run entry to `RuntimeEngine.execute(...)`.

## Remaining Work (Deferred)

- Actual migration of business logic from `bin/`, `compiler/`, `models/`, `nodes/`
- Full Scenario UI integration and persistent safety policy editing
- Deeper adapterization of `models/unitree/unitree_model.py` internals
- Removal of legacy bridges after full dependency migration
- Expanded integration/e2e tests for runtime+safety paths

## Rollback strategy

All new code lives in `frontend/`, `design/`, `shared/`, `legacy/` directories.
To roll back, delete these four top-level directories. No existing files were modified.
