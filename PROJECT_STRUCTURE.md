# Project Structure (Current + Migration Notes)

Last updated: 2026-03-09

This document reflects the current repository layout in `D:\Unitport\EXE\_WIP_`.
It also keeps migration notes for the planned architecture evolution.

## Current Directory Layout

```text
UnitPort/
|-- main.py
|-- README.md
|-- PROJECT_STRUCTURE.md
|-- requirements.txt
|-- instructions.txt
|
|-- compiler/                           # DSL parsing / semantic / lowering / codegen
|
|-- bin/                                # Current UI and app wiring
|   |-- ui.py
|   |-- core/                           # Config, localisation, runtime bridge helpers
|   |-- components/                     # Graph/code/settings UI components
|   |-- compiler/                       # Compiler-side UI support
|   |-- scenario/                       # Scenario panel + runtime console + safety editor
|   `-- layout/
|
|-- system/                             # Backend layered architecture
|   |-- mission/
|   |-- behavior/
|   |-- service/
|   |   |-- adapters/
|   |   |   |-- unitree_sdk2/
|   |   |   |-- spot_sdk/
|   |   |   `-- cyberdog_sdk/
|   |   `-- protocol/
|   |-- runtime/
|   |   |-- interception/
|   |   `-- safety/
|   |-- ir/
|   |   `-- workbench/
|   |-- telemetry/
|   `-- types/
|
|-- nodes/
|   `-- sys_nodes/                      # Built-in node definitions
|
|-- custom_nodes/                       # User/community node packs
|
|-- models/
|   |-- base.py
|   |-- brand_registry.py
|   |-- Unitree/
|   |-- BostionDynamics/
|   `-- XiaoMi/
|
|-- tests/
|   |-- unit/
|   |-- integration/
|   |-- regression/
|   `-- e2e/
|
|-- config/
|-- localisation/
`-- utils/
```

## Layer Responsibilities (Current)

- `compiler/`: Compile pipeline for DSL -> validated artifacts/IR.
- `bin/`: Existing frontend and desktop wiring (current runtime UI entry points).
- `system/`: Mission/Behavior/Service/Runtime/IR/Telemetry/Types layered backend.
- `nodes/` + `custom_nodes/`: Built-in and user-extensible node catalogs.
- `models/`: Vendor assets, SDK mirrors, and model-facing integration code.
- `tests/`: Unit, integration, regression, and e2e test suites.

## Notes About Naming And Paths

- `custom_nodes/` is a top-level directory (not `nodes/custom_nodes/`).
- Vendor model folders are currently cased as:
  - `models/Unitree/`
  - `models/BostionDynamics/`
  - `models/XiaoMi/`
- There is currently no top-level `frontend/` directory; frontend logic is still under `bin/`.

## Migration Target (Planned)

The project intent remains to evolve from `bin/` into clearer frontend domains:

- `frontend/canvas/`
- `frontend/compiler/`
- `frontend/scenario/`

Related planned moves:

1. Split `bin/components/` into dedicated frontend domains (`canvas` and `compiler`).
2. Keep moving execution responsibilities from legacy UI core helpers into `system/runtime/`.
3. Continue consolidating SDK routing in `system/service/service_router.py`.
4. Keep vendor-specific SDK invocation inside `system/service/adapters/*`.
5. Centralize scenario and safety orchestration under `bin/scenario` -> `frontend/scenario` and `system/runtime/safety`.

## Test Commands

```bash
# All tests
python -m unittest discover -s tests -p "test_*.py"

# Unit tests
python -m unittest discover -s tests/unit -p "test_*.py"

# Integration tests
python -m unittest discover -s tests/integration -p "test_*.py"
```

