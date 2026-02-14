# Project Structure (New)

## Directory Layout

```text
UnitPort/
|-- main.py                                 # Entry point
|-- requirements.txt                        # Python dependencies
|-- PROJECT_README.md                       # Product architecture definition
|-- PROJECT_STRUCTURE.md                    # Legacy structure (reference)
|-- PROJECT_STRUCTURE_NEW.md                # New target structure (this file)
|-- instruction.txt                         # Claude execution prompt
|
|-- config/                                 # Global configuration
|   |-- system.ini                          # System-level config
|   |-- user.ini                            # User preferences
|   `-- ui.ini                              # UI style config
|
|-- localisation/                           # i18n resources
|   |-- en.json
|   `-- README.md
|
|-- frontend/                               # Three interaction spaces
|   |-- canvas/                             # Interaction Layer 1: Canvas
|   |   |-- graph_scene.py                  # Mission graph editor scene
|   |   |-- graph_view.py                   # Mission graph editor view
|   |   |-- node_palette.py                 # Node palette and graph tools
|   |   `-- README.md
|   |
|   |-- compiler/                           # Interaction Layer 2: Compiler
|   |   |-- code_editor.py                  # Source editing (Behavior authoring)
|   |   |-- dsl_support.py                  # DSL/AST helper entry
|   |   |-- lint_bridge.py                  # Validation bridge to runtime checks
|   |   `-- README.md
|   |
|   `-- scenario/                           # Interaction Layer 3: Scenario
|       |-- scenario_panel.py               # Scenario setup UI
|       |-- runtime_console.py              # Runtime control and overseen panel
|       |-- safety_policy_editor.py         # Safety protocol configuration
|       `-- README.md
|
|-- design/                                 # Four design layers (core architecture)
|   |-- mission/                            # Design Layer 1: Mission
|   |   |-- mission_model.py                # Task graph model
|   |   |-- mission_planner.py              # Mission composition and routing
|   |   |-- mission_serializer.py           # Mission IR serialization
|   |   `-- README.md
|   |
|   |-- behavior/                           # Design Layer 2: Behavior
|   |   |-- behavior_model.py               # Behavior schema
|   |   |-- behavior_state_machine.py       # Node-internal strategy/state machine
|   |   |-- behavior_compiler_bridge.py     # Canvas/Compiler unified behavior input
|   |   `-- README.md
|   |
|   |-- service/                            # Design Layer 3: Service
|   |   |-- service_registry.py             # Capability and adapter registry
|   |   |-- service_router.py               # Unified service routing
|   |   |-- protocol/                       # Unified service protocol definitions
|   |   |   |-- commands.py
|   |   |   |-- events.py
|   |   |   `-- errors.py
|   |   |
|   |   `-- adapters/                       # Vendor SDK adapters (extensible)
|   |       |-- base_adapter.py             # Adapter ABI
|   |       |-- unitree_sdk2/               # Current implementation
|   |       |   |-- adapter.py
|   |       |   |-- mapper.py
|   |       |   `-- README.md
|   |       `-- README.md
|   |
|   `-- runtime/                            # Design Layer 4: Runtime (with Safety)
|       |-- runtime_engine.py               # Execution orchestrator
|       |-- scheduler.py                    # Task scheduling and arbitration
|       |-- monitor.py                      # Runtime monitoring
|       |-- interception/                   # Compile-time and run-time interception
|       |   |-- compile_guard.py            # Compile checks
|       |   `-- execute_guard.py            # Execute-time checks
|       |-- safety/                         # Runtime embedded Safety subsystem
|       |   |-- safety_policy.py            # Policy model
|       |   |-- safety_checker.py           # Guardrail checks
|       |   |-- emergency_handler.py        # Stop/degrade/rollback
|       |   `-- audit_logger.py             # Safety event audit
|       `-- README.md
|
|-- shared/                                 # Cross-layer shared assets
|   |-- ir/                                 # Unified IR for Mission/Behavior
|   |   |-- workflow_ir.py
|   |   `-- README.md
|   |-- telemetry/                          # State/event bus contracts
|   |   |-- event_bus.py
|   |   `-- README.md
|   `-- types/
|       |-- common_types.py
|       `-- error_codes.py
|
|-- models/                                 # Robot assets and resources
|   |-- base.py                             # Base robot model contract (legacy bridge)
|   |-- __init__.py
|   |-- README.md
|   `-- unitree/
|       |-- unitree_model.py                # Existing model logic (to be adapter-ized)
|       |-- SDK_README.md                   # SDK execution knowledge base
|       |-- unitree_mujoco/
|       `-- unitree_sdk2_python/
|
|-- nodes/                                  # Node definitions and registration
|   |-- __init__.py
|   |-- sys_nodes/
|   |   |-- __init__.py
|   |   |-- base_node.py
|   |   |-- action_nodes.py
|   |   |-- logic_nodes.py
|   |   `-- sensor_nodes.py
|   `-- README.md
|
|-- custom_nodes/                           # User/community node packs
|   |-- __init__.py
|   `-- README.md
|
|-- legacy/                                 # Transitional compatibility space
|   |-- bin_core_bridge/                    # Bridge for old bin/core interfaces
|   |-- bin_components_bridge/              # Bridge for old bin/components interfaces
|   `-- migration_notes.md
|
|-- tests/                                  # Test suites
|   |-- unit/
|   |-- integration/
|   `-- e2e/
|
`-- utils/
    `-- logger.py
```

## Module Responsibilities

### Interaction Layer (`frontend/`)

- `frontend/canvas/`: Mission visual orchestration and node-graph editing.
- `frontend/compiler/`: Behavior source authoring and static validation bridge.
- `frontend/scenario/`: Scenario setup, runtime control, and safety policy UI.

### Design Layer (`design/`)

- `design/mission/`: Project-level workflow orchestration, mission graph definition.
- `design/behavior/`: Fine-grained strategy inside nodes (state machine + parameterized logic).
- `design/service/`: SDK abstraction and vendor adapter routing. This is the expansion core for new SDK onboarding.
- `design/runtime/`: Task scheduling, execution control, monitoring, interception; includes Safety subsystem.

### Shared Layer (`shared/`)

- `shared/ir/`: Unified IR as single semantic source for Canvas + Compiler.
- `shared/telemetry/`: Runtime state/event contracts for monitoring and feedback.
- `shared/types/`: Common data types and error code catalog.

### Compatibility Layer (`legacy/`)

- Transitional bridges to keep current code operational during restructuring.
- Bridges should be removed incrementally after complete migration.

## Design Principles

1. Layer isolation
- Interaction layer does not directly call vendor SDK.
- Design layers communicate through defined interfaces/IR.

2. Single semantic source
- Mission/Behavior authoring from Canvas and Compiler must converge to shared IR.

3. Service extensibility first
- New SDK integration should require only `design/service/adapters/{vendor}` additions and registry update.

4. Runtime embedded safety
- Safety is not passive exception handling; it is proactive interception in compile and execute phases within Runtime.

5. Progressive migration
- Preserve existing behavior through bridge modules and phased replacement.

## Module Dependencies

```text
main.py
  |-- config/ + localisation/
  |
  |-- frontend/
  |    |-- canvas  ---> design/mission + design/behavior + shared/ir
  |    |-- compiler ---> design/behavior + shared/ir
  |    `-- scenario ---> design/runtime
  |
  |-- design/
  |    |-- mission  ---> shared/ir
  |    |-- behavior ---> shared/ir
  |    |-- runtime  ---> design/mission + design/behavior + design/service + shared/telemetry
  |    `-- service  ---> adapters/* ---> models/* + vendor SDK
  |
  |-- shared/ ---> used by frontend + design + tests
  |
  |-- nodes/ + custom_nodes/ ---> design/behavior + design/runtime
  |
  `-- legacy/ ---> compatibility wrappers during migration only
```

## Migration Notes (Must Replan in Current Project)

1. `bin/components/` should be split into `frontend/canvas` and `frontend/compiler`.
2. `bin/core/node_executor.py` and `simulation_thread.py` responsibilities should be absorbed into `design/runtime`.
3. `robot_context.py` routing logic should be incrementally migrated toward `design/service/service_router.py`.
4. `models/unitree/unitree_model.py` should be split into runtime-facing adapter logic and robot asset/simulation resources.
5. Scenario-related settings currently dispersed in UI/config should be centralized into `frontend/scenario` + `design/runtime/safety`.
6. New SDK onboarding must use `design/service/adapters/` only; avoid spreading SDK calls into nodes/ui/runtime.

## Detailed Documentation (Target)

- `frontend/canvas/README.md` - Canvas interaction design
- `frontend/compiler/README.md` - Compiler interaction design
- `frontend/scenario/README.md` - Scenario interaction design
- `design/mission/README.md` - Mission layer
- `design/behavior/README.md` - Behavior layer
- `design/service/README.md` - Service layer and adapter guide
- `design/runtime/README.md` - Runtime and Safety
- `shared/ir/README.md` - Unified IR specification
- `legacy/migration_notes.md` - Migration steps and deprecation plan
