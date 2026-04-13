# UnitPort — Agent Onboarding Knowledge Base

> **Purpose**: 供 AI Agent 快速建立本项目全局视图，理解架构、模块职责、文件位置和改造约束。
> **Last updated**: 2026-04-04

---

## 1. Project Identity

| Key | Value |
|-----|-------|
| Name | **UnitPort** |
| Type | Cross-platform visual robot programming framework |
| Tech Stack | Python 3.11, PySide6 (Qt6), MuJoCo 3.0+, Stable-Baselines3, ONNX, PyTorch |
| Entry Point | `main.py` → `QApplication` → `MainWindow` (`bin/pages/layout/ui.py`) |
| Venv | `.venv311/` (project-local, Python 3.11) |
| Platform | Windows / Linux / macOS |
| Language | Bilingual codebase (Chinese + English); UI strings via `tr()` i18n |
| License | `LICENSE.txt` (project root) |

**One-line summary**: UnitPort 将可视化任务编排 (Canvas)、行为编程 (Compiler)、场景配置 (Scenario) 和 RL 训练 (Training Ground) 统一在一个桌面应用中，支持从仿真到实机的多机器人控制。

---

## 2. Architecture Overview

### 2.1 Data Flow

```
Canvas ──→ Mission ──┐
                      ├──→ Engine (+Safety) ──→ Service ──→ SDK / Robot
Compiler ─→ Behavior ─┘

Scenario ──→ Engine params, execution strategy, safety protocol, sim/real target
```

### 2.2 Backend Layers (`src/system/`)

| Layer | Path | Role |
|-------|------|------|
| **Compiler** | `compiler/` | IR pipeline: `lowering/canvas_to_ir.py`, `codegen/ir_to_code.py`, `lowering/ir_to_canvas.py`, `ir/workflow_ir.py` |
| **IR** | `ir/` | Shared IR dataclasses: `workflow_ir.py`, `layered_contracts.py`, `workbench/` |
| **Mission** | `mission/` | Project-level DAG: `mission_model.py`, `mission_planner.py`, `mission_serializer.py` |
| **Behavior** | `behavior/` | Node-internal logic, state machines, semantic actions, motor protocol |
| **Engine** | `engine/` | **Execution core**: `runtime_engine.py`, `workflow_runner.py`, `node_executor.py`, `contracts.py`, `reactive_loop.py`, `safety/`, `interception/` |
| **Runtime** | `runtime/` | Real-time input & dispatch: `action_dispatcher.py`, `command_bus.py`, `user_input_manager.py`, `input_sources/` |
| **Service** | `service/` | SDK adaptation: `service_router.py`, `adapters/{unitree_sdk2, spot_sdk, cyberdog_sdk}/` |
| **Training** | `training/` | RL: `training_spec.py`, `sb3_trainer.py`, `bundle_exporter.py` (PPO/SAC/TD3, Gymnasium + MuJoCo) |
| **Policy** | `policy/` | Inference: `policy_runner.py`, `inference_engine.py`, `manifest_schema.py` |
| **Skill** | `skill/` | Skill orchestration: `skill_manifest.py`, `transition_validator.py`, `transition_library.py`, `manifest_loader.py`, `manifest_inferrer.py`, `isaac_lab_manifest_parser.py` |
| **Sim** | `sim/` | Simulation core: `pd_controller.py`, `sensor_manager.py` |
| **Nodes** | `nodes/` | Node registry + `sys_nodes/`. API: `register_node()`, `get_node_class()`, `list_node_types()` |
| **Models** | `models/` | Robot model interface: `base.py`, `sdk_manager.py`, `model_registry.py`, `mujoco_asset_registry.py` |
| **Brand Packages** | `brand_packages/` | Brand-specific code: `unitree/`, `boston_dynamics/` |
| **Binding** | `binding/` | SDK binding layer: `base.py`, `bindings/`, `executors/` |
| **Core** | `core/` | Shared infra: `config_manager.py`, `robot_context.py`, `logger.py`, `theme_manager.py`, `localisation.py`, `mission_run_thread.py`, `project_store.py`, `asset_resolver.py`, `utils/` |
| **Telemetry** | `telemetry/` | `event_bus.py` |
| **Types** | `types/` | `common_types.py`, `error_codes.py` |

### 2.3 Frontend Layers (`bin/`)

| Path | Role |
|------|------|
| `pages/layout/ui.py` | MainWindow, top-level shell (3,399 lines) |
| `pages/layout/sidebar_dock.py` | Left sidebar rail + expandable content panel |
| `pages/layout/misc.py` | MainRow (top bar), PageSwitcher, SwitchButton |
| `pages/layout/main_zone_panel.py` | RuntimeZone + MissionZone + ExecutionSummaryBar + DiagnosticsPanel |
| `pages/layout/behavior_panel.py` | Behavior tab panel (timeline, compat audit) |
| `pages/layout/module_cards.py` | Module card widgets |
| `pages/canvas/graph_scene.py` | Visual node-based task builder (main canvas, 9,067 lines) |
| `pages/canvas/code_editor.py` | Python behavior scripting + IR roundtrip |
| `pages/canvas/graph_view.py` | Canvas viewport |
| `pages/canvas/script_editor.py` | Script editor widget |
| `pages/canvas/script_api_analyzer.py` | Script API analysis |
| `pages/canvas/diagnostics_panel.py` | DiagnosticsPanel — error display |
| `pages/canvas/reactive_viewer_overlay.py` | Reactive viewer overlay |
| `pages/scenario/` | Scenario panel, runtime console, safety policy editor |
| `pages/training/training_workspace_window.py` | Training workspace management (5,859 lines) |
| `pages/training/checkpoint_import_dialog.py` | Checkpoint import dialog (local + HuggingFace) |
| `pages/training/training_panel.py` | Training panel widget |
| `pages/homepage/homepage.py` | Home/landing page (1,144 lines) |
| `pages/settings/` | Settings panel, capability inspector, motor weight navigator |
| `pages/setup/setup_wizard.py` | First-run setup wizard |
| `components/overview_panel.py` | Overview panel (mission + training overview content) |
| `components/robot_panel.py` | Robot panel widget |
| `nodes/node_ui_rows.py` | Node parameter UI rows |
| `nodes/training_node_items.py` | Training node UI items (6,133 lines) |

### 2.4 Training Pipeline

```
Canvas training graph
  → TrainingSpecCompiler (src/system/training/training_spec.py)
  → SB3Trainer (PPO/SAC/TD3, Gymnasium + MuJoCo)
  → export_bundle (ONNX + TorchScript + manifest.yaml + source.json)
  → CheckpointRegistry (projects/<slug>/training/exported/<policy_id>/)
  → PolicyRunner.load() + run_episode()
```

Additional training backends:
- `isaac_lab_backend.py` + `isaac_lab_config.py` — Isaac Lab integration
- `behavioral_cloning.py` — BC training support

### 2.5 Compiler Pipeline (IR-based bidirectional)

```
Canvas Graph ←→ IR ←→ Python Code
```

- Canvas → IR: `src/system/compiler/lowering/canvas_to_ir.py`
- IR → Code: `src/system/compiler/codegen/ir_to_code.py`
- IR → Canvas: `src/system/compiler/lowering/ir_to_canvas.py`
- IR data model: `src/system/compiler/ir/workflow_ir.py` (IRNode, IREdge, NodeKind)

Canvas 和 Code 共享同一 IR 作为语义源——两者都不是权威表示。

### 2.6 Skill Orchestration (`src/system/skill/`)

```
SkillManifest → TransitionValidator → TransitionLibrary → Engine dispatch
```

- `skill_manifest.py` — skill manifest dataclass
- `transition_validator.py` — validates state transitions between skills
- `transition_library.py` — transition rule definitions
- `manifest_loader.py` / `manifest_inferrer.py` — load/infer manifests
- `isaac_lab_manifest_parser.py` — parse Isaac Lab manifests into skill format

---

## 3. Complete Directory Map

### 3.1 Top-Level Structure

```
UnitPort/
├── main.py                         # Application entry point (427 lines)
├── requirements.txt                # PySide6, mujoco, numpy, torch, onnxruntime, huggingface_hub, ...
├── install.bat / install.sh        # First-time setup scripts
├── reset.bat / reset.sh            # Environment reset scripts
├── start.sh / s.bat                # Launch scripts
├── instructions.yaml               # Project construction instructions / progress tracking
├── CLAUDE.md                       # AI agent instructions (authoritative)
├── AGENTS.md                       # Agent operation constraints
├── pytest.ini                      # Pytest configuration
│
├── bin/                            # Frontend UI (PySide6)
│   ├── pages/                      #   Page-based UI structure
│   │   ├── layout/                 #     Shell: MainWindow, sidebar, main zone, behavior panel
│   │   ├── canvas/                 #     Canvas: graph scene, code editor, script editor
│   │   ├── scenario/               #     Scenario: panel, console, safety editor
│   │   ├── training/               #     Training: workspace, checkpoint import, panel
│   │   ├── homepage/               #     Home/landing page
│   │   ├── settings/               #     Settings, capability inspector
│   │   └── setup/                  #     First-run setup wizard
│   ├── components/                 #   Shared components: overview panel, robot panel
│   └── nodes/                      #   Node UI rendering: node_ui_rows, training_node_items
│
├── src/                            # Backend source
│   ├── system/                     #   Layered backend architecture
│   │   ├── behavior/               #     Behavior logic + semantic actions
│   │   ├── binding/                #     SDK binding layer
│   │   ├── brand_packages/         #     Brand-specific code (unitree, boston_dynamics)
│   │   ├── compiler/               #     IR-based compiler pipeline
│   │   ├── core/                   #     Shared infra (config, logging, threads, project store)
│   │   ├── engine/                 #     Execution engine (runtime_engine, node_executor, safety)
│   │   ├── ir/                     #     Shared IR dataclasses
│   │   ├── mission/                #     Project-level DAG
│   │   ├── models/                 #     Robot model interface + MuJoCo assets
│   │   ├── nodes/                  #     Node registry + sys_nodes definitions
│   │   ├── policy/                 #     Policy inference stack
│   │   ├── runtime/                #     Real-time input/dispatch (action_dispatcher, command_bus, input_sources)
│   │   ├── service/                #     Vendor SDK adaptation layer
│   │   ├── sim/                    #     Simulation core (PD controller, sensor manager)
│   │   ├── skill/                  #     Skill orchestration (manifest, transitions, validation)
│   │   ├── telemetry/              #     Event bus
│   │   ├── training/               #     RL training pipeline
│   │   └── types/                  #     Shared types + error codes
│   ├── config/                     #   INI config files
│   └── localisation/               #   i18n JSON (en.json, zh.json)
│
├── projects/                       # ProjectStore — per-project workspaces
├── runtime/                        # Vendored SDKs and simulation assets — **read-only**
│   └── sdk/                        #   BostonDynamics, Unitree, XiaoMi SDK mirrors
├── custom_mods/                    # Community hot-plug resources
│   └── motions/                    #   Reference motion data (npy/npz)
├── knowledge_base/                 # Agent knowledge base documents
├── localisation/                   # i18n JSON (en.json, zh.json) — top-level mirror
├── hotfix/                         # Emergency patches
├── logs/                           # Application logs
└── bin/assets/                     # SVG icons, animations, sounds
```

### 3.2 `bin/` — Frontend UI

```
bin/
├── pages/
│   ├── layout/
│   │   ├── ui.py                       # [3399 lines] MainWindow — top settings row, cross-zone wiring,
│   │   │                               #   _on_run/_on_runtime_abort, HF import, compat audit
│   │   ├── sidebar_dock.py             # [459 lines] Fixed left rail navigation
│   │   ├── misc.py                     # [1014 lines] MainRow, PageSwitcher, SwitchButton
│   │   ├── main_zone_panel.py          # [1260 lines] RuntimeZone + MissionZone +
│   │   │                               #   ExecutionSummaryBar + DiagnosticsPanel
│   │   ├── behavior_panel.py           # [5744 lines] Behavior tab panel (timeline, compat audit)
│   │   └── module_cards.py             # [350 lines] Module card widgets
│   ├── canvas/
│   │   ├── graph_scene.py              # [9067 lines] GraphScene — ALL canvas ops (create, serialize, load,
│   │   │                               #   duplicate, group, ungroup, status badges, execution graph)
│   │   ├── graph_view.py               # [487 lines] GraphView — QGraphicsView wrapper
│   │   ├── code_editor.py              # [152 lines] Code editor widget
│   │   ├── script_editor.py            # [1202 lines] Script editor widget
│   │   ├── script_api_analyzer.py      # [748 lines] Script API analysis
│   │   ├── script_builtin_registry.py  # Script built-in function registry
│   │   ├── diagnostics_panel.py        # [386 lines] DiagnosticsPanel — error display
│   │   └── reactive_viewer_overlay.py  # Reactive viewer overlay
│   ├── scenario/
│   │   ├── scenario_panel.py           # Scenario configuration panel
│   │   ├── runtime_console.py          # Runtime console output
│   │   └── safety_policy_editor.py     # Safety policy editor
│   ├── training/
│   │   ├── training_workspace_window.py # [5859 lines] Training workspace management window
│   │   ├── checkpoint_import_dialog.py # [743 lines] Checkpoint import dialog (local + HuggingFace)
│   │   └── training_panel.py           # [230 lines] Training panel widget
│   ├── homepage/
│   │   └── homepage.py                 # [1144 lines] Homepage/welcome screen
│   ├── settings/
│   │   ├── settings_panel.py           # [518 lines] Settings UI panel
│   │   ├── capability_inspector.py     # Robot capability inspector widget
│   │   └── motor_weight_navigator.py   # MotorWeightNavigator — motor topology tree
│   └── setup/
│       └── setup_wizard.py             # [858 lines] First-run setup wizard
├── components/
│   ├── overview_panel.py               # [935 lines] Overview panel
│   ├── mission_overview_content.py     # Mission overview content
│   ├── training_overview_content.py    # Training overview content
│   └── robot_panel.py                  # [611 lines] Robot panel widget
└── nodes/
    ├── node_ui_rows.py                 # [1241 lines] Node parameter UI rows
    └── training_node_items.py          # [6133 lines] Training node UI items
```

### 3.3 `src/system/` — Backend Architecture

```
src/system/
├── engine/                         # Execution engine
│   ├── runtime_engine.py           # [543 lines] RuntimeEngine.execute() — dispatches Path A/B
│   ├── node_executor.py            # [1358 lines] Flow-aware DFS traversal
│   ├── workflow_runner.py          # [396 lines] WorkflowRunner for dict-based execution
│   ├── contracts.py                # [316 lines] RuntimeResult dataclass, DiagnosticsKey constants
│   ├── behavior_invoker.py         # [540 lines] Behavior subgraph invocation
│   ├── reactive_loop.py            # [247 lines] Reactive control loop
│   ├── simulation_runner.py        # Simulation execution runner
│   ├── scheduler.py                # Execution scheduling
│   ├── monitor.py                  # Runtime monitoring
│   ├── result_inspector.py         # Result inspection utilities
│   ├── migration.py                # Runtime data migration
│   ├── interception/
│   │   ├── compile_guard.py        # Pre-compile safety guard
│   │   └── execute_guard.py        # Pre-execute safety guard
│   └── safety/
│       ├── safety_checker.py       # Safety constraint checking
│       ├── safety_policy.py        # Safety policy definition
│       ├── emergency_handler.py    # Emergency stop handling
│       └── audit_logger.py         # Safety audit logging
│
├── runtime/                        # Real-time input & action dispatch
│   ├── action_dispatcher.py        # Action dispatch to robot/sim
│   ├── command_bus.py              # Command bus for runtime commands
│   ├── user_input_manager.py       # User input management
│   └── input_sources/
│       ├── gamepad_source.py       # Gamepad input source
│       └── keyboard_source.py      # Keyboard input source
│
├── skill/                          # Skill orchestration (NEW)
│   ├── skill_manifest.py           # SkillManifest dataclass
│   ├── transition_validator.py     # State transition validation
│   ├── transition_library.py       # Transition rule definitions
│   ├── transition_result.py        # Transition result dataclass
│   ├── manifest_loader.py          # Load skill manifests from disk
│   ├── manifest_inferrer.py        # Infer manifests from checkpoints
│   └── isaac_lab_manifest_parser.py # Parse Isaac Lab manifests
│
├── sim/                            # Simulation core (NEW)
│   ├── pd_controller.py            # PD controller for joint control
│   └── sensor_manager.py           # Sensor data management
│
├── training/                       # RL training pipeline
│   ├── sb3_trainer.py              # [1963 lines] SB3 training runner (PPO/SAC/TD3)
│   ├── unitree_gym_env.py          # [2061 lines] Gymnasium MuJoCo environment
│   ├── training_spec.py            # [1018 lines] TrainingSpecCompiler
│   ├── training_config.py          # [1206 lines] Training configuration dataclasses
│   ├── training_process.py         # Training process management
│   ├── training_run_cache.py       # Training run caching
│   ├── bundle_exporter.py          # [829 lines] Export ONNX + TorchScript + manifest
│   ├── hf_downloader.py            # HuggingFace checkpoint download
│   ├── hf_training_asset_downloader.py  # HF training asset download
│   ├── motion_library.py           # Motion reference library
│   ├── training_asset_registry.py  # Training asset management
│   ├── training_workspace_store.py # Workspace persistence
│   ├── hardware_tuner.py           # Hardware performance tuning
│   ├── obs_contracts.py            # Observation space contracts
│   ├── robot_family.py             # Robot family classification
│   ├── task_module_registry.py     # Task module registration
│   ├── task_template_resolver.py   # Task template resolution
│   ├── task_templates.py           # Task template definitions
│   ├── training_compatibility.py   # Training compatibility checks
│   ├── vis_check_runner.py         # [992 lines] Visual check runner
│   ├── behavioral_cloning.py       # Behavioral cloning training
│   ├── isaac_lab_backend.py        # Isaac Lab training backend
│   ├── isaac_lab_config.py         # Isaac Lab configuration
│   └── loco_mujoco_bridge.py       # Locomotion MuJoCo bridge
│
├── behavior/                       # Behavior logic + semantic actions
│   ├── semantic_action.py          # SemanticActionDescriptor (frozen dataclass)
│   ├── intent_catalog.py           # Canonical intent constants
│   ├── semantic_resolution.py      # Semantic action resolution
│   ├── action_profile.py           # [1322 lines] Action profile comparison
│   ├── action_profile_report.py    # Action profile report generation
│   ├── action_registry.py          # Action registration
│   ├── ir_action_registry.py       # IR-level action registry
│   ├── behavior_compiler_bridge.py # [792 lines] BehaviorCompilerBridge
│   ├── behavior_model.py           # Behavior data model
│   ├── behavior_state_machine.py   # Behavior state machine
│   ├── behavior_artifact.py        # [641 lines] Behavior artifact management
│   ├── hb_channel.py               # [741 lines] HBRuntimeChannel + HBChannelFactory
│   ├── hb_compat_audit.py          # Cross-brand compatibility audit
│   ├── hb_display_state.py         # HB display state management
│   ├── hb_node_catalog.py          # HB node catalog
│   ├── heartbeat_policy.py         # Heartbeat policy management
│   ├── motor_weight_protocol.py    # [585 lines] Motor weight protocol validation
│   ├── motor_param_source.py       # Motor parameter source management
│   ├── motor_registry.py           # [1682 lines] Motor registry
│   ├── motor_topology.py           # Motor topology definition
│   ├── motor_weight_nav_model.py   # [576 lines] Motor weight navigator model
│   ├── protocol_apply_engine.py    # Protocol apply engine
│   ├── editor_ir.py                # Editor IR representation
│   ├── timeline_migration.py       # [629 lines] Timeline data migration
│   ├── timeline_view_model.py      # [642 lines] Timeline view model
│   └── topology_fallback.py        # Topology fallback handling
│
├── policy/                         # Policy inference stack
│   ├── policy_runner.py            # PolicyRunner.load() + run_episode() → EpisodeResult
│   ├── bundle_loader.py            # Loads CheckpointBundle from disk
│   ├── inference_engine.py         # ONNX/TorchScript inference
│   ├── obs_builder.py              # Observation builder
│   ├── obs_assembler.py            # Observation assembler
│   ├── action_applier.py           # Action application to simulation
│   ├── normalizer.py               # Observation normalization
│   ├── compatibility_checker.py    # Policy compatibility checks
│   ├── manifest_schema.py          # Manifest YAML schema
│   ├── joint_name_utils.py         # Joint name mapping utilities
│   ├── joint_reorder.py            # Joint reorder utilities
│   ├── sim_env_context.py          # SimEnvContext for MuJoCo episodes
│   └── policy_command_executor.py  # Policy command execution
│
├── service/                        # Vendor SDK adaptation layer
│   ├── service_router.py           # Service routing
│   ├── service_registry.py         # Service registration
│   ├── checkpoint_registry.py      # CheckpointRegistry — single source of truth for deployed bundles
│   ├── settings_schema.py          # Settings schema definitions
│   ├── settings_validator.py       # Settings validation
│   ├── capability_schema.py        # Capability schema
│   ├── lifecycle.py                # Service lifecycle management
│   ├── reason_codes.py             # Error/status reason codes
│   ├── semantic_action_builder.py  # Semantic action builder
│   ├── telemetry.py                # Service telemetry
│   ├── adapters/
│   │   ├── base_adapter.py         # BaseAdapter abstract interface
│   │   ├── unitree_sdk2/           # Unitree SDK2 adapter + mapper + semantic_actions
│   │   ├── spot_sdk/               # Spot SDK adapter + mapper + semantic_actions
│   │   └── cyberdog_sdk/           # CyberDog SDK adapter + mapper + semantic_actions
│   └── protocol/
│       ├── commands.py             # Protocol command definitions
│       ├── errors.py               # Protocol error types
│       └── events.py               # Protocol event definitions
│
├── core/                           # Shared infrastructure
│   ├── config_manager.py           # Singleton INI config; paths relative to project root
│   ├── robot_context.py            # [584 lines] RobotContext — global robot state, brand mapping
│   ├── logger.py                   # Qt signal-based thread-safe logging
│   ├── theme_manager.py            # UI theme management
│   ├── localisation.py             # i18n: tr("key", "fallback", **kwargs)
│   ├── mission_run_thread.py       # MissionRunThread (QThread) — async execution + cancel
│   ├── mission_persistence.py      # Mission save/load, schema validation
│   ├── project_store.py            # [836 lines] ProjectStore — project workspace management
│   ├── asset_resolver.py           # Asset path resolution
│   ├── data_manager.py             # Data persistence helpers
│   ├── error_ux.py                 # Error UX formatting
│   ├── hf_download_thread.py       # HFDownloadThread (QThread) for HuggingFace downloads
│   ├── hf_training_asset_download_thread.py  # Training asset download thread
│   ├── node_executor.py            # Re-exports engine/node_executor
│   ├── settings_form.py            # Settings domain model + form descriptors
│   ├── simulation_thread.py        # Simulation execution thread
│   ├── train_run_thread.py         # Training run thread (QThread)
│   └── utils/
│       ├── logger.py               # Loguru-based logging utilities
│       ├── path_helper.py          # Path resolution helpers
│       └── project_python.py       # Project Python utilities
│
├── compiler/                       # IR compiler pipeline
│   ├── ir/
│   │   ├── workflow_ir.py          # IRNode, IREdge, NodeKind dataclasses
│   │   └── types.py                # IR type definitions
│   ├── lowering/
│   │   ├── canvas_to_ir.py         # Canvas → IR
│   │   ├── ir_to_canvas.py         # IR → Canvas (roundtrip)
│   │   ├── ast_to_ir.py            # AST → IR
│   │   ├── layout.py               # Layout computation
│   │   └── protocol_compiler.py    # Protocol payload compiler
│   ├── codegen/
│   │   └── ir_to_code.py           # IR → Python code
│   ├── parser/
│   │   ├── lexer.py                # Tokenizer
│   │   ├── parser.py               # Parser
│   │   └── ast_nodes.py            # AST node definitions
│   ├── schema/
│   │   ├── node_schema.py          # Node schema definition
│   │   └── registry.py             # Schema registry
│   ├── semantic/
│   │   ├── validator.py            # Semantic validator
│   │   ├── diagnostics.py          # Diagnostic messages
│   │   └── error_codes.py          # Compiler error codes
│   └── roundtrip/
│       └── normalizer.py           # IR normalizer for roundtrip
│
├── nodes/                          # Node registry
│   ├── __init__.py                 # register_node(), get_node_class(), list_node_types()
│   └── sys_nodes/
│       ├── base_node.py            # Base node class
│       ├── action_nodes.py         # Action nodes (robot commands)
│       ├── logic_nodes.py          # Logic/control flow nodes
│       ├── sensor_nodes.py         # Sensor data nodes
│       ├── utility_nodes.py        # Utility nodes
│       ├── base_control_nodes.py   # Base control flow nodes
│       ├── behavior_node.py        # BehaviorNode — sole runtime executor for trained policies
│       ├── checkpoint_node.py      # CheckpointNode — data provider for policy bundles
│       ├── training_nodes.py       # Training-related nodes
│       ├── manual_control_node.py  # Manual control node (gamepad/keyboard)
│       ├── reactive_loco_node.py   # Reactive locomotion node
│       ├── protocol_emit_node.py   # Protocol emit node
│       └── workflow_boundary_nodes.py # Workflow start/end nodes
│
├── binding/                        # Semantic action binding
│   ├── base.py                     # Base binding interface
│   ├── registry.py                 # Binding registry
│   ├── profiles.py                 # Binding profiles
│   ├── output.py                   # Binding output contracts
│   ├── bindings/
│   │   └── locomotion.py           # Locomotion binding
│   └── executors/
│       ├── base.py                 # Base executor
│       ├── mujoco_executor.py      # Generic MuJoCo executor
│       ├── go2_mujoco_executor.py  # Go2-specific MuJoCo executor
│       ├── spot_mujoco_executor.py # Spot MuJoCo executor
│       ├── sdk_executor.py         # SDK-based executor
│       ├── unitree_sdk_executor.py # Unitree SDK executor
│       └── spot_sdk_executor.py    # Spot SDK executor
│
├── models/                         # Robot integration
│   ├── base.py                     # BaseRobotModel abstract interface
│   ├── sdk_manager.py              # [900 lines] SDK bootstrap manager
│   ├── model_registry.py           # Model registration
│   ├── mujoco_asset_registry.py    # MuJoCo asset registration
│   └── mujoco_menagerie/           # Pre-packaged MuJoCo robot models
│
├── ir/                             # Layered IR contracts
│   ├── layered_contracts.py        # SubgraphIR, PackageMetadata
│   ├── layered_interfaces.py       # Layered IR interfaces
│   ├── workflow_ir.py              # WorkflowIR (system-level)
│   └── workbench/                  # IR workbench utilities
│
├── brand_packages/                 # Brand package registration
│   ├── unitree/unitree_model.py    # Unitree brand model
│   └── boston_dynamics/spot_model.py # Spot brand model
│
├── mission/                        # Mission DAG
│   ├── mission_model.py            # Mission data model
│   ├── mission_planner.py          # Mission planner
│   └── mission_serializer.py       # Mission serializer
│
└── telemetry/event_bus.py          # Event bus for telemetry
```

### 3.4 Data / Config Directories

| Directory | Purpose | Committed? |
|-----------|---------|------------|
| `src/config/` | INI config: `system.ini`, `user.ini`, `ui.ini`, `node_registry.json`, `isaaclab.json`, `setup_state.json` | Yes |
| `localisation/` | i18n JSON: `en.json`, `zh.json` | Yes |
| `bin/assets/` | SVG icons (`icon/`), animations (`anim/`), sounds (`sound/`) | Yes |
| `projects/` | ProjectStore managed per-project workspaces | No |
| `runtime/` | Vendored SDK mirrors (Unitree, BostonDynamics, XiaoMi, MuJoCo menagerie) — **read-only** | Yes |
| `knowledge_base/` | Agent knowledge base documents | Yes |
| `hotfix/` | Emergency patches | No |
| `logs/` | Runtime logs | No |

### 3.5 `custom_mods/` — Community Hot-Plug Resources

所有内容均为 init 后用户自定义下载，可全部删除而不影响程序启动。
`custom_mods/` 本身不是 Python 包（无 `__init__.py`）。

当前仅保留 `motions/` 子目录（参考动作数据 npy/npz）。其余子目录（`training/assets/`, `training/checkpoints/`, `runtime/`, `nodes/`, `archives/` 等）已在最近的重构中清理移除。

---

## 4. Key Design Patterns

### 4.1 Core Patterns

| Pattern | Location | Description |
|---------|----------|-------------|
| **RobotContext** | `src/system/core/robot_context.py` | Global robot state factory; brand mapping ("go2" → "unitree"); hot-swappable |
| **BaseRobotModel** | `src/system/models/base.py` | Abstract adapter: `connect`, `run_action`, `stop`, `get_sensor_data` |
| **Node Registry** | `src/system/nodes/__init__.py` | `register_node()`, `get_node_class()`, `list_node_types()` |
| **NodeExecutor** | `src/system/engine/node_executor.py` | Flow-aware DFS traversal |
| **ConfigManager** | `src/system/core/config_manager.py` | Singleton INI config; paths relative to project root |
| **ProjectStore** | `src/system/core/project_store.py` | Project workspace management, project-level persistence |
| **Qt Signal Logging** | `src/system/core/logger.py` | Thread-safe: `log_info()`, `log_error()`, `log_success()` |
| **Localisation** | `src/system/core/localisation.py` | `tr("key", "fallback", **kwargs)` with JSON translation files |
| **MissionRunThread** | `src/system/core/mission_run_thread.py` | QThread for async execution; emits signals; cancel via `request_cancel()` |
| **CheckpointRegistry** | `src/system/service/checkpoint_registry.py` | Single source of truth for deployed checkpoint bundles |
| **SkillManifest** | `src/system/skill/skill_manifest.py` | Skill definition + transition rules for orchestration |

### 4.2 Engine Execution Paths

`RuntimeEngine.execute(mission_ir, scenario)` dispatches:

- **Path A**: `execution_graph` (dict with `"nodes"` key) → `WorkflowRunner`
- **Path B**: `WorkflowIR` object → `NodeExecutor` (flow-aware DFS)

Both return unified `RuntimeResult` dataclass (`src/system/engine/contracts.py`).

### 4.3 Mission Canvas Execution Chain

```
CheckpointNode.execute()  →  BehaviorNode.execute()  →  MuJoCo episode
```

- `CheckpointNode` = data provider only (never enters workflow queue)
- `BehaviorNode` = sole runtime executor for trained policies

---

## 5. Key Wiring Points (Where Things Connect)

| From | To | How |
|------|----|-----|
| `bin/pages/layout/ui.py` `_on_run()` | `RuntimeEngine` | Gets execution graph → calls execute → dispatches results |
| `bin/pages/layout/ui.py` | `GraphScene` | `graph_scene.set_node_execution_status(nid, status)` for per-node badges |
| `bin/pages/layout/ui.py` | `MissionRunThread` | Async execution wrapper (QThread) |
| `bin/pages/layout/ui.py` | `BehaviorCompilerBridge` | Shared between `runtime_engine.behavior_bridge` and `HBChannelFactory` |
| `GraphScene` | `canvas_to_ir` | `serialize_workflow()` → dict; `get_execution_graph()` → runtime dict |
| `canvas_to_ir` | `ir_to_code` | IR → Python code generation |
| `NodeExecutor` | `BehaviorNode` | Injects `sim_env` (SimEnvContext) before `execute()` |
| `SB3Trainer` | `unitree_gym_env` | Gymnasium environment for RL training |
| `bundle_exporter` | `projects/<slug>/training/exported/<policy_id>/` | Exports ONNX + TorchScript + manifest (Layout v2) |
| `PolicyRunner` | `BundleLoader` | Loads CheckpointBundle → runs inference |
| `SkillManifest` | `TransitionValidator` | Validates skill state transitions |

---

## 6. Important Constraints for Refactoring

### 6.1 Do NOT Break

1. **Existing imports** from `bin/*`, `src/system/nodes/*`, `src/system/models/*`, `src/system/compiler/*` — the project is mid-migration
2. **Node Registry** pattern — `register_node()` / `get_node_class()` must remain stable
3. **RuntimeResult** contract — unified return from both execution paths
4. **GraphScene** serialization — `serialize_workflow()` / `load_workflow()` format compatibility
5. **Mission schema versioning** — `migrate_mission_payload()` must handle older versions
6. **CheckpointRegistry** — `import_local()`, `import_hf_bundle()` interfaces
7. **IR roundtrip** — Canvas ↔ IR ↔ Code bidirectional consistency
8. **SkillManifest** — transition validation contract

### 6.2 Known Deprecations

- `conductor_node.py` and `policy_node.py` — **deleted** (were deprecated, now fully removed)
- `s.bat` — legacy launcher, not canonical

### 6.3 Known Issues

- `bin/pages/layout/behavior_panel.py` — behavior tab still placeholder-heavy
- `src/system/service/settings_schema.py` — `robot_type` lacks dynamic choices from BrandRegistry

---

## 7. Dependencies

### 7.1 Core Runtime

| Package | Version | Purpose |
|---------|---------|---------|
| PySide6 | >=6.5.0 | Qt6 GUI framework |
| mujoco | >=3.0.0 | Physics simulation |
| numpy | >=1.24.0 | Numerical computing |
| PyYAML | >=6.0 | YAML parsing (manifest files) |
| onnxruntime | >=1.17 | ONNX model inference |
| torch | >=2.0.0 | PyTorch (TorchScript inference + training) |
| huggingface_hub | ==1.7.1 | HuggingFace model download |
| psutil | >=5.9 | System resource monitoring |
| cyclonedds | >=0.10.2 | DDS communication (Unitree SDK) |

### 7.2 Dev / Test

| Package | Purpose |
|---------|---------|
| pytest | Test runner |
| stable-baselines3 | RL training (PPO/SAC/TD3) — installed by `install.bat/sh` |

---

## 8. File Size Reference (Largest Files)

| File | Lines | Role |
|------|-------|------|
| `bin/pages/canvas/graph_scene.py` | 9,067 | Canvas — all graph operations |
| `bin/nodes/training_node_items.py` | 6,133 | Training node UI items |
| `bin/pages/training/training_workspace_window.py` | 5,859 | Training workspace UI |
| `bin/pages/layout/behavior_panel.py` | 5,744 | Behavior panel UI |
| `bin/pages/layout/ui.py` | 3,399 | MainWindow |
| `src/system/training/unitree_gym_env.py` | 2,061 | Gymnasium MuJoCo env |
| `src/system/training/sb3_trainer.py` | 1,963 | SB3 training runner |
| `src/system/behavior/motor_registry.py` | 1,682 | Motor registry |
| `src/system/engine/node_executor.py` | 1,358 | Flow-aware node executor |
| `src/system/behavior/action_profile.py` | 1,322 | Action profile comparison |
| `bin/pages/layout/main_zone_panel.py` | 1,260 | Main zone layout |
| `bin/nodes/node_ui_rows.py` | 1,241 | Node parameter UI rows |
| `bin/pages/canvas/script_editor.py` | 1,202 | Script editor widget |
| `src/system/training/training_config.py` | 1,206 | Training configuration |
| `bin/pages/homepage/homepage.py` | 1,144 | Homepage |
| `src/system/training/training_spec.py` | 1,018 | TrainingSpecCompiler |
| `bin/pages/layout/misc.py` | 1,014 | UI miscellaneous helpers |
| `src/system/training/vis_check_runner.py` | 992 | Visual check runner |

---

## 9. Quick Start for Agent

```bash
# Run all tests (Windows)
.venv311\Scripts\python.exe -m pytest tests\ -q

# Compile-check a file
.venv311\Scripts\python.exe -m py_compile path\to\file.py

# Launch the app
.venv311\Scripts\python.exe main.py
```

---

## 10. Module Dependency Flow (Simplified)

```
main.py
  └── bin/pages/layout/ui.py (MainWindow)
        ├── bin/pages/canvas/graph_scene.py (Canvas)
        ├── bin/pages/layout/main_zone_panel.py (Layout)
        ├── bin/pages/layout/behavior_panel.py (Behavior UI)
        ├── bin/pages/scenario/scenario_panel.py (Scenario)
        ├── bin/pages/homepage/homepage.py (Homepage)
        ├── bin/pages/setup/setup_wizard.py (Setup)
        ├── src/system/core/mission_run_thread.py → src/system/engine/runtime_engine.py
        │     ├── src/system/engine/node_executor.py
        │     ├── src/system/engine/workflow_runner.py
        │     ├── src/system/engine/behavior_invoker.py → src/system/behavior/*
        │     └── src/system/engine/reactive_loop.py
        ├── src/system/core/train_run_thread.py → src/system/training/sb3_trainer.py
        │     └── src/system/training/unitree_gym_env.py
        ├── src/system/compiler/lowering/canvas_to_ir.py
        │     └── src/system/compiler/ir/workflow_ir.py
        ├── src/system/compiler/codegen/ir_to_code.py
        ├── src/system/nodes/__init__.py (Node Registry)
        │     └── src/system/nodes/sys_nodes/* (Node Definitions)
        ├── src/system/service/checkpoint_registry.py
        ├── src/system/policy/policy_runner.py
        ├── src/system/skill/skill_manifest.py (Skill Orchestration)
        ├── src/system/core/project_store.py (Project Management)
        └── src/system/models/base.py → src/system/brand_packages/*/
```
