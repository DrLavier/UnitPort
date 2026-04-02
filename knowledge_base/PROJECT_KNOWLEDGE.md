# UnitPort — Agent Onboarding Knowledge Base

> **Purpose**: 供 AI Agent 快速建立本项目全局视图，理解架构、模块职责、文件位置和改造约束。
> **Last updated**: 2026-03-30

---

## 1. Project Identity

| Key | Value |
|-----|-------|
| Name | **UnitPort** |
| Type | Cross-platform visual robot programming framework |
| Tech Stack | Python 3.8+ (migration target 3.12), PySide6 (Qt6), MuJoCo 3.0+, Stable-Baselines3, ONNX, PyTorch |
| Entry Point | `main.py` → `QApplication` → `MainWindow` (`bin/ui.py`) |
| Venv | `.venv311/` (project-local, Python 3.11) |
| Platform | Windows / Linux / macOS |
| Language | Bilingual codebase (Chinese + English); UI strings via `tr()` i18n |
| License | `LICENSE.txt` (project root) |

**One-line summary**: UnitPort 将可视化任务编排 (Canvas)、行为编程 (Compiler)、场景配置 (Scenario) 和 RL 训练 (Training Ground) 统一在一个桌面应用中，支持从仿真到实机的多机器人控制。

---

## 2. Architecture Overview

### 2.1 Four Backend Layers

```
Canvas ──→ Mission ──┐
                      ├──→ Runtime (+Safety) ──→ Service ──→ SDK / Robot
Compiler ─→ Behavior ─┘

Scenario ──→ Runtime params, execution strategy, safety protocol, sim/real target
```

| Layer | Responsibility | Key Directory |
|-------|---------------|---------------|
| **Mission** | "What to do" — project-level DAG task orchestration | `system/mission/` |
| **Behavior** | "How to do" — node-internal logic, state machines, sensor feedback, semantic actions | `system/behavior/` |
| **Service** | Vendor SDK adaptation — adapters for Unitree/Spot/CyberDog | `system/service/` |
| **Runtime** | Execution scheduling, monitoring, safety interception | `system/runtime/` |

### 2.2 Three Frontend Layers

| Layer | Responsibility | Key File |
|-------|---------------|----------|
| **Canvas** | Visual node-based task builder | `bin/components/graph_scene.py` (8900+ lines) |
| **Compiler** | Python behavior scripting + IR roundtrip | `bin/components/code_editor.py`, `compiler/` |
| **Scenario** | Execution config (sim/real, safety, connections) | `bin/scenario/` |

### 2.3 Training Ground Pipeline (Circle 7 / Phase F)

```
Canvas training graph
  → TrainingSpecCompiler (system/training/training_spec.py)
  → SB3Trainer (PPO/SAC, Gymnasium + MuJoCo)
  → export_bundle (ONNX + TorchScript + manifest.yaml + source.json)
  → CheckpointRegistry (custom_mods/training/checkpoints/)
  → PolicyRunner.load() + run_episode()
```

### 2.4 Compiler Pipeline (IR-based bidirectional)

```
Canvas Graph ←→ IR ←→ Python Code
```

- Canvas → IR: `compiler/lowering/canvas_to_ir.py`
- IR → Code: `compiler/codegen/ir_to_code.py`
- IR → Canvas: `compiler/lowering/ir_to_canvas.py`
- IR data model: `compiler/ir/workflow_ir.py` (IRNode, IREdge, NodeKind)

Canvas 和 Code 共享同一 IR 作为语义源——两者都不是权威表示。

---

## 3. Complete Directory Map

### 3.1 Top-Level Structure

```
UnitPort/
├── main.py                         # Application entry point (319 lines)
├── requirements.txt                # PySide6, mujoco, numpy, torch, onnxruntime, huggingface_hub, ...
├── install.bat / install.sh        # First-time setup scripts
├── start.sh / s.bat                # Launch scripts
├── CLAUDE.md                       # AI agent instructions (authoritative)
├── AGENTS.md                       # Agent operation constraints
├── PROJECT_STRUCTURE.md            # Directory layout + migration notes
├── pytest.ini                      # Pytest configuration
│
├── bin/                            # [~41,800 lines] Frontend UI (PySide6)
├── compiler/                       # [~5,150 lines] IR-based compiler pipeline
├── system/                         # [~33,400 lines] Backend layered architecture
├── nodes/                          # [~3,680 lines] Node registry + definitions
├── models/                         # Robot model interface + MuJoCo assets
├── brands_sdk/                     # Vendored SDK mirrors (read-only)
├── tests/                          # [~25,800 lines] Test suites
├── config/                         # INI configuration files
├── localisation/                   # i18n JSON (en.json, zh.json)
├── assets/                         # SVG icons, animations, sounds
├── utils/                          # Path helpers, logging utilities
├── scripts/                        # Build/distribution scripts
├── custom_mods/                    # Community hot-plug resources (see §3.9)
├── training_checkpoints/           # Training-time checkpoint artifacts
├── training_workspaces/            # Per-job training artifacts
├── hotfix/                         # Emergency patches
├── runtime/                        # Vendored SDKs and simulation assets (read-only)
└── logs/                           # Application logs
```

### 3.2 `bin/` — Frontend & App Wiring (~41,800 lines)

```
bin/
├── ui.py                           # [2866 lines] MainWindow — top settings row, cross-zone wiring,
│                                   #   _on_run/_on_runtime_abort, HF import, compat audit
├── core/
│   ├── config_manager.py           # Singleton INI config access (paths relative to project root)
│   ├── data_manager.py             # Data persistence helpers
│   ├── error_ux.py                 # REASON_OPERATOR_TEXT, format_node_diagnostics, error categories
│   ├── hf_download_thread.py       # HFDownloadThread (QThread) for HuggingFace downloads
│   ├── hf_training_asset_download_thread.py  # Training asset download thread
│   ├── localisation.py             # tr("key", "fallback", **kwargs) — i18n
│   ├── logger.py                   # Qt signal-based thread-safe logging
│   ├── mission_persistence.py      # Mission save/load, schema validation, migrate_mission_payload()
│   ├── mission_run_thread.py       # MissionRunThread (QThread) — async execution + cancel
│   ├── node_executor.py            # Re-exports system/runtime/node_executor.NodeExecutor
│   ├── robot_context.py            # RobotContext — global robot state factory, brand mapping
│   ├── settings_form.py            # Settings domain model + form descriptors
│   ├── simulation_thread.py        # Simulation execution thread
│   ├── theme_manager.py            # UI theme management
│   └── train_run_thread.py         # Training run thread (QThread)
├── components/
│   ├── graph_scene.py              # [8927 lines] GraphScene — ALL canvas ops (create, serialize, load,
│   │                               #   duplicate, group, ungroup, status badges, execution graph)
│   ├── graph_view.py               # [487 lines] GraphView — QGraphicsView wrapper
│   ├── code_editor.py              # Code editor widget
│   ├── training_workspace_window.py # [5155 lines] Training workspace management window
│   ├── training_node_items.py      # [5225 lines] Training node UI items
│   ├── training_panel.py           # [230 lines] Training panel widget
│   ├── checkpoint_import_dialog.py # Checkpoint import dialog (local + HuggingFace)
│   ├── settings_panel.py           # Settings UI panel
│   ├── capability_inspector.py     # Robot capability inspector widget
│   ├── diagnostics_panel.py        # DiagnosticsPanel — error display + navigate_requested
│   ├── motor_weight_navigator.py   # MotorWeightNavigator — motor topology tree
│   ├── sidebar_dock.py             # Fixed left rail navigation
│   ├── homepage.py                 # Homepage/welcome screen
│   ├── misc.py                     # Miscellaneous UI helpers
│   ├── module_cards.py             # Module card widgets
│   ├── node_ui_rows.py             # Node parameter UI rows
│   ├── script_api_analyzer.py      # Script API analysis
│   ├── script_builtin_registry.py  # Script built-in function registry
│   └── script_editor.py            # Script editor widget
├── layout/
│   ├── main_zone_panel.py          # [1234 lines] MainZonePanel — runtime_zone + mission_zone +
│   │                               #   ExecutionSummaryBar + DiagnosticsPanel
│   └── behavior_panel.py           # [5744 lines] Behavior tab panel (timeline, compat audit)
└── scenario/
    ├── scenario_panel.py           # Scenario configuration panel
    ├── runtime_console.py          # Runtime console output
    └── safety_policy_editor.py     # Safety policy editor
```

### 3.3 `system/` — Backend Architecture (~33,400 lines)

```
system/
├── training/                       # [~11,000 lines] RL training pipeline
│   ├── sb3_trainer.py              # [1784 lines] SB3 training runner (PPO/SAC)
│   ├── unitree_gym_env.py          # [1871 lines] Gymnasium MuJoCo environment
│   ├── training_spec.py            # [740 lines] TrainingSpecCompiler
│   ├── training_config.py          # Training configuration dataclasses
│   ├── training_process.py         # Training process management
│   ├── training_run_cache.py       # Training run caching
│   ├── bundle_exporter.py          # Export ONNX + TorchScript + manifest
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
│   ├── vis_check_runner.py         # Visual check runner
│   └── loco_mujoco_bridge.py       # Locomotion MuJoCo bridge
│
├── behavior/                       # [~11,400 lines] Behavior logic + semantic actions
│   ├── semantic_action.py          # SemanticActionDescriptor (frozen dataclass)
│   ├── intent_catalog.py           # Canonical intent constants (POSTURE_STAND, LOCOMOTION_FORWARD, ...)
│   ├── semantic_resolution.py      # Semantic action resolution
│   ├── action_profile.py           # Action profile comparison
│   ├── action_profile_report.py    # Action profile report generation
│   ├── action_registry.py          # Action registration
│   ├── ir_action_registry.py       # IR-level action registry
│   ├── behavior_compiler_bridge.py # BehaviorCompilerBridge — connects behavior to compiler
│   ├── behavior_model.py           # Behavior data model
│   ├── behavior_state_machine.py   # Behavior state machine
│   ├── behavior_artifact.py        # Behavior artifact management
│   ├── hb_channel.py               # HBRuntimeChannel + HBChannelFactory
│   ├── hb_compat_audit.py          # Cross-brand compatibility audit
│   ├── hb_display_state.py         # HB display state management
│   ├── hb_node_catalog.py          # HB node catalog
│   ├── heartbeat_policy.py         # Heartbeat policy management
│   ├── motor_weight_protocol.py    # Motor weight protocol validation
│   ├── motor_param_source.py       # Motor parameter source management
│   ├── motor_registry.py           # Motor registry
│   ├── motor_topology.py           # Motor topology definition
│   ├── motor_weight_nav_model.py   # Motor weight navigator model
│   ├── protocol_apply_engine.py    # Protocol apply engine
│   ├── editor_ir.py                # Editor IR representation
│   ├── timeline_migration.py       # Timeline data migration
│   ├── timeline_view_model.py      # Timeline view model
│   └── topology_fallback.py        # Topology fallback handling
│
├── policy/                         # [~2,050 lines] Policy inference stack
│   ├── policy_runner.py            # [501 lines] PolicyRunner.load() + run_episode() → EpisodeResult
│   ├── bundle_loader.py            # Loads CheckpointBundle from disk
│   ├── inference_engine.py         # ONNX/TorchScript inference
│   ├── obs_builder.py              # Observation builder
│   ├── action_applier.py           # Action application to simulation
│   ├── normalizer.py               # Observation normalization
│   ├── compatibility_checker.py    # Policy compatibility checks
│   ├── manifest_schema.py          # Manifest YAML schema
│   ├── joint_name_utils.py         # Joint name mapping utilities
│   ├── sim_env_context.py          # SimEnvContext for MuJoCo episodes
│   └── policy_command_executor.py  # Policy command execution
│
├── service/                        # [~4,250 lines] Vendor SDK adaptation layer
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
├── runtime/                        # [~3,350 lines] Execution engine
│   ├── runtime_engine.py           # [459 lines] RuntimeEngine.execute() — dispatches Path A/B
│   ├── node_executor.py            # [1114 lines] Flow-aware DFS traversal
│   ├── workflow_runner.py          # [396 lines] WorkflowRunner for dict-based execution
│   ├── contracts.py                # RuntimeResult dataclass, DiagnosticsKey constants
│   ├── behavior_invoker.py         # Behavior subgraph invocation
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
├── binding/                        # [~1,275 lines] Semantic action binding
│   ├── base.py                     # Base binding interface
│   ├── registry.py                 # Binding registry
│   ├── profiles.py                 # Binding profiles
│   ├── output.py                   # Binding output contracts
│   └── executors/
│       ├── base.py                 # Base executor
│       ├── mujoco_executor.py      # Generic MuJoCo executor
│       ├── go2_mujoco_executor.py  # Go2-specific MuJoCo executor
│       ├── spot_mujoco_executor.py # Spot MuJoCo executor
│       ├── sdk_executor.py         # SDK-based executor
│       ├── unitree_sdk_executor.py # Unitree SDK executor
│       └── spot_sdk_executor.py    # Spot SDK executor
│
├── ir/                             # Layered IR contracts
│   ├── layered_contracts.py        # SubgraphIR, PackageMetadata
│   ├── layered_interfaces.py       # Layered IR interfaces
│   └── workflow_ir.py              # WorkflowIR (system-level)
│
├── brand_packages/                 # Brand package registration
│   ├── unitree/unitree_model.py    # Unitree brand model
│   └── bostiondynamics/spot_model.py # Spot brand model
│
├── telemetry/event_bus.py          # Event bus for telemetry
├── model_registry.py               # Model registration
├── mujoco_asset_registry.py        # MuJoCo asset registration
└── types/
    ├── common_types.py             # Shared type definitions
    └── error_codes.py              # Error code constants
```

### 3.4 `compiler/` — IR Compiler Pipeline (~5,150 lines)

```
compiler/
├── ir/
│   ├── workflow_ir.py              # [298 lines] IRNode, IREdge, NodeKind dataclasses
│   └── types.py                    # IR type definitions
├── lowering/
│   ├── canvas_to_ir.py             # [636 lines] Canvas → IR
│   ├── ir_to_canvas.py             # IR → Canvas (roundtrip)
│   ├── ast_to_ir.py                # AST → IR
│   ├── layout.py                   # Layout computation
│   └── protocol_compiler.py        # Protocol payload compiler
├── codegen/
│   └── ir_to_code.py               # [659 lines] IR → Python code
├── parser/
│   ├── lexer.py                    # Tokenizer
│   ├── parser.py                   # Parser
│   └── ast_nodes.py                # AST node definitions
├── schema/
│   ├── node_schema.py              # Node schema definition
│   └── registry.py                 # Schema registry
├── semantic/
│   ├── validator.py                # Semantic validator
│   ├── diagnostics.py              # Diagnostic messages
│   └── error_codes.py              # Compiler error codes
└── roundtrip/
    └── normalizer.py               # IR normalizer for roundtrip
```

### 3.5 `nodes/` — Node Registry (~3,680 lines)

```
nodes/
├── __init__.py                     # Node registry: register_node(), get_node_class(), list_node_types()
├── sys_nodes/
│   ├── base_node.py                # [78 lines] Base node class
│   ├── action_nodes.py             # Action nodes (robot commands)
│   ├── logic_nodes.py              # Logic/control flow nodes
│   ├── sensor_nodes.py             # Sensor data nodes
│   ├── utility_nodes.py            # Utility nodes
│   ├── base_control_nodes.py       # Base control flow nodes
│   ├── behavior_node.py            # BehaviorNode — sole runtime executor for trained policies
│   ├── checkpoint_node.py          # CheckpointNode — data provider for policy bundles
│   ├── training_nodes.py           # [1406 lines] Training-related nodes
│   ├── protocol_emit_node.py       # Protocol emit node
│   ├── workflow_boundary_nodes.py  # Workflow start/end nodes
│   ├── conductor_node.py           # DEPRECATED — use BehaviorNode
│   └── policy_node.py              # DEPRECATED — use CheckpointNode + BehaviorNode
└── custom_nodes/                   # User/community custom node packs
```

### 3.6 `models/` — Robot Integration

```
models/
├── base.py                         # BaseRobotModel abstract interface (connect, run_action, stop, get_sensor_data)
├── sdk_manager.py                  # SDK bootstrap manager (sys.executable -m pip)
└── mujoco_menagerie/               # Pre-packaged MuJoCo robot models (Go2, Spot, H1, G1, A1, ...)
```

### 3.7 `tests/` — Test Suites (~25,800 lines)

```
tests/
├── unit/                           # Unit tests
│   ├── test_policy/                # Policy subsystem tests (11 files)
│   ├── test_circle7_*.py           # Training Ground tests
│   ├── test_phase*_*.py            # Phase-based training tests
│   ├── test_p1_checkpoint_import.py
│   ├── test_p2_hf_download.py
│   └── ...
├── integration/                    # Integration tests
│   ├── test_circle7_e2e_train.py
│   ├── test_phase_f_e2e_pipeline.py
│   ├── test_policy_e2e.py
│   └── ...
├── regression/                     # Regression tests
│   ├── test_forward_integration.py
│   └── scene_builder.py
└── fixtures/
    ├── mocks.py
    └── profile_utils.py
```

### 3.8 Data / Config Directories

| Directory | Purpose | Committed? |
|-----------|---------|------------|
| `config/` | INI config: `system.ini`, `user.ini`, `ui.ini`, `rewards_presets.json`, `node_registry.json` | Yes |
| `localisation/` | i18n JSON: `en.json`, `zh.json` | Yes |
| `assets/` | SVG icons (`icon/`), animations (`anim/`), sounds (`sound/`) | Yes |
| `training_checkpoints/` | Training-time checkpoint artifacts | No |
| `training_workspaces/` | Per-job training workspace data | No |
| `runtime/` | Vendored SDK mirrors (Unitree, BostonDynamics, XiaoMi, MuJoCo menagerie) — **read-only** | Yes |
| `hotfix/` | Emergency patches | No |
| `logs/` | Runtime logs | No |

### 3.9 `custom_mods/` — Community Hot-Plug Resources

所有内容均为 init 后用户自定义下载，可全部删除而不影响程序启动。
`custom_mods/` 本身不是 Python 包（无 `__init__.py`）。

```
custom_mods/
├── training/                          # 训练域
│   ├── motions/                       #   参考动作 (npy/npz), 按形态分: quadruped/ biped/ manipulator/ wheeled/ generic/
│   ├── assets/                        #   训练物料包 (SB3 checkpoints, logs, task_template)
│   ├── checkpoints/                   #   部署就绪 policy bundle (ONNX + manifest.yaml + source.json)
│   └── auxiliary/                     #   辅助模型预留 (LLM adapter, 视频/图像生成)
│
├── runtime/                           # 运行域
│   ├── bundles/                       #   Mission workflow 支持包
│   ├── sdk_extensions/                #   SDK 辅助插件
│   └── controllers/                   #   控制器辅助包
│
├── nodes/                             # 社区节点 (drop-in .py, 系统自动扫描注册)
├── workflows/                         # 社区共享 workflow (.unitport)
├── datasets/                          # 数据集 (预留)
└── archives/                          # 社区仓库归档 (git clone 完整仓库, 如 loco-mujoco)
```

| Subdirectory | Registry / Consumer | Hot-plug? |
|---|---|---|
| `training/checkpoints/` | `CheckpointRegistry` (`STORAGE_SUBDIR`) | Yes |
| `training/assets/` | `TrainingAssetRegistry` | Yes |
| `training/motions/` | `MotionLibrary` | Yes |
| `nodes/` | `src/system/nodes/__init__.py` 自动扫描 | Yes |
| `workflows/` | 用户手动加载 | Yes |
| `archives/` | 待实现自动解包分配逻辑 | Yes |

---

## 4. Key Design Patterns

### 4.1 Core Patterns

| Pattern | Location | Description |
|---------|----------|-------------|
| **RobotContext** | `bin/core/robot_context.py` | Global robot state factory; brand mapping ("go2" → "unitree"); hot-swappable |
| **BaseRobotModel** | `models/base.py` | Abstract adapter: `connect`, `run_action`, `stop`, `get_sensor_data` |
| **Node Registry** | `nodes/__init__.py` | `register_node()`, `get_node_class()`, `list_node_types()` |
| **NodeExecutor** | `system/runtime/node_executor.py` | Flow-aware DFS traversal |
| **ConfigManager** | `bin/core/config_manager.py` | Singleton INI config; paths relative to project root |
| **Qt Signal Logging** | `bin/core/logger.py` | Thread-safe: `log_info()`, `log_error()`, `log_success()` |
| **Localisation** | `bin/core/localisation.py` | `tr("key", "fallback", **kwargs)` with JSON translation files |
| **MissionRunThread** | `bin/core/mission_run_thread.py` | QThread for async execution; emits signals; cancel via `request_cancel()` |
| **CheckpointRegistry** | `system/service/checkpoint_registry.py` | Single source of truth for deployed checkpoint bundles |

### 4.2 Runtime Execution Paths

`RuntimeEngine.execute(mission_ir, scenario)` dispatches:

- **Path A**: `execution_graph` (dict with `"nodes"` key) → `WorkflowRunner`
- **Path B**: `WorkflowIR` object → `NodeExecutor` (flow-aware DFS)

Both return unified `RuntimeResult` dataclass (`system/runtime/contracts.py`).

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
| `bin/ui.py` `_on_run()` | `RuntimeEngine` | Gets execution graph → calls execute → dispatches results |
| `bin/ui.py` | `GraphScene` | `graph_scene.set_node_execution_status(nid, status)` for per-node badges |
| `bin/ui.py` | `MissionRunThread` | Async execution wrapper (QThread) |
| `bin/ui.py` | `BehaviorCompilerBridge` | Shared between `runtime_engine.behavior_bridge` and `HBChannelFactory` |
| `GraphScene` | `canvas_to_ir` | `serialize_workflow()` → dict; `get_execution_graph()` → runtime dict |
| `canvas_to_ir` | `ir_to_code` | IR → Python code generation |
| `NodeExecutor` | `BehaviorNode` | Injects `sim_env` (SimEnvContext) before `execute()` |
| `SB3Trainer` | `unitree_gym_env` | Gymnasium environment for RL training |
| `bundle_exporter` | `custom_mods/training/checkpoints/` | Exports ONNX + TorchScript + manifest |
| `PolicyRunner` | `BundleLoader` | Loads CheckpointBundle → runs inference |

---

## 6. Important Constraints for Refactoring

### 6.1 Do NOT Break

1. **Existing imports** from `bin/*`, `nodes/*`, `models/*`, `compiler/*` — the project is mid-migration
2. **Node Registry** pattern — `register_node()` / `get_node_class()` must remain stable
3. **RuntimeResult** contract — unified return from both execution paths
4. **GraphScene** serialization — `serialize_workflow()` / `load_workflow()` format compatibility
5. **Mission schema versioning** — currently at v1.4; `migrate_mission_payload()` must handle older versions
6. **CheckpointRegistry** — `import_local()`, `import_hf_bundle()` interfaces
7. **IR roundtrip** — Canvas ↔ IR ↔ Code bidirectional consistency

### 6.2 Known Deprecations

- `nodes/sys_nodes/conductor_node.py` — DEPRECATED, replaced by `BehaviorNode`
- `nodes/sys_nodes/policy_node.py` — DEPRECATED, replaced by `CheckpointNode + BehaviorNode`
- `s.bat` — legacy launcher, not canonical

### 6.3 Known Issues

- `bin/layout/behavior_panel.py` — behavior tab still placeholder-heavy
- `system/service/settings_schema.py` — `robot_type` lacks dynamic choices from BrandRegistry
- Pre-existing test failures (12 GUI + 9 SDK-absent): do not regress beyond these

### 6.4 Migration Target

Planned evolution from `bin/` to clearer frontend domains:

```
bin/components/ → frontend/canvas/ + frontend/compiler/
bin/scenario/   → frontend/scenario/
```

Authority hierarchy for conflicts: `PROJECT_README.md` > `PROJECT_STRUCTURE_NEW.md` > current module docs.

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
| stable-baselines3 | RL training (PPO/SAC) — installed by `install.bat/sh` |

---

## 8. File Size Reference (Largest Files)

| File | Lines | Role |
|------|-------|------|
| `bin/components/graph_scene.py` | 8,927 | Canvas — all graph operations |
| `bin/layout/behavior_panel.py` | 5,744 | Behavior panel UI |
| `bin/components/training_workspace_window.py` | 5,155 | Training workspace UI |
| `bin/components/training_node_items.py` | 5,225 | Training node UI items |
| `bin/ui.py` | 2,866 | MainWindow |
| `system/training/unitree_gym_env.py` | 1,871 | Gymnasium MuJoCo env |
| `system/training/sb3_trainer.py` | 1,784 | SB3 training runner |
| `nodes/sys_nodes/training_nodes.py` | 1,406 | Training node definitions |
| `bin/layout/main_zone_panel.py` | 1,234 | Main zone layout |
| `system/runtime/node_executor.py` | 1,114 | Flow-aware node executor |

---

## 9. Quick Start for Agent

```bash
# Run all tests (Windows)
.venv311\Scripts\python.exe -m pytest tests\ -q

# Run unit tests only
.venv311\Scripts\python.exe -m pytest tests\unit\ -q

# Compile-check a file
.venv311\Scripts\python.exe -m py_compile path\to\file.py

# Launch the app
.venv311\Scripts\python.exe main.py
```

---

## 10. Module Dependency Flow (Simplified)

```
main.py
  └── bin/ui.py (MainWindow)
        ├── bin/components/graph_scene.py (Canvas)
        ├── bin/layout/main_zone_panel.py (Layout)
        ├── bin/layout/behavior_panel.py (Behavior UI)
        ├── bin/scenario/scenario_panel.py (Scenario)
        ├── bin/core/mission_run_thread.py → system/runtime/runtime_engine.py
        │     ├── system/runtime/node_executor.py
        │     ├── system/runtime/workflow_runner.py
        │     └── system/runtime/behavior_invoker.py → system/behavior/*
        ├── bin/core/train_run_thread.py → system/training/sb3_trainer.py
        │     └── system/training/unitree_gym_env.py
        ├── compiler/lowering/canvas_to_ir.py
        │     └── compiler/ir/workflow_ir.py
        ├── compiler/codegen/ir_to_code.py
        ├── nodes/__init__.py (Node Registry)
        │     └── nodes/sys_nodes/* (Node Definitions)
        ├── system/service/checkpoint_registry.py
        ├── system/policy/policy_runner.py
        └── models/base.py → system/brand_packages/*/
```
