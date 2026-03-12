# UnitPort - Unified Robot Programming Framework

[![Website](https://img.shields.io/badge/Website-uniport.ai-blue)](https://uniport.ai)
[![Python](https://img.shields.io/badge/Python-3.11.9-blue.svg)](https://www.python.org/downloads/)

> [!WARNING]
> **WIP (Work In Progress)**  
> This project is under active development.  
> **End-user local feature implementation is not supported yet.**

A cross-platform visual robot programming framework that unifies **task orchestration (Canvas)**, **behavior programming (Compiler)**, and **scenario configuration (Scenario)** into a consistent engineering system.

**🌐 Visit us at [uniport.ai](https://uniport.ai)**

## Core Value Proposition

- **Simulation-to-Real Deployment**: Execute the same task seamlessly in simulation or on physical robots
- **Multi-Robot Support**: Vendor-agnostic design through Service adapter layer (Unitree, Boston Dynamics, and more)
- **Safety-First Runtime**: Built-in execution interception and constraint system at compile-time and runtime
- **Visual + Code**: Low-code Canvas for task flow + Python Compiler for fine-grained behavior control

---

## Quick Start

UnitPort currently uses a project-local, built-in virtual environment: `.venv311`
(Python `3.11.9`).

If the environment is missing or needs to be rebuilt, use the platform-native
install script for your OS.

Windows:

```bat
REM Clone repository
git clone https://github.com/DrLavier/UnitPort.git
cd UnitPort

REM Optional: rebuild/update the local environment
install.bat

REM Run application
.venv311\Scripts\python.exe main.py
```

Linux:

```bash
git clone https://github.com/DrLavier/UnitPort.git
cd UnitPort

chmod +x install.sh start.sh
./install.sh
./start.sh
```

## Environment Model

### Current Strategy: project-local `.venv311` (Python 3.11.9)

- The repository currently uses the built-in `.venv311\` virtual environment.
- `.venv311\pyvenv.cfg` identifies the environment version as Python `3.11.9`.
- On Windows, `install.bat` recreates or repairs `.venv311\` and installs
  dependencies there. It does not install project packages into the global
  Python environment.
- On Linux, `install.sh` creates or repairs `.venv311/` and launches through
  `start.sh`, which also sets Linux-specific runtime environment variables.
- Launch on Windows with `.venv311\Scripts\python.exe main.py`.
- Launch on Linux with `./start.sh`.
- If `runtime\python\python.exe` or `runtime/python/python` is present
  (packaged runtime, see below), the platform install/start scripts prefer it
  automatically over `.venv311`.

### Runtime requirements

| Component | Required | Notes |
|---|---|---|
| Python 3.11 | Only when rebuilding env | Used by `install.bat` or `install.sh` if `.venv311` must be recreated |
| `.venv311\` | Yes | Project-local Python 3.11.9 environment used to run the app |
| `runtime\cyclonedds\` | Optional | Needed for Unitree SDK hardware integration |

### Packaged runtime (future / maintainers)

Run `scripts\build_runtime.bat` to download a self-contained Python 3.11 embeddable
runtime and CycloneDDS into `runtime\`. Once present, the platform install/start flow
uses it instead of the system Python — no global Python required at all.

See `scripts\README.txt` for the full maintainer build pipeline.

---

## Architecture Overview

UnitPort is built on a **4-layer design system** with **3 interaction layers**:

### Design Layers (Backend)

```
┌─────────────────────────────────────────────────────────────────┐
│  Mission Layer    │ Task orchestration & flow composition       │
├─────────────────────────────────────────────────────────────────┤
│  Behavior Layer   │ Action logic, state machines & strategies   │
├─────────────────────────────────────────────────────────────────┤
│  Service Layer    │ Vendor SDK adaptation & capability mapping  │
├─────────────────────────────────────────────────────────────────┤
│  Runtime Layer    │ Execution scheduling, monitoring & Safety   │
│   └─ Safety       │ Compile/pre-exec/exec/post-exec intercept  │
└─────────────────────────────────────────────────────────────────┘
```

### Interaction Layers (Frontend)

```
[Canvas]     Visual task builder (node-based programming)
   ↓
[Compiler]   Python behavior scripting (parameters, logic, plugins)
   ↓
[Scenario]   Execution config (sim/real, safety rules, connection)
   ↓
[Runtime] → [Service] → [Robot SDK]
```

### Key Design Principles

| Layer | Principle | Responsibility |
|-------|-----------|----------------|
| **Mission** | Describes "what to do", not "how" | Project-level task flow orchestration |
| **Behavior** | Describes "how to do", encapsulated as reusable nodes | Node-internal logic, state machines, sensor feedback |
| **Service** | Unified interface abstraction over vendor SDKs | Protocol translation, capability mapping, SDK calls |
| **Runtime** | Event-driven, observable, interruptible with Safety | Task scheduling, resource arbitration, safety interception |

---

## Project Structure

```
UnitPort/
├── main.py                 # Application entry point
├── compiler/               # DSL compiler pipeline (parser/lowering/codegen)
├── config/                 # Configuration files
│   ├── system.ini         # System settings
│   ├── user.ini           # User preferences
│   └── ui.ini             # UI style configuration
├── localisation/          # i18n translation files
│   └── en.json            # English translations
├── bin/
│   ├── ui.py             # Main window interface
│   ├── compiler/         # Compiler-side UI bindings
│   ├── core/             # Framework core (config, logging, theme, i18n)
│   │   ├── robot_context.py   # RobotContext (global state manager)
│   │   └── README.md           # Core framework documentation
│   ├── scenario/         # Scenario-related UI panels
│   └── components/       # Shared UI components (graph editor, code editor)
│       └── README.md           # UI component documentation
├── nodes/                 # Node system
│   ├── sys_nodes/        # Built-in system nodes (do not modify)
│   └── README.md         # Node design documentation
├── custom_nodes/          # Community/user custom nodes
├── system/                # Backend system layers
│   ├── mission/          # Mission orchestration
│   ├── behavior/         # Behavior logic layer
│   ├── service/          # Vendor adapter/service layer
│   ├── runtime/          # Runtime execution/safety layer
│   ├── ir/              # Unified IR entry
│   │   ├── workflow_ir.py
│   │   ├── README.md
│   │   └── workbench/   # IR development workspace
│   │       ├── IR_DEVELOPER_MANUAL.md
│   │       ├── code/
│   │       ├── templates/
│   │       ├── readable/
│   │       └── tmp/
│   ├── telemetry/       # Event/telemetry contracts
│   └── types/           # Shared types/error codes
└── models/               # Robot integration layer
    ├── base.py           # BaseRobotModel (abstract interface)
    ├── Unitree/          # Unitree robot support (Go2/A1/B1)
    ├── BostionDynamics/  # Boston Dynamics vendor assets/SDK mirrors
    ├── XiaoMi/           # Xiaomi CyberDog assets/SDK mirrors
    └── README.md         # Robot integration documentation
```

---

## Features

### Visual Programming (Canvas)
- Drag-and-drop node-based task composition
- Real-time graph visualization
- Task flow validation and error checking

### Behavior Scripting (Compiler)
- Python-based behavior definition
- Parameter templates and fine-tuning
- Plugin/agent integration (LLM, sensors, custom logic)

### Scenario Management
- Simulation/real robot switching
- Safety protocol configuration
- Environment and connection setup
- Reproducible execution profiles

### Multi-Robot Support
- **Current**: Unitree Go2, A1, B1
- **Architecture**: Extensible to Boston Dynamics, ANYbotics, and more
- **RobotContext Pattern**: Hot-swappable robot models without code changes

### Safety System (Runtime-Embedded)
- **Compile-time**: Parameter boundary checks, capability validation
- **Pre-execution**: Environment and connection verification
- **Runtime**: Threshold monitoring, resource conflict detection, timeout handling
- **Post-execution**: Graceful degradation, rollback, emergency stop, audit logs

### MuJoCo Simulation
- Physics-accurate robot simulation
- Sensor feedback emulation
- Sim-to-real transfer validation

### Internationalization
- Multi-language support (English, Chinese, more)
- Easy translation contribution workflow

---

## Multi-Robot Support Architecture

UnitPort uses a **RobotContext pattern** for vendor-agnostic design:

```python
# In UI: User selects robot
RobotContext.set_robot_type("go2")

# In Action Nodes: Generic execution
RobotContext.run_action('stand')  # Automatically routed to correct SDK

# RobotContext handles:
# - Brand mapping: "go2" → "unitree"
# - Model factory: Creates UnitreeModel("go2")
# - SDK adaptation: Translates to Unitree SDK calls
```

### Adding New Robot Brands

See [models/README.md](models/README.md) for detailed instructions on adding support for new robot brands.

---

## End-to-End Workflow

1. **Build Mission**: Use Canvas to compose task flow with nodes
2. **Configure Behavior**: Define node-internal logic via Canvas + Compiler
3. **Set Scenario**: Configure sim/real target, safety rules, connection params
4. **Execute**: Runtime schedules tasks with Safety interception
5. **Adapt**: Service layer translates to vendor SDK
6. **Monitor**: Runtime provides unified status feedback and error handling

---

## Configuration

| File | Purpose |
|------|---------|
| `config/system.ini` | System paths, simulation parameters |
| `config/user.ini` | User preferences (theme, language) |
| `config/ui.ini` | UI styling (fonts, colors, layout) |

---

## Internationalization (i18n)

All user-facing text uses the localisation system:

```python
from bin.core.localisation import tr

# Usage
message = tr("status.ready", "Ready")
```

**Contributing translations**: See [localisation/README.md](localisation/README.md)

---

## Extension Development

### Adding Custom Nodes
See [custom_nodes/README.md](custom_nodes/README.md) for node creation guidelines.

### Adding Robot Support
See [models/README.md](models/README.md) for robot integration instructions.

### Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Follow architecture principles in [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
4. Submit a pull request

---

## Tech Stack

- **GUI Framework**: PySide6
- **Simulation**: MuJoCo 3.0+
- **Robot SDKs**: Unitree SDK 2, (extensible to others)
- **Language**: Python 3.11.9

---

## Documentation

| Topic | Location |
|-------|----------|
| Architecture Overview | [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) |
| Framework Core | [bin/core/README.md](bin/core/README.md) |
| UI Components | [bin/components/README.md](bin/components/README.md) |
| Node System | [nodes/README.md](nodes/README.md) |
| IR Workbench | [system/ir/workbench/IR_DEVELOPER_MANUAL.md](system/ir/workbench/IR_DEVELOPER_MANUAL.md) |
| IR Template (v0.1) | [system/ir/workbench/templates/ir_template_v0_1.yaml](system/ir/workbench/templates/ir_template_v0_1.yaml) |
| Robot Integration | [models/README.md](models/README.md) |
| Internationalization | [localisation/README.md](localisation/README.md) |

---

## Design Principles (Must Follow)

- **Single Semantic Source**: Canvas and Compiler converge to unified task semantics
- **Loose Coupling**: Mission/Behavior never directly call vendor SDKs
- **Hot-Swappable**: Service adapters are pluggable without Runtime changes
- **Auditable**: All interceptions, exceptions, rollbacks are traceable
- **Sim-to-Real**: Same task validates in simulation before real execution

---

## Community & Support

- **Website**: [uniport.ai](https://uniport.ai)
- **Repository**: [DrLavier/UnitPort](https://github.com/DrLavier/UnitPort)
- **Issues**: [GitHub Issues](https://github.com/DrLavier/UnitPort/issues)
- **Discussions**: [GitHub Discussions](https://github.com/DrLavier/UnitPort/discussions)

---

## License

No top-level `LICENSE` file is present in this workspace snapshot. Add an explicit
license file before publishing redistribution terms in this README.

