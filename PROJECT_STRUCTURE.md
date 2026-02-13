# Project Structure

## Directory Layout

```
UnitPort/
├── main.py                          # Entry point + config path definitions
├── requirements.txt                 # Python dependencies
├── config/                          # Configuration directory
│   ├── system.ini                   # System configuration
│   ├── user.ini                     # User preferences
│   └── ui.ini                       # UI style configuration
│
├── localisation/                    # Localisation (i18n)
│   ├── en.json                      # English translations
│   └── README.md                    # Localisation guide
│
├── bin/                             # Core code
│   ├── ui.py                        # Main window
│   ├── core/                        # Framework Module ★
│   │   ├── config_manager.py        # Configuration manager
│   │   ├── data_manager.py          # Data manager
│   │   ├── logger.py                # Logging system
│   │   ├── theme_manager.py         # Theme manager
│   │   ├── localisation.py          # Localisation manager
│   │   ├── robot_context.py         # Global robot state manager (CRITICAL) ★★
│   │   ├── node_executor.py         # Node execution engine
│   │   ├── simulation_thread.py     # Simulation thread
│   │   └── README.md                # Module documentation
│   └── components/                  # UI Design Module ★
│       ├── graph_scene.py           # Graph editor scene
│       ├── graph_view.py            # Graph editor view
│       ├── code_editor.py           # Code editor
│       ├── module_cards.py          # Module cards
│       └── README.md                # Module documentation
│
├── nodes/                           # Node Registry ★
│   └── __init__.py                  # Unified registry (nodes/sys_nodes + custom_nodes)
│
├── nodes/sys_nodes/                 # System Nodes (Built-in) ★
│   ├── __init__.py
│   ├── base_node.py                 # Node base class
│   ├── action_nodes.py              # Action nodes
│   ├── logic_nodes.py               # Logic nodes
│   └── sensor_nodes.py              # Sensor nodes
│
├── custom_nodes/                    # Custom Nodes (Community) ★
│   ├── __init__.py                  # Custom node registry
│   └── README.md                    # Custom node guide
│
├── models/                          # Robot Integration ★
│   ├── base.py                      # Robot model base class
│   ├── __init__.py                  # Model registry
│   ├── README.md                    # Module documentation
│   └── unitree/                     # Unitree implementation
│       ├── unitree_model.py         # Control logic
│       ├── unitree_mujoco/          # MuJoCo simulation
│       └── unitree_sdk2_python/     # SDK
│
└── utils/                           # Utility modules
    └── logger.py                    # File logging
```

## Module Responsibilities

### Framework (`bin/core/`)

Core infrastructure: config, logging, theme, localisation, execution engine.

**Critical Component: `robot_context.py`**
- Global robot state manager (singleton pattern)
- Maps robot types to brands (go2 → unitree, etc.)
- Factory for creating brand-specific robot models
- All action nodes MUST use RobotContext, not direct model imports

### UI Design (`bin/components/`)

User interface: main window, graph editor, code editor, module palette.

### Node System (`nodes/`, `nodes/sys_nodes/`, `custom_nodes/`)

- `nodes/`: Unified registry loading from both nodes/sys_nodes and custom_nodes
- `nodes/sys_nodes/`: Built-in system nodes (action, logic, sensor)
- `custom_nodes/`: Community/user-defined custom nodes

**Important**: Action/sensor nodes use `RobotContext` for robot-agnostic execution.

### Robot Integration (`models/`)

Robot model abstraction: base class, brand-specific implementations.

**Structure**:
```
models/
├── base.py              # BaseRobotModel (abstract)
├── __init__.py          # Model registry
└── {brand}/             # Brand-specific implementation
    └── {brand}_model.py # e.g., unitree_model.py
```

**Adding new robot brand**: See README.md "Adding New Robot Brand" section.

## Config Path Management

All config file paths are defined centrally in `main.py`:

```python
# main.py
CONFIG_DIR = PROJECT_ROOT / "config"
SYSTEM_CONFIG_PATH = CONFIG_DIR / "system.ini"
USER_CONFIG_PATH = CONFIG_DIR / "user.ini"
UI_CONFIG_PATH = CONFIG_DIR / "ui.ini"
LOCALISATION_DIR = PROJECT_ROOT / "localisation"
```

Other modules should use these paths via `main.get_config_path()` or through `ConfigManager`.

## Module Dependencies

```
main.py
   │
   ├── config/           → Configuration files
   │
   ├── localisation/     → Translation files
   │
   ├── bin/core/         → Framework services
   │   └── Used by all other modules
   │
   ├── bin/components/   → UI components
   │   └── Depends on: core, nodes, models
   │
   ├── nodes/            → Node registry
   │   ├── nodes/sys_nodes/    → Built-in nodes
   │   └── custom_nodes/ → Custom nodes
   │
   └── models/           → Robot models
       └── Depends on: core
```

## Detailed Documentation

- [bin/core/README.md](bin/core/README.md) - Framework Module
- [bin/components/README.md](bin/components/README.md) - UI Design Module
- [nodes/README.md](nodes/README.md) - Node Design Module
- [models/README.md](models/README.md) - Robot Integration Module
- [localisation/README.md](localisation/README.md) - Localisation Guide
- [custom_nodes/README.md](custom_nodes/README.md) - Custom Nodes Guide
