# UnitPort - Robot Visual Programming Platform

A PySide6-based visual robot control system supporting graphical programming and MuJoCo simulation.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run application
python main.py
```

## Project Structure

```
UnitPort/
├── main.py                 # Entry point (config path definitions)
├── config/                 # Configuration files
│   ├── system.ini         # System config
│   ├── user.ini           # User preferences
│   └── ui.ini             # UI style config
├── localisation/          # Translation files (i18n)
│   └── en.json            # English translations
├── bin/                   # UI and core logic
│   ├── ui.py             # Main window
│   ├── core/             # Framework (→ bin/core/README.md)
│   └── components/       # UI components (→ bin/components/README.md)
├── nodes/                 # Node registry (→ nodes/README.md)
├── nodes/sys_nodes/      # Built-in system nodes
├── custom_nodes/         # Community/user custom nodes
└── models/               # Robot integration (→ models/README.md)
```

## Department Documentation

| Module | Path | Responsibilities |
|--------|------|------------------|
| Framework | [bin/core/README.md](bin/core/README.md) | Config, logging, theme, localisation |
| UI Design | [bin/components/README.md](bin/components/README.md) | Main window, graph editor, code editor |
| Node Design | [nodes/README.md](nodes/README.md) | Node base class, action/logic/sensor nodes |
| Robot Integration | [models/README.md](models/README.md) | Robot models, Unitree integration, simulation |

## Features

- Visual node-based programming
- Auto code generation
- MuJoCo simulation support
- Unitree Go2/A1/B1 robot support
- Light/Dark theme switching
- Localisation support (i18n)

## Configuration Files

| File | Description |
|------|-------------|
| `config/system.ini` | System config (paths, simulation params) |
| `config/user.ini` | User preferences (theme) |
| `config/ui.ini` | UI style (fonts, colors) |

## Localisation

All user-facing text should use the localisation system:

```python
from bin.core.localisation import tr

# In code
message = tr("status.ready", "Ready")
```

Translation files are in `localisation/` directory. See [localisation/README.md](localisation/README.md) for details.

**Important**: When adding new features, always use `tr()` for user-facing text to maintain i18n compatibility.

## Node System

Nodes are organized into two categories:

- **nodes/sys_nodes/**: Built-in system nodes (do not modify)
- **custom_nodes/**: Community and user-defined nodes

See [custom_nodes/README.md](custom_nodes/README.md) for creating custom nodes.

## Extension Development

**Adding new nodes**: See [custom_nodes/README.md](custom_nodes/README.md)

**Adding new robots**: See [models/README.md](models/README.md)

## Tech Stack

- GUI: PySide6
- Simulation: MuJoCo 3.0+
- Robot SDK: Unitree SDK 2

## Architecture: Multi-Robot Support

### Critical Design Rule: RobotContext

UnitPort uses a **RobotContext** pattern to support multiple robot brands while keeping action nodes generic. This is a critical architectural decision.

```
┌─────────────────────────────────────────────────────────────────┐
│                         UI Layer                                │
│   ┌─────────────┐                                               │
│   │ robot_combo │ ──► RobotContext.set_robot_type("go2")       │
│   └─────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    bin/core/robot_context.py                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ RobotContext (Global State Manager)                       │  │
│  │  - _current_robot_type: str                               │  │
│  │  - _current_robot_model: BaseRobotModel                   │  │
│  │  - ROBOT_BRAND_MAP: {"go2": "unitree", "a1": "unitree"}  │  │
│  │                                                           │  │
│  │  + set_robot_type(type) → updates global state            │  │
│  │  + get_robot_model() → returns correct model instance     │  │
│  │  + run_action(name) → delegates to model                  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Model Layer                               │
│  models/                                                        │
│  ├── base.py           # BaseRobotModel (abstract)             │
│  ├── unitree/                                                   │
│  │   └── unitree_model.py  # UnitreeModel(go2/a1/b1/...)       │
│  ├── boston_dynamics/      # (future)                          │
│  │   └── bd_model.py       # BDModel(spot/...)                 │
│  └── ...                                                        │
└─────────────────────────────────────────────────────────────────┘
```

### How It Works

1. **UI Selection**: When user selects a robot in `robot_combo`, it calls:
   ```python
   RobotContext.set_robot_type("go2")
   ```

2. **Brand Routing**: RobotContext maps robot type to brand:
   ```python
   ROBOT_BRAND_MAP = {
       "go2": "unitree",
       "a1": "unitree",
       "spot": "boston_dynamics",  # future
   }
   ```

3. **Model Factory**: Creates the correct model instance:
   ```python
   def _create_model_for_brand(brand, robot_type):
       if brand == "unitree":
           from models.unitree import UnitreeModel
           return UnitreeModel(robot_type)
       elif brand == "boston_dynamics":
           from models.boston_dynamics import BDModel
           return BDModel(robot_type)
   ```

4. **Action Nodes**: Use RobotContext, NOT direct model imports:
   ```python
   # CORRECT - in action_nodes.py
   from bin.core.robot_context import RobotContext
   RobotContext.run_action('stand')

   # WRONG - do NOT do this
   from models.unitree import UnitreeModel
   model = UnitreeModel('go2')
   model.run_action('stand')
   ```

### Adding New Robot Brand

To add support for a new robot brand:

1. **Create model directory**:
   ```
   models/
   └── new_brand/
       ├── __init__.py
       └── new_brand_model.py
   ```

2. **Implement BaseRobotModel**:
   ```python
   # models/new_brand/new_brand_model.py
   from models.base import BaseRobotModel

   class NewBrandModel(BaseRobotModel):
       def run_action(self, action_name, **kwargs):
           # Brand-specific implementation
           pass
   ```

3. **Register in RobotContext** (`bin/core/robot_context.py`):
   ```python
   ROBOT_BRAND_MAP = {
       # ... existing entries
       "new_robot_type": "new_brand",
   }

   def _create_model_for_brand(cls, brand, robot_type):
       # ... existing code
       elif brand == "new_brand":
           from models.new_brand import NewBrandModel
           return NewBrandModel(robot_type)
   ```

4. **Register in models/__init__.py**:
   ```python
   try:
       from .new_brand import NewBrandModel
       register_model("new_brand", NewBrandModel)
   except ImportError:
       pass
   ```

### Key Principles

| Principle | Description |
|-----------|-------------|
| **Single Source of Truth** | `RobotContext` is THE global robot state manager |
| **Lazy Initialization** | Robot model created only when first needed |
| **Brand Abstraction** | Action nodes never import brand-specific code |
| **Hot-Swappable** | Changing robot type updates model automatically |
| **Fail-Safe** | Missing SDK gracefully falls back to simulation mode |
