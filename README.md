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
