# Framework Module

Core infrastructure including configuration management, data persistence, logging system, theme management, and localisation.

## Responsibilities

Provides global services for UI, node system, and robot integration modules.

## Files

```
bin/core/
├── config_manager.py    # Configuration manager
├── data_manager.py      # Data manager (thread-safe)
├── logger.py            # Logging system (UI integrated)
├── theme_manager.py     # Theme manager
├── localisation.py      # Localisation manager (i18n)
├── node_executor.py     # Node execution engine
├── simulation_thread.py # Simulation thread base class
└── __init__.py

config/
├── system.ini           # System configuration
├── user.ini             # User preferences
└── ui.ini               # UI style configuration

localisation/
├── en.json              # English translations
└── README.md            # Localisation guide

main.py                  # Application entry point
utils/
└── logger.py            # File logging utility
```

## Core Modules

### ConfigManager (`config_manager.py`)

Singleton configuration manager for system.ini and user.ini.

```python
from bin.core.config_manager import ConfigManager

config = ConfigManager()
robot = config.get('SIMULATION', 'default_robot', fallback='go2')
config.set('PREFERENCES', 'theme', 'dark', config_type='user')
```

**Main interfaces**:
- `get(section, option, fallback, config_type)` - Read config
- `get_int/float/bool()` - Typed read
- `set(section, option, value, config_type)` - Write config
- `get_path(path_key)` - Get path (auto convert to absolute)

### DataManager (`data_manager.py`)

Thread-safe data manager with INI/JSON support and caching.

```python
from bin.core.data_manager import load_ini, get_ini_value, up_ini

load_ini('config/system.ini')
robot = get_ini_value('config/system.ini', 'SIMULATION', 'default_robot')
up_ini('config/user.ini', 'PREFERENCES', 'theme', 'dark')
```

**Features**:
- File-level locking for concurrent access
- Auto-caching to reduce IO
- `force_reload` for cache refresh

### Logger (`logger.py`)

Qt signal-based logging system with thread-safe UI output.

```python
from bin.core.logger import log_info, log_error, log_success, log_warning

log_info("Application started")
log_success("Operation complete")
log_error("Error occurred")
log_info("Loading...", typer=True)  # Typewriter effect
```

**Log levels**:
- `log_debug()` - Debug info
- `log_info()` - General info
- `log_warning()` - Warning
- `log_error()` - Error
- `log_success()` - Success

### LocalisationManager (`localisation.py`)

Internationalisation support with JSON translation files.

```python
from bin.core.localisation import tr, tr_list, get_localisation

# Get translated text
text = tr("toolbar.new", "New")

# Get translated list
features = tr_list("modules.logic_control.features")

# With format arguments
message = tr("status.ready", "Ready | Robot: {robot}", robot="go2")

# Change language
loc = get_localisation()
loc.load_language("en")
```

**Features**:
- JSON-based translation files
- Format string support
- Language hot-switching
- Automatic fallback to default language

### ThemeManager (`theme_manager.py`)

Theme management with Light/Dark switching.

```python
from bin.core.theme_manager import set_theme, get_color, get_font_size

set_theme('dark')
bg = get_color('bg')
font_size = get_font_size('size_normal')
```

### NodeExecutor (`node_executor.py`)

Node execution engine with topological sorting and code generation.

```python
from bin.core.node_executor import NodeExecutor

executor = NodeExecutor()
executor.add_node(node_id, node_type, node_data)
executor.add_connection(from_node, from_port, to_node, to_port)
executor.build_execution_order()
code = executor.to_code()
```

### SimulationThread (`simulation_thread.py`)

Simulation thread base class with Qt signals for UI notification.

**Signals**:
- `simulation_started` - Simulation started
- `progress_updated` - Progress update
- `simulation_finished` - Simulation finished
- `error_occurred` - Error occurred

## Configuration Files

### system.ini

| Section | Key | Description |
|---------|-----|-------------|
| PATH | project_root | Project root directory |
| PATH | unitree_sdk | Unitree SDK path |
| PATH | unitree_mujoco | MuJoCo simulation path |
| SIMULATION | default_robot | Default robot type |
| SIMULATION | available_robots | Available robot list |
| MUJOCO | timestep | Simulation timestep |
| UI | window_width/height | Window dimensions |
| DEBUG | debug_mode | Debug mode switch |

### user.ini

| Section | Key | Description |
|---------|-----|-------------|
| PREFERENCES | theme | Theme (light/dark) |
| RECENT | last_project | Last project path |

### ui.ini

| Section | Key | Description |
|---------|-----|-------------|
| Font | family, size_* | Font configuration |
| NodeColors | *_start, *_end | Node gradient colors |
| Light/Dark | bg, text_* etc | Theme colors |

## Development Guidelines

1. **Singleton pattern**: ConfigManager, DataManager, LogSignal are singletons
2. **Thread safety**: Use Qt signals or file locks for cross-thread operations
3. **Path handling**: Use `get_path()` for auto relative/absolute conversion
4. **Logging**: Use `log_*` functions instead of `print()`
5. **Localisation**: Use `tr()` for all user-facing text
