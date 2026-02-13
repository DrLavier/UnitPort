# Robot Integration Module

Adaptation and integration for different robot brands including base class, brand implementations, and simulation integration.

## Responsibilities

Implement unified abstraction interface for robot hardware/simulation, support hot-pluggable multi-brand robot extensions.

## Files

```
models/
├── base.py                      # Robot model abstract base class
├── __init__.py                  # Model registry center
└── unitree/                     # Unitree brand implementation
    ├── unitree_model.py         # Unitree control logic
    ├── __init__.py
    ├── unitree_mujoco/          # MuJoCo simulation package
    │   ├── unitree_robots/      # Robot model files
    │   │   ├── go2/scene.xml
    │   │   ├── a1/scene.xml
    │   │   ├── b1/scene.xml
    │   │   └── b2/
    │   ├── simulate_python/     # Python simulation interface
    │   └── example/             # Example code
    └── unitree_sdk2_python/     # Unitree SDK
        ├── unitree_sdk2py/      # SDK core library
        └── example/             # SDK examples

bin/core/
└── simulation_thread.py         # Simulation thread (shared with Framework)

config/
└── system.ini                   # Path and simulation configuration
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│              Application Layer (UI/Nodes)        │
├─────────────────────────────────────────────────┤
│          Unified Interface (BaseRobotModel)      │
├──────────────┬──────────────┬───────────────────┤
│  Unitree     │  Other Brand │   Future          │
│  Model       │  Model       │   Extensions      │
├──────────────┼──────────────┼───────────────────┤
│ unitree_sdk  │ other_sdk    │                   │
│ mujoco_sim   │ gazebo_sim   │                   │
└──────────────┴──────────────┴───────────────────┘
```

## Core Modules

### BaseRobotModel (`base.py`)

Robot model abstract base class defining unified interface.

```python
from models.base import BaseRobotModel

class BaseRobotModel(ABC):
    def __init__(self, robot_type: str):
        self.robot_type = robot_type

    @abstractmethod
    def initialize(self) -> bool: ...

    @abstractmethod
    def load_model(self) -> bool: ...

    @abstractmethod
    def run_action(self, action_name: str, **kwargs) -> bool: ...

    @abstractmethod
    def get_available_actions(self) -> List[str]: ...

    @abstractmethod
    def get_sensor_data(self) -> Dict[str, Any]: ...

    @abstractmethod
    def stop(self): ...
```

**Required methods**:

| Method | Description |
|--------|-------------|
| `initialize()` | Initialize robot/simulation environment |
| `load_model()` | Load robot model |
| `run_action(name, **kwargs)` | Execute specified action |
| `get_available_actions()` | Return available action list |
| `get_sensor_data()` | Return sensor data dictionary |
| `stop()` | Stop all actions |

### Model Registry (`__init__.py`)

Manages all robot model registration and retrieval.

```python
from models import register_model, get_model, list_models

# Register model
register_model("unitree", UnitreeModel)

# Get model instance
model = get_model("go2")  # Returns UnitreeModel instance

# List all models
models = list_models()  # ['go2', 'a1', 'b1', ...]
```

### UnitreeModel (`unitree/unitree_model.py`)

Unitree robot specific implementation.

```python
from models.unitree import UnitreeModel

model = UnitreeModel("go2")
model.initialize()
model.load_model()
model.run_action("stand")
data = model.get_sensor_data()
model.stop()
```

**Supported robot types**:
- `go2` - Unitree Go2 quadruped robot
- `a1` - Unitree A1
- `b1` - Unitree B1
- `b2` - Unitree B2

**Supported actions**:

| Action | Method | Description |
|--------|--------|-------------|
| stand | `run_action("stand")` | Stand |
| sit | `run_action("sit")` | Sit |
| walk | `run_action("walk", speed=0.5)` | Walk |
| lift_right_leg | `run_action("lift_right_leg")` | Lift right leg |

**Sensor data**:

```python
{
    'imu': {
        'orientation': [x, y, z, w],
        'angular_velocity': [wx, wy, wz],
        'linear_acceleration': [ax, ay, az]
    },
    'joint_states': {
        'position': [...],
        'velocity': [...],
        'effort': [...]
    },
    'foot_contact': [True, False, True, True],
    'odometry': {
        'position': [x, y, z],
        'velocity': [vx, vy, vz]
    }
}
```

## MuJoCo Simulation Integration

### Model File Structure

```
unitree_mujoco/unitree_robots/
├── go2/
│   ├── scene.xml          # Scene configuration
│   ├── go2.xml            # Robot model
│   └── meshes/            # Mesh files
├── a1/
│   └── ...
└── b1/
    └── ...
```

### Simulation Parameters

```ini
# config/system.ini
[MUJOCO]
gl_backend = glfw
timestep = 0.002
keep_window_time = 5.0
```

## Adding New Robot Brands

### Step 1: Create Directory Structure

```
models/
└── your_brand/
    ├── __init__.py
    ├── your_robot_model.py
    └── sdk/                  # Optional: Brand SDK
```

### Step 2: Implement Model Class

```python
# models/your_brand/your_robot_model.py
from models.base import BaseRobotModel

class YourRobotModel(BaseRobotModel):
    def __init__(self, robot_type: str):
        super().__init__(robot_type)
        self.sdk = None

    def initialize(self) -> bool:
        # Initialize SDK or simulation environment
        return True

    def load_model(self) -> bool:
        # Load robot model
        return True

    def run_action(self, action_name: str, **kwargs) -> bool:
        actions = {
            'stand': self._stand,
            'walk': self._walk,
        }
        if action_name in actions:
            return actions[action_name](**kwargs)
        return False

    def get_available_actions(self) -> List[str]:
        return ['stand', 'walk', 'sit']

    def get_sensor_data(self) -> Dict[str, Any]:
        return {
            'imu': {...},
            'joint_states': {...}
        }

    def stop(self):
        # Stop actions
        pass
```

### Step 3: Register Model

```python
# models/__init__.py
from .your_brand import YourRobotModel

register_model("your_robot_type", YourRobotModel)
```

### Step 4: Configure Paths

```ini
# config/system.ini
[PATH]
your_brand_sdk = ./models/your_brand/sdk
your_brand_models = ./models/your_brand/models

[SIMULATION]
available_robots = go2,a1,b1,your_robot_type
```

## Simulation Thread Mechanism

```
MainWindow (UI thread)
       │
       └─► SimulationThread (Worker thread)
           │
           ├─ simulation_started ──┐
           ├─ progress_updated  ───┼─► UI update
           ├─ simulation_finished ─┤
           └─ error_occurred ──────┘

           Execution:
           1. Receive action and robot_model
           2. Call robot_model.run_action()
           3. Emit signals to notify UI
```

## Development Guidelines

1. **Inherit base class**: All robot models must inherit `BaseRobotModel`
2. **Unified interface**: Implement all abstract methods, maintain interface consistency
3. **Thread safety**: Action execution in separate thread, notify UI via signals
4. **Error handling**: Catch exceptions, return meaningful error messages
5. **Resource management**: Release all resources in `stop()`
