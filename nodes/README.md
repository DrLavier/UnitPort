# Node System

Unified node registry that loads nodes from both `sys_nodes/` (built-in) and `custom_nodes/` (community).

## Directory Structure

```
nodes/                    # Unified registry
├── __init__.py          # Registry loading sys_nodes + custom_nodes
└── README.md            # This file

sys_nodes/               # Built-in system nodes (DO NOT MODIFY)
├── __init__.py
├── base_node.py         # Node base class
├── action_nodes.py      # Action nodes
├── logic_nodes.py       # Logic nodes
└── sensor_nodes.py      # Sensor nodes

custom_nodes/            # Community/user custom nodes
├── __init__.py          # Custom node registry
└── README.md            # Custom node guide
```

## Node Types

### System Nodes (sys_nodes/)

Built-in nodes that ship with the application:

| Type | Class | Description |
|------|-------|-------------|
| `action_execution` | ActionExecutionNode | Execute robot actions |
| `stop` | StopNode | Stop robot motion |
| `if` | IfNode | Conditional branch |
| `while_loop` | WhileLoopNode | Loop execution |
| `comparison` | ComparisonNode | Compare values |
| `sensor_input` | SensorInputNode | Read sensor data |

### Custom Nodes (custom_nodes/)

Community and user-defined nodes. See [custom_nodes/README.md](../custom_nodes/README.md).

## Usage

### Import Nodes

```python
from nodes import (
    BaseNode,
    ActionExecutionNode,
    IfNode,
    SensorInputNode
)
```

### Registry Functions

```python
from nodes import (
    register_node,
    get_node_class,
    create_node,
    list_node_types,
    list_system_nodes,
    list_custom_nodes
)

# Register a node
register_node("my_node", MyNodeClass)

# Get node class
NodeClass = get_node_class("action_execution")

# Create node instance
node = create_node("action_execution", "node_1")

# List all nodes
all_types = list_node_types()

# List only system nodes
sys_types = list_system_nodes()

# List only custom nodes
custom_types = list_custom_nodes()
```

## Creating Custom Nodes

For creating custom nodes, see [custom_nodes/README.md](../custom_nodes/README.md).

## BaseNode Interface

All nodes must inherit from `BaseNode` and implement:

```python
class MyNode(BaseNode):
    def __init__(self, node_id: str):
        super().__init__(node_id, "my_node")
        self.inputs = {'in': None}
        self.outputs = {'out': None}
        self.parameters = {}

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute node logic"""
        return {'out': result}

    def get_display_name(self) -> str:
        """Return display name"""
        return "My Node"

    def get_description(self) -> str:
        """Return description"""
        return "Node description"

    def to_code(self) -> str:
        """Generate Python code"""
        return "# Generated code\n"
```

## Development Guidelines

1. **System nodes**: Do not modify files in `sys_nodes/` - they are maintained by the core team
2. **Custom nodes**: Add your nodes to `custom_nodes/` directory
3. **Unique types**: Node type identifiers must be unique across all nodes
4. **Error handling**: Handle errors gracefully in `execute()`
5. **Code generation**: Generate valid, runnable Python code in `to_code()`
