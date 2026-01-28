# Custom Nodes

This directory is for community and user-defined custom nodes.

## Quick Start

1. Create a new Python file (e.g., `my_nodes.py`)
2. Define your node class inheriting from `BaseNode`
3. Register it in `__init__.py`
4. Restart the application

## Example Custom Node

```python
# my_nodes.py
from sys_nodes.base_node import BaseNode

class DelayNode(BaseNode):
    """Custom delay node"""

    def __init__(self, node_id: str):
        super().__init__(node_id, "delay")
        self.inputs = {'in': None}
        self.outputs = {'out': None}
        self.parameters = {'seconds': 1.0}

    def execute(self, inputs):
        import time
        seconds = self.get_parameter('seconds', 1.0)
        time.sleep(seconds)
        return {'out': inputs.get('in', {})}

    def get_display_name(self):
        return "Delay"

    def get_description(self):
        return "Wait for specified seconds"

    def to_code(self):
        seconds = self.get_parameter('seconds', 1.0)
        return f"import time\\ntime.sleep({seconds})\\n"
```

## Registering Your Node

In `__init__.py`:

```python
from .my_nodes import DelayNode
CUSTOM_NODES['delay'] = DelayNode
```

## Guidelines

- Node type names must be unique across all nodes
- Implement all required methods: `execute()`, `get_display_name()`, `get_description()`
- Handle exceptions in `execute()` - return error status instead of raising
- Generate valid, runnable Python code in `to_code()`
- Test your node before sharing

## Sharing Nodes

To share your custom nodes with the community:
1. Create a GitHub repository
2. Include installation instructions
3. Document node parameters and behavior
