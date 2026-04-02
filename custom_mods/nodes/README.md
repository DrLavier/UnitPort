# Custom Nodes (Drop-in Directory)

This directory is for community and user-defined custom nodes.
**No `__init__.py` needed** — the system scans this directory automatically on startup.

## How It Works

1. Drop any `.py` file into this directory
2. The system discovers all `BaseNode` subclasses in each file
3. Each discovered node is registered by its `node_type` and becomes available in the canvas
4. Registration is recorded in `src/config/node_registry.json` under `custom.registered`

Files starting with `_` are ignored.

## Custom Node Protocol

Each custom node must:

1. Inherit from `BaseNode`
2. Set a unique `node_type` in `__init__`
3. Implement all required methods

```python
# my_nodes.py — drop this file into custom_mods/nodes/
from src.system.nodes.sys_nodes.base_node import BaseNode


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
        return f"import time\ntime.sleep({seconds})\n"
```

## Rules

- `node_type` must be unique across all nodes (system + custom)
- If a custom node type collides with a system node type, the system node wins
- Handle exceptions in `execute()` — return error status instead of raising
- Generate valid Python code in `to_code()`
- Deleting all files from this directory will NOT break the application
