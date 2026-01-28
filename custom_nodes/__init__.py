#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom Nodes Module
Community and user-defined custom nodes

HOW TO ADD CUSTOM NODES:
========================

1. Create a new Python file in this directory (e.g., my_nodes.py)

2. Import the BaseNode class:
   from sys_nodes.base_node import BaseNode

3. Define your custom node class:

   class MyCustomNode(BaseNode):
       def __init__(self, node_id: str):
           super().__init__(node_id, "my_custom")
           self.inputs = {'in': None}
           self.outputs = {'out': None}
           self.parameters = {'my_param': 'default'}

       def execute(self, inputs):
           # Your logic here
           return {'out': {'result': 'value'}}

       def get_display_name(self):
           return "My Custom Node"

       def get_description(self):
           return "Description of my custom node"

       def to_code(self):
           return "# Generated code\\n"

4. Register your node in this __init__.py file:

   from .my_nodes import MyCustomNode
   CUSTOM_NODES['my_custom'] = MyCustomNode

5. Restart the application - your node will be available!

GUIDELINES:
===========
- Node type names must be unique
- Use descriptive display names
- Implement all abstract methods from BaseNode
- Handle errors gracefully in execute()
- Generate valid Python code in to_code()
"""

from typing import Dict, Type

# Import base node for custom node development
from sys_nodes.base_node import BaseNode

# Registry for custom nodes
# Format: {'node_type': NodeClass}
CUSTOM_NODES: Dict[str, Type[BaseNode]] = {}

# ============================================================================
# REGISTER YOUR CUSTOM NODES BELOW
# ============================================================================

# Example:
# from .my_nodes import MyCustomNode
# CUSTOM_NODES['my_custom'] = MyCustomNode

# ============================================================================
# AUTO-DISCOVERY (Optional - for advanced users)
# ============================================================================

def discover_custom_nodes():
    """
    Auto-discover custom node files in this directory.
    Called by the node registry during initialization.
    """
    import os
    import importlib
    from pathlib import Path

    custom_dir = Path(__file__).parent

    for file in custom_dir.glob("*.py"):
        if file.name.startswith("_"):
            continue

        module_name = file.stem
        try:
            module = importlib.import_module(f".{module_name}", package="custom_nodes")

            # Look for classes that inherit from BaseNode
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and
                    issubclass(attr, BaseNode) and
                    attr is not BaseNode):
                    # Register using the node_type from an instance
                    try:
                        temp_instance = attr("temp")
                        node_type = temp_instance.node_type
                        if node_type not in CUSTOM_NODES:
                            CUSTOM_NODES[node_type] = attr
                    except:
                        pass
        except Exception:
            pass


def get_custom_nodes() -> Dict[str, Type[BaseNode]]:
    """Get all registered custom nodes"""
    return CUSTOM_NODES.copy()


__all__ = [
    'BaseNode',
    'CUSTOM_NODES',
    'discover_custom_nodes',
    'get_custom_nodes'
]
