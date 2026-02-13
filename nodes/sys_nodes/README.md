# System Node Development Guide

This directory (`nodes/sys_nodes/`) contains **built-in system nodes** used by the core runtime.
These nodes are **maintained by the core team** and are **not meant for community edits**.
If you are adding or updating nodes here, follow the¹æ·¶ below.

> Default language is English. Any user-facing strings must go through `tr()` (localisation).
> UI styles are defined in `config/ui.ini`; do not hardcode them in node logic.

---

## 1. File & Class Conventions

- Each node must inherit from `BaseNode`.
- One file can contain multiple related node classes, but keep categories clear:
  - `action_nodes.py` for action/execution nodes
  - `logic_nodes.py` for control/flow nodes
  - `sensor_nodes.py` for sensor/input nodes
- Class names should be clear and descriptive: `MoveArmNode`, `IfNode`, `SensorInputNode`.
- Node type IDs must be unique across the whole project.

---

## 2. Registration

All system nodes are registered in `nodes/__init__.py`.

Example:

```python
from nodes.sys_nodes.logic_nodes import IfNode
register_node("if", IfNode)
```

---

## 3. Required Methods

Each node must implement:

- `__init__(self, node_id: str)`
- `execute(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]`
- Optionally override:
  - `get_display_name()`
  - `get_description()`
  - `get_input_ports()`
  - `get_output_ports()`

---

## 4. Execution Contract

Nodes receive inputs based on port connections and must return a dict for outputs.
Example output:

```python
return {"result": True}
```

Keep execution **pure and deterministic** if possible. External I/O must be explicit.

---

## 5. Localisation

All user-facing text should go through localisation:

```python
from bin.core.localisation import tr

return tr("node.if.display", "If")
```

Add new translation keys in `localisation/en.json`.

---

## 6. Style Rules

- Do not hardcode UI colors or styles in node logic.
- UI layout is handled in `bin/components/graph_scene.py`.
- Node classes only define logic and metadata.

---

## 7. Example Node (Fully Commented)

Below is a fully annotated example node. This is **for reference only**.

```python
from typing import Dict, Any, List
from bin.core.localisation import tr
from .base_node import BaseNode


class ExampleThresholdNode(BaseNode):
    """
    Example node that compares an input value to a threshold.

    Inputs:
      - value: numeric input value
      - threshold: numeric threshold

    Outputs:
      - passed: True if value >= threshold
      - value: pass-through of input value
    """

    def __init__(self, node_id: str):
        # node_type must be unique across the system
        super().__init__(node_id, "example_threshold")

    def get_display_name(self) -> str:
        # UI-visible name, use localisation
        return tr("node.example_threshold.display", "Threshold")

    def get_description(self) -> str:
        # Short description for tooltips or help panels
        return tr("node.example_threshold.desc", "Compare value against a threshold")

    def get_input_ports(self) -> List[str]:
        # These names must match the ports used in graph UI / executor
        return ["value", "threshold"]

    def get_output_ports(self) -> List[str]:
        # These names must match the outputs used in executor and downstream nodes
        return ["passed", "value"]

    def execute(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute node logic.

        inputs: Values collected from upstream node outputs.
        context: Shared runtime context (robot handles, config, etc.).
        """
        # Defensive extraction with defaults
        value = inputs.get("value", 0)
        threshold = inputs.get("threshold", 0)

        try:
            # Convert inputs to float where possible
            value_f = float(value)
            threshold_f = float(threshold)
        except (ValueError, TypeError):
            # If conversion fails, return a safe output
            return {
                "passed": False,
                "value": value
            }

        # Perform the comparison
        passed = value_f >= threshold_f

        # Return outputs as a dict
        return {
            "passed": passed,
            "value": value_f
        }
```

---

## 8. Testing Checklist

- Node can be created and connected in the graph UI
- Inputs map correctly from upstream nodes
- Outputs are returned in expected shape
- No hardcoded UI text or styles
- Localisation keys exist and are valid

