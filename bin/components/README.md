# UI Design Module

User interface design and implementation including main window, graph editor, code editor, and module palette.

## Responsibilities

Build visual programming interface, handle user interactions, display node graphs and generated code.

## Files

```
bin/
├── ui.py                      # Main window
└── components/
    ├── graph_scene.py         # Graph editor scene (core)
    ├── graph_view.py          # Graph editor view
    ├── code_editor.py         # Code editor
    ├── module_cards.py        # Module card palette
    └── __init__.py

config/
└── ui.ini                     # UI style configuration
```

## Layout Structure

```
┌─────────────────────────────────────────────────────────────────┐
│  Toolbar (Robot selection, New, Open, Save, Export, Run, Lang) │
├────────┬────────┬───────────────────────┬───────────────────────┤
│        │        │                       │                       │
│  Log   │ Module │      Graph Editor     │     Code Editor       │
│ Panel  │ Palette│       (Canvas)        │   (Auto-generated)    │
│ 300px  │ 280px  │       720px           │       400px           │
│        │        │                       │                       │
├────────┴────────┴───────────────────────┴───────────────────────┤
│                          Status Bar                              │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### MainWindow (`ui.py`)

Main window class managing overall layout and toolbar.

```python
from bin.ui import MainWindow

window = MainWindow()
window.show()
```

**Features**:
- Toolbar: Robot selection, file operations, run control, language switch
- Status bar: Current status display
- Left log panel
- Center graph editor
- Right code display

### GraphScene (`graph_scene.py`)

Graph editor scene managing nodes and connections.

```python
from bin.components.graph_scene import GraphScene

scene = GraphScene()
scene.set_code_editor(code_editor)
scene.set_robot_type('go2')
```

**Main interfaces**:
- `create_node(name, pos, features, gradient)` - Create node
- `regenerate_code()` - Regenerate code
- `set_robot_type(robot_type)` - Set robot type

**Node structure**:
```
Node Item (QGraphicsRectItem)
├── Title area
├── Input ports (left circles)
├── Output ports (right circles)
└── Parameter area
```

**Connection (ConnectionItem)**:
- Bezier curve connection
- Endpoint drag reconnection
- Selection highlight effect

### GraphView (`graph_view.py`)

Graph editor view handling zoom, pan, and drag-drop.

```python
from bin.components.graph_view import GraphView

view = GraphView()
view.setScene(scene)
```

**Interactions**:
- Mouse wheel zoom
- Middle button pan
- Drag from module palette to create nodes

### CodeEditor (`code_editor.py`)

Code display editor showing auto-generated Python code.

```python
from bin.components.code_editor import CodeEditor

editor = CodeEditor()
editor.set_code("# Generated code\nrobot.stand()")
code = editor.get_code()
```

**Main interfaces**:
- `set_code(code)` - Set code content
- `get_code()` - Get code content
- `append_code(code)` - Append code

### ModulePalette (`module_cards.py`)

Node library panel with collapsible groups (ComfyUI-style), supporting drag and double-click.

```python
from bin.components.module_cards import ModulePalette

palette = ModulePalette()
```

**Node library groups**:
- System Nodes
  - Action Nodes
  - Base Nodes
  - Logic Nodes
  - Sensor Nodes
- Custom Nodes

## Interaction Flow

```
User drags node from library
       ↓
GraphView.dropEvent() receives drop
       ↓
GraphScene.create_node() creates node
       ↓
User connects node ports
       ↓
GraphScene._create_connection() creates connection
       ↓
GraphScene.regenerate_code() generates code
       ↓
CodeEditor.set_code() updates display
```

## Style Configuration (ui.ini)

### Font Configuration

```ini
[Font]
family = Arial
size_mini = 9
size_small = 10
size_normal = 12
size_large = 14
```

### Node Colors

```ini
[NodeColors]
action_start = #4CAF50
action_end = #2E7D32
logic_start = #2196F3
logic_end = #1565C0
sensor_start = #FF9800
sensor_end = #E65100
```

### Theme Colors

```ini
[Light]
bg = #f5f5f5
card_bg = #ffffff
text_primary = #212121

[Dark]
bg = #1e1e1e
card_bg = #2d2d2d
text_primary = #ffffff
```

## Development Guidelines

1. **Theme adaptation**: Use `get_color()` for colors, don't hardcode
2. **Signal communication**: UI updates via Qt signals for thread safety
3. **Responsive layout**: Use QSplitter for adjustable layouts
4. **Node rendering**: Use QGraphicsScene/View framework
5. **Localisation**: Use `tr()` for all user-facing text

## Node Layout Contract (Mandatory)

The following rules are mandatory when developing or extending node UIs in `graph_scene.py`:

1. **Fixed node-content padding to border**
   - Node content area must keep fixed margins from node borders.
   - Use the fixed constants in `GraphScene.create_node()`:
     - `_NODE_PADDING_X`
     - `_NODE_PADDING_Y`
     - `_NODE_PADDING_RIGHT`
     - `_NODE_PADDING_BOTTOM`
   - Do not use ad-hoc per-node border spacing.

2. **Row-based composition only**
   - Add node content strictly as rows through `NodeRowStack.add_row(...)`.
   - Do not place free-floating widgets in node content.
   - Input/output ports must align to the target row center via `_port_y(...)` or widget center mapping (`_widget_center_in_node(...)` for embedded square zones such as `ConditionSet.left_box`).

3. **Fixed row height and fixed row spacing**
   - Row spacing is fixed by `_ROW_SPACING`.
   - Row height is fixed by `_ROW_HEIGHT` and enforced by `NodeRowStack(row_height=...)`.
   - New row widgets must be compatible with this fixed-height row system.

4. **Dynamic row visibility must use collapse API**
   - For mode-dependent rows (for example Wait/Gate/Cancel), do not use only `row.setVisible(False)`.
   - Always route visibility through `NodeRowStack.set_row_collapsed(...)` (or the wrapper `_set_row_visible_compact(...)` in `GraphScene.create_node()`).
   - Reason: plain hide/show can leave phantom spacing in `QVBoxLayout`, causing abnormal row gaps and content overflow.

5. **Documentation compliance**
   - Any new node type or node UI pattern must follow this contract.
   - If a special case is needed, document the reason and implementation details in this file before merging.

6. **Port dot-kind compatibility (hard rule)**
   - `main_dot` can connect only to `main_dot`.
   - `sub_dot` can connect only to `sub_dot`.
   - Do not allow mixed `main_dot -> sub_dot` or `sub_dot -> main_dot` links, even when channel/data types match.

7. **sub_dot type color + type-safe wiring**
   - `sub_dot` border color must represent `data_type` (for example `bool/string/int/float/function/...`).
   - Connection rule for `sub_dot`: same `data_type` only.
   - "Same color can connect, different color cannot" is mandatory for data ports.

8. **sub_dot side placement rule (hard rule)**
   - `sub_dot` input ports (`io == in`) must be placed on the left side of node rows.
   - `sub_dot` output ports (`io == out`) must be placed on the right side of node rows.
   - Do not place input `sub_dot` on the right or output `sub_dot` on the left.

## Behavior Node Protocol sub_dot Contract (Step 5 — schema v1.4)

### condition port type

The Behavior node's `condition` input sub_dot has `data_type="protocol"` (since
schema v1.4).  Only output ports also typed `"protocol"` can connect to it at
interaction time.  The canvas enforces this via `_can_connect_ports()`.

### Border color states (status communication, not decoration)

| State | Color | Meaning |
|-------|-------|---------|
| `protocol_none` | grey `#6b7280` | No connection on condition port / legacy mode |
| `protocol_valid` | green `#22c55e` | Connected port is `protocol`-typed **and** runtime confirmed valid payload |
| `protocol_invalid` | amber `#f59e0b` | Type mismatch **or** runtime `protocol_status ∈ {invalid, stale}` or `reason=INVALID_PROTOCOL` |

Runtime result is authoritative: `apply_behavior_protocol_states_from_run_result(run_result)`
is called after every mission run and overrides the design-time state.
`refresh_style()` preserves the active `_protocol_state` so theme changes do not
reset the border.

### Migration compat for files saved before schema v1.4

Old mission files (schema < 1.4) may contain Behavior `condition` connections
where the source port was typed `"bool"` (pre-Step-4 contract).  These load
successfully via a migration-compat gate (`_loading_workflow=True` skips the
type check for this slot only).  The node renders with a `protocol_invalid`
border to prompt the user to rewire the connection to a `"protocol"`-typed
output.  See `bin/core/mission_persistence.py → migrate_mission_payload()` for
the programmatic migration-info API.
