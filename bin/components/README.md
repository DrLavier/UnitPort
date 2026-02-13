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
