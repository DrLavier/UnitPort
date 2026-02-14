# Canvas Layer

`frontend/canvas` hosts mission-graph interaction modules.

Current status:
- `graph_scene.py` and `graph_view.py` bridge legacy canvas components.
- `node_palette.py` bridges legacy module palette.

Migration note:
- Keep UI behavior stable while runtime logic moves into `design/runtime`.
