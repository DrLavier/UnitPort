# Compiler Layer

`frontend/compiler` provides source authoring and validation entry points.

Current status:
- `code_editor.py` bridges legacy editor UI.
- `dsl_support.py` parses DSL source.
- `lint_bridge.py` runs parse + lower + semantic checks.
