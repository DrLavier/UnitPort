# Unified IR

`system/ir` re-exports canonical IR types from `compiler/ir/workflow_ir.py`.

All new modules should import IR from `system.ir` to keep a single semantic entry path.

## Layered IR Scaffolding

The layered contracts for Mission/Behavior/Command/Safety/Observability/Execution/Subgraph live in:

- `system/ir/layered_contracts.py`
- `system/ir/layered_interfaces.py`

For frontend integration details, see:

- `system/ir/FRONTEND_UI_INTERFACE_CHECKLIST.txt`

## IR Workbench

IR design and transpilation iteration files live in `system/ir/workbench/`, including:

- `code/`: IR authoring/transpilation code (experimental and production-bound helpers).
- `readable/`: human-readable generated IR files for inspection/review.
- `templates/`: canonical readable IR templates.
- `tmp/`: temporary files and scratch outputs.
- `IR_DEVELOPER_MANUAL.md`: onboarding and conventions for IR agents/developers.


