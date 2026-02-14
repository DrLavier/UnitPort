# Unified IR

`shared/ir` re-exports canonical IR types from `compiler/ir/workflow_ir.py`.

All new modules should import IR from `shared.ir` to keep a single semantic entry path.
