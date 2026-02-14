# Runtime Layer

`design/runtime` orchestrates execution, monitoring, interception, and safety.

Modules:
- `runtime_engine.py`: single execution entry.
- `workflow_runner.py`: legacy graph control-flow runner.
- `interception/*`: compile/execute guards.
- `safety/*`: policy, checker, emergency handling, audit.
