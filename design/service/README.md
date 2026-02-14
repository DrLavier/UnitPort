# Service Layer

`design/service` is the unified routing layer between runtime and vendor SDK adapters.

Modules:
- `service_registry.py`: adapter registry.
- `service_router.py`: action/sensor routing APIs.
- `adapters/`: vendor implementations.
- `protocol/`: command/event/error contracts.
