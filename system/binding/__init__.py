"""system.binding — Backend Binding infrastructure for Behavior IR execution.

Package layout:
    output.py       -- BindingOutput dataclass (Step 1)
    base.py         -- ActionBinding abstract base (Step 2)
    registry.py     -- BackendBindingRegistry + default_registry singleton (Step 3)
    profiles.py     -- LocomotionProfile per-brand/model data (Step 6)
    executors/      -- OperationExecutor interfaces and implementations (Steps 4, 7)
    bindings/       -- Concrete ActionBinding implementations (Steps 3/5)

Importing this package guarantees that all built-in bindings are registered
into ``default_registry`` before any caller touches it.
"""

# Trigger import-time registration of all built-in bindings.
import system.binding.bindings  # noqa: F401
