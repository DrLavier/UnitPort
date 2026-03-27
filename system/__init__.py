"""Lightweight package exports for the system layer.

Keep package import side effects minimal so utility modules such as
``system.model_registry`` can be imported during bootstrap without pulling in
the entire service/runtime stack and creating circular imports.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple


__all__ = [
    "MissionPlanner",
    "MissionSerializer",
    "MissionModel",
    "BehaviorModel",
    "BehaviorCompilerBridge",
    "BehaviorStateMachine",
    "ServiceRegistry",
    "ServiceRouter",
    "RuntimeEngine",
]


_EXPORT_MAP: Dict[str, Tuple[str, str]] = {
    "MissionPlanner": (".mission", "MissionPlanner"),
    "MissionSerializer": (".mission", "MissionSerializer"),
    "MissionModel": (".mission", "MissionModel"),
    "BehaviorModel": (".behavior", "BehaviorModel"),
    "BehaviorCompilerBridge": (".behavior", "BehaviorCompilerBridge"),
    "BehaviorStateMachine": (".behavior", "BehaviorStateMachine"),
    "ServiceRegistry": (".service", "ServiceRegistry"),
    "ServiceRouter": (".service", "ServiceRouter"),
    "RuntimeEngine": (".runtime", "RuntimeEngine"),
}


def __getattr__(name: str) -> Any:
    export = _EXPORT_MAP.get(name)
    if export is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = export
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
