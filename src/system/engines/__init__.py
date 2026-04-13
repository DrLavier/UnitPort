"""Engine management package.

Each simulation / physics engine backend lives in its own subpackage:

    engines/
        isaac/          — NVIDIA Isaac Lab + Isaac Sim
        newton/         — (future) NVIDIA Newton physics
        ...

Public helpers re-exported here for convenience.
"""

from src.system.engines.base import ENGINES_DEFAULT_ROOT, engine_default_dir  # noqa: F401
from src.system.engines.registry import EngineRegistry, get_engine_registry  # noqa: F401
