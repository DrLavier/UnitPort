"""Scene presets — sub-folder aggregator.

The single source of truth lives in :mod:`scripts.scenes.registry`,
which holds the ``Scene`` dataclass and the process-local ``_REGISTRY``.
Each built-in scene is a separate file under ``builtin/`` exporting an
``ENTRY: Scene`` constant; the registry imports them at module load.
"""

from scripts.scenes.registry import (
    Scene,
    SceneValidationError,
    clear_registry,
    get_scene,
    has_scene,
    list_review_scenes,
    list_scenes,
    register_scene,
    rescan,
)


__all__ = [
    "Scene",
    "SceneValidationError",
    "register_scene",
    "get_scene",
    "has_scene",
    "list_scenes",
    "list_review_scenes",
    "clear_registry",
    "rescan",
]
