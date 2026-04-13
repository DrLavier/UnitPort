"""Mission-side tools that consume the P1–P4 runtime layer.

P5 ships :mod:`robustness_matrix` — the Bundle Robustness Matrix
described in knowledge_base/mission_design.yaml §8.
"""

from .robustness_matrix import (
    EpisodeStats,
    FieldRow,
    RobustnessMatrix,
    RobustnessReport,
    idle_runner,
    policy_runner_factory,
)
from .field_thumbnail import (
    clear_cache as clear_thumbnail_cache,
    field_thumbnail_qimage,
    render_field_thumbnail,
)

__all__ = [
    "EpisodeStats",
    "FieldRow",
    "RobustnessMatrix",
    "RobustnessReport",
    "idle_runner",
    "policy_runner_factory",
    "render_field_thumbnail",
    "field_thumbnail_qimage",
    "clear_thumbnail_cache",
]
