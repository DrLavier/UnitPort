"""Motion pipeline — phase_2 of AMP_design.yaml.

Receive, validate, and 3D-preview user-supplied reference motion files.
This package provides the backend infrastructure for the AMP consumption
path (discriminator transition sampling) and coexists with the existing
``unitree_gym_env._init_reference_motion`` tracking path.

Per AMP_design.yaml §3.reference_motion.architectural_principle:
**Additive extension** — there is no new ``MotionSourceNode`` or
``MotionDatasetSpec``. The existing ``ReferenceMotionNode`` (phase_2
gains 4 new parameters) and ``ReferenceMotionConfig`` (gains 4 new
optional fields) remain the single entry point.

See ``knowledge_base/AMP_design.yaml`` §3 for the contract this
package implements.
"""
from __future__ import annotations

from src.system.training.motion.clip import (
    MotionClip,
    MotionClipError,
)
from src.system.training.motion.loader import (
    MotionLoader,
    NpyLoader,
    AMPLegGymLoader,
    get_loader,
    LoaderError,
)
from src.system.training.motion.validator import (
    MotionValidationError,
    validate_clip,
)
from src.system.training.motion.preview_renderer import (
    FramePose,
    MotionPreviewBundle,
    build_preview_bundle,
)

__all__ = [
    # clip
    "MotionClip",
    "MotionClipError",
    # loader
    "MotionLoader",
    "NpyLoader",
    "AMPLegGymLoader",
    "get_loader",
    "LoaderError",
    # validator
    "MotionValidationError",
    "validate_clip",
    # preview
    "FramePose",
    "MotionPreviewBundle",
    "build_preview_bundle",
]
