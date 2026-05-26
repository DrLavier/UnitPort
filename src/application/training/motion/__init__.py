# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""application.training.motion — Reference-motion pipeline.

Stage 5 — receive, validate, and surface reference motion clips for the
AMP-PPO consumer (Stage 8) and the tracking reward path. Two file
formats supported out of the box: ``unitport_npy`` (legacy joint-pos
trajectories) and ``amp_legged_gym`` (DeepMimic-style 61-float frames).

Public surface:
    * :class:`MotionClip` / :exc:`MotionClipError`
    * :class:`MotionLoader` + :func:`get_loader` factory
    * :func:`validate_clip` (clip-only) and :func:`validate_clip_against_robot`
      (cross-check vs ``registers.robots.RobotSpec``)
    * Filename → task-tag inference helpers
"""
from __future__ import annotations

from application.training.motion.clip import MotionClip, MotionClipError
from application.training.motion.labels import (
    infer_task_tag,
    list_available_tags,
    load_manifest,
    load_manifest_weights,
    resolve_task_tags,
)
from application.training.motion.loader import (
    AMPLegGymLoader,
    LoaderError,
    MotionLoader,
    NpyLoader,
    get_loader,
    mirror_motion_clip,
)
from application.training.motion.validator import (
    MotionValidationError,
    validate_clip,
    validate_clip_against_robot,
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
    "mirror_motion_clip",
    # labels
    "infer_task_tag",
    "load_manifest",
    "load_manifest_weights",
    "resolve_task_tags",
    "list_available_tags",
    # validator
    "MotionValidationError",
    "validate_clip",
    "validate_clip_against_robot",
]
