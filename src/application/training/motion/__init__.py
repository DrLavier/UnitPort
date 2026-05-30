# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""application.training.motion — Reference-motion pipeline.

Receive, validate, and surface reference motion clips for the AMP-PPO
consumer and the tracking reward path. Two file formats supported out
of the box: ``unitport_npy`` (legacy joint-pos trajectories) and
``amp_legged_gym`` (DeepMimic-style 61-float frames).

Public surface:
    * :class:`MotionClip` / :exc:`MotionClipError` — numeric payload.
    * :class:`ReferenceMotionContract` — schema wrapping a clip with
      ``consumption_mode`` + ``source_info`` + optional target binding.
    * :class:`MotionLoader` / :data:`LOADER_REGISTRY` /
      :func:`register_loader` / :func:`list_loader_formats` /
      :func:`get_loader` — loader registration + dispatch.
    * :func:`validate_clip` and :func:`validate_clip_against_robot`
      (clip-side structural / cross-robot checks).
    * :func:`validate_contract` (contract-side schema checks).
    * Filename → task-tag inference helpers.
"""
from __future__ import annotations

from application.training.motion.clip import MotionClip, MotionClipError
from application.training.motion.contract import (
    ALLOWED_CONSUMPTION_MODES,
    CONTRACT_SCHEMA_VERSION,
    H0_CONSUMPTION_MODES,
    ReferenceMotionContract,
    ReferenceMotionContractError,
    SourceInfo,
    validate_contract,
)
from application.training.motion.labels import (
    infer_task_tag,
    list_available_tags,
    load_manifest,
    load_manifest_weights,
    resolve_task_tags,
)
from application.training.motion.loaders import (
    AMPLegGymLoader,
    LOADER_REGISTRY,
    LoaderError,
    MotionLoader,
    NpyLoader,
    get_loader,
    list_loader_formats,
    mirror_motion_clip,
    register_loader,
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
    # contract
    "ALLOWED_CONSUMPTION_MODES",
    "CONTRACT_SCHEMA_VERSION",
    "H0_CONSUMPTION_MODES",
    "ReferenceMotionContract",
    "ReferenceMotionContractError",
    "SourceInfo",
    "validate_contract",
    # loader
    "MotionLoader",
    "NpyLoader",
    "AMPLegGymLoader",
    "LOADER_REGISTRY",
    "register_loader",
    "list_loader_formats",
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
