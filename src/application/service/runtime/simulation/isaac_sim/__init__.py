# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Isaac Sim review subprocess wiring.

Bundle-only review path for the Export node's "Launch Review" button when
``review_backend=isaac_sim``. Mirrors the Mujoco path in shape but spawns
``il_review_launcher.py`` as an Isaac Sim subprocess (Kit cannot be loaded
in-process alongside the main UnitPort Qt app).

Per RELEASE/CLAUDE.md §1.9 (portable artifacts): the subprocess loads the
policy from the bundle's ONNX, reads physics + obs/action layout from
``bundle.manifest.deploy_contract``, and resolves the robot USD via the
SKU registry — it never reaches back into ``<project>/training/runs/``,
the canvas, or the in-memory spec. The bundle dir is the only input.
"""

from .review_session import IsaacSimReviewTask
from .pose_review_session import IsaacSimPoseReviewTask

__all__ = ["IsaacSimReviewTask", "IsaacSimPoseReviewTask"]
