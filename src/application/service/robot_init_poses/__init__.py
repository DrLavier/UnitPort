# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Init pose presets — factory baselines + user-saved custom poses.

Exports the service singleton and the data models. Importers should
always go through this facade:

    from application.service.robot_init_poses import (
        get_init_pose_service, InitPoseOverride, InitPosePreset,
    )
"""

from __future__ import annotations

from .model import (
    InitPoseOverride,
    InitPosePreset,
    SOURCE_FACTORY,
    SOURCE_USER,
)
from .service import (
    InitPoseService,
    get_init_pose_service,
)

__all__ = [
    "InitPoseOverride",
    "InitPosePreset",
    "InitPoseService",
    "get_init_pose_service",
    "SOURCE_FACTORY",
    "SOURCE_USER",
]
