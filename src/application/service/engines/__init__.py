# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Engines service — thin facade over registers.backends + per-engine user state.

Public entry points:

    from application.service.engines import get_engine_service

    svc = get_engine_service()
    for eid in svc.list_known_engines():
        st = svc.status(eid)             # {available, enabled, version, path}
        local = svc.get_local(eid)       # {enabled, root, registered}
        servers = svc.list_servers(eid)
"""

from __future__ import annotations

from .service import (
    CLOUD_TOKEN,
    LOCAL_NONE,
    LOCAL_PREFIX,
    EngineService,
    build_local_target_options,
    format_isaac_install_label,
    get_engine_service,
    link_kind_for_data,
)

__all__ = [
    "EngineService",
    "get_engine_service",
    "build_local_target_options",
    "format_isaac_install_label",
    "link_kind_for_data",
    "LOCAL_PREFIX",
    "LOCAL_NONE",
    "CLOUD_TOKEN",
]
