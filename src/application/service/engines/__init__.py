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

from .service import EngineService, get_engine_service

__all__ = ["EngineService", "get_engine_service"]
