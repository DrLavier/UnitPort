# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""topic_registry — declarative per-brand topic / msg_type / QoS table.

Each brand adapter declares its topic surface at module-load time via
:func:`register_brand`; the registry exposes :func:`list_for_brand` so:

- :class:`BaseROS2Adapter` consumes it in ``_subscribe_declared`` to wire
  every "obs" (observation/telemetry) topic on session open.
- :class:`Ros2BrownfieldConnectTask` (P6) consumes it to compute a topic
  hash for the connected brand and compare against the live ROS2 graph
  (Phase 4 inspector).
- Future "what topics does my robot expose?" UI consumers (e.g. Phase 6
  sparkline picker) consume it to list available streams without knowing
  the brand specifics.

Roles (caller-defined, free-form strings — convention only):

- ``obs``    — telemetry subscriptions (joint_states / imu / battery / ...)
- ``cmd``    — outbound publications (cmd_vel / body_pose / ...)
- ``aux``    — optional auxiliary I/O (aux_command / lcd / ...)
- ``status`` — heartbeat / status outputs

The registry is module-level state; adapters register at import time, the
connector reads at session-open time. There is no "unregister" — the
registry is append-only within a process. Tests can call
:func:`clear_caches` between cases.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class TopicSpec:
    """One declared topic surface in the adapter's topic table."""

    topic: str
    msg_type: str
    qos_profile: str       # name resolvable by qos_profiles.resolve(...)
    role: str              # "obs" / "cmd" / "aux" / "status" (free-form convention)


# Module-level state: brand_id → role → list of TopicSpecs.
_REGISTRY: Dict[str, Dict[str, List[TopicSpec]]] = {}
_LOCK = threading.RLock()


def register_brand(brand_id: str, specs: List[TopicSpec]) -> None:
    """Register every TopicSpec for the given brand_id.

    Called by each brand adapter package's ``__init__.py`` at import time.
    Duplicate (brand_id, topic) entries silently override the older entry —
    this is intentional so a brand package can hot-patch its table without
    a process restart during development.
    """
    if not brand_id:
        raise ValueError("brand_id must be non-empty")
    with _LOCK:
        per_role: Dict[str, List[TopicSpec]] = _REGISTRY.setdefault(brand_id, {})
        for spec in specs:
            bucket = per_role.setdefault(spec.role, [])
            # Replace any existing spec with the same topic in this role.
            bucket[:] = [s for s in bucket if s.topic != spec.topic]
            bucket.append(spec)


def list_for_brand(brand_id: str, role: str = "") -> List[TopicSpec]:
    """Return every TopicSpec registered for ``brand_id``.

    When ``role`` is given, only specs in that role are returned. When empty,
    the full table across roles is returned (sorted by role then topic for
    deterministic ordering — needed for P6 mapping-hash stability).
    """
    with _LOCK:
        per_role = _REGISTRY.get(brand_id, {})
        if role:
            return list(per_role.get(role, []))
        out: List[TopicSpec] = []
        for r in sorted(per_role.keys()):
            out.extend(sorted(per_role[r], key=lambda s: s.topic))
        return out


def roles_for_brand(brand_id: str) -> List[str]:
    """List every role currently registered for ``brand_id``."""
    with _LOCK:
        return sorted(_REGISTRY.get(brand_id, {}).keys())


def clear_caches() -> None:
    """Test helper — drop all brand entries."""
    with _LOCK:
        _REGISTRY.clear()


__all__ = [
    "TopicSpec",
    "register_brand",
    "list_for_brand",
    "roles_for_brand",
    "clear_caches",
]
