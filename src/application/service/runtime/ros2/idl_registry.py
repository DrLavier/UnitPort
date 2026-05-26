# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""IDL type registry for NativeDDSBridge.

Resolves ROS2 msg_type strings (e.g. ``geometry_msgs/msg/Twist``) to
cyclonedds IdlStruct dataclasses.

Phase 2 simplification vs DEMO: only the **core** registry + a user-override
hook are wired. Brand sub-registries arrive in Phase 5 (with ``BaseROS2Adapter``
+ brand packages); the ``brand_id`` parameter is accepted for forward
compatibility but ignored.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Type

from application.service.runtime.ros2.idl_messages import CORE_REGISTRY
from application.service.runtime.ros2.native_dds_errors import IDLNotFound


_USER_OVERRIDE: Dict[str, Type[Any]] = {}
_LOCK = threading.RLock()


def register_user_override(msg_type: str, cls: Type[Any]) -> None:
    """Register a user-supplied override for a core message.

    Provided as a forward-compat hook for the v0.2 ``compile-msgs`` tool;
    application code should not call it directly in Phase 2.
    """
    with _LOCK:
        _USER_OVERRIDE[msg_type] = cls


def resolve(msg_type: str, brand_id: Optional[str] = None) -> Type[Any]:
    """Return the IdlStruct dataclass for ``msg_type``.

    ``brand_id`` is accepted but ignored in Phase 2. Phase 5 will dispatch to
    brand-specific registries for non-core messages.

    Raises :class:`IDLNotFound` with the searched sources in the detail dict
    if no match.
    """
    searched: List[str] = []

    with _LOCK:
        override = _USER_OVERRIDE.get(msg_type)
    if override is not None:
        return override
    searched.append("user_override")

    cls = CORE_REGISTRY.get(msg_type)
    if cls is not None:
        return cls
    searched.append("core")

    if brand_id:
        searched.append(f"brand:{brand_id} (Phase 5 — not implemented)")

    raise IDLNotFound.for_msg_type(msg_type, searched)


def list_registered(brand_id: Optional[str] = None) -> List[str]:
    """Every msg_type currently resolvable. ``brand_id`` ignored in Phase 2."""
    out = set(CORE_REGISTRY.keys())
    with _LOCK:
        out.update(_USER_OVERRIDE.keys())
    return sorted(out)


def clear_caches() -> None:
    """Test helper — drop the override cache."""
    with _LOCK:
        _USER_OVERRIDE.clear()
