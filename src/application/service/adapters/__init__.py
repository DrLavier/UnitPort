# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""application.service.adapters — Brand semantic boundary for RELEASE.

Adapters translate UnitPort core's brand-agnostic command/sensor/capability
contracts to a specific robot's wire protocol. Each brand lives in its own
subpackage (e.g. ``mangdang_ros2/``) and registers its topic table with
:mod:`topic_registry` at module load time.

Resolution: ``AdapterFactory.build(sku, strategy)`` — never instantiate
adapter classes directly outside tests; the factory handles caching, dotted-
path resolution, and structured error reporting.

CLAUDE.md §1.1 (Multi-Brand Inclusiveness) — core never mentions brand
strings; all brand-specific code lives in the per-brand subpackage. Adding
a new brand is exactly one new subpackage + one ``adapter`` field in
``registers/data/robots_canonical.json``; core code is not edited.

Public API (the only names callers should import from this package):

- :class:`BaseAdapter` — abstract contract every adapter satisfies
- :class:`BaseROS2Adapter` — concrete base for ROS2-via-DDS adapters
- :class:`AdapterFactory` — static factory: SKU + strategy → instance
- :func:`available_strategies` — UI combo population helper
- :class:`AdapterUnavailable` — typed error for connector phase failures
- :class:`CapabilitySet` + per-kind capabilities (Teleop / Estop / Arm / Pid)
- :class:`TopicSpec` + :func:`register_brand` / :func:`list_for_brand`
- :class:`TeleopPump` + :func:`zero_twist`
- :class:`HeartbeatPublisher`
"""

from __future__ import annotations

from application.service.adapters.base_adapter import (
    AdapterUnavailable,
    BaseAdapter,
)
from application.service.adapters.base_ros2_adapter import BaseROS2Adapter
from application.service.adapters.capabilities import (
    ArmCapability,
    CapabilitySet,
    EstopCapability,
    PidCapability,
    TeleopCapability,
)
from application.service.adapters.factory import AdapterFactory, available_strategies
from application.service.adapters.heartbeat_publisher import HeartbeatPublisher
from application.service.adapters.teleop_pump import TeleopPump, zero_twist
from application.service.adapters.topic_registry import (
    TopicSpec,
    list_for_brand,
    register_brand,
    roles_for_brand,
)


__all__ = [
    "AdapterFactory",
    "AdapterUnavailable",
    "ArmCapability",
    "BaseAdapter",
    "BaseROS2Adapter",
    "CapabilitySet",
    "EstopCapability",
    "HeartbeatPublisher",
    "PidCapability",
    "TeleopCapability",
    "TeleopPump",
    "TopicSpec",
    "available_strategies",
    "list_for_brand",
    "register_brand",
    "roles_for_brand",
    "zero_twist",
]
