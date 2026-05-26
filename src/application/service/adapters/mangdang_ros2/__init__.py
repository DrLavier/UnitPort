# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Mangdang Mini Pupper v2 host-side adapter package.

Module-load registers the brand's full topic table with
:mod:`topic_registry` so :class:`BaseROS2Adapter._subscribe_declared` can
walk the obs role at session open without any per-brand import knowledge
in the base class.

Public API: :class:`MangdangROS2Adapter` — instantiated only via
:class:`AdapterFactory`; never construct directly.
"""

from __future__ import annotations

from application.service.adapters.mangdang_ros2.adapter import MangdangROS2Adapter
from application.service.adapters.mangdang_ros2.topic_table import ALL_SPECS
from application.service.adapters.topic_registry import register_brand


# Register the brand's topic table once at import time. Idempotent — the
# registry replaces same-(brand,topic,role) entries on duplicate calls.
register_brand(MangdangROS2Adapter.BRAND_ID, ALL_SPECS)


__all__ = ["MangdangROS2Adapter"]
