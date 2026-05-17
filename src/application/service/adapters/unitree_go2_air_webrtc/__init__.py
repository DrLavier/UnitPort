"""Unitree Go2 Air host-side adapter package.

Unitree does not expose a DDS surface for the Go2 Air SKU; the only
field-viable host link is the RoboVerse ``go2_webrtc`` library shipped
as a brand SDK under
``custom_mods/runtime/sdk_extensions/Unitree/go2_webrtc/`` and a thin
:class:`Go2WebRTCBridge` core wrapper. This package wires the two
together as a :class:`BaseAdapter` so :class:`AdapterFactory` can
resolve it from ``robots_canonical.json``.

Module-load registers the brand's topic table with :mod:`topic_registry`
so future P6-style "what topics does this brand stream?" consumers can
query without per-brand import knowledge.

Public API: :class:`Go2AirWebRTCAdapter` -- instantiated only via
:class:`AdapterFactory`; never construct directly.
"""

from __future__ import annotations

from application.service.adapters.topic_registry import register_brand
from application.service.adapters.unitree_go2_air_webrtc.adapter import (
    Go2AirWebRTCAdapter,
)
from application.service.adapters.unitree_go2_air_webrtc.topic_table import ALL_SPECS


# Register the brand's topic table once at import time. Idempotent --
# the registry replaces same-(brand, topic, role) entries on duplicate
# calls, so reloading the package during development is safe.
register_brand(Go2AirWebRTCAdapter.BRAND_ID, list(ALL_SPECS))


__all__ = ["Go2AirWebRTCAdapter"]
