# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""BridgeProtocol — interface for ROS2 bridge implementations.

In RELEASE Phase 2 only :class:`NativeDDSBridge` implements this Protocol;
a docker / subprocess variant may join later for debugging. ``BaseROS2Adapter``
(Phase 5) will hold ``self._client: Optional[BridgeProtocol]`` so the dispatch
between native and any future implementation is mechanical (no conditionals).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class BridgeProtocol(Protocol):
    """Shared surface for ROS2 bridge implementations."""

    def start(self) -> None:
        """Open the underlying transport (DDS participant or subprocess).

        MUST NOT block longer than ~2s. Raises on hard failure (no return-False
        soft fail).
        """
        ...

    def stop(self, timeout: float = 5.0) -> None:
        """Tear down the transport. Idempotent."""
        ...

    def is_alive(self) -> bool:
        """True iff the transport is currently open."""
        ...

    def on_error(self, cb: Callable[[str], None]) -> None:
        """Register a callback invoked on async transport errors.

        The callback MUST be thread-safe; for native DDS it fires from a
        cyclonedds reader thread.
        """
        ...

    def subscribe(
        self,
        topic: str,
        msg_type: str,
        qos: Optional[Dict[str, Any]] = None,
        timeout: float = 3.0,
    ) -> bool:
        """Subscribe to ``topic`` with the given ``msg_type`` and QoS dict.

        QoS dict keys: reliability / history / depth / durability / deadline /
        liveliness.
        """
        ...

    def publish(
        self,
        topic: str,
        msg_type: str,
        payload: Dict[str, Any],
        qos: Optional[Dict[str, Any]] = None,
        timeout: float = 1.0,
        wait_ack: bool = False,
    ) -> bool:
        """Publish ``payload`` (dict matching msg_type IDL fields) on ``topic``."""
        ...

    def ping(self, timeout: float = 2.0) -> bool:
        """Check if the bridge is reachable. Native mirrors ``is_alive()``."""
        ...

    def latest(self, topic: str) -> Optional[Dict[str, Any]]:
        """Most recent payload received on ``topic``, as a plain dict."""
        ...

    def rate_hz(self, topic: str) -> float:
        """Sliding-window publish rate (Hz) for ``topic``. 0.0 if unsubscribed."""
        ...

    def subscription_stats(self) -> Dict[str, Dict[str, float]]:
        """Snapshot of per-topic metrics: ``{topic: {rate_hz, last_ts}}``."""
        ...

    def start_recording(
        self, topics: List[str], out_dir: str, timeout: float = 5.0
    ) -> bool:
        """Begin recording to a rosbag-compatible file. Optional capability."""
        ...

    def stop_recording(self, timeout: float = 5.0) -> bool:
        """Stop an active recording and flush the output."""
        ...
