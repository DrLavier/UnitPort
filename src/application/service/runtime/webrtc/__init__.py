# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""application.service.runtime.webrtc — WebRTC data-channel bridges.

Sibling of :mod:`application.service.runtime.ros2`. Hosts in-process
bridges that wrap vendor-specific WebRTC clients (e.g. Unitree Go2 sport
API over the RoboVerse ``go2-webrtc`` library) and expose a synchronous
publish/subscribe surface to the rest of RELEASE.

The vendor library itself is treated as a brand SDK: cloned by
:class:`SdkManager` into
``custom_mods/runtime/sdk_extensions/<Brand>/<project>/`` and installed
into the project venv. Bridge code in this package is RELEASE core — it
imports the SDK as a regular Python package.
"""

from application.service.runtime.webrtc.go2_webrtc_bridge import Go2WebRTCBridge

__all__ = ["Go2WebRTCBridge"]
