#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""UnitPort handshake probe — active side of the bidirectional handshake.

Runs as a long-lived daemon under ``unitport-handshake.service``. Its job:

    1. For the first ``BOOT_WINDOW_SEC`` seconds after service start, poll
       the host's hello endpoint every ``FAST_INTERVAL_SEC`` seconds. This
       gives the user a ~instant switch when they boot the robot with the
       cable already plugged into a running UnitPort host.

    2. After the boot window, fall back to ``SLOW_INTERVAL_SEC`` forever.
       Still-listening-in-background mode — a host that starts later still
       gets picked up within a few seconds.

When a poll succeeds, the robot self-switches into UnitPort mode via
``systemctl isolate unitport.target`` (unless already in UnitPort mode, in
which case the poll is just a liveness heartbeat and no action is taken).

Host side: ``HostHelloServer`` (Python on the host running inside the
UnitPort desktop app) binds to ``192.168.55.2:9998`` and answers
``GET /unitport-hello`` with a small JSON body. 192.168.55.2 is the
static lease our dnsmasq pins to the host's RNDIS/ECM MAC.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

HOST_HELLO_URL = os.environ.get(
    "UNITPORT_HOST_HELLO_URL",
    "http://192.168.55.2:9998/unitport-hello",
)
MODE_FILE = "/etc/unitport/mode"
BOOT_WINDOW_SEC = float(os.environ.get("UNITPORT_HANDSHAKE_BOOT_WINDOW_SEC", "10.0"))
FAST_INTERVAL_SEC = float(os.environ.get("UNITPORT_HANDSHAKE_FAST_INTERVAL_SEC", "1.0"))
SLOW_INTERVAL_SEC = float(os.environ.get("UNITPORT_HANDSHAKE_SLOW_INTERVAL_SEC", "5.0"))
PROBE_TIMEOUT_SEC = float(os.environ.get("UNITPORT_HANDSHAKE_PROBE_TIMEOUT_SEC", "1.0"))


def _log(msg: str) -> None:
    sys.stderr.write(f"[unitport-handshake] {msg}\n")
    sys.stderr.flush()


def _read_mode() -> str:
    try:
        with open(MODE_FILE, "r", encoding="utf-8") as fh:
            return fh.read().strip() or "unknown"
    except OSError:
        return "unknown"


def _write_mode(mode: str) -> None:
    try:
        os.makedirs(os.path.dirname(MODE_FILE), exist_ok=True)
        with open(MODE_FILE, "w", encoding="utf-8") as fh:
            fh.write(mode.strip() + "\n")
    except OSError as exc:
        _log(f"mode marker write failed: {exc}")


def _probe_host() -> bool:
    try:
        req = urllib.request.Request(HOST_HELLO_URL, method="GET")
        with urllib.request.urlopen(req, timeout=PROBE_TIMEOUT_SEC) as resp:
            if getattr(resp, "status", 200) != 200:
                return False
            raw = resp.read(4096)
            # We don't require any particular body — presence of 200 is enough,
            # but try to parse for logging.
            try:
                body = json.loads(raw.decode("utf-8", errors="replace"))
                if isinstance(body, dict) and body.get("accepted"):
                    return True
                # Older host versions may not echo `accepted`; treat any
                # well-formed JSON dict as success.
                return isinstance(body, dict)
            except ValueError:
                # Non-JSON reply still counts as "host reachable".
                return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _switch_to_unitport() -> bool:
    """Additively start unitport.target. Idempotent and non-destructive.

    HISTORY: older versions of this helper used ``systemctl isolate
    unitport.target`` which Conflicted with default.target and
    transiently took down sshd / Flask (:8080) / the vendor ROS2
    bringup — breaking the Robot Control Panel. See
    memory/architecture_additive_not_exclusive.md. We now use
    ``systemctl start`` so default.target keeps running underneath.
    """
    try:
        proc = subprocess.run(
            ["systemctl", "start", "unitport.target"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode == 0:
            _write_mode("on")
            _log("started unitport.target — UnitPort overlay ACTIVE alongside vendor stack")
            return True
        _log(
            "systemctl start unitport.target failed: "
            f"{(proc.stderr or proc.stdout).strip()[:200]}"
        )
        return False
    except subprocess.TimeoutExpired:
        _log("systemctl start unitport.target timed out after 30s")
        return False
    except Exception as exc:  # noqa: BLE001
        _log(f"systemctl start unitport.target raised: {exc}")
        return False


def main() -> int:
    _log(
        f"probing {HOST_HELLO_URL} every {FAST_INTERVAL_SEC:.1f}s for "
        f"{BOOT_WINDOW_SEC:.0f}s, then every {SLOW_INTERVAL_SEC:.1f}s"
    )
    start = time.monotonic()
    in_boot_window = True
    last_seen_ok = False
    while True:
        now = time.monotonic()
        if in_boot_window and (now - start) > BOOT_WINDOW_SEC:
            in_boot_window = False
            if not last_seen_ok:
                _log(
                    f"no host reply within {BOOT_WINDOW_SEC:.0f}s — staying in "
                    "factory mode, switching to background polling"
                )

        ok = _probe_host()
        if ok:
            if not last_seen_ok:
                _log("host hello reachable")
            last_seen_ok = True
            # Only self-switch when we're not already in UnitPort mode.
            # Saves a pointless isolate on every poll during a live session.
            if _read_mode() != "on":
                _switch_to_unitport()
        else:
            if last_seen_ok:
                _log("host hello unreachable")
            last_seen_ok = False

        time.sleep(FAST_INTERVAL_SEC if in_boot_window else SLOW_INTERVAL_SEC)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
