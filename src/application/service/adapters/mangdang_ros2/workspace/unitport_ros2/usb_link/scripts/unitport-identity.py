#!/usr/bin/env python3
"""UnitPort robot-side identity + handshake service.

Authoritative identity record lives at ``/etc/unitport/identity.yaml`` on
the robot. This script serves it over HTTP at 192.168.55.1:9999 and
writes it during Start Deploy via the ``--probe`` subcommand.

Routes::

    GET  /identity    → YAML / JSON identity record (read-only)
    POST /handshake   → additive switch: ``systemctl start unitport.target``
    POST /disconnect  → additive switch: ``systemctl stop unitport.target``
    POST /upgrade     → reserved (workspace upgrade hook, not yet wired here)

CLI subcommands::

    unitport-identity                # serve HTTP (default; same as previous)
    unitport-identity --probe        # run minimal hardware probe → identity.yaml
    unitport-identity --print        # dump the current identity.yaml

Stdlib only — no pip deps so factory images stay lean. YAML is emitted
by hand because PyYAML isn't on every base image; a tiny line-based
emitter is sufficient for the flat IdentityRecord schema.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional

ROBOT_ID_FILE = "/etc/unitport/robot_id"          # legacy flat-KV (back-compat)
IDENTITY_FILE = "/etc/unitport/identity.yaml"     # authoritative record (single source of truth)
MODE_FILE = "/etc/unitport/mode"
# Static daemon identity for HTTP Server: header. The actual UnitPort
# version and workspace fingerprint live in identity.yaml on disk and are
# served from there — they are NOT module constants. Host-side bootstrap
# writes them via `unitport-identity --probe --unitport-version=X
# --workspace-fingerprint=Y`.
DAEMON_VERSION = "1.0"
BIND_HOST = "192.168.55.1"
BIND_PORT = 9999

_MAX_POST_BYTES = 8 * 1024


# ---------------------------------------------------------------------------
# Identity record IO
# ---------------------------------------------------------------------------


def _read_identity_yaml() -> Dict[str, Any]:
    """Return the identity record as a dict, or {} if missing/unreadable.

    Stdlib YAML parser would be nice but isn't shipped — instead we accept
    JSON-style flat YAML (key: value, single-line lists) which is what
    ``--probe`` writes. PyYAML on the host parses the same file fine.
    """
    try:
        with open(IDENTITY_FILE, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return {}
    try:
        # Files we write are JSON-superset YAML; try JSON first.
        return json.loads(text)
    except ValueError:
        pass
    return _parse_flat_yaml(text)


def _parse_flat_yaml(text: str) -> Dict[str, Any]:
    """Parse a one-level-nesting YAML subset (what ``--probe`` writes)."""
    out: Dict[str, Any] = {}
    stack: list = [(out, -1)]
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        while stack and stack[-1][1] >= indent:
            stack.pop()
        parent = stack[-1][0]
        if ":" not in line:
            continue
        key, _, value = line.lstrip().partition(":")
        key = key.strip()
        value = value.strip()
        if not value:
            child: Dict[str, Any] = {}
            parent[key] = child
            stack.append((child, indent))
            continue
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            parent[key] = (
                [v.strip().strip("'\"") for v in inner.split(",") if v.strip()]
                if inner else []
            )
            continue
        if value.lower() in ("true", "false"):
            parent[key] = (value.lower() == "true")
            continue
        try:
            parent[key] = int(value)
            continue
        except ValueError:
            pass
        try:
            parent[key] = float(value)
            continue
        except ValueError:
            pass
        if value == "null":
            parent[key] = None
            continue
        parent[key] = value.strip("'\"")
    return out


def _emit_yaml(value: Any, indent: int = 0) -> str:
    """Hand-rolled YAML emitter for the IdentityRecord schema (flat-ish)."""
    pad = "  " * indent
    lines: list[str] = []
    if isinstance(value, dict):
        if not value:
            return "{}"
        for k, v in value.items():
            if isinstance(v, dict):
                if not v:
                    lines.append(f"{pad}{k}: {{}}")
                else:
                    lines.append(f"{pad}{k}:")
                    lines.append(_emit_yaml(v, indent + 1))
            elif isinstance(v, list):
                if not v:
                    lines.append(f"{pad}{k}: []")
                elif all(isinstance(x, (str, int, float, bool)) for x in v):
                    inner = ", ".join(_yaml_scalar(x) for x in v)
                    lines.append(f"{pad}{k}: [{inner}]")
                else:
                    lines.append(f"{pad}{k}:")
                    for x in v:
                        sub = _emit_yaml(x, indent + 2)
                        first, _, rest = sub.partition("\n")
                        lines.append(f"{pad}  - {first.lstrip()}")
                        if rest:
                            lines.append(rest)
            else:
                lines.append(f"{pad}{k}: {_yaml_scalar(v)}")
        return "\n".join(lines)
    if isinstance(value, list):
        return ", ".join(_yaml_scalar(x) for x in value)
    return _yaml_scalar(value)


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    s = str(value)
    if s == "" or any(c in s for c in ":#[]{},&*!|>'\"%@`") or s.strip() != s:
        return json.dumps(s)
    return s


def _write_identity_yaml(record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(IDENTITY_FILE), exist_ok=True)
    text = _emit_yaml(record) + "\n"
    tmp = IDENTITY_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, IDENTITY_FILE)


# ---------------------------------------------------------------------------
# Legacy flat-KV reader (back-compat with old /etc/unitport/robot_id)
# ---------------------------------------------------------------------------


def _load_legacy_robot_id() -> Dict[str, str]:
    data = {
        "hostname": socket.gethostname(),
        "serial": "unknown",
        "platform": "unknown",
        "carrier": "unknown",
        "form_factor": "unknown",
        "transport": "usb_ethernet_gadget",
    }
    try:
        with open(ROBOT_ID_FILE, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                data[key.strip()] = value.strip()
    except OSError:
        pass
    return data


def _read_mode() -> str:
    try:
        with open(MODE_FILE, "r", encoding="utf-8") as fh:
            return fh.read().strip() or "unknown"
    except OSError:
        return "unknown"


def _write_mode(mode: str) -> None:
    try:
        with open(MODE_FILE, "w", encoding="utf-8") as fh:
            fh.write(mode.strip() + "\n")
    except OSError as exc:
        sys.stderr.write(f"[unitport-identity] mode marker write failed: {exc}\n")


# ---------------------------------------------------------------------------
# HTTP /identity payload assembly
# ---------------------------------------------------------------------------


def _identity_payload() -> Dict[str, Any]:
    """Compose the response for GET /identity.

    Single source of truth: ``identity.yaml`` on disk. The version and
    fingerprint are written there by the host's bootstrap step (via
    ``--probe --unitport-version --workspace-fingerprint``) and read here
    verbatim — no RAM-stamped module constants, no setdefault. If those
    fields are missing the host treats empty strings as "robot needs
    bootstrap", which is the correct semantic.

    Legacy fallback: when identity.yaml is absent we synthesise a record
    from the flat-KV ``robot_id`` file (back-compat with pre-2026-04-27
    images). The version + fingerprint fields are empty strings in the
    fallback path — host UI will surface this as "not provisioned".
    """
    record = _read_identity_yaml()
    if record:
        # Always-fresh runtime fields (mode flips on every handshake/disconnect).
        record["mode"] = _read_mode()
        # Ensure the two host-comparison fields exist as keys even when
        # absent so the host's payload-shape check stays stable.
        record.setdefault("unitport_version", "")
        record.setdefault("workspace_fingerprint", "")
        return record

    legacy = _load_legacy_robot_id()
    legacy["robot_id"] = legacy.get("serial", "unknown")
    legacy["unitport_version"] = ""
    legacy["workspace_fingerprint"] = ""
    legacy["mode"] = _read_mode()
    legacy["schema_version"] = 0  # 0 = legacy KV, 1+ = new identity.yaml
    return legacy


# ---------------------------------------------------------------------------
# systemctl wrapper (additive only — never isolate)
# ---------------------------------------------------------------------------


def _systemctl(action: str, target: str) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            ["systemctl", action, target],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode == 0:
            return True, ""
        return False, (proc.stderr or proc.stdout or "").strip()[:300]
    except subprocess.TimeoutExpired:
        return False, f"systemctl {action} timed out after 30s"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


class IdentityHandler(BaseHTTPRequestHandler):
    server_version = "unitport-identity/" + DAEMON_VERSION

    def _send_json(self, status: int, body: dict) -> None:
        payload = json.dumps(body, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _send_empty(self, status: int) -> None:
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _read_post_json(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        length = max(0, min(length, _MAX_POST_BYTES))
        if length == 0:
            return {}
        try:
            raw = self.rfile.read(length)
            data = json.loads(raw.decode("utf-8", errors="replace"))
            return data if isinstance(data, dict) else {}
        except (ValueError, UnicodeDecodeError):
            return {}

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.rstrip("/") or "/"
        if path == "/identity":
            self._send_json(200, _identity_payload())
            return
        self._send_empty(404)

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.rstrip("/") or "/"
        body = self._read_post_json()

        if path == "/handshake":
            prev = _read_mode()
            ok, detail = _systemctl("start", "unitport.target")
            if ok:
                _write_mode("on")
            self._send_json(200 if ok else 500, {
                "ok": ok,
                "prev_mode": prev,
                "new_mode": "on" if ok else prev,
                "robot_id": _identity_payload().get("robot_id", "unknown"),
                "session_hint": str(body.get("session_id") or ""),
                "detail": detail,
            })
            return

        if path == "/disconnect":
            prev = _read_mode()
            ok, detail = _systemctl("stop", "unitport.target")
            if ok:
                _write_mode("off")
            self._send_json(200 if ok else 500, {
                "ok": ok,
                "prev_mode": prev,
                "new_mode": "off" if ok else prev,
                "detail": detail,
            })
            return

        self._send_empty(404)

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("[unitport-identity] " + (fmt % args) + "\n")


# ---------------------------------------------------------------------------
# --probe — minimal hardware probe → identity.yaml
# ---------------------------------------------------------------------------


def _probe_robot_id() -> str:
    """UUID bound to MAC + serial fingerprint. Stable across reflashes."""
    legacy = _load_legacy_robot_id()
    serial = legacy.get("serial", "")
    if serial and serial != "unknown":
        # Deterministic UUID from serial so re-probes don't churn it.
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"unitport.robot.{serial}"))
    # No serial — use a MAC-based fingerprint.
    try:
        mac = uuid.getnode()
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"unitport.mac.{mac:012x}"))
    except Exception:
        return str(uuid.uuid4())


def _probe_board_model() -> str:
    for path in (
        "/sys/firmware/devicetree/base/model",
        "/proc/device-tree/model",
    ):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                model = fh.read().strip("\x00").strip()
                if model:
                    return model
        except OSError:
            continue
    return ""


def _probe_ros_distro() -> str:
    for candidate in ("humble", "iron", "jazzy", "rolling"):
        if os.path.isdir(f"/opt/ros/{candidate}"):
            return candidate
    return os.environ.get("ROS_DISTRO", "")


def run_probe(write: bool = True, *, brand: str = "", model: str = "",
              form_factor: str = "custom",
              extends_factory_stack: bool = False,
              unitport_version: str = "",
              workspace_fingerprint: str = "") -> Dict[str, Any]:
    """Compose a minimal IdentityRecord from local probes + caller-provided
    UnitPort metadata.

    ``unitport_version`` and ``workspace_fingerprint`` are written to the
    record's top-level fields. Host's ``bootstrap.sh`` invokes us with
    these values from the env vars set during SSH execution
    (UNITPORT_VERSION, UNITPORT_WORKSPACE_FINGERPRINT) — that flow is the
    single source of truth for the host's Upgrade/Latest-Build state.
    """
    legacy = _load_legacy_robot_id()
    record: Dict[str, Any] = {
        "schema_version": 1,
        "robot_id": _probe_robot_id(),
        "hostname": socket.gethostname(),
        "serial": legacy.get("serial") if legacy.get("serial") not in ("", "unknown") else None,
        "calibrated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "calibration_revision": int(time.time()),
        "unitport_version": unitport_version,
        "workspace_fingerprint": workspace_fingerprint,
        "vendor": {
            "brand": brand or legacy.get("platform", "") or "",
            "model": model or legacy.get("carrier", "") or "",
            "form_factor": form_factor or legacy.get("form_factor", "custom") or "custom",
            "extends_factory_stack": bool(extends_factory_stack),
        },
        "ros2": {
            "distro": _probe_ros_distro(),
            "rmw_implementation": os.environ.get("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp"),
            "domain_id_default": int(os.environ.get("ROS_DOMAIN_ID", "42") or "42"),
        },
        "network": {
            "primary_transport": legacy.get("transport", "usb_ethernet_gadget"),
            "usb_ip": BIND_HOST,
            "identity_port": BIND_PORT,
        },
        "hardware": {
            "joints": [],
            "cameras": [],
            "lidars": [],
            "imus": [],
            "controllers_available": [],
            "aux_capabilities": [],
            "board_model": _probe_board_model(),
        },
        "provenance": {
            "inspector_version": unitport_version,
            "general_build_version": "",
            "brand_overlay_version": "",
            "sources": {},
        },
    }
    if write:
        _write_identity_yaml(record)
    return record


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _serve() -> int:
    try:
        server = HTTPServer((BIND_HOST, BIND_PORT), IdentityHandler)
    except OSError as exc:
        sys.stderr.write(
            f"[unitport-identity] bind {BIND_HOST}:{BIND_PORT} failed: {exc}\n"
            "  is usb0 up with 192.168.55.1? check netplan + gadget service.\n"
        )
        return 1
    sys.stderr.write(
        f"[unitport-identity] listening on {BIND_HOST}:{BIND_PORT} "
        "(GET /identity, POST /handshake, POST /disconnect)\n"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="UnitPort robot-side identity service.",
    )
    parser.add_argument("--probe", action="store_true",
                        help="Run hardware probe and write /etc/unitport/identity.yaml")
    parser.add_argument("--print", dest="print_record", action="store_true",
                        help="Print the current identity record (YAML on stdout)")
    parser.add_argument("--brand", default="", help="Vendor brand id (probe hint)")
    parser.add_argument("--model", default="", help="Vendor model id (probe hint)")
    parser.add_argument("--form-factor", default="custom",
                        help="Form factor (probe hint)")
    parser.add_argument("--extends-factory-stack", action="store_true",
                        help="Brand keeps factory ROS2 stack alive (additive overlay)")
    parser.add_argument("--unitport-version", default=os.environ.get("UNITPORT_VERSION", ""),
                        help="UnitPort workspace version string (defaults to "
                             "$UNITPORT_VERSION; written to identity.yaml)")
    parser.add_argument("--workspace-fingerprint",
                        default=os.environ.get("UNITPORT_WORKSPACE_FINGERPRINT", ""),
                        help="UnitPort workspace content fingerprint (defaults to "
                             "$UNITPORT_WORKSPACE_FINGERPRINT; written to identity.yaml)")
    parser.add_argument("--no-write", action="store_true",
                        help="With --probe: print the record instead of writing")
    args = parser.parse_args(argv)

    if args.print_record:
        record = _read_identity_yaml() or _identity_payload()
        sys.stdout.write(_emit_yaml(record) + "\n")
        return 0

    if args.probe:
        record = run_probe(
            write=not args.no_write,
            brand=args.brand,
            model=args.model,
            form_factor=args.form_factor,
            extends_factory_stack=args.extends_factory_stack,
            unitport_version=args.unitport_version,
            workspace_fingerprint=args.workspace_fingerprint,
        )
        if args.no_write:
            sys.stdout.write(_emit_yaml(record) + "\n")
        else:
            sys.stderr.write(f"[unitport-identity] wrote {IDENTITY_FILE}\n")
        return 0

    return _serve()


if __name__ == "__main__":
    sys.exit(main())
