# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Structured errors for NativeDDSBridge.

Every error carries a stable ``code`` string and a ``detail`` dict that gets
propagated verbatim into diagnostics so the GUI Runtime Console can render
actionable messages without sniffing exception strings.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional


class NativeDDSBridgeError(Exception):
    """Base class for all NativeDDSBridge hard-failures."""

    code: str = "native_dds_error"

    def __init__(self, message: str, detail: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.detail = dict(detail or {})

    def as_diagnostics(self) -> Dict[str, Any]:
        return {"code": self.code, "message": str(self), **self.detail}


class CycloneDDSLoadFailed(NativeDDSBridgeError):
    code = "cyclonedds_load_failed"

    @classmethod
    def from_import_error(cls, exc: Exception) -> "CycloneDDSLoadFailed":
        msg = (
            f"cyclonedds-python failed to load (ImportError: {exc}). "
            f"Reinstall via `.venv311\\Scripts\\pip install --force-reinstall cyclonedds`."
        )
        return cls(msg, {"import_error": str(exc)})


class IDLNotFound(NativeDDSBridgeError):
    code = "idl_not_found"

    @classmethod
    def for_msg_type(cls, msg_type: str, searched: Iterable[str]) -> "IDLNotFound":
        paths = list(searched)
        msg = (
            f"IDL definition for {msg_type!r} not found. "
            f"Searched {len(paths)} path(s). "
            f"Register it via application.service.runtime.ros2.idl_registry.register_user_override() "
            f"or extend the core idl_messages package."
        )
        return cls(msg, {"msg_type": msg_type, "searched": paths})


class IDLCompileFailed(NativeDDSBridgeError):
    code = "idl_compile_failed"


class DomainUnreachable(NativeDDSBridgeError):
    code = "domain_unreachable"

    @classmethod
    def make(
        cls,
        domain_id: int,
        robot_hostname: str,
        timeout: float,
        interfaces: Iterable[str],
    ) -> "DomainUnreachable":
        ifaces = list(interfaces)
        msg = (
            f"DDS discovery failed on domain {domain_id} within {timeout:.1f}s. "
            f"Robot '{robot_hostname or 'unknown'}' did not respond on any of "
            f"the {len(ifaces)} enumerated interface(s): {ifaces}. Check: "
            f"(a) robot is powered and ROS2 stack is running, "
            f"(b) host and robot share the same network, "
            f"(c) ROS_DOMAIN_ID matches on both sides."
        )
        return cls(
            msg,
            {
                "domain_id": domain_id,
                "robot_hostname": robot_hostname,
                "timeout": timeout,
                "interfaces": ifaces,
            },
        )


class QoSMismatch(NativeDDSBridgeError):
    code = "qos_mismatch"


class TypeCollision(NativeDDSBridgeError):
    code = "type_collision"


class WriteFailed(NativeDDSBridgeError):
    code = "write_failed"

    @classmethod
    def make(cls, topic: str, timeout: float, detail: str) -> "WriteFailed":
        msg = (
            f"DDS write on {topic} failed after {timeout:.1f}s: {detail}. "
            f"Reliable QoS may be backed up; check robot subscriber liveliness."
        )
        return cls(msg, {"topic": topic, "timeout": timeout, "detail": detail})
