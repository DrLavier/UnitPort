#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BindingOutput — the standardized backend operation contract (Step 1).

BindingOutput is the single structured object that crosses the boundary between
the semantic Binding layer and the execution Executor layer.  It carries
everything an executor needs to perform a backend operation without knowing
anything about semantic intent or IR structure.

Design constraints
------------------
- Frozen dataclass: immutable after construction; safe to cache and share.
- No SDK imports; no MuJoCo imports; no Qt imports.
- `backend` must be exactly "sdk" or "mujoco".
- Serialisable to plain dict via `.to_dict()`.

Supported operations
--------------------
- "velocity_move"  -- linear + angular velocity command (first supported op)
"""

from __future__ import annotations

import copy
import types
from dataclasses import dataclass
from typing import Any, Dict, Mapping

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BACKEND_SDK = "sdk"
BACKEND_MUJOCO = "mujoco"
_VALID_BACKENDS = frozenset({BACKEND_SDK, BACKEND_MUJOCO})

OPERATION_VELOCITY_MOVE = "velocity_move"


# ---------------------------------------------------------------------------
# BindingOutput
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BindingOutput:
    """Structured backend operation produced by a Binding.

    Attributes
    ----------
    operation:
        Backend operation name.  Currently only ``"velocity_move"`` is
        defined.  Future operations extend this set without touching IR code.
    params:
        Backend-ready parameters as a plain dict.  Content is operation-
        specific; for ``velocity_move`` the keys are ``vx``, ``vy``,
        ``vyaw``, and ``duration``.  Stored as an immutable copy.
    intent_id:
        The originating semantic intent identifier, e.g.
        ``"locomotion.forward"``.  Preserved for diagnostics and tracing.
    brand:
        Robot brand, e.g. ``"unitree"``.
    robot_type:
        Model within the brand, e.g. ``"go2"`` or ``"go2w"``.
    backend:
        Execution backend target.  Must be exactly ``"sdk"`` or ``"mujoco"``.
    """

    operation: str
    params: Mapping[str, Any]
    intent_id: str
    brand: str
    robot_type: str
    backend: str

    # ------------------------------------------------------------------
    # Post-init validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        if self.backend not in _VALID_BACKENDS:
            raise ValueError(
                f"BindingOutput.backend must be 'sdk' or 'mujoco', got {self.backend!r}"
            )
        if not self.operation:
            raise ValueError("BindingOutput.operation must be a non-empty string")
        if not self.intent_id:
            raise ValueError("BindingOutput.intent_id must be a non-empty string")
        # Replace params with a MappingProxyType wrapping a deep copy.
        # This gives true immutability: the external dict reference cannot
        # mutate stored values (deep copy), and the proxy rejects item
        # assignment at runtime (MappingProxyType).  We must go through
        # object.__setattr__ because frozen=True blocks normal assignment.
        object.__setattr__(
            self,
            "params",
            types.MappingProxyType(copy.deepcopy(dict(self.params))),
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe plain-dict representation.

        params is returned as a regular dict (copy) so callers can serialise
        it to JSON without special handling for MappingProxyType.
        """
        return {
            "operation": self.operation,
            "params": dict(self.params),
            "intent_id": self.intent_id,
            "brand": self.brand,
            "robot_type": self.robot_type,
            "backend": self.backend,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BindingOutput":
        """Reconstruct a BindingOutput from a plain dict (e.g. after JSON load)."""
        return cls(
            operation=str(data.get("operation", "")),
            params=dict(data.get("params") or {}),
            intent_id=str(data.get("intent_id", "")),
            brand=str(data.get("brand", "")),
            robot_type=str(data.get("robot_type", "")),
            backend=str(data.get("backend", "")),
        )

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def velocity_move(
        cls,
        *,
        vx: float,
        vy: float,
        vyaw: float,
        duration: float,
        intent_id: str,
        brand: str,
        robot_type: str,
        backend: str,
    ) -> "BindingOutput":
        """Convenience constructor for a velocity_move BindingOutput."""
        return cls(
            operation=OPERATION_VELOCITY_MOVE,
            params={
                "vx": float(vx),
                "vy": float(vy),
                "vyaw": float(vyaw),
                "duration": float(duration),
            },
            intent_id=intent_id,
            brand=brand,
            robot_type=robot_type,
            backend=backend,
        )
