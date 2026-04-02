#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BackendBindingRegistry — resolves ActionBindings by (intent_id, brand, robot_type, backend).

Step 3 of the binding architecture.

Fallback resolution order (strict)
-----------------------------------
1. exact match      — (intent_id, brand,  robot_type, backend)
2. model wildcard   — (intent_id, brand,  "",          backend)
3. brand wildcard   — (intent_id, "",     "",           backend)
4. global fallback  — (intent_id, "",     "",           "")

Returns ``None`` when no binding is found; never raises on a miss.

Registration key format
-----------------------
``"{intent_id}|{brand}|{robot_type}|{backend}"``
Wildcard segments use the empty string ``""``.

Module-level singleton
----------------------
``default_registry`` is the shared registry used by the rest of the system.
Use ``register()`` / ``resolve()`` module-level helpers to interact with it.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from src.system.binding.base import ActionBinding

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_KEY_SEP = "|"


def _make_key(intent_id: str, brand: str, robot_type: str, backend: str) -> str:
    return f"{intent_id}{_KEY_SEP}{brand}{_KEY_SEP}{robot_type}{_KEY_SEP}{backend}"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class BackendBindingRegistry:
    """Registry that maps (intent_id, brand, robot_type, backend) to ActionBinding.

    Instantiate a fresh registry for testing; use ``default_registry`` for
    production code.
    """

    def __init__(self) -> None:
        self._store: Dict[str, ActionBinding] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        binding: ActionBinding,
        *,
        brand: str = "",
        robot_type: str = "",
        backend: str = "",
    ) -> None:
        """Register a binding for the given (brand, robot_type, backend) scope.

        Parameters
        ----------
        binding:
            A concrete ``ActionBinding`` instance.  Its ``intent_id`` property
            forms the primary key segment.
        brand:
            Robot brand scope.  Empty string ``""`` means "any brand"
            (wildcard).
        robot_type:
            Model scope within the brand.  Empty string ``""`` means "any
            model" (wildcard).  Must be ``""`` when ``brand`` is ``""``.
        backend:
            Execution backend scope.  Must be ``"sdk"``, ``"mujoco"``, or
            ``""`` (wildcard / global fallback).

        Notes
        -----
        - Registration is additive: registering the same key twice replaces
          the previous binding (last write wins).
        - ``intent_id`` is taken from ``binding.intent_id``; it is not
          passed as a separate argument.
        """
        if not isinstance(binding, ActionBinding):
            raise TypeError(
                f"binding must be an ActionBinding instance, got {type(binding).__name__!r}"
            )
        if backend not in ("sdk", "mujoco", ""):
            raise ValueError(
                f"backend must be 'sdk', 'mujoco', or '' (wildcard), got {backend!r}"
            )
        if brand == "" and robot_type != "":
            raise ValueError(
                "robot_type must be '' when brand is '' "
                f"(wildcard brand implies wildcard model), got robot_type={robot_type!r}"
            )
        key = _make_key(binding.intent_id, brand, robot_type, backend)
        self._store[key] = binding

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(
        self,
        intent_id: str,
        brand: str,
        robot_type: str,
        backend: str,
    ) -> Optional[ActionBinding]:
        """Return the best matching ActionBinding, or ``None`` if none exists.

        Fallback order (first match wins):
        1. exact:          (intent_id, brand,  robot_type, backend)
        2. model wildcard: (intent_id, brand,  "",          backend)
        3. brand wildcard: (intent_id, "",     "",           backend)
        4. global fallback:(intent_id, "",     "",           "")

        Parameters
        ----------
        intent_id:
            Semantic intent to look up.
        brand:
            Robot brand of the active model.
        robot_type:
            Model identifier within the brand.
        backend:
            Execution backend — ``"sdk"`` or ``"mujoco"``.

        Returns
        -------
        ActionBinding or None
            The most-specific registered binding, or ``None`` when no
            registration matches.  Never raises on a miss.
        """
        candidates = _fallback_keys(intent_id, brand, robot_type, backend)
        for key in candidates:
            binding = self._store.get(key)
            if binding is not None:
                return binding
        return None

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def registered_keys(self) -> List[str]:
        """Return a snapshot of all registered keys (for diagnostics / tests)."""
        return list(self._store.keys())

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: str) -> bool:
        return key in self._store


# ---------------------------------------------------------------------------
# Fallback key sequence (also useful for unit tests)
# ---------------------------------------------------------------------------

def _fallback_keys(
    intent_id: str,
    brand: str,
    robot_type: str,
    backend: str,
) -> List[str]:
    """Return the ordered list of lookup keys for the fallback resolution chain."""
    seen: set = set()
    keys: List[str] = []

    candidates = [
        (intent_id, brand,  robot_type, backend),  # 1. exact
        (intent_id, brand,  "",         backend),  # 2. model wildcard
        (intent_id, "",     "",         backend),  # 3. brand wildcard
        (intent_id, "",     "",         ""),        # 4. global fallback
    ]
    for args in candidates:
        k = _make_key(*args)
        if k not in seen:
            seen.add(k)
            keys.append(k)
    return keys


# ---------------------------------------------------------------------------
# Module-level singleton + convenience helpers
# ---------------------------------------------------------------------------

default_registry = BackendBindingRegistry()


def register(
    binding: ActionBinding,
    *,
    brand: str = "",
    robot_type: str = "",
    backend: str = "",
) -> None:
    """Register a binding in the module-level default registry."""
    default_registry.register(binding, brand=brand, robot_type=robot_type, backend=backend)


def resolve(
    intent_id: str,
    brand: str,
    robot_type: str,
    backend: str,
) -> Optional[ActionBinding]:
    """Resolve a binding from the module-level default registry."""
    return default_registry.resolve(intent_id, brand, robot_type, backend)
