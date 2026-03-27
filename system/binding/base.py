#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ActionBinding — abstract base class for semantic-to-backend translation (Step 2).

Design constraints
------------------
- Each concrete subclass handles exactly one intent_id (declared via `intent_id` property).
- `bind()` must not execute anything; it only translates and packages.
- Bindings must not import or reference any SDK or MuJoCo modules directly.
- `bind()` is the only legal path for semantic-to-backend translation.
"""

from __future__ import annotations

import abc
from typing import Any, Dict

from system.binding.output import BindingOutput


class ActionBinding(abc.ABC):
    """Abstract base for a single-intent semantic-to-backend binding.

    Subclasses translate one semantic intent (identified by ``intent_id``)
    into a ``BindingOutput`` for a specific backend target.  They must not
    perform any I/O, SDK calls, or execution of any kind.

    Subclass contract
    -----------------
    1. Override ``intent_id`` to return the exact intent string this binding
       handles (e.g. ``"locomotion.forward"``).
    2. Override ``bind()`` to perform the translation.
    3. Do not import ``unitree_sdk2py``, ``mujoco``, or any other hardware /
       simulation library.
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    @abc.abstractmethod
    def intent_id(self) -> str:
        """The semantic intent identifier this binding handles.

        Must be a non-empty, valid intent ID string (dot-separated ASCII
        lowercase segments, e.g. ``"locomotion.forward"``).  The Backend
        Binding Registry uses this value as the primary lookup key.
        """

    # ------------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def bind(
        self,
        intent_id: str,
        ir_params: Dict[str, Any],
        brand: str,
        robot_type: str,
        backend: str,
    ) -> BindingOutput:
        """Translate a semantic intent into a backend operation.

        Parameters
        ----------
        intent_id:
            The semantic intent being resolved.  Must equal ``self.intent_id``
            (callers should assert this; registry guarantees it).
        ir_params:
            Normalized IR-layer parameters from the authored behavior node.
            Keys and value ranges are intent-specific (e.g. ``speed`` in
            ``[0.0, 1.0]``, ``duration`` in seconds).
        brand:
            Robot brand, e.g. ``"unitree"``.
        robot_type:
            Model within the brand, e.g. ``"go2"`` or ``"go2w"``.
        backend:
            Execution backend target — exactly ``"sdk"`` or ``"mujoco"``.

        Returns
        -------
        BindingOutput
            Fully populated backend operation object.  The returned object
            must be a valid, immutable ``BindingOutput`` instance.

        Raises
        ------
        ValueError
            If the provided arguments cannot produce a valid ``BindingOutput``
            (e.g. an unrecognised backend value, missing required ir_param).
        """
