# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""CommandContractModel — live canvas preview of the v1 CommandContract.

One :class:`QObject` per ``training_motion`` :class:`NodeItem`, created
lazily through :func:`get_contract_model` and stored on the scene. The
model runs the SAME derivation the export path serializes —
``derive_command_contract(strict=False)`` — so the Command Pipe
Inspector and the IL Observation injected section render exactly what a
bundle exported from this canvas will carry (WYSIWYG byte equality,
acceptance A1). UI code re-implementing any part of the derivation is a
violation guarded by the A4 purity test.

Recompute triggers (debounced to one pass per event-loop tick):
  * ``AppSignals.canvas_param_changed``   — any param edit on this canvas
    (training_motion params, the Robot node's ``asset_id``, ...);
  * ``AppSignals.canvas_edge_changed``    — actor-pipe / robot-pipe /
    command-pipe wiring changes;
  * ``AppSignals.canvas_topology_changed``— node spawn / delete / load.

The model never mutates the node and never caches across param changes.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from application.training.command_schema import (
    ContractResult,
    derive_command_contract,
)


def resolve_training_motion_family(node) -> Optional[str]:
    """Soft family resolution for the canvas preview boundary.

    Walks ``training_motion.actor_pipe`` ← ``actor_setting.robot_pipe`` ←
    Robot node ``asset_id`` → canonical SKU → ``families[0]`` (the same
    dispatch key the gait registry and the export path use).

    Returns ``None`` on ANY unresolved link. This is the sanctioned
    preview softening (design §2): the emitter turns ``None`` into a
    ``FAMILY_UNKNOWN`` unresolved entry which the UI renders loudly —
    nothing is silently substituted.
    """

    def _input_src(n, port_name: str):
        for p in (getattr(n, "_in_ports", None) or []):
            if str(getattr(getattr(p, "spec", None), "name", "") or "") != port_name:
                continue
            for conn in (getattr(p, "connections", None) or []):
                sp = getattr(conn, "src_port", None)
                if sp is None:
                    continue
                src = sp.parent_node()
                if src is not None:
                    return src
        return None

    if node is None:
        return None
    actor = _input_src(node, "actor_pipe")
    if actor is None:
        return None
    if str(getattr(getattr(actor, "manifest", None), "id", "") or "") != "actor_setting":
        return None
    robot = _input_src(actor, "robot_pipe")
    if robot is None:
        return None
    if str(getattr(getattr(robot, "manifest", None), "id", "") or "") != "robot":
        return None
    asset_id = str((getattr(robot, "params", None) or {}).get("asset_id", "") or "")
    if not asset_id:
        return None
    try:
        from registers.robots import get_robot_spec, resolve_id

        sku = resolve_id(asset_id) or asset_id
        families = list(getattr(get_robot_spec(sku), "families", []) or [])
    except Exception:
        # Unresolvable SKU on the preview boundary → FAMILY_UNKNOWN notice
        # in the Inspector (rendered, not swallowed). The export path
        # fail-louds on the same condition.
        return None
    return str(families[0]) if families else None


class CommandContractModel(QObject):
    """Owns one :class:`ContractResult`, keyed to one training_motion node."""

    contract_changed = pyqtSignal()

    def __init__(self, node_item, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._node = node_item
        self._result: Optional[ContractResult] = None
        self._pending = False
        from application.service.signals import get_app_signals

        sig = get_app_signals()
        sig.canvas_param_changed.connect(self._on_param_changed)
        sig.canvas_edge_changed.connect(self._on_canvas_changed)
        sig.canvas_topology_changed.connect(self._on_canvas_changed)
        self.schedule_recompute()

    # ------------------------------------------------------------------

    @property
    def node(self):
        return self._node

    @property
    def result(self) -> Optional[ContractResult]:
        """Current :class:`ContractResult`; computed synchronously on first
        access so headless consumers (tests) need no event loop."""
        if self._result is None:
            self._recompute()
        return self._result

    def is_stale(self) -> bool:
        """True when the owning node left the scene (deleted)."""
        return self._node is None or self._node.scene() is None

    # ------------------------------------------------------------------

    def _file_id(self) -> str:
        sc = self._node.scene() if self._node is not None else None
        page = getattr(sc, "_page_ref", None) if sc is not None else None
        return str(getattr(page, "_file_id", "") or "")

    def _on_param_changed(self, file_id: str, _node_id: str, _key: str, _value) -> None:
        # Any param on this canvas may affect the contract (item edits on
        # the training_motion node, the Robot node's asset_id swap, ...);
        # the per-tick debounce absorbs the over-approximation.
        if str(file_id or "") == self._file_id():
            self.schedule_recompute()

    def _on_canvas_changed(self, file_id: str) -> None:
        if str(file_id or "") == self._file_id():
            self.schedule_recompute()

    def schedule_recompute(self) -> None:
        """Debounced recompute: at most one derivation per event-loop tick."""
        if self._pending:
            return
        self._pending = True
        QTimer.singleShot(0, self._recompute)

    def _recompute(self) -> None:
        self._pending = False
        node = self._node
        if node is None or node.scene() is None:
            return  # deleted node — stale model, consumers drop it
        family = resolve_training_motion_family(node)
        params = dict(getattr(node, "params", None) or {})
        self._result = derive_command_contract(params, family, strict=False)
        self.contract_changed.emit()


def get_contract_model(node_item) -> CommandContractModel:
    """Scene-scoped lazy registry: one model per training_motion node.

    Stored on the scene (``_contract_models``) so all consumers — the
    Command Pipe Inspector and every IL Observation node resolving its
    Command Pipe input — share one instance per node.
    """
    sc = node_item.scene()
    if sc is None:
        # Off-scene node (should not happen for live consumers): build an
        # unshared model; it will be discarded with the node.
        return CommandContractModel(node_item)
    store = getattr(sc, "_contract_models", None)
    if store is None:
        store = {}
        sc._contract_models = store
    key = str(getattr(node_item, "node_id", "") or id(node_item))
    model = store.get(key)
    if model is None or model.node is not node_item or model.is_stale():
        model = CommandContractModel(node_item)
        store[key] = model
    return model


__all__ = [
    "CommandContractModel",
    "get_contract_model",
    "resolve_training_motion_family",
]
