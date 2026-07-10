# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""CanvasAgentBridge — apply NL-orchestration agent mutations to the live canvas.

Two load-bearing low-level interfaces for natural-language canvas orchestration
(see ``knowledge_base/nl_canvas_orchestration_blueprint.md`` §2 Mutation API /
§6 persistence). The future agent harness drives the canvas exclusively through
this object — it is the single seam between the harness and the live editor.

1. **Real-time mutation sync.** ``place_node`` / ``remove_node`` / ``connect`` /
   ``disconnect`` / ``set_param`` apply to the live :class:`CanvasPage` so the
   display updates immediately, and each emits a typed, color-coded line into the
   :class:`AiBuildPanel` so the user watches the agent work. These mirror the
   blueprint's five Mutation-API primitives.

2. **Checkpoint / restore.** :meth:`checkpoint` snapshots the whole canvas (taken
   before a prompt runs); :meth:`reset_to_checkpoint` reverts to it — the Reset
   button's safety net against a bad agent run.

Design notes:
  * Mutations go through the page's *raw* primitives (no undo bookkeeping, no
    auto-save): an agent run is a coherent batch whose recovery mechanism is the
    checkpoint (the Reset button), NOT the user's Ctrl+Z history, and whose
    persistence is the user's normal save once they accept the result. Keeping
    agent micro-edits out of the undo stack avoids tangling them with the user's
    manual edits.
  * Fail-loud (§8): a rejected mutation (unknown node / schema / edge) does not
    silently no-op — it logs an error line and returns a falsy result so the
    caller can react. It never substitutes a wrong default.
  * Threading: these methods touch Qt widgets and MUST be called on the GUI
    thread. A worker-thread harness marshals through Qt signals (the same
    pattern as ``canvas_param_changed``), exactly like the rest of the edit path.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from unitport_sdk import log_warning, tr

from .ai_build_panel import AiBuildPanel, LogSegment


class CanvasAgentBridge:
    """Single seam: agent mutations + checkpoint/restore over a live CanvasPage."""

    def __init__(self, page: Any, panel: AiBuildPanel) -> None:
        self._page = page
        self._panel = panel
        self._checkpoint: Optional[Dict[str, Any]] = None

    # -- interface 1: live mutation sync (blueprint C1 primitives) -----------

    def place_node(
        self,
        schema_id: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        x: float = 0.0,
        y: float = 0.0,
        instance_id: Optional[str] = None,
    ) -> Optional[str]:
        """Spawn a node; return its instance id, or None on rejection."""
        item = self._page._spawn_node_raw(
            schema_id, x=x, y=y, instance_id=instance_id, params=params
        )
        if item is None:
            return self._fail("place_node", f"unknown or duplicate node '{schema_id}'")
        node_id = item.node_id
        self._event("canvas.ai_build.ev_place", "Placed [[node:{node}]] node", node=node_id)
        return node_id

    def remove_node(self, instance_id: str) -> bool:
        """Delete a node (and its edges); False if it does not exist."""
        if instance_id not in self._page._instances:
            return bool(self._fail("remove_node", f"no node '{instance_id}'"))
        self._page._delete_node_raw(instance_id)
        self._event(
            "canvas.ai_build.ev_remove", "Removed [[node:{node}]] node", node=instance_id
        )
        return True

    def connect(
        self, src_node: str, src_port: str, dst_node: str, dst_port: str
    ) -> Optional[str]:
        """Wire ``src_node.src_port`` → ``dst_node.dst_port``; return edge id."""
        edge = self._page._connect_raw(src_node, src_port, dst_node, dst_port)
        if edge is None:
            return self._fail(
                "connect", f"{src_node}.{src_port} -> {dst_node}.{dst_port}"
            )
        edge_id = str(getattr(edge, "edge_id", "") or "")
        self._event(
            "canvas.ai_build.ev_connect",
            "Wired [[node:{src}]] -> [[node:{dst}]] [[param:{port}]]",
            src=src_node,
            dst=dst_node,
            port=dst_port,
        )
        return edge_id

    def disconnect(self, edge_id: str) -> bool:
        """Remove an edge by id; False if no such edge."""
        if not self._page._disconnect_edge_by_id_raw(edge_id):
            return bool(self._fail("disconnect", f"no edge '{edge_id}'"))
        self._event(
            "canvas.ai_build.ev_disconnect",
            "Disconnected edge [[param:{edge}]]",
            edge=edge_id,
        )
        return True

    def set_param(self, instance_id: str, key: str, value: Any) -> bool:
        """Write a node param (display syncs live); False if node missing."""
        if instance_id not in self._page._instances:
            return bool(self._fail("set_param", f"no node '{instance_id}'"))
        self._page._set_param_raw(instance_id, str(key), value)
        self._event(
            "canvas.ai_build.ev_set",
            "Set [[param:{key}]] = [[value:{value}]] on [[node:{node}]]",
            key=str(key),
            value=self._fmt_value(value),
            node=instance_id,
        )
        return True

    # -- interface 2: checkpoint / restore ----------------------------------

    def checkpoint(self) -> None:
        """Snapshot the entire canvas. Call before an agent run starts.

        If the canvas is in a non-serializable state (``to_workflow_dict``
        raises), this fails LOUD but does not crash the GUI: it surfaces a red
        error line, leaves no checkpoint, and lets :meth:`has_checkpoint` report
        the truth so Reset never offers a snapshot that does not exist. This is a
        visible, honest failure — not a swallowed exception (§8).
        """
        try:
            self._checkpoint = copy.deepcopy(self._page.to_workflow_dict())
        except Exception as exc:  # serialization is the canvas's own contract
            self._checkpoint = None
            log_warning(f"[ai.bridge] checkpoint failed: {exc!r}")
            self._panel.append_markup(
                tr(
                    "canvas.ai_build.checkpoint_failed",
                    "[[error:Could not snapshot the canvas — Reset is unavailable for this run.]]",
                )
            )

    def has_checkpoint(self) -> bool:
        return self._checkpoint is not None

    def reset_to_checkpoint(self) -> bool:
        """Revert the canvas to the last checkpoint. False if none exists."""
        if self._checkpoint is None:
            self._panel.append_markup(
                tr(
                    "canvas.ai_build.no_checkpoint",
                    "[[warning:No checkpoint yet — send a prompt first.]]",
                )
            )
            return False
        # from_workflow_dict clears + rebuilds the scene; feed a deep copy so the
        # stored checkpoint survives repeated resets unmutated.
        self._page.from_workflow_dict(copy.deepcopy(self._checkpoint))
        self._panel.append_markup(
            tr(
                "canvas.ai_build.reverted",
                "[[success:Reverted the canvas to the pre-prompt checkpoint.]]",
            )
        )
        return True

    # -- interface 3: agent-editing highlight (AI Build v2) ------------------

    def highlight_block(self, schema_ids: Any, on: bool) -> None:
        """Highlight every live node whose schema_id is in ``schema_ids``.

        Visual feedback for "the agent is editing this block". GUI thread only
        (the worker harness marshals this through ``BridgeMutationSink``)."""
        wanted = set(schema_ids or ())
        for item in self._page._instances.values():
            if getattr(item.manifest, "id", "") in wanted:
                item.set_agent_active(bool(on))

    def highlight_node(self, instance_id: str, on: bool) -> None:
        """Highlight one live node by instance id (the just-mutated node)."""
        item = self._page._instances.get(instance_id)
        if item is not None:
            item.set_agent_active(bool(on))

    def clear_highlights(self) -> None:
        """Clear the agent-editing highlight on every node (run end / cancel)."""
        for item in self._page._instances.values():
            item.set_agent_active(False)

    def apply_layout(self, plan: Any) -> None:
        """Arrange the canvas per a :class:`LayoutPlan` (GUI thread only).

        Delegates the geometry-heavy work (move nodes + rebuild the titled block
        region boxes tightly around their live members) to the page, which owns
        ``_instances`` / ``_groups`` / the scene. Cosmetic — no C4 event line."""
        self._page.apply_ai_build_layout(
            getattr(plan, "positions", {}) or {},
            getattr(plan, "regions", ()) or (),
        )

    # -- harness-facing log helpers -----------------------------------------

    def log(self, markup: str) -> None:
        """Append one ``[[role:text]]``-tagged line (harness convenience)."""
        self._panel.append_markup(markup)

    def log_event(self, segments: List[LogSegment]) -> None:
        """Append one typed (text, role) segment line (harness convenience)."""
        self._panel.append_segments(segments)

    # -- internals ----------------------------------------------------------

    def _event(self, tr_key: str, default_markup: str, **fields: Any) -> None:
        """Render a tr template (carrying ``[[role:..]]`` markup) into the log.

        ``tr_key`` is deliberately NOT named ``key`` so a ``key=`` field (from
        ``set_param``) cannot collide with it.
        """
        template = tr(tr_key, default_markup)
        try:
            rendered = template.format(**fields)
        except (KeyError, IndexError):
            # Translator dropped/renamed a placeholder — fall back to the default.
            rendered = default_markup.format(**fields)
        self._panel.append_markup(rendered)
        # Every applied primitive routes through here → refresh the live
        # Configuration preview (debounced on the page) so the sections follow
        # the agent's edits in real time. Best-effort UI feedback.
        refresh = getattr(self._page, "ai_build_schedule_config_refresh", None)
        if callable(refresh):
            try:
                refresh()
            except Exception:  # cosmetic refresh must never break a mutation
                pass

    def _fail(self, op: str, detail: str) -> None:
        """Log a fail-loud error line and return None (falsy)."""
        log_warning(f"[ai.bridge] {op} rejected: {detail}")
        template = tr("canvas.ai_build.ev_failed", "[[error:{op} failed — {detail}]]")
        try:
            rendered = template.format(op=op, detail=detail)
        except (KeyError, IndexError):
            rendered = f"[[error:{op} failed — {detail}]]"
        self._panel.append_markup(rendered)
        return None

    @staticmethod
    def _fmt_value(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:g}"
        return str(value)


__all__ = ["CanvasAgentBridge"]
