# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""AI Build config-version store — the producer behind the Configuration combobox.

The redesigned :class:`~application.ui.canvas.ai_build_panel.AiBuildPanel` exposes
a Configuration source combobox seeded with a single ``Origin`` entry and a
``set_config_sources`` seam, but shipped with **no producer** (a documented
BACKEND TODO). This is that producer, kept **Qt-free** so it is unit-testable and
so the panel stays a dumb view:

  * ``Origin`` is the canvas *before the first* AI Build run of the session.
  * every run that actually changes the canvas appends a numbered version
    (``v1``, ``v2``, …) — "if the agent modified anything, a new version MUST
    appear in the dropdown".
  * selecting an entry returns that snapshot so the page can apply it to the live
    canvas; selecting ``Origin`` is the user's manual revert path.

Change detection ignores cosmetic node positions (a moved node is not a config
change) — it compares the *config fingerprint* (nodes' id/schema/params + edges),
so a run that only re-lays-out the graph does not spawn a spurious version.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["ConfigVersion", "AiBuildVersionStore", "ORIGIN_KEY", "config_fingerprint"]

#: The always-present first combobox entry (the pre-first-run canvas).
ORIGIN_KEY = "origin"


@dataclass(frozen=True)
class ConfigVersion:
    """One recorded canvas config version."""

    key: str
    label: str
    snapshot: Dict[str, Any]


def config_fingerprint(canvas: Dict[str, Any]) -> str:
    """A position-independent fingerprint of a canvas' CONFIG.

    Includes node identity/schema/params and the edge set (ordered
    deterministically); excludes ``ui`` coordinates and other cosmetic keys, so
    two canvases that differ only by node placement compare equal.
    """
    nodes = sorted(
        (
            {
                "id": str(n.get("id")),
                "schema_id": str(n.get("schema_id")),
                "params": n.get("params") or {},
            }
            for n in (canvas.get("nodes") or [])
        ),
        key=lambda n: n["id"],
    )
    edges = sorted(
        (
            {
                "source_node": str(e.get("source_node")),
                "source_port": str(e.get("source_port")),
                "target_node": str(e.get("target_node")),
                "target_port": str(e.get("target_port")),
            }
            for e in (canvas.get("edges") or [])
        ),
        key=lambda e: (
            e["source_node"], e["source_port"], e["target_node"], e["target_port"]
        ),
    )
    payload = {
        "backend": canvas.get("backend"),
        "robot_id": canvas.get("robot_id"),
        "nodes": nodes,
        "edges": edges,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


@dataclass
class AiBuildVersionStore:
    """Origin + numbered versions for one canvas' AI Build session."""

    _origin: Optional[Dict[str, Any]] = None
    _versions: List[ConfigVersion] = field(default_factory=list)
    _seq: int = 0

    def set_origin_if_absent(self, canvas: Dict[str, Any]) -> None:
        """Capture ``Origin`` (the pre-first-run canvas) exactly once."""
        if self._origin is None:
            self._origin = copy.deepcopy(canvas)

    def has_origin(self) -> bool:
        return self._origin is not None

    def record(
        self, canvas: Dict[str, Any], *, label: Optional[str] = None
    ) -> Optional[ConfigVersion]:
        """Append a version iff ``canvas`` differs (by config fingerprint) from
        the last known state (the latest version, else Origin).

        Returns the new :class:`ConfigVersion`, or ``None`` when nothing changed
        (so a no-op / aborted run does not fabricate a version). Origin is
        captured here too if it was never set (defensive).
        """
        self.set_origin_if_absent(canvas)
        baseline = self._versions[-1].snapshot if self._versions else self._origin
        if baseline is not None and config_fingerprint(baseline) == config_fingerprint(canvas):
            return None
        self._seq += 1
        key = f"v{self._seq}"
        version = ConfigVersion(
            key=key,
            label=label.strip() if (label and label.strip()) else key,
            snapshot=copy.deepcopy(canvas),
        )
        self._versions.append(version)
        return version

    def sources(self) -> List[Tuple[str, str]]:
        """``[(key, label), …]`` for :meth:`AiBuildPanel.set_config_sources`
        (Origin is prepended by the panel, not listed here)."""
        return [(v.key, v.label) for v in self._versions]

    def latest_key(self) -> str:
        """Combobox key to select after a fresh record (the newest version, or
        Origin when there are none)."""
        return self._versions[-1].key if self._versions else ORIGIN_KEY

    def snapshot_for(self, key: str) -> Optional[Dict[str, Any]]:
        """The canvas snapshot for a combobox key (a deep copy so callers cannot
        mutate the store); ``None`` for an unknown key."""
        if str(key) == ORIGIN_KEY:
            return copy.deepcopy(self._origin) if self._origin is not None else None
        for v in self._versions:
            if v.key == str(key):
                return copy.deepcopy(v.snapshot)
        return None
