# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Quadruped gait presets — Walk These Ways style parameterised gaits.

Each preset is a named bundle of continuous parameters that together
describe one gait:

  * ``frequency``    — foot stepping frequency in Hz
  * ``phase``        — per-foot phase offset in [0, 1] for (FL, FR, RL, RR).
                        The phase clock runs 0..1 per step cycle; a foot
                        is in stance when its local phase is in the lower
                        half, swing in the upper half. Relative offsets
                        are what defines the gait class:

                            trot   (0.0, 0.5, 0.5, 0.0)  diagonal pairs
                            bound  (0.0, 0.0, 0.5, 0.5)  front/back pairs
                            pace   (0.0, 0.5, 0.0, 0.5)  left/right pairs
                            walk   (0.0, 0.25, 0.5, 0.75) sequential
                            pronk  (0.0, 0.0, 0.0, 0.0)  all in phase

  * ``body_height``  — commanded base height in metres
  * ``step_height``  — commanded swing foot clearance in metres

The runtime CommandBus snaps the seven gait command dimensions to a
preset when the user presses a D-pad shortcut; between presets the
controller interpolates continuously, so the policy always sees a
smooth command trajectory instead of hard mode switches (this matches
Walk These Ways §3 — "multiplicity of behavior through parameter
interpolation").

Each preset lives in its own file under ``presets_data/`` and exports
``ENTRY: GaitPreset`` plus ``ORDER: int`` for stable list ordering. The
aggregator scans the sub-package at module load and materialises
``DEFAULT_PRESETS`` as a sorted list.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# GaitPreset dataclass
# ---------------------------------------------------------------------------


@dataclass
class GaitPreset:
    """Single named gait snapshot — Walk These Ways §3."""

    name: str
    frequency: float                                  # Hz
    phase: Tuple[float, float, float, float]          # (FL, FR, RL, RR) in [0, 1]
    body_height: float                                # metres
    step_height: float                                # metres

    def __post_init__(self) -> None:
        if len(self.phase) != 4:
            raise ValueError(
                f"GaitPreset {self.name!r}: phase must be a 4-tuple, "
                f"got length {len(self.phase)}"
            )
        for i, p in enumerate(self.phase):
            if not (0.0 <= float(p) <= 1.0):
                raise ValueError(
                    f"GaitPreset {self.name!r}: phase[{i}]={p} out of "
                    f"[0, 1] — phases are normalised clock offsets"
                )
        if self.frequency <= 0:
            raise ValueError(
                f"GaitPreset {self.name!r}: frequency must be positive, "
                f"got {self.frequency}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "frequency": float(self.frequency),
            "phase": [float(x) for x in self.phase],
            "body_height": float(self.body_height),
            "step_height": float(self.step_height),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GaitPreset":
        phase_raw = d.get("phase") or (0.0, 0.5, 0.5, 0.0)
        if not isinstance(phase_raw, (list, tuple)) or len(phase_raw) != 4:
            phase_raw = (0.0, 0.5, 0.5, 0.0)
        return cls(
            name=str(d.get("name", "unnamed")),
            frequency=float(d.get("frequency", 2.5) or 2.5),
            phase=(
                float(phase_raw[0]),
                float(phase_raw[1]),
                float(phase_raw[2]),
                float(phase_raw[3]),
            ),
            body_height=float(d.get("body_height", 0.35) or 0.35),
            step_height=float(d.get("step_height", 0.08) or 0.08),
        )


# ---------------------------------------------------------------------------
# Bundled default presets — aggregated from presets_data/<name>.py
# ---------------------------------------------------------------------------


def _collect_presets() -> List[GaitPreset]:
    pkg = importlib.import_module("scripts.gait.presets_data")
    items: List[Tuple[int, GaitPreset]] = []
    for m in pkgutil.iter_modules(pkg.__path__):
        if m.name.startswith("_"):
            continue
        mod = importlib.import_module(f"scripts.gait.presets_data.{m.name}")
        entry = getattr(mod, "ENTRY", None)
        if entry is None:
            continue
        order = int(getattr(mod, "ORDER", 1_000_000))
        items.append((order, entry))
    items.sort(key=lambda io: (io[0], io[1].name))
    return [e for _, e in items]


DEFAULT_PRESETS: List[GaitPreset] = _collect_presets()


def default_presets() -> List[GaitPreset]:
    """Return a fresh copy of the bundled presets (safe to mutate)."""
    return [
        GaitPreset(
            name=p.name,
            frequency=p.frequency,
            phase=tuple(p.phase),
            body_height=p.body_height,
            step_height=p.step_height,
        )
        for p in DEFAULT_PRESETS
    ]


def default_presets_json() -> List[Dict[str, Any]]:
    """Serialised form of the bundled presets for storage on canvas nodes."""
    return [p.to_dict() for p in DEFAULT_PRESETS]


__all__ = [
    "GaitPreset",
    "DEFAULT_PRESETS",
    "default_presets",
    "default_presets_json",
]
