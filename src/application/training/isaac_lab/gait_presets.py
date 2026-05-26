# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Quadruped gait presets — Walk These Ways style parameterised gaits.

Each preset is a named 7-tuple of continuous parameters that together
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

No Qt imports. The presets are loaded at module import time and
re-exported as a plain list; the canvas gait_preset_table widget
uses this list as its seed on new Training Commands nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
        # CLAUDE.md §1.8: gait presets are robot-geometry sensitive.
        # The previous defaults (body_height=0.35, step_height=0.08,
        # frequency=2.5, phase=trot) were Go2-class — substituting them
        # for a humanoid (G1 body_height ≈ 0.7) produces a gait command
        # impossible to track. Refuse to fabricate; the caller MUST
        # provide every field.
        def _required(key: str, caster):
            if key not in d or d[key] is None:
                raise ValueError(
                    f"[GaitPreset.from_dict] required key {key!r} is "
                    f"missing — refusing to substitute Go2-class default "
                    f"(CLAUDE.md §1.8). Fix the preset source."
                )
            try:
                return caster(d[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"[GaitPreset.from_dict] {key}={d[key]!r} is not "
                    f"coercible to {caster.__name__}."
                ) from exc

        phase_raw = _required("phase", lambda v: v)
        if not isinstance(phase_raw, (list, tuple)) or len(phase_raw) != 4:
            raise ValueError(
                f"[GaitPreset.from_dict] phase must be a 4-element "
                f"sequence (per-leg clock offsets), got {phase_raw!r}."
            )
        return cls(
            name=str(_required("name", str)),
            frequency=_required("frequency", float),
            phase=(
                float(phase_raw[0]),
                float(phase_raw[1]),
                float(phase_raw[2]),
                float(phase_raw[3]),
            ),
            body_height=_required("body_height", float),
            step_height=_required("step_height", float),
        )


# ---------------------------------------------------------------------------
# Bundled default presets
# ---------------------------------------------------------------------------


DEFAULT_PRESETS: List[GaitPreset] = [
    GaitPreset(
        name="trot",
        frequency=2.5,
        phase=(0.0, 0.5, 0.5, 0.0),
        body_height=0.35,
        step_height=0.08,
    ),
    GaitPreset(
        name="walk",
        frequency=1.5,
        phase=(0.0, 0.25, 0.5, 0.75),
        body_height=0.30,
        step_height=0.05,
    ),
    GaitPreset(
        name="bound",
        frequency=3.0,
        phase=(0.0, 0.0, 0.5, 0.5),
        body_height=0.32,
        step_height=0.10,
    ),
    GaitPreset(
        name="pace",
        frequency=2.5,
        phase=(0.0, 0.5, 0.0, 0.5),
        body_height=0.33,
        step_height=0.07,
    ),
    GaitPreset(
        name="pronk",
        frequency=3.5,
        phase=(0.0, 0.0, 0.0, 0.0),
        body_height=0.35,
        step_height=0.15,
    ),
]


def default_presets() -> List[GaitPreset]:
    """Return a fresh copy of the bundled presets (safe to mutate)."""
    return [GaitPreset(**p.to_dict()) if False else
            GaitPreset(
                name=p.name,
                frequency=p.frequency,
                phase=tuple(p.phase),
                body_height=p.body_height,
                step_height=p.step_height,
            )
            for p in DEFAULT_PRESETS]


def default_presets_json() -> List[Dict[str, Any]]:
    """Serialised form of the bundled presets for storage on canvas nodes."""
    return [p.to_dict() for p in DEFAULT_PRESETS]


__all__ = [
    "GaitPreset",
    "DEFAULT_PRESETS",
    "default_presets",
    "default_presets_json",
]
