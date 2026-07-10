# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Gait presets — Walk These Ways style parameterised gaits.

Each preset is a named tuple of continuous parameters describing one
gait:

  * ``frequency``    — foot stepping frequency in Hz. ``0.0`` is
                       supported and denotes the "stand" / "frozen"
                       gait: the phase clock does not advance and feet
                       stay at their initial phase offsets. (Prior
                       releases forbade 0.0; the new explicit semantic
                       avoids the §8-anti-pattern of users encoding
                       "stand" as ``freq=0.01 + step_h=0.0``.)
  * ``phase``        — per-foot phase offset in [0, 1]. Tuple width
                       depends on family: quadruped = 4 (FL, FR, RL,
                       RR), biped/humanoid = 2 (L, R). The phase clock
                       runs 0..1 per step cycle; a foot is in stance
                       when its local phase is in the lower half, swing
                       in the upper half. Relative offsets define the
                       gait class.

                            Quadruped:
                              trot   (0.0, 0.5, 0.5, 0.0)  diagonal pairs
                              bound  (0.0, 0.0, 0.5, 0.5)  front/back pairs
                              pace   (0.0, 0.5, 0.0, 0.5)  left/right pairs
                              walk   (0.0, 0.25, 0.5, 0.75) sequential
                              pronk  (0.0, 0.0, 0.0, 0.0)  all in phase

                            Biped/humanoid:
                              walk   (0.0, 0.5)            alternating
                              run    (0.0, 0.5)            alternating, fast
                              stand  (0.0, 0.0)            in phase, freq=0
                              skip   (0.0, 0.0)            in phase, freq>0

  * ``body_height``  — commanded base height in metres
  * ``step_height``  — commanded swing foot clearance in metres

The runtime CommandBus snaps the gait command dimensions to a preset
when the user presses a D-pad shortcut; between presets the controller
interpolates continuously, so the policy always sees a smooth command
trajectory instead of hard mode switches (this matches Walk These Ways
§3 — "multiplicity of behavior through parameter interpolation").

The phase-tuple width is **family-keyed**: the canonical phase count
per family is sourced from
:func:`registers.gait_commands.get_gait_command` (the same registry the
env_cfg_compiler dispatches on). The presets here are bundled per
family; the canvas widget seeds the family-matching subset on new
Training Commands nodes.

No Qt imports. The presets are loaded at module import time and
re-exported as plain lists; the canvas gait_preset_table widget uses
this list as its seed on new Training Commands nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# GaitPreset dataclass
# ---------------------------------------------------------------------------


@dataclass
class GaitPreset:
    """Single named gait snapshot — Walk These Ways §3.

    Phase tuple width is family-keyed (quadruped=4, biped/humanoid=2);
    enforced at construction via the registry-supplied phase_count for
    the explicit ``family`` argument. The family is required because the
    bare tuple shape is ambiguous (a 2-tuple cannot be unambiguously
    classed as biped without external knowledge of the source registry).
    """

    name: str
    frequency: float                                  # Hz; 0.0 == stand
    phase: Tuple[float, ...]                          # per-foot in [0, 1]
    body_height: float                                # metres
    step_height: float                                # metres
    family: str = "quadruped"

    def __post_init__(self) -> None:
        # Phase width validated against the registry-supplied phase_count
        # for the declared family. The registry is the single source of
        # truth for phase shape -- the prior hardcoded ``len(phase) == 4``
        # check assumed all presets were quadruped and would have rejected
        # every biped preset.
        try:
            from registers.gait_commands import get_gait_command
            spec = get_gait_command(self.family)
            expected = spec.phase_count
        except Exception as exc:
            raise ValueError(
                f"GaitPreset {self.name!r}: cannot resolve phase_count "
                f"for family={self.family!r}; the gait_commands registry "
                f"must be loaded and the family must have an entry "
                f"(direct or alias_of). Underlying error: {exc!r}"
            ) from exc
        if len(self.phase) != expected:
            raise ValueError(
                f"GaitPreset {self.name!r} (family={self.family!r}): "
                f"phase must be a {expected}-tuple per the gait_commands "
                f"registry, got length {len(self.phase)}"
            )
        for i, p in enumerate(self.phase):
            if not (0.0 <= float(p) <= 1.0):
                raise ValueError(
                    f"GaitPreset {self.name!r}: phase[{i}]={p} out of "
                    f"[0, 1] — phases are normalised clock offsets"
                )
        # Allow frequency == 0.0 explicitly to denote "stand" (frozen
        # phase clock). Negative frequencies are still rejected.
        if self.frequency < 0:
            raise ValueError(
                f"GaitPreset {self.name!r}: frequency must be >= 0 "
                f"(0 denotes stand / frozen clock), got {self.frequency}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "frequency": float(self.frequency),
            "phase": [float(x) for x in self.phase],
            "body_height": float(self.body_height),
            "step_height": float(self.step_height),
            "family": self.family,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GaitPreset":
        # CLAUDE.md §1.8: gait presets are robot-geometry sensitive.
        # The previous defaults (body_height=0.35, step_height=0.08,
        # frequency=2.5, phase=trot) were Go2-class — substituting them
        # for a humanoid (G1 body_height ≈ 0.7) produces a gait command
        # impossible to track. Refuse to fabricate; the caller MUST
        # provide every field. ``family`` is required because phase
        # width is family-keyed.
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
        if not isinstance(phase_raw, (list, tuple)):
            raise ValueError(
                f"[GaitPreset.from_dict] phase must be a sequence "
                f"(per-leg clock offsets), got {phase_raw!r}."
            )
        family = d.get("family", "quadruped")
        return cls(
            name=str(_required("name", str)),
            frequency=_required("frequency", float),
            phase=tuple(float(x) for x in phase_raw),
            body_height=_required("body_height", float),
            step_height=_required("step_height", float),
            family=str(family),
        )


# ---------------------------------------------------------------------------
# Bundled default presets — per-family
# ---------------------------------------------------------------------------


# Quadruped presets (Go2-class body geometry: body_height ~0.30-0.35 m,
# step_height ~0.05-0.15 m, frequency 1.5-3.5 Hz). These match the prior
# DEFAULT_PRESETS verbatim so the quadruped emit path stays R6
# byte-identical at the preset-literal layer.
DEFAULT_QUADRUPED_PRESETS: List[GaitPreset] = [
    GaitPreset(
        name="trot",
        frequency=2.5,
        phase=(0.0, 0.5, 0.5, 0.0),
        body_height=0.35,
        step_height=0.08,
        family="quadruped",
    ),
    GaitPreset(
        name="walk",
        frequency=1.5,
        phase=(0.0, 0.25, 0.5, 0.75),
        body_height=0.30,
        step_height=0.05,
        family="quadruped",
    ),
    GaitPreset(
        name="bound",
        frequency=3.0,
        phase=(0.0, 0.0, 0.5, 0.5),
        body_height=0.32,
        step_height=0.10,
        family="quadruped",
    ),
    GaitPreset(
        name="pace",
        frequency=2.5,
        phase=(0.0, 0.5, 0.0, 0.5),
        body_height=0.33,
        step_height=0.07,
        family="quadruped",
    ),
    GaitPreset(
        name="pronk",
        frequency=3.5,
        phase=(0.0, 0.0, 0.0, 0.0),
        body_height=0.35,
        step_height=0.15,
        family="quadruped",
    ),
]


# Biped / humanoid presets (G1/H1-class body geometry: body_height
# ~0.65-0.75 m, step_height ~0.04-0.10 m, frequency 1.0-2.0 Hz).
# ``stand`` carries frequency=0.0 to explicitly freeze the phase clock --
# the prior contortion "freq=0.01 + step_h=0.0" was §8 silent-fallback
# anti-pattern (a value pretending to be valid). The relaxed validator
# above accepts freq=0 as a documented gait class.
DEFAULT_BIPED_PRESETS: List[GaitPreset] = [
    GaitPreset(
        name="walk",
        frequency=1.5,
        phase=(0.0, 0.5),
        body_height=0.72,
        step_height=0.06,
        family="biped",
    ),
    GaitPreset(
        name="run",
        frequency=2.0,
        phase=(0.0, 0.5),
        body_height=0.70,
        step_height=0.08,
        family="biped",
    ),
    GaitPreset(
        name="stand",
        frequency=0.0,
        phase=(0.0, 0.0),
        body_height=0.72,
        step_height=0.00,
        family="biped",
    ),
    GaitPreset(
        name="skip",
        frequency=1.8,
        phase=(0.0, 0.0),
        body_height=0.70,
        step_height=0.10,
        family="biped",
    ),
]


# Legacy export name -- the quadruped presets were the only built-ins
# until 7-gamma; keep the name so cross-codebase imports of
# DEFAULT_PRESETS continue to work as the quadruped list.
DEFAULT_PRESETS: List[GaitPreset] = DEFAULT_QUADRUPED_PRESETS


_FAMILY_PRESETS: Dict[str, List[GaitPreset]] = {
    "quadruped": DEFAULT_QUADRUPED_PRESETS,
    "biped": DEFAULT_BIPED_PRESETS,
    # humanoid aliases to biped in the gait_commands registry; mirror
    # that here so default_presets_for_family("humanoid") yields the
    # biped list without an extra lookup.
    "humanoid": DEFAULT_BIPED_PRESETS,
}


def default_ranges_for_family(family: str) -> Dict[str, Tuple[float, float]]:
    """Return the bundled default (lo, hi) ranges for the named family.

    Keys: ``"frequency"`` / ``"body_height"`` / ``"step_height"``.

    SINGLE READ of ``registers/data/gait_commands_catalog.json``'s
    ``default_ranges`` block (catalog 1.1.0) — the one place the numbers
    live. Aliases (humanoid) resolve through the registry alias chain.
    Consumed by the training-side ``CommandSchema`` channel emission, the
    ``CommandContract`` emitter, and the IsaacLab env_cfg_compiler's gait
    Cfg emitters. The canvas may override per-range; these are the
    fallback defaults when the canvas didn't specify.

    Raises ``KeyError`` (preserved historical API) when the family is
    unknown to the registry or its entry declares no ``default_ranges``
    — never substitutes another family's geometry-class values (§8).
    """
    from registers.gait_commands import (
        GaitCommandValidationError,
        get_gait_command,
    )

    try:
        spec = get_gait_command(family)
    except GaitCommandValidationError as exc:
        raise KeyError(
            f"gait_presets.default_ranges_for_family: family={family!r} is "
            f"unknown to the gait_commands registry ({exc})"
        ) from exc
    if spec.default_ranges is None:
        raise KeyError(
            f"gait_presets.default_ranges_for_family: family={family!r} "
            f"(resolved_from={spec.resolved_from!r}) declares no "
            f"default_ranges in gait_commands_catalog.json — add the "
            f"block to the family entry (or the user overlay) instead of "
            f"borrowing another family's geometry-class values"
        )
    return {
        "frequency": spec.default_ranges["frequency"],
        "body_height": spec.default_ranges["body_height"],
        "step_height": spec.default_ranges["step_height"],
    }


def default_phase_offsets_for_family(family: str) -> Tuple[float, ...]:
    """Return the per-foot phase offsets of the first bundled preset for
    the named family.

    Used as the per-channel ``default`` for the gait phase
    :class:`CommandChannel` entries. quadruped -> trot
    ``(0.0, 0.5, 0.5, 0.0)``; biped/humanoid -> walk ``(0.0, 0.5)``.
    """
    presets = default_presets_for_family(family)
    if not presets:
        raise KeyError(
            f"gait_presets.default_phase_offsets_for_family: no bundled "
            f"presets for family={family!r}"
        )
    return tuple(float(x) for x in presets[0].phase)


def default_presets() -> List[GaitPreset]:
    """Return a fresh copy of the bundled quadruped presets.

    Preserved name for backward-compatible callers that still expect the
    quadruped list (the only built-ins prior to 7-gamma)."""
    return [
        GaitPreset(
            name=p.name,
            frequency=p.frequency,
            phase=tuple(p.phase),
            body_height=p.body_height,
            step_height=p.step_height,
            family=p.family,
        )
        for p in DEFAULT_QUADRUPED_PRESETS
    ]


def default_presets_for_family(family: str) -> List[GaitPreset]:
    """Return a fresh copy of the bundled presets for the named family.

    Resolves aliases via the gait_commands registry so callers can pass
    ``"humanoid"`` and receive the biped list."""
    bundle = _FAMILY_PRESETS.get(family)
    if bundle is None:
        try:
            from registers.gait_commands import get_gait_command
            spec = get_gait_command(family)
            # If the family resolves through the registry but we still
            # don't have a preset bundle for it locally, surface a
            # specific raise rather than a silent fallback (§8).
            raise KeyError(
                f"gait_presets: no bundled default presets for family="
                f"{family!r} (registry resolves it to class_name="
                f"{spec.class_name!r}); add a DEFAULT_<FAMILY>_PRESETS "
                f"entry in gait_presets.py"
            )
        except KeyError:
            raise
        except Exception as exc:
            raise KeyError(
                f"gait_presets: family={family!r} is unknown to the "
                f"gait_commands registry and has no bundled default "
                f"presets. Underlying error: {exc!r}"
            ) from exc
    return [
        GaitPreset(
            name=p.name,
            frequency=p.frequency,
            phase=tuple(p.phase),
            body_height=p.body_height,
            step_height=p.step_height,
            family=p.family,
        )
        for p in bundle
    ]


def default_presets_for_phase_count(phase_count: int) -> List[GaitPreset]:
    """Return a fresh copy of the bundled presets for the given phase
    count (4 -> quadruped, 2 -> biped/humanoid).

    Used by the env_cfg_compiler's preset-phase literal helper which
    only knows the registry-supplied ``phase_count`` for the resolved
    gait command. Raises if no family matches the requested count -- no
    silent fallback to a wrong-width list.
    """
    if phase_count == 4:
        return default_presets_for_family("quadruped")
    if phase_count == 2:
        return default_presets_for_family("biped")
    raise ValueError(
        f"gait_presets.default_presets_for_phase_count: no bundled "
        f"presets for phase_count={phase_count} (supported: 2, 4). Add "
        f"DEFAULT_<FAMILY>_PRESETS for the new family and extend the "
        f"_FAMILY_PRESETS / default_presets_for_phase_count map."
    )


def default_presets_json() -> List[Dict[str, Any]]:
    """Serialised form of the bundled quadruped presets for storage on
    canvas nodes (legacy export name)."""
    return [p.to_dict() for p in DEFAULT_QUADRUPED_PRESETS]


def default_presets_json_for_family(family: str) -> List[Dict[str, Any]]:
    """Serialised form of the bundled per-family presets, used by the
    canvas widget to seed Training Commands nodes when the bound robot's
    family is biped/humanoid."""
    return [p.to_dict() for p in default_presets_for_family(family)]


__all__ = [
    "GaitPreset",
    "DEFAULT_PRESETS",
    "DEFAULT_QUADRUPED_PRESETS",
    "DEFAULT_BIPED_PRESETS",
    "default_presets",
    "default_presets_for_family",
    "default_presets_for_phase_count",
    "default_presets_json",
    "default_presets_json_for_family",
    "default_ranges_for_family",
    "default_phase_offsets_for_family",
]
