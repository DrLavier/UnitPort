# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Deploy-side stick → velocity command mapping.

At deploy time a controller (gamepad / keyboard / UI) produces a NORMALISED
command in ``[-1, 1]`` per channel. The policy, however, was trained on REAL
velocity commands (m/s, rad/s) sampled from the per-item ranges. The bridge
between the two — documented in ``training/command_schema.py`` and configured
on the Training Motion node (``mapping_mode`` / ``deadzone`` / ``curve_exponent``
+ the four ``gain_*`` levers that set each channel's range) — is applied HERE,
once, before the command enters the observation.

This is the consumer half of the contract the exporter writes into
``deploy_contract.commands`` (channels + mapping params). Reference formulas
(``hi``/``|lo|`` is the per-direction gain, picked by stick sign):

    linear:       vel = shaped × gain,          shaped = s
    deadzone:     vel = shaped × gain,          shaped = sign(s)·max(0,(|s|-dz)/(1-dz))
    exponential:  vel = shaped × gain,          shaped = sign(s)·|s|**p

with ``gain = hi`` for ``shaped ≥ 0`` and ``gain = |lo|`` for ``shaped < 0``
(so an asymmetric forward/backward envelope maps correctly), then clipped to
``[lo, hi]`` when the channel sets ``runtime_clip``.

Bundles whose ``commands`` block is empty (legacy / pre-mapping exports) yield
``None`` from :func:`build_command_mapper`; the caller then passes the command
through unchanged (identity) — the historical behaviour — and warns once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

MAPPING_LINEAR = "linear"
MAPPING_DEADZONE = "deadzone"
MAPPING_EXPONENTIAL = "exponential"
_SUPPORTED_MODES = frozenset({MAPPING_LINEAR, MAPPING_DEADZONE, MAPPING_EXPONENTIAL})


@dataclass(frozen=True)
class _Channel:
    name: str
    low: float
    high: float
    runtime_clip: bool


@dataclass(frozen=True)
class CommandMapper:
    """Maps a normalised ``[-1, 1]`` command vector to real velocity units.

    Channels are ordered to match the deploy command vector (the velocity
    channels ``lin_vel_x, lin_vel_y, ang_vel_z`` come first — the exporter
    orders them that way). Extra channels (e.g. gait) past the supplied vector
    are ignored; a shorter channel list than the vector leaves the tail
    untouched (identity), so the mapper never silently drops command entries.
    """

    channels: List[_Channel]
    mode: str
    deadzone: float
    curve_exponent: float

    def map(self, command: Optional[Sequence[float]]) -> Optional[List[float]]:
        if command is None:
            return None
        out: List[float] = [float(c) for c in command]
        n = min(len(out), len(self.channels))
        for i in range(n):
            out[i] = self._map_channel(float(out[i]), self.channels[i])
        return out

    # ------------------------------------------------------------------

    def _shape(self, s: float) -> float:
        """Sign-preserving normalised shaping in [-1, 1] per the mode."""
        if self.mode == MAPPING_DEADZONE:
            dz = self.deadzone
            a = abs(s)
            if a <= dz:
                return 0.0
            denom = 1.0 - dz
            if denom <= 1e-9:
                return 0.0
            mag = (a - dz) / denom
            return mag if s >= 0.0 else -mag
        if self.mode == MAPPING_EXPONENTIAL:
            p = self.curve_exponent if self.curve_exponent > 0.0 else 1.0
            mag = abs(s) ** p
            return mag if s >= 0.0 else -mag
        # linear (default)
        return s

    def _map_channel(self, s: float, ch: _Channel) -> float:
        shaped = self._shape(s)
        gain = ch.high if shaped >= 0.0 else abs(ch.low)
        vel = shaped * gain
        if ch.runtime_clip:
            vel = max(ch.low, min(ch.high, vel))
        return vel


def build_command_mapper(commands: Any) -> Optional[CommandMapper]:
    """Build a :class:`CommandMapper` from a ``deploy_contract.commands`` dict.

    Returns ``None`` when the block is missing/empty or carries no continuous
    channels — the caller then treats the command as already-real (identity
    passthrough), which is exactly the pre-mapping behaviour. Raises on a
    malformed (non-dict / bad channel) block rather than silently degrading
    (§8): a present-but-broken mapping is a real export bug, not a fallback.
    """
    if not commands:
        return None
    if not isinstance(commands, dict):
        raise ValueError(
            f"deploy_contract.commands: expected dict, got {type(commands).__name__}"
        )
    raw_channels = commands.get("channels") or []
    if not isinstance(raw_channels, list):
        raise ValueError("deploy_contract.commands.channels must be a list")

    channels: List[_Channel] = []
    for c in raw_channels:
        if not isinstance(c, dict):
            raise ValueError("deploy_contract.commands.channels: each entry must be a dict")
        # Only continuous velocity-like channels participate in stick mapping.
        kind = str(c.get("kind", "continuous"))
        if kind not in ("continuous", ""):
            continue
        name = str(c.get("name", ""))
        if not name:
            continue
        try:
            low = float(c["low"])
            high = float(c["high"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"deploy_contract.commands.channels[{name!r}]: low/high "
                f"missing or non-numeric ({exc})"
            )
        channels.append(_Channel(
            name=name,
            low=low,
            high=high,
            runtime_clip=bool(c.get("runtime_clip", True)),
        ))

    if not channels:
        return None

    mode = str(commands.get("mapping_mode", MAPPING_LINEAR) or MAPPING_LINEAR).strip().lower()
    if mode not in _SUPPORTED_MODES:
        mode = MAPPING_LINEAR
    return CommandMapper(
        channels=channels,
        mode=mode,
        deadzone=float(commands.get("deadzone", 0.0) or 0.0),
        curve_exponent=float(commands.get("curve_exponent", 1.0) or 1.0),
    )


__all__ = ["CommandMapper", "build_command_mapper",
           "MAPPING_LINEAR", "MAPPING_DEADZONE", "MAPPING_EXPONENTIAL"]
