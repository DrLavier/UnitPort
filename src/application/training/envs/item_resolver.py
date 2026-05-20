"""Active-item dispatcher for composite per-motion rewards.

Maps the env's current ``command`` vector to a soft membership over
``MotionConfig.training_items`` so each step's reward can blend per-item
reward terms instead of hard-switching at command-range boundaries
(CLAUDE.md §1.8 — no silent fallback to global terms).

Channels in ``command_ranges`` are read by canonical name (vx/vy/wyaw/...)
and projected onto the 3-dim ``env._command`` layout ``[vx, vy, wyaw]``
used by ``GenericMujocoEnv``. Channels absent from an item's range
specification are treated as unconstrained on that axis (per-axis
membership = 1).
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


# Map canonical channel names → index into ``GenericMujocoEnv._command``.
# The env's command vector is the fixed [vx, vy, wyaw] triple — see
# generic_mujoco_env.py:520-523. Aliases follow what TrainingMotionNode
# / command_schema.py emit into ``training_items[*].command_ranges``.
_CHANNEL_TO_IDX: Dict[str, int] = {
    "vx": 0, "lin_vel_x": 0, "v_x": 0,
    "vy": 1, "lin_vel_y": 1, "v_y": 1,
    "wyaw": 2, "ang_vel_z": 2, "w_z": 2, "wz": 2, "yaw_rate": 2,
}


def _channel_range(item_payload: Dict[str, Any]) -> Dict[int, Tuple[float, float]]:
    """Extract ``{cmd_index: (lo, hi)}`` from an enriched item payload.

    Skips channels with unrecognised names or malformed spans (loud at
    nothing; if every item ends up with an empty dict, the env's
    constructor will raise — that is the §1.8 trigger).
    """
    out: Dict[int, Tuple[float, float]] = {}
    if not isinstance(item_payload, dict):
        return out
    ranges = item_payload.get("command_ranges")
    if not isinstance(ranges, dict):
        return out
    for chan, span in ranges.items():
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            continue
        idx = _CHANNEL_TO_IDX.get(str(chan).lower())
        if idx is None:
            continue
        try:
            lo = float(span[0])
            hi = float(span[1])
        except (TypeError, ValueError):
            continue
        if hi < lo:
            lo, hi = hi, lo
        out[idx] = (lo, hi)
    return out


def _membership_score(
    cmd: np.ndarray,
    chan_ranges: Dict[int, Tuple[float, float]],
    blend_width: float,
) -> float:
    """Trapezoidal membership of ``cmd`` in the item's hyperbox.

    For each constrained channel ``i`` with range ``[lo, hi]`` the per-
    axis score is ``ramp(x − lo, w) · ramp(hi − x, w)`` where ``ramp``
    is a linear 0→1 ramp of width ``w_i = min(blend_width, (hi − lo)/2)``
    — clamping the blend width to at most each item's half-range keeps
    the score saturated at 1 inside the box (so a narrow / tight item
    isn't penalised against a wide one with the same centre). Outside
    the range, the score decays linearly to 0 across ``w_i``.

    The full score multiplies per-axis terms; items with no channel
    constraints score 0 so they never win a tie — the env-level
    "nearest-centre" fallback picks them up only when every other
    score also vanishes.
    """
    if not chan_ranges:
        return 0.0
    score = 1.0
    for idx, (lo, hi) in chan_ranges.items():
        span = hi - lo
        # Cap the per-axis ramp width at the half-range so a tight
        # box still saturates at 1 in its centre; below 1e-3 we treat
        # the axis as a Dirac (only an exact value matches).
        w_i = max(min(float(blend_width), 0.5 * span), 1e-3)
        x = float(cmd[idx])
        left = min(1.0, max(0.0, (x - lo) / w_i))
        right = min(1.0, max(0.0, (hi - x) / w_i))
        score *= left * right
    return float(score)


class ItemResolver:
    """Pre-compile per-item channel ranges; resolve item weights at step time.

    Construction is cheap and pure; the env caches one instance and
    queries :meth:`weights` once per step. ``item_id_order`` fixes the
    one-hot ordering for any obs-side exposure.
    """

    def __init__(
        self,
        training_items: Dict[str, Any],
        item_id_order: Sequence[str],
        blend_width: float = 0.10,
    ) -> None:
        if not item_id_order:
            raise ValueError(
                "ItemResolver: item_id_order is empty. terms_by_item is "
                "set but no item ids were supplied — see CLAUDE.md §1.8 "
                "(no silent fallback to the global rewards bag)."
            )
        if not isinstance(training_items, dict) or not training_items:
            raise ValueError(
                "ItemResolver: training_items is empty. Composite rewards "
                "need MotionConfig.training_items[*].command_ranges to map "
                "the command vector to an active item — see CLAUDE.md §1.8."
            )

        self._order: List[str] = [str(i) for i in item_id_order]
        self._ranges: List[Dict[int, Tuple[float, float]]] = []
        missing: List[str] = []
        for item_id in self._order:
            payload = training_items.get(item_id)
            chan = _channel_range(payload or {})
            if not chan:
                missing.append(item_id)
            self._ranges.append(chan)
        if len(missing) == len(self._order):
            raise ValueError(
                "ItemResolver: every item in terms_by_item lacks usable "
                f"command_ranges (items={self._order!r}). The TrainingMotion "
                "node must declare per-item command envelopes — see "
                "CLAUDE.md §1.8."
            )
        self._blend_width = max(float(blend_width), 1e-3)
        # Cache for fast argmax fallback.
        self._centers: List[Dict[int, float]] = [
            {idx: 0.5 * (lo + hi) for idx, (lo, hi) in r.items()}
            for r in self._ranges
        ]

    @property
    def item_ids(self) -> List[str]:
        return list(self._order)

    @property
    def blend_width(self) -> float:
        return self._blend_width

    def weights(self, cmd: np.ndarray) -> Dict[str, float]:
        """Return ``{item_id: alpha}`` with the alphas summing to 1.

        When every score is ~0 (``cmd`` is several ``blend_width``s
        outside every item's hyperbox), falls back to the nearest item
        by L2 distance to its range center — better than a uniform
        smear that would average rewards across unrelated motions.
        """
        scores = np.array(
            [_membership_score(cmd, r, self._blend_width) for r in self._ranges],
            dtype=np.float64,
        )
        total = float(scores.sum())
        if total > 1e-9:
            scores /= total
            return {iid: float(scores[i]) for i, iid in enumerate(self._order)}
        nearest = self._nearest_item_idx(cmd)
        out: Dict[str, float] = {iid: 0.0 for iid in self._order}
        out[self._order[nearest]] = 1.0
        return out

    def _nearest_item_idx(self, cmd: np.ndarray) -> int:
        best_i = 0
        best_d = float("inf")
        for i, centers in enumerate(self._centers):
            if not centers:
                continue
            d = 0.0
            for idx, c in centers.items():
                d += (float(cmd[idx]) - c) ** 2
            if d < best_d:
                best_d = d
                best_i = i
        return best_i


__all__ = ["ItemResolver"]
