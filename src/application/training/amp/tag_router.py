# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Command → motion_tag routing for command-conditioned AMP sampling.

Connects the Training Motion Node's design intent (each task_item binds a
command range to a reference clip) to the discriminator's expert
sampling: when env *i* is tracking a command that falls inside task
``T``'s ``command_ranges``, the disc's expert transitions for env *i*
must come from the clip(s) bound to task ``T`` — **not** a uniform mix
over all enabled clips.

Without this routing, policy tracking "walk +x" is compared by the
discriminator against a random mix of stand / trot / turn expert
transitions. No policy can match that union, so ``disc_acc`` floors at
~0.93/1.00 regardless of how close policy motion gets to the trot
mocap. This module is the missing link.

Pure Python + numpy + torch. No Isaac Lab / no RSL-RL deps so it's
unit-testable in the main venv.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# Sentinel for "no matching task" — callers should treat this as a
# fallback (e.g., skip sampling for that row or fall back to uniform).
UNKNOWN_TAG_ID: int = -1


@dataclass(frozen=True)
class TagRegistry:
    """Stable ``{task_item.id → int}`` mapping built once at runner init.

    The int ids are indices into the ordered list of task_items that
    fed the builder, so they stay stable across a single training run.
    They are NOT meant to persist across runs (a checkpoint reload with
    a different task_items order will remap).
    """

    name_to_id: Dict[str, int]
    id_to_name: Tuple[str, ...]

    @property
    def num_tags(self) -> int:
        return len(self.id_to_name)

    def get_id(self, name: str) -> int:
        return self.name_to_id.get(name, UNKNOWN_TAG_ID)


def build_tag_registry(task_items: Sequence[Any]) -> TagRegistry:
    """Build ``{id → int}`` in the order task_items were declared.

    Each element of *task_items* must expose ``.id`` (string).
    """
    names = tuple(str(getattr(t, "id", "")) for t in task_items)
    if any(not n for n in names):
        raise ValueError(
            "build_tag_registry: all task_items must have a non-empty id"
        )
    return TagRegistry(
        name_to_id={n: i for i, n in enumerate(names)},
        id_to_name=names,
    )


# ---------------------------------------------------------------------------
# Command → tag matching
# ---------------------------------------------------------------------------


_ZERO_CMD_ABS_TOL: float = 0.05
_DEFAULT_STAND_TAG_NAME: str = "stand"


def command_to_task_id(
    command: Sequence[float],
    task_items: Sequence[Any],
    *,
    zero_tol: float = _ZERO_CMD_ABS_TOL,
) -> Optional[str]:
    """Point-in-box lookup: which ``task_item.id`` does *command* belong to?

    Parameters
    ----------
    command:
        3-float sequence ``(lin_vel_x, lin_vel_y, ang_vel_z)``.
    task_items:
        Ordered list of objects with ``.id`` (str) and
        ``.command_ranges`` (``Dict[channel_name → (lo, hi)]``).
    zero_tol:
        Absolute tolerance for the zero-command fast path. When all
        channels are within ``±zero_tol`` of zero, returns the task
        whose id is ``"stand"`` (if present), bypassing range checks.
        Zero-command carries no direction signal, so trusting the
        point-in-box on three noisy zeros would coin-flip between
        overlapping tasks.

    Returns
    -------
    The matching ``task_item.id``, or ``None`` if no task's
    ``command_ranges`` contain the point.

    Ordering semantics: stable — first match wins (iteration order of
    *task_items*). If the user wants a specific priority, they should
    list the more specific task first. Overlapping ranges are a canvas
    design choice.
    """
    lin_vel_x, lin_vel_y, ang_vel_z = (float(command[0]), float(command[1]), float(command[2]))

    # Zero-command fast path — map to "stand" if it exists.
    if (
        abs(lin_vel_x) < zero_tol
        and abs(lin_vel_y) < zero_tol
        and abs(ang_vel_z) < zero_tol
    ):
        for t in task_items:
            if str(getattr(t, "id", "")) == _DEFAULT_STAND_TAG_NAME:
                return _DEFAULT_STAND_TAG_NAME
        # Fall through if no explicit stand task exists.

    channels = {
        "lin_vel_x": lin_vel_x,
        "lin_vel_y": lin_vel_y,
        "ang_vel_z": ang_vel_z,
    }

    for t in task_items:
        ranges = getattr(t, "command_ranges", None) or {}
        if not ranges:
            continue
        inside = True
        for ch_name, value in channels.items():
            rng = ranges.get(ch_name)
            if rng is None:
                # Missing channel = no constraint on that axis.
                continue
            try:
                lo, hi = float(rng[0]), float(rng[1])
            except (TypeError, ValueError, IndexError):
                inside = False
                break
            # Small slop so boundary values don't fall through.
            if value < lo - zero_tol or value > hi + zero_tol:
                inside = False
                break
        if inside:
            return str(getattr(t, "id", ""))

    return None


def command_to_tag_id_batch(
    commands: "np.ndarray | Any",
    task_items: Sequence[Any],
    registry: TagRegistry,
    *,
    zero_tol: float = _ZERO_CMD_ABS_TOL,
) -> np.ndarray:
    """Vectorised ``command_to_task_id`` over a batch of envs.

    Parameters
    ----------
    commands:
        ``(num_envs, >=3)`` numpy array or torch tensor. Only the first
        three columns (lin_vel_x, lin_vel_y, ang_vel_z) are read.
    task_items:
        Same semantics as ``command_to_task_id``.
    registry:
        Built once via ``build_tag_registry``.
    zero_tol:
        Forwarded to ``command_to_task_id``.

    Returns
    -------
    ``(num_envs,)`` int64 numpy array of tag ids. Envs whose command
    matches no task get ``UNKNOWN_TAG_ID = -1``.
    """
    # Accept both numpy and torch tensors.
    if hasattr(commands, "detach"):
        cmd_np = commands.detach().cpu().numpy()
    else:
        cmd_np = np.asarray(commands)
    if cmd_np.ndim != 2 or cmd_np.shape[1] < 3:
        raise ValueError(
            f"command_to_tag_id_batch: expected (num_envs, >=3), "
            f"got shape {cmd_np.shape}"
        )

    num_envs = cmd_np.shape[0]
    out = np.full((num_envs,), UNKNOWN_TAG_ID, dtype=np.int64)
    for i in range(num_envs):
        name = command_to_task_id(cmd_np[i, :3], task_items, zero_tol=zero_tol)
        if name is not None:
            tid = registry.get_id(name)
            if tid >= 0:
                out[i] = tid
    return out


# ---------------------------------------------------------------------------
# Clip partitioning by task_item
# ---------------------------------------------------------------------------


def partition_clips_by_task_item(
    clips: Sequence[Any],
    task_items: Sequence[Any],
) -> Dict[str, List[Any]]:
    """Group loaded clips into the expert pool for each task_item.

    A clip belongs to task ``T``'s pool if **either**:

    1. its loaded source filename equals the tail of ``T.clip_ref``
       (explicit user binding from the canvas);  OR
    2. its ``task_tag`` attribute equals ``T.motion_tag`` (tag-based
       match, covers users who point ``clip_ref`` at a pack directory
       and rely on the tag registry).

    Returns a dict keyed by ``task_item.id``. Tasks with no matching
    clips are omitted — the caller decides whether to error, skip, or
    fall back.
    """
    result: Dict[str, List[Any]] = {}
    for ti in task_items:
        t_id = str(getattr(ti, "id", ""))
        if not t_id:
            continue
        clip_ref = getattr(ti, "clip_ref", None) or ""
        motion_tag = getattr(ti, "motion_tag", None) or ""
        ref_tail = ""
        if clip_ref:
            # Parse "pack:<pkg>:<subdir>/<file>" or absolute path.
            ref_tail = Path(clip_ref.replace("pack:", "").split(":")[-1]).name

        matching: List[Any] = []
        seen_ids = set()
        for c in clips:
            cid = id(c)
            # Explicit clip_ref filename match
            if ref_tail:
                src_path = ""
                meta = getattr(c, "metadata", None) or {}
                if isinstance(meta, dict):
                    src_path = str(meta.get("source_path", ""))
                if src_path and Path(src_path).name == ref_tail:
                    matching.append(c)
                    seen_ids.add(cid)
                    continue
            # Tag match fallback
            c_tag = getattr(c, "task_tag", "") or ""
            if motion_tag and c_tag == motion_tag and cid not in seen_ids:
                matching.append(c)
                seen_ids.add(cid)

        if matching:
            result[t_id] = matching
    return result
