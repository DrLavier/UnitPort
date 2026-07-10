# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Deterministic canvas auto-layout for AI Build (pure, Qt-free).

The AI Build harness places nodes **position-blind**: the ``place_node`` tool
exposes no coordinates on purpose — the model must not do coordinate math (it
can't track occupancy and would be non-deterministic, §11). Left alone, every
node lands at the origin and they all overlap. This module is the deterministic
SYSTEM step (same shape as ``_apply_requirements`` / ``_apply_pd_auto``) that
arranges them, and it spends **zero tokens**:

  * **block = column**, laid out left→right in the canonical :data:`BLOCKS`
    dataflow order (robot → actor_env → rewards → training → export) — so same
    kind clusters and the whole graph reads left-to-right along its data flow;
  * **within a column**, nodes are stacked top-down using their **real heights**
    (from the snapshot ``ui``) so variable-height nodes (a tall rewards node)
    never overlap the next one;
  * **one titled region per non-empty block** gives the visual partition
    ("分区") — the member list only; the Qt apply step
    (:meth:`application.ui.canvas.page.CanvasPage.apply_ai_build_layout`)
    computes the exact pixel rect from live bounding boxes.

Pure over the ``to_workflow_dict``-shape snapshot dict → headless-testable and
idempotent (the plan depends only on schema_id + node id + heights, never on the
current positions, so re-running it is stable). The Qt side only *applies* it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .blocks import BLOCKS, block_for_schema

__all__ = [
    "LayoutPlan",
    "RegionSpec",
    "compute_layout",
    "COL_PITCH",
    "ROW_GAP",
    "MARGIN_X",
    "MARGIN_Y",
    "COL_GAP",
]

# Geometry (scene units), in a LOCAL frame (top-left ~= (MARGIN_X, MARGIN_Y)); the
# Qt apply step translates the whole layout onto the current view centre. Columns
# are packed left→right using each column's OWN widest node plus ``COL_GAP`` of
# clear space, so the per-block region boxes (node bbox + padding) never overlap —
# wide enough that the boxes read as distinct partitions, not a cramped pile.
COL_GAP = 200.0
ROW_GAP = 56.0
MARGIN_X = 80.0
MARGIN_Y = 80.0
_DEFAULT_W = 220.0
_DEFAULT_H = 130.0
#: Nodes whose schema is not assigned to any block share this trailing lane.
_UNMAPPED_LANE = "__unmapped__"


@dataclass(frozen=True)
class RegionSpec:
    """One titled partition box around a block's nodes.

    Carries the member ids + the block title (key + default) only — the Qt apply
    step computes the exact rect from live member bounding boxes and resolves the
    title through ``tr(title_key, title)`` so localisation stays at the UI edge.
    """

    region_id: str
    title_key: str
    title: str
    member_ids: Tuple[str, ...]


@dataclass(frozen=True)
class LayoutPlan:
    """positions: ``node_id -> (x, y)`` top-left; regions: per-block partition
    boxes (one per non-empty block, in :data:`BLOCKS` order)."""

    positions: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    regions: Tuple[RegionSpec, ...] = ()


def _node_size(node: Dict[str, Any]) -> Tuple[float, float]:
    """(width, height) from the node's serialized ``ui``, with safe defaults.

    A snapshot from a live canvas carries the real reflowed height (a rewards
    node with many rows is tall); a fake/minimal snapshot may omit ``ui`` → the
    default keeps stacking non-overlapping (just less tight)."""
    ui = node.get("ui") or {}
    try:
        w = float(ui.get("width") or _DEFAULT_W)
        h = float(ui.get("height") or _DEFAULT_H)
    except (TypeError, ValueError):
        w, h = _DEFAULT_W, _DEFAULT_H
    return (max(1.0, w), max(1.0, h))


def _id_sort_key(node_id: str) -> Tuple[str, int, str]:
    """Stable within-column order: group by schema prefix, then numeric suffix.

    ``actor_setting_2`` sorts after ``actor_setting_1`` and both stay adjacent
    (same kind clusters inside a column); a suffix-less id sorts first."""
    nid = str(node_id)
    prefix, _sep, suffix = nid.rpartition("_")
    if prefix and suffix.isdigit():
        return (prefix, int(suffix), "")
    return (nid, -1, nid)


def compute_layout(snapshot: Dict[str, Any]) -> LayoutPlan:
    """Assign every node a non-overlapping ``(x, y)`` in its block's column and
    build one titled region per non-empty block.

    Deterministic + idempotent. Unmapped nodes (a schema not in any block — e.g.
    a ``note``) are parked in a trailing lane to the right of the last block
    rather than hidden at the origin (§8 — visible, not silently lost), and get
    no titled region box (they are not training config)."""
    nodes = list(snapshot.get("nodes") or [])
    if not nodes:
        return LayoutPlan()

    # bucket nodes by lane (block id, or the trailing unmapped lane)
    by_lane: Dict[str, List[Dict[str, Any]]] = {}
    for node in nodes:
        block = block_for_schema(str(node.get("schema_id") or ""))
        lane = block.id if block is not None else _UNMAPPED_LANE
        by_lane.setdefault(lane, []).append(node)

    # Column order: present blocks in BLOCKS order, then the unmapped lane last.
    # Sequential packing (no empty-block gaps) — each column takes its OWN widest
    # node's width, and the next column starts COL_GAP of clear space to the right
    # so the region boxes never touch.
    lane_order = [b.id for b in BLOCKS if b.id in by_lane]
    if _UNMAPPED_LANE in by_lane:
        lane_order.append(_UNMAPPED_LANE)

    positions: Dict[str, Tuple[float, float]] = {}
    x = MARGIN_X
    for lane in lane_order:
        lane_nodes = sorted(
            by_lane[lane], key=lambda n: _id_sort_key(str(n.get("id")))
        )
        col_w = max((_node_size(n)[0] for n in lane_nodes), default=_DEFAULT_W)
        y = MARGIN_Y
        for node in lane_nodes:
            _w, h = _node_size(node)
            positions[str(node.get("id"))] = (x, y)
            y += h + ROW_GAP
        x += col_w + COL_GAP

    # regions: one per non-empty BLOCK (the unmapped lane gets no titled box).
    regions: List[RegionSpec] = []
    for block in BLOCKS:
        lane_nodes = by_lane.get(block.id) or []
        if not lane_nodes:
            continue
        members = tuple(
            str(n.get("id"))
            for n in sorted(lane_nodes, key=lambda n: _id_sort_key(str(n.get("id"))))
        )
        regions.append(
            RegionSpec(
                region_id=f"aibuild_region_{block.id}",
                title_key=block.title_key,
                title=block.default_title,
                member_ids=members,
            )
        )
    return LayoutPlan(positions=positions, regions=tuple(regions))
