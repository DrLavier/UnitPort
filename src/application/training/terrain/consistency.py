# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Cross-engine terrain consistency gate (施工规划 v2 Step 3).

Both engines derive their terrain from the same :class:`HeightField`, but
through different code paths — MuJoCo transposes into a ``<hfield>`` and
renders it with its native heightfield collision; IsaacLab keeps the
array as-is and int16-discretises it into a trimesh. This gate proves the
two actually agree: it builds the MuJoCo model, ray-casts its surface at a
set of sample nodes, compares against the IsaacLab int16 grid height at
the SAME physical points, and **fails loud** (§8) if any sample exceeds
the tolerance.

Why this catches real bugs, not just self-consistency
-----------------------------------------------------
MuJoCo's grid is the *transpose* of IsaacLab's. An asymmetric terrain
therefore only agrees node-for-node if BOTH derivations got the
row/col→x/y mapping right (PV④). A transpose error on either side makes
an asymmetric field diverge massively here — so this gate is the
end-to-end guard for the orientation/extent wiring, run at export/build.

Tolerance (NOT the PD 1e-3)
---------------------------
The legitimate per-node difference is the IsaacLab int16 quantisation
(≤ ``vertical_scale / 2``); MuJoCo carries the un-quantised float height.
So the default tolerance is a small multiple of ``vertical_scale`` (plus
a floor), padded above quantisation. This is deliberately separate from
the PD calibration's 1e-3 — terrain divergence is geometric quantisation,
not serialization round-off (施工规划 v2 §4: keep thresholds separate).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np

from application.training.terrain.contract import HeightField
from application.training.terrain.isaaclab_lowering import (
    DEFAULT_VERTICAL_SCALE,
    heightfield_to_isaaclab,
)
from application.training.terrain.mujoco_lowering import inject_heightfield


class TerrainConsistencyError(RuntimeError):
    """Raised when the MuJoCo and IsaacLab derivations of a heightfield
    disagree beyond tolerance — refuse to ship a bundle whose two engines
    would train on different ground (§8)."""


@dataclass(frozen=True)
class ConsistencyResult:
    """Outcome of :func:`check_cross_engine_consistency`."""

    verdict: str                 # "pass" | "fail"
    n_samples: int
    n_nodes_total: int
    tol: float
    max_abs_diff: float
    mean_abs_diff: float
    worst_point_world: Tuple[float, float]
    worst_mujoco_h: float
    worst_isaaclab_h: float


def _mujoco_model_from_heightfield(hf: HeightField):
    """Build a bare MjModel holding only the injected terrain, centred at
    the world origin. Fail-loud (no flat fallback)."""
    import mujoco  # noqa: PLC0415

    spec = mujoco.MjSpec()
    inject_heightfield(mujoco, spec, hf, center=(0.0, 0.0), replace_planes=False)
    model = spec.compile()
    return mujoco, model


def _mujoco_surface_height(mujoco, model, data, x: float, y: float) -> Optional[float]:
    gid = np.zeros(1, dtype=np.int32)
    dist = mujoco.mj_ray(
        model, data,
        np.array([x, y, 1.0e4]), np.array([0.0, 0.0, -1.0]),
        None, 1, -1, gid,
    )
    if gid[0] < 0:
        return None
    return 1.0e4 - float(dist)


def check_cross_engine_consistency(
    hf: HeightField,
    *,
    vertical_scale: float = DEFAULT_VERTICAL_SCALE,
    tol: Optional[float] = None,
    max_samples: int = 1024,
    raise_on_fail: bool = True,
) -> ConsistencyResult:
    """Compare the MuJoCo and IsaacLab renderings of ``hf`` at grid nodes.

    Samples up to ``max_samples`` grid nodes (evenly subsampled if the
    grid is larger; the drop is logged, never silent — plan "no silent
    caps"). For each node: MuJoCo surface via ray-cast on the compiled
    ``<hfield>`` (centred frame), IsaacLab surface as
    ``int16[i, j] * vertical_scale`` (corner frame, same physical point).

    Returns a :class:`ConsistencyResult`; raises
    :class:`TerrainConsistencyError` when the verdict is ``"fail"`` and
    ``raise_on_fail`` is set.
    """
    conv = heightfield_to_isaaclab(hf, vertical_scale=vertical_scale)
    int16 = conv["heights_int16"]
    vscale = conv["vertical_scale"]
    n_rows, n_cols = int(conv["n_rows"]), int(conv["n_cols"])
    size_x = float(hf.size_x)
    size_y = float(hf.size_y)

    if tol is None:
        tol = max(2.0 * vscale, 0.01)

    # Candidate sample nodes (i, j); evenly subsample if over the budget.
    ii, jj = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
    nodes = np.stack([ii.ravel(), jj.ravel()], axis=1)
    n_total = int(nodes.shape[0])
    if n_total > max_samples:
        sel = np.linspace(0, n_total - 1, max_samples).round().astype(int)
        sel = np.unique(sel)
        nodes = nodes[sel]
        try:
            from unitport_sdk import log_warning
            log_warning(
                f"[terrain.consistency] grid has {n_total} nodes; sampling "
                f"{len(nodes)} (max_samples={max_samples}). Increase "
                f"max_samples to check every node."
            )
        except Exception:  # noqa: BLE001 - logging must never break the gate
            pass

    mujoco, model = _mujoco_model_from_heightfield(hf)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Small inset so edge nodes don't ray-miss the hfield boundary.
    eps = 1.0e-3
    half_x, half_y = size_x / 2.0, size_y / 2.0

    diffs: List[float] = []
    worst = (0.0, (0.0, 0.0), 0.0, 0.0)  # (diff, (x,y), mj_h, isaac_h)
    n_used = 0
    for (i, j) in nodes:
        i = int(i); j = int(j)
        # Canonical corner coords → centred MuJoCo coords.
        cx = i / (n_rows - 1) * size_x
        cy = j / (n_cols - 1) * size_y
        wx = float(np.clip(cx - half_x, -half_x + eps, half_x - eps))
        wy = float(np.clip(cy - half_y, -half_y + eps, half_y - eps))
        mj_h = _mujoco_surface_height(mujoco, model, data, wx, wy)
        if mj_h is None:
            raise TerrainConsistencyError(
                f"check_cross_engine_consistency: MuJoCo ray missed the "
                f"terrain at world ({wx:.4f}, {wy:.4f}) — the injected hfield "
                f"does not cover its own footprint (lowering bug)."
            )
        isaac_h = float(int16[i, j]) * vscale
        d = abs(mj_h - isaac_h)
        diffs.append(d)
        if d > worst[0]:
            worst = (d, (wx, wy), float(mj_h), isaac_h)
        n_used += 1

    arr = np.asarray(diffs, dtype=np.float64)
    max_abs = float(arr.max()) if arr.size else 0.0
    mean_abs = float(arr.mean()) if arr.size else 0.0
    verdict = "pass" if max_abs <= tol else "fail"

    result = ConsistencyResult(
        verdict=verdict,
        n_samples=n_used,
        n_nodes_total=n_total,
        tol=float(tol),
        max_abs_diff=max_abs,
        mean_abs_diff=mean_abs,
        worst_point_world=worst[1],
        worst_mujoco_h=worst[2],
        worst_isaaclab_h=worst[3],
    )

    if verdict == "fail" and raise_on_fail:
        raise TerrainConsistencyError(
            f"Cross-engine terrain mismatch: max |MuJoCo - IsaacLab| = "
            f"{max_abs:.5f} m > tol {tol:.5f} m at world {worst[1]} "
            f"(MuJoCo={worst[2]:.5f}, IsaacLab={worst[3]:.5f}). The two "
            f"engines would train on different ground — fix the lowering / "
            f"raise the vertical_scale, do not ship this bundle (§8)."
        )
    return result


__all__ = [
    "TerrainConsistencyError",
    "ConsistencyResult",
    "check_cross_engine_consistency",
]
