# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""``unitport_npy`` — legacy tracking-only loader.

Reads a ``.npy`` file of shape ``(n_frames, dof)`` joint positions and
wraps it in a :class:`ReferenceMotionContract`. No AMP payload (no
joint velocities, no root pose, no toe positions) — the clip is only
consumable by tracking-style training paths.

``fps`` is supplied by the caller because ``.npy`` files don't carry
timing metadata.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

from application.training.motion.clip import MotionClip
from application.training.motion.contract import (
    CONTRACT_SCHEMA_VERSION,
    ReferenceMotionContract,
    SourceInfo,
    validate_contract,
)
from application.training.motion.labels import infer_task_tag
from application.training.motion.loaders.base import LoaderError, MotionLoader
from application.training.motion.normalizer import normalize_motion_clip


class NpyLoader(MotionLoader):
    """Load a ``.npy`` file of shape ``(n_frames, dof)`` joint positions.

    The resulting clip has **no AMP payload** (``has_amp_payload() ==
    False``) — only the tracking reward consumes it. The caller supplies
    ``fps`` explicitly because .npy files don't carry timing metadata.
    """

    format_id = "unitport_npy"

    def __init__(self, fps: float = 50.0) -> None:
        self._fps = float(fps)

    def load(
        self,
        path: Union[Path, str],
        *,
        target_sku: str,
        target_family: str,
    ) -> ReferenceMotionContract:
        p = Path(path)
        if not p.is_file():
            raise LoaderError(f"NpyLoader: file not found: {p}")
        try:
            arr = np.load(str(p))
        except Exception as exc:  # noqa: BLE001
            raise LoaderError(f"NpyLoader: np.load failed for {p}: {exc}") from exc

        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 2:
            raise LoaderError(
                f"NpyLoader: expected 2-D array (n_frames, dof), "
                f"got shape {arr.shape} in {p}"
            )
        if arr.shape[0] < 2:
            raise LoaderError(
                f"NpyLoader: motion too short ({arr.shape[0]} frames) in {p}"
            )

        clip = MotionClip(
            name=p.stem,
            format_id=self.format_id,
            fps=self._fps,
            loop_mode="wrap",
            motion_weight=1.0,
            task_tag=infer_task_tag(p.name),
            joint_pos=arr,
            amp_obs_dim=0,
            metadata={"source_path": str(p.resolve())},
        )
        # Identity-only normaliser today (no override on disk → same
        # MotionClip object back). A future non-identity override
        # raises loud rather than silently mis-translate.
        clip = normalize_motion_clip(clip, override=None)
        contract = ReferenceMotionContract(
            schema_version=CONTRACT_SCHEMA_VERSION,
            consumption_mode=self.default_consumption_mode,
            clip=clip,
            source_info=SourceInfo(source_path=str(p.resolve())),
            target_family=target_family,
            target_sku=target_sku,
        )
        validate_contract(contract)
        return contract


__all__ = ["NpyLoader"]
