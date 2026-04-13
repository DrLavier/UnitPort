"""Motion preview bundle builder — UI-neutral dataclass layer.

The 3D preview panel (``bin/pages/training/motion_preview_panel.py``) is
a pure view — it consumes a ``MotionPreviewBundle`` and renders whatever
is inside. **Zero file I/O happens in the panel.** All parsing,
sampling, and geometry prep lives here.

This separation serves two purposes:

1. The panel can be swapped between pyqtgraph / QtQuick3D / matplotlib
   without touching loader code. All three can digest the same bundle.
2. Unit tests don't need a Qt application — they instantiate bundles
   from in-memory ``MotionClip`` objects and assert on the dataclass
   contents directly.

Per AMP_design.yaml §3.reference_motion.new_modules.preview_renderer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

import numpy as np

from src.system.training.motion.clip import MotionClip

if TYPE_CHECKING:
    from src.system.training.robot_assets.registry import RobotAsset


# ---------------------------------------------------------------------------
# Per-frame pose
# ---------------------------------------------------------------------------


@dataclass
class FramePose:
    """One frame of the 3D preview — root pose + joint positions.

    All fields are plain numpy arrays so the consuming UI layer can
    convert to whatever its 3D backend wants (OpenGL buffers, Qt3D
    vertex data, matplotlib lines, …) without re-doing math.
    """

    root_pos: np.ndarray        # shape (3,) — world xyz, float32
    root_quat: np.ndarray       # shape (4,) — (x, y, z, w), float32
    joint_pos: np.ndarray       # shape (dof,) — canonical order, float32

    @classmethod
    def zero(cls, dof: int) -> "FramePose":
        return cls(
            root_pos=np.zeros(3, dtype=np.float32),
            root_quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            joint_pos=np.zeros(dof, dtype=np.float32),
        )


# ---------------------------------------------------------------------------
# Preview bundle
# ---------------------------------------------------------------------------


@dataclass
class MotionPreviewBundle:
    """Everything the UI needs to render a motion preview.

    Attributes
    ----------
    name:
        Clip name (for window title / logs).
    format_id:
        ``"amp_legged_gym"`` or ``"unitport_npy"``. The panel can label
        the format in the metadata sidebar.
    fps:
        Source fps. The panel's play-rate selector multiplies this.
    duration_s:
        Total clip length in seconds.
    frames:
        List of ``FramePose`` — one entry per source frame, in order.
    joint_names:
        Optional canonical joint ordering. When present, the panel can
        show a per-joint time-series alongside the 3D view.
    foot_link_names:
        Optional. If the panel wants to highlight foot links in the 3D
        view, it uses these names against the robot asset's link tree.
    skeleton_parents:
        Optional parent-index list ``parent_of[i] = j``. Tracking-only
        clips without a RobotAsset won't have this — the panel falls
        back to a simple "bone-per-joint" stick figure.
    robot_name:
        Which RobotAsset this preview is relative to, or empty string.
    warnings:
        Non-fatal messages (tracking-only clip can't show toe contacts,
        missing robot asset means no skeleton topology, etc.).
    """

    name: str
    format_id: str
    fps: float
    duration_s: float
    frames: List[FramePose]

    joint_names: List[str] = field(default_factory=list)
    foot_link_names: List[str] = field(default_factory=list)
    skeleton_parents: List[int] = field(default_factory=list)
    robot_name: str = ""
    warnings: List[str] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def dof(self) -> int:
        return int(self.frames[0].joint_pos.shape[0]) if self.frames else 0


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_preview_bundle(
    clip: MotionClip,
    robot_asset: Optional["RobotAsset"] = None,
) -> MotionPreviewBundle:
    """Convert a ``MotionClip`` (+ optional robot asset) into a preview bundle.

    The motion clip's numeric arrays are copied out into per-frame
    ``FramePose`` records. Tracking-only clips (no root pose, no root
    quat) get synthesized zero-pose root entries — the panel still
    renders the joint motion, just without world translation.

    Passing a ``robot_asset`` populates the skeleton metadata
    (joint_names, foot_link_names). Without one, the panel falls back
    to its generic stick-figure renderer.
    """
    warnings: List[str] = []
    n = clip.n_frames
    dof = clip.dof
    if n == 0 or dof == 0:
        raise ValueError(
            f"build_preview_bundle: clip '{clip.name}' is empty "
            f"(n_frames={n}, dof={dof})"
        )

    # Root pos
    if clip.root_pos is not None:
        root_pos_arr = clip.root_pos.astype(np.float32, copy=False)
    else:
        root_pos_arr = np.zeros((n, 3), dtype=np.float32)
        warnings.append(
            "Clip has no root position — rendering at origin. "
            "(Normal for tracking-only .npy clips.)"
        )

    # Root quat
    if clip.root_quat is not None:
        root_quat_arr = clip.root_quat.astype(np.float32, copy=False)
    else:
        root_quat_arr = np.tile(
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (n, 1)
        )
        warnings.append(
            "Clip has no root orientation — rendering with identity quaternion."
        )

    joint_pos_arr = clip.joint_pos.astype(np.float32, copy=False)

    frames: List[FramePose] = [
        FramePose(
            root_pos=root_pos_arr[i].copy(),
            root_quat=root_quat_arr[i].copy(),
            joint_pos=joint_pos_arr[i].copy(),
        )
        for i in range(n)
    ]

    joint_names: List[str] = []
    foot_links: List[str] = []
    robot_name = ""
    if robot_asset is not None:
        joint_names = list(robot_asset.joint_names)
        foot_links = list(robot_asset.foot_link_names)
        robot_name = robot_asset.asset_id
        if joint_names and len(joint_names) != dof:
            warnings.append(
                f"RobotAsset '{robot_asset.asset_id}' has "
                f"{len(joint_names)} joints but clip has dof={dof}. "
                f"Joint labels disabled in preview."
            )
            joint_names = []

    return MotionPreviewBundle(
        name=clip.name,
        format_id=clip.format_id,
        fps=clip.fps,
        duration_s=clip.duration_s,
        frames=frames,
        joint_names=joint_names,
        foot_link_names=foot_links,
        skeleton_parents=[],   # RobotAsset doesn't store link-tree parent
                               # indices yet; panel falls back to stick figure.
        robot_name=robot_name,
        warnings=warnings,
    )
