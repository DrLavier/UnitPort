"""MjSimEnv — concrete SimEnvContext for the Mission runtime layer.

This is the class that closes BUG-1: BehaviorNode no longer needs a
monkey-patched ``SimEnvContext`` smuggled in by ``vis_check_runner``.
StartNode constructs an ``MjSimEnv(actor, field)`` directly, threads it
through the canvas as a port value, and BehaviorNode hands it to
``PolicyRunner.load(bundle, env)``.

Inheritance contract
--------------------
``MjSimEnv`` **subclasses** :class:`src.system.policy.sim_env_context.SimEnvContext`.
It does NOT replace the base dataclass — that base carries
``mj_model`` / ``mj_data`` / ``joint_names`` / ``control_frequency_hz`` /
``sensor_manager`` / ``overlay_info`` fields plus the Phase 6
``configure_sensors`` / ``get_sensor_readings`` / ``set_overlay`` methods.
We override the four abstract methods (``sim_step`` / ``reset`` /
``render`` / ``is_terminated``) with concrete MuJoCo implementations.

The construction logic is cannibalized from
``src/system/training/vis_check_runner.py`` — specifically the
``SimEnvContext(...)`` site (~line 700) and the surrounding monkey-patch
of ``sim_step`` / ``reset`` / ``render`` / ``is_terminated``. The
difference is that here those become real subclass methods instead of
runtime attribute assignments.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from src.system.policy.sim_env_context import SimEnvContext

from .mj_actor import MjActor
from .mj_field import MjField


# ----------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------

# PolicyRunner overwrites this in load() with the bundle's
# control_frequency_hz, but we need a value at construction time so the
# dataclass is well-defined for callers who never call PolicyRunner
# (e.g. raw physics smoke tests).
_DEFAULT_CONTROL_HZ: float = 50.0

# Termination guard: the base of a quadruped at this height (or lower)
# means it has fallen. PolicyRunner can override via attribute set, and
# new actors can supply their own threshold via the constructor.
_DEFAULT_FALL_HEIGHT_M: float = 0.10


@dataclass(eq=False)
class MjSimEnv(SimEnvContext):
    """Concrete :class:`SimEnvContext` backed by an :class:`MjActor`
    placed inside an :class:`MjField`.

    Use :meth:`build` instead of the dataclass ``__init__`` — it wires
    the actor + field, populates the inherited base fields, and returns
    a ready-to-use env. The dataclass init signature is preserved for
    pickling and for callers that already hold mj_model / mj_data
    references.
    """

    # ------------------------------------------------------------------
    # Extra fields beyond SimEnvContext
    # ------------------------------------------------------------------
    actor: Optional[MjActor] = None
    field_spec: Optional[MjField] = None
    fall_height_m: float = _DEFAULT_FALL_HEIGHT_M

    # Optional default qpos written by PolicyRunner._align_env_default_pose.
    # Stored as a public attribute so the legacy fall-through path
    # (``setattr(env, "_default_qpos_full", ...)``) keeps working.
    _default_qpos_full: Optional[Any] = None

    # Optional viewer attached after construction (P6 hooks this up;
    # P1 leaves it None and ``render()`` is a no-op).
    _viewer: Optional[Any] = None
    _viewer_sync: Optional[Callable[[], None]] = None
    _next_frame_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Construction helper
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        actor: MjActor,
        field: Optional[MjField] = None,
        control_frequency_hz: float = _DEFAULT_CONTROL_HZ,
    ) -> "MjSimEnv":
        """Wire an actor + field into an MjSimEnv.

        ``field`` defaults to :meth:`MjField.flat_ground` so that the
        StartNode "zero config" path can call ``MjSimEnv.build(actor)``
        directly.
        """
        if field is None:
            field = MjField.flat_ground()

        mj_model = field.compose(actor)
        # Use the actor's mj_data when the field is a pass-through over
        # the actor's own pre-composed scene; otherwise build a fresh
        # data buffer for the composed model. P1 only has the
        # pass-through case, so this is straightforward.
        if mj_model is actor.mj_model:
            mj_data = actor.mj_data
        else:
            import mujoco

            mj_data = mujoco.MjData(mj_model)

        return cls(
            mj_model=mj_model,
            mj_data=mj_data,
            joint_names=list(actor.joint_names),
            control_frequency_hz=float(control_frequency_hz),
            adapter=None,
            actor=actor,
            field_spec=field,
        )

    # ------------------------------------------------------------------
    # SimEnvContext abstract method implementations
    # ------------------------------------------------------------------

    def sim_step(self) -> None:
        """Advance the physics by one ``mj_step``."""
        import mujoco

        mujoco.mj_step(self.mj_model, self.mj_data)

    def reset(self) -> None:
        """Reset mj_data, then re-apply the default qpos if PolicyRunner
        has aligned the bundle's training pose into us.

        ``_default_qpos_full`` is set by
        ``PolicyRunner._align_env_default_pose`` and carries the
        bundle's default joint positions in **qpos_space order**
        (i.e. one float per non-free joint, NOT the full mj_data.qpos
        layout). For a go2 with a floating base, that's 12 leg-joint
        floats — NOT the full 19 qpos slots. We must therefore walk
        the model's joint table and write each default into the
        corresponding ``jnt_qposadr`` slot, leaving the 7-slot
        floating base untouched.
        """
        import mujoco
        import numpy as np

        mujoco.mj_resetData(self.mj_model, self.mj_data)

        if self._default_qpos_full is not None:
            try:
                base = np.asarray(self._default_qpos_full, dtype=float).flatten()
                # Walk the model's non-free joints in their natural
                # order — this matches the qpos_space ordering produced
                # by joint_spaces_from_mj_model (which excludes free
                # and ball joints).
                joint_index = 0
                for jid in range(int(self.mj_model.njnt)):
                    jtype = int(self.mj_model.jnt_type[jid])
                    # Skip free + ball joints (they own multiple qpos
                    # slots and aren't in qpos_space).
                    if jtype == int(mujoco.mjtJoint.mjJNT_FREE):
                        continue
                    if jtype == int(mujoco.mjtJoint.mjJNT_BALL):
                        continue
                    if joint_index >= base.shape[0]:
                        break
                    qposadr = int(self.mj_model.jnt_qposadr[jid])
                    self.mj_data.qpos[qposadr] = float(base[joint_index])
                    joint_index += 1
                mujoco.mj_forward(self.mj_model, self.mj_data)
            except Exception:
                # Default-pose alignment is best-effort; reset still
                # succeeded even if the override failed.
                pass

    def render(self) -> None:
        """No-op by default. If a passive viewer has been attached via
        :meth:`attach_viewer` we call its ``sync()`` and pace against
        wall-clock so Mission playback runs at real-time speed instead
        of as-fast-as-the-CPU-can-step.
        """
        if self._viewer_sync is None:
            return
        try:
            self._viewer_sync()
        except Exception:
            return
        try:
            dt = float(self.mj_model.opt.timestep)
        except Exception:
            return
        if dt <= 0.0:
            return
        now = time.perf_counter()
        target = self._next_frame_time
        if target is None or now - target > 0.25:
            # First frame, or we fell far behind (e.g. paused viewer) —
            # resync the wall clock instead of busy-catching up.
            self._next_frame_time = now + dt
            return
        remaining = target - now
        if remaining > 0.0:
            time.sleep(remaining)
        self._next_frame_time = target + dt

    def is_terminated(self) -> bool:
        """Default termination: base z below the fall threshold.

        For a free-floating root (typical legged-robot MJCF), qpos[2] is
        the world-frame base height. For fixed-base models qpos[2] is a
        joint angle and the threshold check is meaningless — in that
        case we return False so the episode runs to ``max_steps``.
        """
        try:
            if int(self.mj_model.nq) < 3:
                return False
            # Heuristic: only treat qpos[2] as base_z when the model has
            # a free joint at the root. ``mj_model.jnt_type[0] == 0``
            # corresponds to ``mjJNT_FREE``.
            if int(self.mj_model.jnt_type[0]) != 0:
                return False
            base_z = float(self.mj_data.qpos[2])
        except Exception:
            return False
        return base_z < float(self.fall_height_m)

    # ------------------------------------------------------------------
    # Optional viewer attachment (used by Mission render path)
    # ------------------------------------------------------------------

    def attach_viewer(self, viewer: Any) -> None:
        """Bind a ``mujoco.viewer`` passive handle so :meth:`render`
        becomes a viewer sync.

        BehaviorNode / Mission render path can call this after
        ``mujoco.viewer.launch_passive(env.mj_model, env.mj_data)``.
        We deliberately do NOT manage viewer lifetime here; the caller
        is responsible for ``with`` blocks and teardown.
        """
        self._viewer = viewer
        self._viewer_sync = getattr(viewer, "sync", None)
        self._next_frame_time = None

    # ------------------------------------------------------------------
    # set_default_qpos contract used by PolicyRunner._align_env_default_pose
    # ------------------------------------------------------------------

    def set_default_qpos(self, values: Any) -> None:
        """PolicyRunner-facing setter for the bundle-aligned default pose."""
        self._default_qpos_full = values
