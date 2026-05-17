"""MjSimEnv — concrete SimEnvContext for the Mission runtime layer.

Subclasses :class:`application.service.runtime.policy.sim_env_context.SimEnvContext`
and overrides the four abstract methods (``sim_step`` / ``reset`` /
``render`` / ``is_terminated``) with concrete MuJoCo implementations.

Ported from DEMO/src/system/runtime/simulation/mujoco/mj_sim_env.py
verbatim apart from the import path swap and the new convenience
constructor :meth:`from_sku` that uses RELEASE's RobotAssetService.

reset() — lines 155–177 in DEMO — is a Phase plan invariant (R6):
preserve the joint-walking permutation so that the bundle's default
qpos is written into the right ``jnt_qposadr`` slots regardless of
free-joint position.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from application.service.runtime.policy.sim_env_context import SimEnvContext

log = logging.getLogger(__name__)

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

    Use :meth:`build` (raw actor+field) or :meth:`from_sku` (resolves
    via RobotAssetService) instead of the dataclass ``__init__`` — they
    wire the actor + field, populate the inherited base fields, and
    return a ready-to-use env. The dataclass init signature is preserved
    for pickling and for callers that already hold mj_model / mj_data
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

    # Bundle-provided IL training init root position [x, y, z]. When
    # present, ``reset()`` writes ``qpos[free_qposadr + 2] = init_z``
    # as the authoritative spawn base z — this matches the
    # ``ArticulationCfg.InitialStateCfg.pos`` the policy was trained
    # against. None means legacy bundle without the field; ``reset()``
    # falls back to a contact-based ground-clearance heuristic.
    _il_init_base_pos: Optional[Any] = None

    # UI-driven init joint pos override (bundle joint order, one float
    # per non-free non-ball joint). When non-None, ``reset()`` writes
    # these values into ``jnt_qposadr`` slots in place of
    # ``_default_qpos_full``. Used by the Simulation card's "Init Pose
    # Override" subsection to spawn the robot from arbitrary preset
    # poses (lying, fall-from-height, user-saved). ``None`` means no
    # override active → standard bundle default path.
    _init_joint_pos_override: Optional[Any] = None

    # Optional viewer attached after construction (P6 hooks this up;
    # P1 leaves it None and ``render()`` is a no-op).
    _viewer: Optional[Any] = None
    _viewer_sync: Optional[Callable[[], None]] = None
    _next_frame_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Construction helpers
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

        # NOTE: do NOT copy ``key_qpos[home]`` into ``qpos0`` here.
        # ``qpos0`` is not a "default qpos to load on reset"; for
        # hinge joints it is the *reference angle* against which
        # ``qpos[i]`` is offset when computing the body rotation
        # (the physical angle applied = ``qpos[i] - qpos0[i]``).
        # Aligning ``qpos0`` to the home keyframe therefore makes
        # ``mj_resetDataKeyframe`` collapse every hinge to its design
        # pose (legs straight), even though ``d.qpos`` looks correct
        # — the per-joint angle written into kinematics is
        # ``key_qpos - qpos0 = 0``. That's exactly the silent failure
        # mode that produced the "four-limbs-straight, robot
        # launches at 3.47 m/s on episode 0" sim2sim regression.
        # The cosmetic goal (viewer Reset / ``mj_resetData`` landing
        # on a standing pose instead of the design pose) is left
        # unsolved; correct sim2sim physics matters far more.

        return cls(
            mj_model=mj_model,
            mj_data=mj_data,
            joint_names=list(actor.joint_names),
            control_frequency_hz=float(control_frequency_hz),
            adapter=None,
            actor=actor,
            field_spec=field,
        )

    @classmethod
    def from_sku(
        cls,
        sku: str,
        field: Optional[MjField] = None,
        control_frequency_hz: float = _DEFAULT_CONTROL_HZ,
    ) -> "MjSimEnv":
        """Convenience: SKU → MjActor.from_sku → build(actor, field).

        Single-line entry point for canvas / smoke / Phase 4 SB3 trainer
        code that only knows the robot SKU and wants the default
        flat-ground field.
        """
        actor = MjActor.from_sku(sku)
        return cls.build(actor=actor, field=field, control_frequency_hz=control_frequency_hz)

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

        Reset strategy (sim2sim correctness):

        1. If the MJCF carries a keyframe (``<key qpos="...">``) — prefer
           one named ``home`` / ``stand`` / ``initial`` / ``default``,
           else use ``keyframe 0`` — load it via
           ``mj_resetDataKeyframe``. This is the only place the model
           carries a known-good *standing* base z; ``mj_resetData``
           alone uses ``qpos0`` (derived from ``<body pos="...">``)
           which for menagerie quadrupeds is the design-time *rest*
           pose with legs straight down, ~0.18 m too high. Without the
           keyframe path, overlaying bundle leg defaults onto that
           rest pose drops the robot in mid-air at episode start ⇒
           OOD obs ⇒ "joints twist + fall immediately" failure mode.
        2. Otherwise fall back to ``mj_resetData`` (qpos0).
        3. Either way, walk non-free non-ball joints and overwrite
           each ``jnt_qposadr`` slot with ``_default_qpos_full`` (set
           by PolicyRunner._align_env_default_pose, in qpos_space
           order = one float per actuated joint, NOT full nq).
        4. Ground-clamp ``qpos[base_z]`` against the lowest robot
           collision-geom z so the standing footprint always sits on
           the floor. Trivial on menagerie standing keyframes (the
           home key already lands the foot at the right z, so
           |delta| < 0.01 m and the clamp is a no-op). The clamp
           DOES fire when the keyframe is a lying / curled pose: the
           joint overlay rewrites the joints to the standing config
           (which is what the policy was trained against), but the
           keyframe's ``base_z`` is still at the lying height — the
           standing footprint then penetrates 0.1+ m and mj_step
           would launch the robot. We do NOT use
           ``_il_init_base_pos[2]`` (= spec.actor.init_pos_z): it is
           the IsaacLab training spawn altitude in the USD asset
           frame, unrelated to where the MJCF's standing footprint
           actually lands. We also do NOT react to mujoco's reported
           ``contact.dist`` here: menagerie quadrupeds at standing
           pose naturally report a couple cm of foot-capsule
           penetration that mj_step resolves cleanly on step 1.
           Diagnostic log surfaces ``init_source=ground_clamp`` plus
           the delta so a wrong-pose keyframe is visible at a
           glance.
        """
        import mujoco
        import numpy as np

        # Step 1+2: pick a keyframe (preferred standing pose) or fall
        # back to qpos0.
        used_keyframe: Optional[str] = None
        try:
            if int(self.mj_model.nkey) > 0:
                preferred = ("home", "stand", "stand_up", "initial", "default")
                kid = -1
                for name in preferred:
                    candidate = int(
                        mujoco.mj_name2id(
                            self.mj_model, int(mujoco.mjtObj.mjOBJ_KEY), name
                        )
                    )
                    if candidate >= 0:
                        kid = candidate
                        used_keyframe = name
                        break
                if kid < 0:
                    # No named match; use keyframe 0 (most menagerie
                    # models ship exactly one keyframe — the home pose).
                    kid = 0
                    raw_name = mujoco.mj_id2name(
                        self.mj_model, int(mujoco.mjtObj.mjOBJ_KEY), 0
                    )
                    used_keyframe = str(raw_name) if raw_name else "key0"
                mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, kid)
            else:
                mujoco.mj_resetData(self.mj_model, self.mj_data)
        except Exception:
            # Robust against MJCFs without keyframes or with malformed
            # keyframe metadata — fall back to the plain reset path.
            mujoco.mj_resetData(self.mj_model, self.mj_data)
            used_keyframe = None

        base_lifted_by: float = 0.0
        init_source: str = "keyframe"  # "il_init_base_pos" | "contact_fallback" | "keyframe"
        # Choose joint angle source: UI override wins over bundle default.
        # When override is set but bundle has no default (RobotReviewTask
        # path), this branch still fires using override values alone.
        override_active = self._init_joint_pos_override is not None
        joint_pos_source = (
            self._init_joint_pos_override
            if override_active
            else self._default_qpos_full
        )
        if joint_pos_source is not None:
            try:
                base = np.asarray(joint_pos_source, dtype=float).flatten()
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

                # Locate freejoint root. ``qpos[free_qposadr + 2]`` is
                # the base z; rigid-attachment FK means modifying it
                # lifts the whole robot uniformly. For fixed-base
                # robots (no FREE joint) we skip both base-z paths
                # below — qpos[2] there is a joint angle, not base z.
                has_freejoint_root = False
                free_qposadr = -1
                for jid in range(int(self.mj_model.njnt)):
                    if int(self.mj_model.jnt_type[jid]) == int(
                        mujoco.mjtJoint.mjJNT_FREE
                    ):
                        has_freejoint_root = True
                        free_qposadr = int(self.mj_model.jnt_qposadr[jid])
                        break

                # base_z policy:
                #
                # * We do NOT use ``_il_init_base_pos[2]`` — it is the
                #   IsaacLab training spawn altitude in the USD asset
                #   frame (default 0.4 m), unrelated to where the
                #   MJCF's standing footprint actually lands.
                # * We do NOT lift base_z just because mujoco reports
                #   contact penetration. menagerie quadrupeds at
                #   standing pose naturally report a couple cm of
                #   foot-sphere/calf-capsule penetration (capsule vs
                #   plane boundary math); mj_step resolves it with
                #   normal force on step 1. Reacting to it would
                #   launch the robot.
                #
                # We DO ground-clamp the base when the keyframe-
                # supplied ``base_z`` is grossly inconsistent with
                # the standing joint configuration we just wrote into
                # qpos. Example: an MJCF whose home keyframe is a
                # lying / curled pose ships ``base_z ≈ 0.10 m``, but
                # ``_default_qpos_full`` rewrites the joints to the
                # standing config (which the policy was trained
                # against) — needing ``base_z ≈ 0.27 m`` to actually
                # stand. Without correction the standing footprint
                # penetrates the floor by 10+ cm and mj_step kicks
                # the robot up violently.
                #
                # Detection: scan robot collision geoms' world z; if
                # the lowest one is far from the target half-burial
                # depth, shift base_z to bring it back. The check is
                # a no-op (``|delta| < 0.01 m``) on menagerie
                # standing keyframes — the lowest collision geom is
                # the foot sphere center sitting at z ≈ +0.004 m,
                # which is within tolerance of TARGET_MIN_Z. Only the
                # lying-keyframe / wrong-design cases trigger a real
                # shift, and they're surfaced as ``init_source=
                # ground_clamp`` in the diagnostic log.
                mujoco.mj_forward(self.mj_model, self.mj_data)
                if has_freejoint_root and free_qposadr >= 0:
                    robot_geom_min_z = float("inf")
                    for gid in range(int(self.mj_model.ngeom)):
                        bodyid = int(self.mj_model.geom_bodyid[gid])
                        # body_rootid==0 means the geom belongs to
                        # worldbody (floor, lights, props) — not the
                        # robot. Skip.
                        if int(self.mj_model.body_rootid[bodyid]) == 0:
                            continue
                        contype = int(self.mj_model.geom_contype[gid])
                        conaffinity = int(
                            self.mj_model.geom_conaffinity[gid]
                        )
                        # contype==0 and conaffinity==0 → visual-only
                        # geom (mesh under class="visual"). Skip.
                        if (contype | conaffinity) == 0:
                            continue
                        gz = float(self.mj_data.geom_xpos[gid][2])
                        if gz < robot_geom_min_z:
                            robot_geom_min_z = gz
                    # Target: lowest robot collision-geom center
                    # sits just above the floor (≈+5 mm). For
                    # menagerie quadrupeds at standing pose the
                    # lowest collision geom is the foot sphere
                    # center, which the keyframe already places at
                    # z ≈ +0.004 m — that's within the no-op
                    # tolerance below, so menagerie standing keys
                    # are untouched. For a lying / curled keyframe
                    # the standing-joint overlay drops the lowest
                    # geom well below the floor (-0.10 m range), and
                    # the clamp lifts base_z to bring it back to
                    # +5 mm. The actual penetration of the foot
                    # geom into the floor (sphere bottom = center -
                    # radius ≈ -0.017 m) is what menagerie's hand-
                    # tuned home keyframes themselves produce; the
                    # next mj_step resolves it with normal force.
                    TARGET_MIN_Z = 0.005
                    # No-op tolerance: don't move base for sub-2-cm
                    # offsets. Menagerie standing keys land within
                    # this band; only real keyframe-vs-joint
                    # mismatches (lying poses, miscalibrated
                    # base_z, etc.) exceed it.
                    GROUND_CLAMP_TOL = 0.02
                    if robot_geom_min_z != float("inf"):
                        delta = TARGET_MIN_Z - robot_geom_min_z
                        if abs(delta) > GROUND_CLAMP_TOL:
                            z_idx = free_qposadr + 2
                            self.mj_data.qpos[z_idx] = float(
                                self.mj_data.qpos[z_idx] + delta
                            )
                            base_lifted_by = delta
                            init_source = "ground_clamp"
                            mujoco.mj_forward(
                                self.mj_model, self.mj_data
                            )
                            log.info(
                                "MjSimEnv.reset: ground-clamp · "
                                "keyframe %r left lowest robot geom "
                                "at z=%.4f m (target %.4f m); "
                                "shifted base_z by %+.4f m so the "
                                "standing footprint sits on the "
                                "floor.",
                                used_keyframe,
                                robot_geom_min_z,
                                TARGET_MIN_Z,
                                delta,
                            )

                    max_penetration = 0.0
                    ncon = int(self.mj_data.ncon)
                    contact_details: list = []
                    for i in range(ncon):
                        c = self.mj_data.contact[i]
                        dist = float(c.dist)
                        if dist < 0.0 and -dist > max_penetration:
                            max_penetration = -dist
                        contact_details.append(
                            (i, float(c.dist), int(c.geom1), int(c.geom2))
                        )
                    if max_penetration > 1e-6:
                        log.info(
                            "MjSimEnv.reset: spawn-time geometric "
                            "penetration max=%.3f m at keyframe=%r "
                            "(no base_z correction; physics step will "
                            "resolve via normal force).",
                            max_penetration, used_keyframe,
                        )
                    # One-shot deep-dive diagnostic: which geom pairs
                    # have the deepest penetration, plus each leg's
                    # foot/calf world position. This lets us tell the
                    # difference between "foot barely buries 0.017 m
                    # into floor (normal)" and "something else has
                    # 0.18 m of penetration (kinematic chain or
                    # collision-mesh bug)".
                    if not getattr(
                        type(self), "_logged_contact_diag", False
                    ):
                        try:
                            contact_details.sort(key=lambda t: t[1])
                            top = contact_details[:8]
                            lines = []
                            for idx, dist, g1, g2 in top:
                                n1 = mujoco.mj_id2name(
                                    self.mj_model,
                                    int(mujoco.mjtObj.mjOBJ_GEOM),
                                    g1,
                                )
                                n2 = mujoco.mj_id2name(
                                    self.mj_model,
                                    int(mujoco.mjtObj.mjOBJ_GEOM),
                                    g2,
                                )
                                lines.append(
                                    f"  contact[{idx:2d}] dist={dist:+.4f} "
                                    f"{(n1 or f'#{g1}')!s} <-> "
                                    f"{(n2 or f'#{g2}')!s}"
                                )
                            log.info(
                                "MjSimEnv.reset: ncon=%d top contacts "
                                "(deepest penetration first):\n%s",
                                ncon, "\n".join(lines) if lines else "  (none)",
                            )
                            foot_names = ("FL", "FR", "RL", "RR")
                            foot_lines = []
                            for name in foot_names:
                                gid = mujoco.mj_name2id(
                                    self.mj_model,
                                    int(mujoco.mjtObj.mjOBJ_GEOM),
                                    name,
                                )
                                if gid >= 0:
                                    p = self.mj_data.geom_xpos[gid]
                                    foot_lines.append(
                                        f"  geom {name!r} xpos="
                                        f"({float(p[0]):+.4f},"
                                        f" {float(p[1]):+.4f},"
                                        f" {float(p[2]):+.4f})"
                                    )
                                else:
                                    foot_lines.append(
                                        f"  geom {name!r} not found"
                                    )
                            log.info(
                                "MjSimEnv.reset: foot world positions "
                                "(geom_xpos):\n%s",
                                "\n".join(foot_lines),
                            )
                        except Exception:
                            log.exception(
                                "MjSimEnv.reset: contact-diagnostic dump "
                                "failed (non-fatal)"
                            )
                        type(self)._logged_contact_diag = True
            except Exception:
                # Default-pose alignment is best-effort; reset still
                # succeeded even if the override failed.
                log.exception(
                    "MjSimEnv.reset: default_pos overlay / base_z "
                    "init failed"
                )

        # mj_resetDataKeyframe / mj_resetData write qpos but do NOT run
        # forward kinematics; xpos / xmat / xanchor stay at whatever
        # state the previous step / qpos0-design-pose left them in. The
        # passive viewer renders bodies via xpos, so without this call
        # Review Robot's default path (no override → no joint overlay
        # branch → no mj_forward) shows the design-time rest pose
        # (legs straight down) even though qpos already holds the home
        # keyframe (legs bent). Propagate qpos → xpos here so every
        # reset path lands on a viewer-correct state.
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # One-shot diagnostic so the chosen reset path is visible in
        # CMD logs the first time an env is reset. Includes spawn
        # qpos[7:nq] so the user can verify joints actually landed in
        # the IL bent pose vs all-zero (= overlay link broken).
        if not getattr(type(self), "_logged_reset_strategy", False):
            try:
                nq = int(self.mj_model.nq)
                base_z = float(self.mj_data.qpos[2]) if nq >= 3 else float("nan")
                joints_after = (
                    self.mj_data.qpos[7:nq].tolist() if nq >= 7 else []
                )
                dq_full = (
                    np.asarray(self._default_qpos_full).flatten().tolist()
                    if self._default_qpos_full is not None
                    else None
                )
                log.info(
                    "MjSimEnv.reset: keyframe=%r base_z=%.4f "
                    "base_lifted_by=%.4fm init_source=%s "
                    "il_init_base_pos=%s default_qpos_full=%s "
                    "joint_pos_override=%s",
                    used_keyframe,
                    base_z,
                    base_lifted_by,
                    init_source,
                    (
                        list(self._il_init_base_pos)
                        if self._il_init_base_pos is not None
                        else None
                    ),
                    "set" if self._default_qpos_full is not None else "none",
                    "set" if override_active else "none",
                )
                log.info(
                    "MjSimEnv.reset: spawn joints qpos[7:nq]=%s",
                    ["%.3f" % v for v in joints_after],
                )
                log.info(
                    "MjSimEnv.reset: _default_qpos_full content=%s",
                    (
                        ["%.3f" % v for v in dq_full]
                        if dq_full is not None
                        else "None"
                    ),
                )
            except Exception:
                pass
            type(self)._logged_reset_strategy = True

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

    def set_il_init_base_pos(self, pos: Any) -> None:
        """PolicyRunner-facing setter for the bundle's IL init root pos.

        Accepts a length-3 sequence ``[x, y, z]`` (meters) or ``None``.
        When non-None, ``reset()`` writes the z component into the
        freejoint root's ``qpos`` slot as the authoritative spawn base
        altitude, bypassing the contact-based ground-clearance heuristic.
        x/y are stored but not currently applied at spawn (keyframe
        origin is kept; future deploy-side domain rand can consume
        them).
        """
        if pos is None:
            self._il_init_base_pos = None
            return
        try:
            triple = tuple(float(v) for v in pos)
        except (TypeError, ValueError):
            self._il_init_base_pos = None
            return
        if len(triple) != 3:
            self._il_init_base_pos = None
            return
        self._il_init_base_pos = triple

    def set_init_joint_pos_override(self, qpos_bundle: Any) -> None:
        """UI-facing setter for the init-pose-override joint angles.

        ``qpos_bundle`` is a sequence of floats in bundle joint order
        (one per non-free non-ball joint, matching the layout
        ``set_default_qpos`` expects). ``None`` clears the override so
        ``reset()`` falls back to ``_default_qpos_full`` (the bundle's
        training-time pose).

        Values are NOT validated against joint limits here; MuJoCo's
        ``mj_forward`` clamps to URDF/MJCF limits during reset. This
        is consistent with ``set_default_qpos`` semantics.
        """
        if qpos_bundle is None:
            self._init_joint_pos_override = None
            return
        try:
            import numpy as np

            arr = np.asarray(qpos_bundle, dtype=float).flatten()
        except Exception:
            self._init_joint_pos_override = None
            return
        if arr.size == 0:
            self._init_joint_pos_override = None
            return
        self._init_joint_pos_override = arr
