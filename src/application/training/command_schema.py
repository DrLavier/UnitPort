# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""CommandSchema — unified training / sim / sim2real command contract.

Single source of truth for the ``command`` sub-vector that enters the
policy observation. Lives in both:

  * :class:`TrainingJobSpec` — compile-time, consumed by the IL / SB3
    training runners as the velocity sampler's source of truth.
  * :class:`DeployContract` — runtime, stored inside ``manifest.yaml``
    under ``deploy_contract.commands`` so the deployment CommandBus
    knows the exact channel layout, ranges, and runtime clip behaviour
    the policy was trained against.

The bundle-embedded form is :class:`CommandContract` (contract v1):
velocity channels (per-channel union over enabled training items) +
family-keyed gait channels, each carrying an :class:`ObsEncoding`
(command space vs obs space are two different sums). ONE derivation
body — :func:`derive_command_contract` — feeds both the export path
(strict, raises) and the canvas preview path (collects unresolved
conditions for the UI). ``trigger`` / ``enum`` channel kinds ship in
the v1 vocabulary but have no producer yet (reserved for skill items).

No Qt imports. Pure data layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Channel kinds
# ---------------------------------------------------------------------------

CHANNEL_CONTINUOUS = "continuous"
CHANNEL_DISCRETE = "discrete"
CHANNEL_BUTTON = "button"


# ---------------------------------------------------------------------------
# Gait channel dispatch helper -- family-keyed via registry.gait_commands
#
# The training-side CommandSchema and the IsaacLab env_cfg_compiler are
# the two consumers of the family-keyed gait command shape. The
# env_cfg_compiler dispatches via _resolve_gait_spec (7-beta/7-gamma);
# this helper is the matching dispatch on the CommandSchema side so the
# two paths cannot drift in channel count, channel name, or per-channel
# default.
#
# Two registries are queried by family id:
#   * registers.gait_commands.get_gait_command(family) ->
#     phase_names tuple (load-bearing for channel naming);
#   * isaac_lab.gait_presets.default_ranges_for_family /
#     default_phase_offsets_for_family (single source of truth for
#     bundled fallback ranges and phase defaults; shared with the
#     env_cfg_compiler gait Cfg emitters).
#
# Quadruped channel naming + order is byte-identical to the prior
# hardcoded 7-channel emit (gait_frequency / gait_phase_fl / fr / rl /
# rr / body_height / step_height). Biped channels derive from
# phase_names ("L", "R") -> gait_phase_l / gait_phase_r, total 5
# channels (1 freq + 2 phase + body_h + step_h).
# ---------------------------------------------------------------------------


def _build_gait_channels(
    family: str,
    *,
    freq_range: Tuple[float, float],
    body_height_range: Tuple[float, float],
    step_height_range: Tuple[float, float],
    runtime_clip: bool,
) -> List["CommandChannel"]:
    """Return the ordered gait channel list for ``family``.

    The channel set is derived from ``registers.gait_commands.get_gait_command``:
    one ``gait_frequency`` continuous channel, one ``gait_phase_<name>``
    channel per entry in ``spec.phase_names``, then ``body_height`` and
    ``step_height``. Phase channel order matches the registry's
    ``phase_names`` tuple, which is load-bearing for the policy obs
    sub-vector layout.

    Per-channel defaults: frequency / body_h / step_h take the
    range midpoint; phase channels take the family's bundled
    first-preset offsets (quadruped trot pattern, biped alternating).
    All defaults sourced via :mod:`gait_presets` so the env_cfg_compiler
    emitters and this helper share one source of truth.
    """
    from registers.gait_commands import get_gait_command
    from application.training.isaac_lab.gait_presets import (
        default_phase_offsets_for_family,
    )

    spec = get_gait_command(family)
    phase_offsets = default_phase_offsets_for_family(family)
    if len(phase_offsets) != spec.phase_count:
        raise RuntimeError(
            f"_build_gait_channels: bundled phase offsets for family="
            f"{family!r} has length {len(phase_offsets)} but the "
            f"gait_commands registry declares phase_count="
            f"{spec.phase_count}. Align "
            f"gait_presets._FAMILY_PRESETS[{family!r}] with the "
            f"registry spec.phase_count, or update the bundled first "
            f"preset's phase tuple."
        )

    freq_lo, freq_hi = float(freq_range[0]), float(freq_range[1])
    bh_lo, bh_hi = float(body_height_range[0]), float(body_height_range[1])
    sh_lo, sh_hi = float(step_height_range[0]), float(step_height_range[1])

    channels: List["CommandChannel"] = [
        CommandChannel(
            name="gait_frequency",
            kind=CHANNEL_CONTINUOUS,
            low=freq_lo,
            high=freq_hi,
            unit="Hz",
            default=(freq_lo + freq_hi) * 0.5,
            runtime_clip=runtime_clip,
            binding="",
        ),
    ]
    for phase_name, phase_default in zip(spec.phase_names, phase_offsets):
        channels.append(
            CommandChannel(
                name=f"gait_phase_{str(phase_name).lower()}",
                kind=CHANNEL_CONTINUOUS,
                low=0.0,
                high=1.0,
                unit="",
                default=float(phase_default),
                runtime_clip=runtime_clip,
                binding="",
            )
        )
    channels.append(
        CommandChannel(
            name="body_height",
            kind=CHANNEL_CONTINUOUS,
            low=bh_lo,
            high=bh_hi,
            unit="m",
            default=(bh_lo + bh_hi) * 0.5,
            runtime_clip=runtime_clip,
            binding="",
        )
    )
    channels.append(
        CommandChannel(
            name="step_height",
            kind=CHANNEL_CONTINUOUS,
            low=sh_lo,
            high=sh_hi,
            unit="m",
            default=(sh_lo + sh_hi) * 0.5,
            runtime_clip=runtime_clip,
            binding="",
        )
    )
    return channels


# ---------------------------------------------------------------------------
# Stick → velocity mapping modes (deployment-side function shape)
#
# These modes affect how the runtime CommandBus turns a normalised
# stick reading in [-1, 1] into a velocity command in the trained
# range — they do NOT change the training sampler, which continues to
# sample uniformly over the full range regardless of mode. The policy
# is always trained on the complete envelope; the curve just reshapes
# the user-facing control feel.
#
# Runtime reference formulas (``fwd`` / ``bwd`` picked by stick sign):
#
#   linear:       vel = stick × gain
#   deadzone:     vel = sign(stick) × gain × (|stick| - dz) / (1 - dz)
#                       (0 when |stick| < dz)
#   exponential:  vel = sign(stick) × gain × |stick| ** p
# ---------------------------------------------------------------------------

MAPPING_LINEAR = "linear"
MAPPING_DEADZONE = "deadzone"
MAPPING_EXPONENTIAL = "exponential"

SUPPORTED_MAPPING_MODES = {
    MAPPING_LINEAR,
    MAPPING_DEADZONE,
    MAPPING_EXPONENTIAL,
}


# ---------------------------------------------------------------------------
# CommandChannel
# ---------------------------------------------------------------------------

@dataclass
class CommandChannel:
    """One named entry in the policy command vector.

    Fields
    ------
    name:
        Channel identifier used everywhere downstream: policy obs
        sub-vector key, CommandBus binding, manifest key. Must be unique
        within a :class:`CommandSchema`.
    kind:
        ``"continuous"`` | ``"discrete"`` | ``"button"``.
    low / high:
        Inclusive range for continuous channels; for discrete channels
        low = 0 and high = ``n_levels - 1``; for buttons both are 0/1.
    unit:
        Human-readable unit label for the UI and manifest
        (``"m/s"`` / ``"rad/s"`` / ``"Hz"`` / ``""``).
    default:
        Fallback value when the channel is not actively driven (e.g. no
        gamepad input). Also the value used by the episode sampler's
        "zero command" branch.
    runtime_clip:
        When True, the deployment CommandBus clips any incoming value
        to ``[low, high]`` before handing it to the policy. Strongly
        recommended to keep on — protects against out-of-distribution
        runtime commands from stick drift / sensor noise.
    binding:
        Optional CommandBus binding identifier (e.g.
        ``"gamepad.left_stick_y"``). Empty string = not bound, must be
        provided programmatically or by a task planner.
    required_reward:
        Reward term ids whose presence is required when this channel is
        active. PR-3 consistency-check hook. Empty list in PR-1.
    """

    name: str
    kind: str
    low: float
    high: float
    unit: str = ""
    default: float = 0.0
    runtime_clip: bool = True
    binding: str = ""
    required_reward: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.kind not in (CHANNEL_CONTINUOUS, CHANNEL_DISCRETE, CHANNEL_BUTTON):
            raise ValueError(
                f"CommandChannel {self.name!r}: kind must be one of "
                f"continuous/discrete/button, got {self.kind!r}"
            )
        if self.high < self.low:
            raise ValueError(
                f"CommandChannel {self.name!r}: high ({self.high}) < "
                f"low ({self.low})"
            )

    def range_tuple(self) -> Tuple[float, float]:
        return (float(self.low), float(self.high))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "low": float(self.low),
            "high": float(self.high),
            "unit": self.unit,
            "default": float(self.default),
            "runtime_clip": bool(self.runtime_clip),
            "binding": self.binding,
            "required_reward": list(self.required_reward),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CommandChannel":
        return cls(
            name=str(d.get("name", "")),
            kind=str(d.get("kind", CHANNEL_CONTINUOUS)),
            low=float(d.get("low", 0.0)),
            high=float(d.get("high", 0.0)),
            unit=str(d.get("unit", "") or ""),
            default=float(d.get("default", 0.0) or 0.0),
            runtime_clip=bool(d.get("runtime_clip", True)),
            binding=str(d.get("binding", "") or ""),
            required_reward=list(d.get("required_reward") or []),
        )


# ---------------------------------------------------------------------------
# TaskItem
# ---------------------------------------------------------------------------

@dataclass
class TaskItem:
    """One training task (Stand/Walk/Turn/Pace/...) with its command envelope and clip binding.

    A TaskItem bundles three things that must stay consistent:
      * ``motion_tag`` — discriminator filter tag for AMP
      * ``command_ranges`` — per-channel (low, high) the env samples from
        when this task is active
      * ``clip_ref`` — optional "pack:<pkg>:<subdir>/<file>" pointing at
        the reference clip for style alignment (None → no reference)

    Produced by TrainingMotionNode; consumed by the AMP data provider
    (per-env task filtering) and env command sampler (per-env range).
    """

    id: str
    motion_tag: str
    command_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    clip_ref: Optional[str] = None
    # Optional second reference clip for the negative-direction half of a
    # bidirectional task (backward walk, CW turn, …). None if the task is
    # single-direction (stand) or the user only wired one clip.
    clip_ref_reverse: Optional[str] = None
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "motion_tag": str(self.motion_tag),
            "command_ranges": {
                str(k): [float(v[0]), float(v[1])]
                for k, v in self.command_ranges.items()
            },
            "clip_ref": None if self.clip_ref is None else str(self.clip_ref),
            "clip_ref_reverse": (
                None if self.clip_ref_reverse is None else str(self.clip_ref_reverse)
            ),
            "weight": float(self.weight),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaskItem":
        if not isinstance(d, dict):
            return cls(id="", motion_tag="")
        raw_ranges = d.get("command_ranges") or {}
        ranges: Dict[str, Tuple[float, float]] = {}
        if isinstance(raw_ranges, dict):
            for k, v in raw_ranges.items():
                if not isinstance(v, (list, tuple)) or len(v) != 2:
                    continue
                try:
                    ranges[str(k)] = (float(v[0]), float(v[1]))
                except (TypeError, ValueError):
                    continue
        clip = d.get("clip_ref")
        clip_rev = d.get("clip_ref_reverse")
        return cls(
            id=str(d.get("id", "") or ""),
            motion_tag=str(d.get("motion_tag", "") or ""),
            command_ranges=ranges,
            clip_ref=None if clip is None else str(clip),
            clip_ref_reverse=None if clip_rev is None else str(clip_rev),
            weight=float(d.get("weight", 1.0) or 1.0),
        )


# ---------------------------------------------------------------------------
# CommandSchema
# ---------------------------------------------------------------------------

@dataclass
class CommandSchema:
    """Ordered list of :class:`CommandChannel` + sampler metadata.

    The iteration order of ``channels`` IS the policy obs sub-vector
    order. Do not reorder on the way through — the runtime policy
    assumes the exact same ordering as training.
    """

    channels: List[CommandChannel] = field(default_factory=list)

    task_items: List["TaskItem"] = field(default_factory=list)
    """Per-task command envelopes + clip bindings. Populated by the
    ``training_items`` ingress branch of :meth:`from_node_dict`. Empty
    when the schema was built from the gain-based / legacy range path.
    Consumed by the AMP data provider and the env command sampler."""

    # Episode sampler controls (moved here from ILVelocityCommandNode)
    resampling_time_range: Tuple[float, float] = (10.0, 10.0)
    """How often (seconds) the sampler draws a fresh command during an
    episode. ``(lo, hi)`` = uniform sample of resample interval."""

    zero_command_probability: float = 0.0
    """Probability that a freshly-sampled episode uses the all-zero
    command instead of a random point in the channel ranges. Forces the
    policy to learn stationary stance."""

    cmd_step_change_prob: float = 0.01
    """Per-policy-step probability of an abrupt command step-change,
    independent of :attr:`resampling_time_range`. Simulates joystick
    flips (vel_cmd jumping 0→max within 1–2 frames). When triggered,
    the env resamples ``commands`` immediately from the channel ranges
    (curriculum scaling applied via the standard sampler path). 0.0
    disables. 0.01 ≈ one step-change per 100 steps on average."""

    # --- Heading-command mode (legged_gym parity) ---
    heading_command: bool = False
    """When True the yaw channel (``ang_vel_z``) is closed-loop: each
    resample draws a target world-heading in :attr:`heading_range`, and
    every control step the env recomputes ``ang_vel_z = clip(
    heading_control_stiffness * wrap_to_pi(target − current_yaw),
    <item ang_vel_z range>)``. Mirrors legged_gym's
    ``commands.heading_command`` (ang vel command from heading error).
    Default False = open-loop yaw sampling (byte-identical legacy)."""

    heading_control_stiffness: float = 0.5
    """Proportional gain mapping heading error (rad) → yaw rate (rad/s).
    legged_gym uses 0.5. Only consumed when :attr:`heading_command`."""

    heading_range: Tuple[float, float] = (-3.141592653589793, 3.141592653589793)
    """``(lo, hi)`` world-heading target sample range (rad). legged_gym
    uses ``[-π, π]``. Only consumed when :attr:`heading_command`."""

    # --- Deployment-side stick → velocity mapping ---
    mapping_mode: str = MAPPING_LINEAR
    """``linear`` | ``deadzone`` | ``exponential``. Runtime CommandBus
    uses this + the parameters below to shape raw stick input. Training
    sampling is NOT affected (always uniform over channel range)."""

    deadzone: float = 0.0
    """Dead-zone threshold (absolute stick value). Only honoured when
    ``mapping_mode == "deadzone"``. Typical 0.05–0.15."""

    curve_exponent: float = 1.0
    """Power-curve exponent. Only honoured when
    ``mapping_mode == "exponential"``. ``1.0`` = linear, ``2.0`` =
    squared (soft low / aggressive high), ``0.5`` = sqrt (aggressive
    low / soft high)."""

    # --- P2: Walk These Ways gait parameterisation ---
    gait_enabled: bool = False
    """When True, seven extra channels (``gait_frequency``,
    ``gait_phase_fl / fr / rl / rr``, ``body_height``, ``step_height``)
    are appended to :attr:`channels`. Training sampler draws them
    uniformly from the ranges below; runtime uses the preset table
    as D-pad shortcuts with continuous interpolation between picks."""

    # REGISTRY PIN — these three shell defaults mirror
    # registers/data/gait_commands_catalog.json families.quadruped
    # .default_ranges (the single source; read via
    # gait_presets.default_ranges_for_family everywhere family context
    # exists). They CANNOT read the registry at runtime: CommandSchema is
    # constructed inside the Isaac Lab Kit worker venv
    # (il_train_launcher.from_dict), where importing registers →
    # unitport_sdk → PyQt6 crashes. Drift is caught by
    # test_command_contract_v1.test_command_schema_shell_defaults_pin.
    gait_frequency_range: Tuple[float, float] = (1.5, 3.5)
    body_height_range: Tuple[float, float] = (0.28, 0.40)
    step_height_range: Tuple[float, float] = (0.03, 0.15)

    gait_presets: List[Dict[str, Any]] = field(default_factory=list)
    """Serialised list of :class:`GaitPreset` dicts. Not consumed by
    the training sampler (which is uniform), only by the deployment
    CommandBus for D-pad preset cycling. Empty list ⇒ use the bundled
    defaults from :mod:`gait_presets`."""

    # Calibration metadata (populated by the controller calibration
    # wizard in PR-3). Empty dict in PR-1.
    calibration: Dict[str, Any] = field(default_factory=dict)

    def channel_names(self) -> List[str]:
        return [c.name for c in self.channels]

    def get_channel(self, name: str) -> Optional[CommandChannel]:
        for ch in self.channels:
            if ch.name == name:
                return ch
        return None

    def total_dim(self) -> int:
        """Dimension of the command vector as seen by the policy.
        Discrete / button channels count as 1 dim each (they are
        passed as normalised floats)."""
        return len(self.channels)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channels": [c.to_dict() for c in self.channels],
            "task_items": [t.to_dict() for t in self.task_items],
            "resampling_time_range": [
                float(self.resampling_time_range[0]),
                float(self.resampling_time_range[1]),
            ],
            "zero_command_probability": float(self.zero_command_probability),
            "cmd_step_change_prob": float(self.cmd_step_change_prob),
            "heading_command": bool(self.heading_command),
            "heading_control_stiffness": float(self.heading_control_stiffness),
            "heading_range": [
                float(self.heading_range[0]),
                float(self.heading_range[1]),
            ],
            "mapping_mode": str(self.mapping_mode),
            "deadzone": float(self.deadzone),
            "curve_exponent": float(self.curve_exponent),
            "gait_enabled": bool(self.gait_enabled),
            "gait_frequency_range": [
                float(self.gait_frequency_range[0]),
                float(self.gait_frequency_range[1]),
            ],
            "body_height_range": [
                float(self.body_height_range[0]),
                float(self.body_height_range[1]),
            ],
            "step_height_range": [
                float(self.step_height_range[0]),
                float(self.step_height_range[1]),
            ],
            "gait_presets": [dict(p) for p in self.gait_presets],
            "calibration": dict(self.calibration),
        }

    @classmethod
    def from_dict(cls, d: Any) -> "CommandSchema":
        if not isinstance(d, dict):
            return cls()
        channels_raw = d.get("channels") or []
        channels: List[CommandChannel] = []
        seen: set = set()
        for c in channels_raw:
            if not isinstance(c, dict):
                continue
            ch = CommandChannel.from_dict(c)
            if not ch.name or ch.name in seen:
                continue
            seen.add(ch.name)
            channels.append(ch)
        task_items_raw = d.get("task_items") or []
        task_items: List[TaskItem] = []
        if isinstance(task_items_raw, list):
            for t in task_items_raw:
                if not isinstance(t, dict):
                    continue
                ti = TaskItem.from_dict(t)
                if not ti.id:
                    continue
                task_items.append(ti)
        rt = d.get("resampling_time_range") or [10.0, 10.0]
        if isinstance(rt, (list, tuple)) and len(rt) == 2:
            resample = (float(rt[0]), float(rt[1]))
        else:
            resample = (10.0, 10.0)
        mode = str(d.get("mapping_mode", MAPPING_LINEAR) or MAPPING_LINEAR)
        if mode not in SUPPORTED_MAPPING_MODES:
            mode = MAPPING_LINEAR

        def _r(key: str, default: Tuple[float, float]) -> Tuple[float, float]:
            raw = d.get(key)
            if isinstance(raw, (list, tuple)) and len(raw) == 2:
                try:
                    return (float(raw[0]), float(raw[1]))
                except (TypeError, ValueError):
                    pass
            return default

        presets = d.get("gait_presets") or []
        if not isinstance(presets, list):
            presets = []
        return cls(
            channels=channels,
            task_items=task_items,
            resampling_time_range=resample,
            zero_command_probability=float(d.get("zero_command_probability", 0.0) or 0.0),
            cmd_step_change_prob=float(d.get("cmd_step_change_prob", 0.01) or 0.0),
            heading_command=bool(d.get("heading_command", False)),
            heading_control_stiffness=float(d.get("heading_control_stiffness", 0.5) or 0.5),
            heading_range=_r("heading_range", (-3.141592653589793, 3.141592653589793)),
            mapping_mode=mode,
            deadzone=float(d.get("deadzone", 0.0) or 0.0),
            curve_exponent=float(d.get("curve_exponent", 1.0) or 1.0),
            gait_enabled=bool(d.get("gait_enabled", False)),
            gait_frequency_range=_r("gait_frequency_range", (1.5, 3.5)),
            body_height_range=_r("body_height_range", (0.28, 0.40)),
            step_height_range=_r("step_height_range", (0.03, 0.15)),
            gait_presets=[dict(p) for p in presets if isinstance(p, dict)],
            calibration=dict(d.get("calibration") or {}),
        )

    # ------------------------------------------------------------------
    # Node-parameter ingress (PR-1 velocity-only shape)
    # ------------------------------------------------------------------

    @classmethod
    def from_node_dict(
        cls,
        params: Dict[str, Any],
        *,
        family: str = "quadruped",
    ) -> "CommandSchema":
        """Build a CommandSchema from a :class:`TrainingMotionNode`
        parameter dict.

        Primary ingress path: **gain-based lever coupling**. The four
        gain fields ``gain_lin_forward / gain_lin_backward /
        gain_lin_lateral / gain_yaw`` define both

          1. the training velocity ranges (sampled uniformly), and
          2. the deployment stick → velocity mapping
             (``vel = stick × gain``).

        This is the single source of truth for training/deployment
        equivalence — both read the same four numbers.

        Legacy fallback path: if the gain fields are missing but the
        deprecated ``lin_vel_x_range`` / ``lin_vel_y_range`` /
        ``ang_vel_z_range`` JSON fields are present, they are honoured
        so old canvases keep compiling.

        ``family`` (keyword-only) drives the family-keyed gait dispatch:
        the gait channel list and per-family fallback ranges are sourced
        via :func:`_build_gait_channels` + ``gait_presets.default_ranges_for_family``
        so the channel set matches the bound robot's gait_commands
        registry entry. Defaults to ``"quadruped"`` for backward compat
        with the legacy zero-arg call sites. Pass the bound robot's
        ``families[0]`` to enable biped/humanoid emit."""
        def _range(key: str, default_lo: float, default_hi: float) -> Tuple[float, float]:
            raw = params.get(key, f"[{default_lo}, {default_hi}]")
            try:
                import json as _json
                v = _json.loads(str(raw))
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    return (float(v[0]), float(v[1]))
            except Exception:
                pass
            return (default_lo, default_hi)

        def _f(key: str, default: float) -> float:
            raw = params.get(key)
            if raw is None or raw == "":
                return default
            try:
                return float(raw)
            except (TypeError, ValueError):
                return default

        def _has(key: str) -> bool:
            raw = params.get(key)
            return raw is not None and str(raw).strip() != ""

        def _b(key: str, default: bool) -> bool:
            raw = params.get(key)
            if raw is None:
                return default
            return str(raw).strip().lower() == "true"

        runtime_clip = _b("runtime_clip", True)

        # --- P3 primary path: TrainingMotionNode-style per-task items ---
        #
        # When ``params["training_items"]`` is a non-empty dict, the node
        # is driving the schema via the task registry: each entry picks a
        # ``TrainingItem`` (Stand / Walk / Turn / Pace / ...), scales its
        # command_template by a per-item speed pair, optionally overrides
        # individual channel ranges, and binds a reference clip. We then:
        #   * build one :class:`TaskItem` per enabled entry,
        #   * union all per-item channel ranges into the schema's
        #     :class:`CommandChannel` list (policy obs shape),
        #   * fall through to the same gait / resample / mapping handling
        #     as the gain path so those global knobs still apply.
        #
        # If ``training_items`` is absent/empty we fall through unchanged
        # to the existing gain-based / legacy range logic below.
        _ti_raw = params.get("training_items")
        if isinstance(_ti_raw, dict) and _ti_raw:
            try:
                from application.training.training_item_registry import (
                    get_item as get_training_item,
                )
            except Exception:
                get_training_item = None  # type: ignore

            # speed-scaling note: we use ``effective = (lo * smax, hi * smax)``
            # i.e. both bounds scale by the high end of the speed pair.
            # smin is NOT applied to range bounds here — the env sampler
            # honours smin later as a "minimum requested magnitude". This
            # keeps the per-channel envelope monotone in smax and avoids
            # collapsing symmetric channels (-a, a) into a non-symmetric
            # window when smin > 0.
            def _pair(v: Any, default: Tuple[float, float]) -> Tuple[float, float]:
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    try:
                        return (float(v[0]), float(v[1]))
                    except (TypeError, ValueError):
                        return default
                return default

            task_items_built: List[TaskItem] = []
            union_ranges: Dict[str, Tuple[float, float]] = {}

            for item_id, entry in _ti_raw.items():
                if not isinstance(entry, dict):
                    continue
                if not bool(entry.get("enabled", True)):
                    continue
                item_def = None
                if get_training_item is not None:
                    try:
                        item_def = get_training_item(str(item_id))
                    except Exception:
                        item_def = None
                if item_def is None:
                    continue

                # SINGLE derivation shared with the SB3 ItemResolver /
                # command sampler (registers.commands.derive_command_ranges) so
                # both backends sample identical per-channel envelopes.
                from registers.commands import derive_command_ranges as _derive_cr
                _adv = entry.get("advanced")
                _ov = _adv.get("command_overrides") if isinstance(_adv, dict) else None
                eff_ranges: Dict[str, Tuple[float, float]] = {
                    str(k): (float(v[0]), float(v[1]))
                    for k, v in _derive_cr(item_def, entry.get("speed"), _ov).items()
                }

                clip_val = entry.get("clip")
                clip_ref = None if clip_val in (None, "") else str(clip_val)
                clip_rev_val = entry.get("clip_reverse")
                clip_ref_reverse = (
                    None if clip_rev_val in (None, "") else str(clip_rev_val)
                )

                tag = str(
                    getattr(item_def, "default_motion_tag", None)
                    or getattr(item_def, "motion_tag", "")
                    or ""
                )
                task_items_built.append(TaskItem(
                    id=str(item_id),
                    motion_tag=tag,
                    command_ranges=dict(eff_ranges),
                    clip_ref=clip_ref,
                    clip_ref_reverse=clip_ref_reverse,
                    weight=float(entry.get("weight", 1.0) or 1.0),
                ))

                for ch_name, (lo, hi) in eff_ranges.items():
                    if ch_name in union_ranges:
                        prev_lo, prev_hi = union_ranges[ch_name]
                        union_ranges[ch_name] = (min(prev_lo, lo), max(prev_hi, hi))
                    else:
                        union_ranges[ch_name] = (lo, hi)

            if task_items_built:
                _standard_binding = {
                    "lin_vel_x": "gamepad.left_stick_y",
                    "lin_vel_y": "gamepad.left_stick_x",
                    "ang_vel_z": "gamepad.right_stick_x",
                }

                def _unit_for(ch_name: str) -> str:
                    if ch_name.startswith("lin_vel"):
                        return "m/s"
                    if ch_name.startswith("ang_vel"):
                        return "rad/s"
                    return ""

                # Stable ordering: preferred known channels first, then
                # remaining channels sorted alphabetically for determinism.
                preferred = ["lin_vel_x", "lin_vel_y", "ang_vel_z"]
                ordered: List[str] = [n for n in preferred if n in union_ranges]
                ordered += sorted(n for n in union_ranges.keys() if n not in preferred)

                channels_ti: List[CommandChannel] = []
                for ch_name in ordered:
                    lo, hi = union_ranges[ch_name]
                    channels_ti.append(CommandChannel(
                        name=ch_name,
                        kind=CHANNEL_CONTINUOUS,
                        low=lo,
                        high=hi,
                        unit=_unit_for(ch_name),
                        default=0.0,
                        runtime_clip=runtime_clip,
                        binding=_standard_binding.get(ch_name, ""),
                    ))

                raw_mode_ti = str(params.get("mapping_mode", MAPPING_LINEAR) or MAPPING_LINEAR).strip().lower()
                if raw_mode_ti not in SUPPORTED_MAPPING_MODES:
                    raw_mode_ti = MAPPING_LINEAR

                # Family-keyed default range fallbacks: read from
                # gait_presets.default_ranges_for_family so both this
                # path and the env_cfg_compiler Cfg emitters share a
                # single source of truth (7-delta).
                from application.training.isaac_lab.gait_presets import (
                    default_ranges_for_family,
                )
                _fam_ranges_ti = default_ranges_for_family(family)
                gait_on_ti = _b("gait_enabled", False)
                freq_lo_ti, freq_hi_ti = _range(
                    "gait_frequency_range", *_fam_ranges_ti["frequency"]
                )
                bh_lo_ti, bh_hi_ti = _range(
                    "body_height_range", *_fam_ranges_ti["body_height"]
                )
                sh_lo_ti, sh_hi_ti = _range(
                    "step_height_range", *_fam_ranges_ti["step_height"]
                )

                if gait_on_ti:
                    # Family-keyed gait channel dispatch: derive the
                    # ordered channel list from
                    # registers.gait_commands.get_gait_command(family).
                    # Quadruped: 7 channels byte-identical to the
                    # prior hardcoded list (gait_frequency /
                    # gait_phase_fl/fr/rl/rr / body_height /
                    # step_height). Biped/humanoid: 5 channels (1
                    # freq + 2 phase + body_h + step_h).
                    channels_ti.extend(_build_gait_channels(
                        family,
                        freq_range=(freq_lo_ti, freq_hi_ti),
                        body_height_range=(bh_lo_ti, bh_hi_ti),
                        step_height_range=(sh_lo_ti, sh_hi_ti),
                        runtime_clip=runtime_clip,
                    ))

                import json as _json_ti
                raw_presets_ti = params.get("gait_presets", "")
                presets_list_ti: List[Dict[str, Any]] = []
                if raw_presets_ti:
                    try:
                        parsed = _json_ti.loads(str(raw_presets_ti))
                        if isinstance(parsed, list):
                            presets_list_ti = [p for p in parsed if isinstance(p, dict)]
                    except Exception:
                        pass
                if not presets_list_ti:
                    # Fixed import path (was application.training.gait_presets
                    # which never existed; actual module ships under
                    # isaac_lab/). Silent except removed -- if the
                    # bundled presets cannot be imported, that is a real
                    # repo-level broken-ness and must surface as a
                    # raise, not be masked as an empty preset list (the
                    # F-beta1 lesson: fix the bug + remove the
                    # fallback that masked the bug).
                    from application.training.isaac_lab.gait_presets import (
                        default_presets_json_for_family,
                    )
                    presets_list_ti = default_presets_json_for_family(family)

                return cls(
                    channels=channels_ti,
                    task_items=task_items_built,
                    resampling_time_range=_range("resampling_time_range", 10.0, 10.0),
                    zero_command_probability=_f("zero_command_probability", 0.0),
                    cmd_step_change_prob=_f("cmd_step_change_prob", 0.01),
                    heading_command=_b("heading_command", False),
                    heading_control_stiffness=_f("heading_control_stiffness", 0.5),
                    heading_range=_range(
                        "heading_range", -3.141592653589793, 3.141592653589793
                    ),
                    mapping_mode=raw_mode_ti,
                    deadzone=_f("deadzone", 0.0),
                    curve_exponent=_f("curve_exponent", 1.0),
                    gait_enabled=gait_on_ti,
                    gait_frequency_range=(freq_lo_ti, freq_hi_ti),
                    body_height_range=(bh_lo_ti, bh_hi_ti),
                    step_height_range=(sh_lo_ti, sh_hi_ti),
                    gait_presets=presets_list_ti,
                )
            # No usable items resolved → fall through to gain-based path.

        # --- Primary: gain-based lever coupling ---
        have_gains = any(
            _has(k) for k in (
                "gain_lin_forward", "gain_lin_backward",
                "gain_lin_lateral", "gain_yaw",
            )
        )
        if have_gains:
            gain_fwd = _f("gain_lin_forward", 2.0)
            gain_bwd = _f("gain_lin_backward", 1.0)
            gain_lat = _f("gain_lin_lateral", 0.5)
            gain_yaw = _f("gain_yaw", 1.5)
            lin_x_lo, lin_x_hi = -abs(gain_bwd), abs(gain_fwd)
            lin_y_lo, lin_y_hi = -abs(gain_lat), abs(gain_lat)
            yaw_lo, yaw_hi = -abs(gain_yaw), abs(gain_yaw)
        else:
            # --- Legacy fallback: explicit range fields ---
            lin_x_lo, lin_x_hi = _range("lin_vel_x_range", -1.0, 2.0)
            lin_y_lo, lin_y_hi = _range("lin_vel_y_range", -0.5, 0.5)
            yaw_lo, yaw_hi = _range("ang_vel_z_range", -1.5, 1.5)

        channels = [
            CommandChannel(
                name="lin_vel_x", kind=CHANNEL_CONTINUOUS,
                low=lin_x_lo, high=lin_x_hi, unit="m/s",
                default=0.0, runtime_clip=runtime_clip,
                binding="gamepad.left_stick_y",
            ),
            CommandChannel(
                name="lin_vel_y", kind=CHANNEL_CONTINUOUS,
                low=lin_y_lo, high=lin_y_hi, unit="m/s",
                default=0.0, runtime_clip=runtime_clip,
                binding="gamepad.left_stick_x",
            ),
            CommandChannel(
                name="ang_vel_z", kind=CHANNEL_CONTINUOUS,
                low=yaw_lo, high=yaw_hi, unit="rad/s",
                default=0.0, runtime_clip=runtime_clip,
                binding="gamepad.right_stick_x",
            ),
        ]

        raw_mode = str(params.get("mapping_mode", MAPPING_LINEAR) or MAPPING_LINEAR).strip().lower()
        if raw_mode not in SUPPORTED_MAPPING_MODES:
            raw_mode = MAPPING_LINEAR

        # --- P2: gait channels + ranges + presets ---
        # Family-keyed default range fallbacks (single source of truth
        # shared with the env_cfg_compiler Cfg emitters, 7-delta).
        from application.training.isaac_lab.gait_presets import (
            default_presets_json_for_family,
            default_ranges_for_family,
        )
        _fam_ranges = default_ranges_for_family(family)
        gait_on = _b("gait_enabled", False)
        freq_lo, freq_hi = _range(
            "gait_frequency_range", *_fam_ranges["frequency"]
        )
        bh_lo, bh_hi = _range(
            "body_height_range", *_fam_ranges["body_height"]
        )
        sh_lo, sh_hi = _range(
            "step_height_range", *_fam_ranges["step_height"]
        )

        if gait_on:
            # Family-keyed gait channel dispatch. Ordering inside
            # _build_gait_channels is STABLE because the policy obs
            # reads channels in order; the registry's phase_names tuple
            # is load-bearing for the per-foot phase channel order.
            # Quadruped emit is byte-identical to the prior hardcoded
            # 7-channel list (R6 acceptance).
            channels.extend(_build_gait_channels(
                family,
                freq_range=(freq_lo, freq_hi),
                body_height_range=(bh_lo, bh_hi),
                step_height_range=(sh_lo, sh_hi),
                runtime_clip=runtime_clip,
            ))

        # Parse gait presets (JSON string or list). Empty -> fall back
        # to the bundled family-matching default preset table.
        import json as _json
        raw_presets = params.get("gait_presets", "")
        presets_list: List[Dict[str, Any]] = []
        if raw_presets:
            try:
                parsed = _json.loads(str(raw_presets))
                if isinstance(parsed, list):
                    presets_list = [p for p in parsed if isinstance(p, dict)]
            except Exception:
                pass
        if not presets_list:
            # Fixed import path (was application.training.gait_presets
            # which never existed; the actual module is under isaac_lab/).
            # The legacy broken path silently fell back to an empty
            # presets list at the upstream task-items branch's
            # except Exception, and was an outright ImportError on this
            # gain-based path -- in either case the bundled biped
            # presets shipped in 7-gamma never reached the schema.
            presets_list = default_presets_json_for_family(family)

        return cls(
            channels=channels,
            resampling_time_range=_range("resampling_time_range", 10.0, 10.0),
            zero_command_probability=_f("zero_command_probability", 0.0),
            cmd_step_change_prob=_f("cmd_step_change_prob", 0.01),
            heading_command=_b("heading_command", False),
            heading_control_stiffness=_f("heading_control_stiffness", 0.5),
            heading_range=_range(
                "heading_range", -3.141592653589793, 3.141592653589793
            ),
            mapping_mode=raw_mode,
            deadzone=_f("deadzone", 0.0),
            curve_exponent=_f("curve_exponent", 1.0),
            gait_enabled=gait_on,
            gait_frequency_range=(freq_lo, freq_hi),
            body_height_range=(bh_lo, bh_hi),
            step_height_range=(sh_lo, sh_hi),
            gait_presets=presets_list,
        )


# ---------------------------------------------------------------------------
# CommandContract v1 — bundle-embedded policy command contract
# ---------------------------------------------------------------------------
#
# ``deploy_contract.commands`` carries the serialized form of this model
# (``contract_version: 1``). ONE derivation body —
# :func:`derive_command_contract` — feeds BOTH boundaries:
#
#   * export (``strict=True``): any unresolved condition RAISES
#     :class:`CommandContractError` (§8 at the artifact boundary);
#   * canvas preview (``strict=False``): unresolved conditions are collected
#     on :class:`ContractResult.unresolved` and MUST be rendered by the UI.
#
# UI code re-implementing any part of the derivation is a WYSIWYG violation.
#
# The contract speaks COMMAND SPACE: a gait phase offset is ONE scalar
# channel. The sin/cos expansion lives only inside :class:`ObsEncoding`,
# whose dims derive from the ``registers.gait_commands`` ``obs_layout`` pins
# (never re-declared here). Total command dim and total obs dim are two
# different sums — exactly the distinction the Command Pipe Inspector footer
# displays.

CONTRACT_VERSION = 1

# Contract channel kinds. A DIFFERENT vocabulary from the training-side
# CommandChannel kinds (continuous/discrete/button): the contract ships the
# deployment-facing v1 vocabulary. Only ``continuous`` is producible today;
# ``trigger`` / ``enum`` are reserved for skill items (no producer yet).
CONTRACT_KIND_CONTINUOUS = "continuous"
CONTRACT_KIND_TRIGGER = "trigger"
CONTRACT_KIND_ENUM = "enum"
CONTRACT_KINDS = (
    CONTRACT_KIND_CONTINUOUS,
    CONTRACT_KIND_TRIGGER,
    CONTRACT_KIND_ENUM,
)

CONTRACT_SOURCE_VELOCITY = "velocity_item_union"
CONTRACT_SOURCE_GAIT = "gait_registry"
CONTRACT_SOURCE_SKILL = "skill"
CONTRACT_SOURCES = (
    CONTRACT_SOURCE_VELOCITY,
    CONTRACT_SOURCE_GAIT,
    CONTRACT_SOURCE_SKILL,
)

# Latch semantics for trigger / enum channels — per-channel, authored in the
# skill_item (skill_command_path_design.md §6). ``pulse`` = one-shot, auto-clears;
# ``hold`` = active while the bound input is held. ``None`` for continuous/gait
# channels. Presence-gated additive on the v1 contract (no version bump): the
# ``latch`` key is emitted only when set, so non-trigger channels are byte-identical.
CONTRACT_LATCH_PULSE = "pulse"
CONTRACT_LATCH_HOLD = "hold"
_CONTRACT_LATCH_MODES = (CONTRACT_LATCH_PULSE, CONTRACT_LATCH_HOLD)

# Latch modes the *trained policy* actually supports for a skill trigger, declared
# on the contract as a capability (``supported_latch``) so the controller panel can
# gate its latch selector. Slice 3's SkillTriggerCommand only trains the decaying
# PULSE; a sustained HOLD would feed the policy an off-distribution constant it never
# trained on (illusion of success — banned, §8). When a later slice adds training-side
# hold, extend this to (pulse, hold) and every existing panel auto-unlocks with NO
# binding-layer change. Old bundles carrying no supported_latch default to this list
# with a loud WARN (§8(c) legacy) — never to "all supported".
_SKILL_TRAINED_LATCH_MODES = (CONTRACT_LATCH_PULSE,)

OBS_LAYOUT_IDENTITY = "identity"
OBS_LAYOUT_SIN_COS = "sin_cos"

OBS_GROUP_POLICY = "policy"
OBS_GROUP_CRITIC = "critic"
_CONTRACT_OBS_GROUPS = (OBS_GROUP_POLICY, OBS_GROUP_CRITIC)

# Expected obs_layout pin term order for gait entries in
# registers/data/gait_commands_catalog.json. The encoder refuses to guess
# when the registry grows a term it does not know how to map onto channels
# (fail-loud instead of mis-shaping the policy obs).
_GAIT_OBS_PIN_ORDER = ("frequency", "phase_sin_cos", "body_height", "step_height")

# Informational deployment-side binding hints for the standard velocity trio.
# Consumed by the controller binding panel as *defaults*; never by canvas UI.
_STANDARD_VELOCITY_BINDINGS: Dict[str, str] = {
    "lin_vel_x": "gamepad.left_stick_y",
    "lin_vel_y": "gamepad.left_stick_x",
    "ang_vel_z": "gamepad.right_stick_x",
}

# Unresolved condition codes. Also the i18n key anchors at the UI boundary
# (``Unresolved.message`` is the English fallback).
UNRESOLVED_FAMILY_UNKNOWN = "FAMILY_UNKNOWN"
UNRESOLVED_FAMILY_NO_GAIT = "FAMILY_NO_GAIT"
UNRESOLVED_NO_ENABLED_ITEMS = "NO_ENABLED_ITEMS"
UNRESOLVED_UNKNOWN_ITEM = "UNKNOWN_ITEM"
UNRESOLVED_MALFORMED_PARAM = "MALFORMED_PARAM"


@dataclass
class Unresolved:
    """One condition that kept the contract from fully resolving."""

    code: str
    message: str


class CommandContractError(RuntimeError):
    """Strict-path failure of :func:`derive_command_contract`.

    Carries the unresolved list so export pipelines can surface every
    condition at once (mirrors the ``SpecValidationError`` message shape).
    """

    def __init__(self, unresolved: List[Unresolved]):
        self.unresolved = list(unresolved)
        msg = "command contract derivation failed:\n  " + "\n  ".join(
            f"[error:{u.code}] {u.message}" for u in self.unresolved
        )
        super().__init__(msg)


def _req(d: Dict[str, Any], key: str, ctx: str) -> Any:
    if key not in d:
        raise ValueError(f"{ctx}: required key {key!r} is missing")
    return d[key]


def _req_number(d: Dict[str, Any], key: str, ctx: str) -> float:
    v = _req(d, key, ctx)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(
            f"{ctx}.{key}: expected a number, got {type(v).__name__}={v!r}"
        )
    return float(v)


@dataclass
class ObsEncoding:
    """How one contract channel projects into the policy observation.

    ``groups`` records which obs groups the COMPILER auto-injects the
    channel's obs term into (``env_cfg_compiler._observations_cfg``):
    gait channels → ``["policy", "critic"]`` (gait obs terms are appended
    to both groups); velocity channels → ``[]`` because velocity command
    obs is NOT compiler-injected — it enters through the user's
    ``velocity_command`` obs term. An empty ``groups`` therefore means
    "delivered via a user obs term, not injected"; the IL Observation
    node's injected section renders only channels with non-empty groups
    (no double counting against the user term list).
    """

    layout: str
    obs_dims: int
    groups: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.layout, str) or not self.layout.strip():
            raise ValueError("ObsEncoding.layout must be a non-empty string")
        if isinstance(self.obs_dims, bool) or not isinstance(self.obs_dims, int):
            raise ValueError(
                f"ObsEncoding.obs_dims must be an int, got "
                f"{type(self.obs_dims).__name__}={self.obs_dims!r}"
            )
        if self.obs_dims < 0:
            raise ValueError(
                f"ObsEncoding.obs_dims must be >= 0, got {self.obs_dims}"
            )
        if not isinstance(self.groups, list):
            raise ValueError("ObsEncoding.groups must be a list of group names")
        seen: set = set()
        for g in self.groups:
            if g not in _CONTRACT_OBS_GROUPS:
                raise ValueError(
                    f"ObsEncoding.groups entry {g!r} is not one of "
                    f"{list(_CONTRACT_OBS_GROUPS)}"
                )
            if g in seen:
                raise ValueError(f"ObsEncoding.groups: duplicate entry {g!r}")
            seen.add(g)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layout": self.layout,
            "obs_dims": int(self.obs_dims),
            "groups": list(self.groups),
        }

    @classmethod
    def from_dict(cls, d: Any) -> "ObsEncoding":
        ctx = "CommandContract channel obs_encoding"
        if not isinstance(d, dict):
            raise ValueError(f"{ctx}: expected dict, got {type(d).__name__}")
        layout = _req(d, "layout", ctx)
        obs_dims = _req(d, "obs_dims", ctx)
        groups = _req(d, "groups", ctx)
        if isinstance(obs_dims, bool) or not isinstance(obs_dims, int):
            raise ValueError(f"{ctx}.obs_dims: expected int, got {obs_dims!r}")
        if not isinstance(groups, list):
            raise ValueError(
                f"{ctx}.groups: expected list, got {type(groups).__name__}"
            )
        return cls(
            layout=str(layout),
            obs_dims=obs_dims,
            groups=[str(g) for g in groups],
        )


@dataclass
class ContractChannel:
    """One named entry in the v1 command contract.

    The first five keys (``name`` / ``kind`` / ``low`` / ``high`` /
    ``runtime_clip``) are exactly what the runtime ``CommandMapper``
    consumes — a v1 block is a strict superset of the legacy 6-key
    channel entries, so the mapper loads it unchanged (acceptance A5).
    """

    name: str
    kind: str
    low: float
    high: float
    unit: str
    default: float
    runtime_clip: bool
    source: str
    obs_encoding: ObsEncoding
    suggested_binding: Optional[str] = None
    latch: Optional[str] = None          # pulse/hold for trigger/enum; None for continuous
    window_s: Optional[float] = None     # trigger post-pulse decay window (s); deploy rebuilds
                                         # the training SkillTriggerCommand envelope from it
    supported_latch: Optional[List[str]] = None   # capability: latch modes the trained
                                         # policy supports (trigger/enum); the controller
                                         # panel gates its latch selector to these

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("ContractChannel.name must be a non-empty string")
        if self.kind not in CONTRACT_KINDS:
            raise ValueError(
                f"ContractChannel {self.name!r}: kind must be one of "
                f"{list(CONTRACT_KINDS)}, got {self.kind!r}"
            )
        if self.source not in CONTRACT_SOURCES:
            raise ValueError(
                f"ContractChannel {self.name!r}: source must be one of "
                f"{list(CONTRACT_SOURCES)}, got {self.source!r}"
            )
        for fld_name in ("low", "high", "default"):
            v = getattr(self, fld_name)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(
                    f"ContractChannel {self.name!r}: {fld_name} must be a "
                    f"number, got {type(v).__name__}={v!r}"
                )
        self.low = float(self.low)
        self.high = float(self.high)
        self.default = float(self.default)
        if self.high < self.low:
            raise ValueError(
                f"ContractChannel {self.name!r}: high ({self.high}) < "
                f"low ({self.low})"
            )
        if not isinstance(self.runtime_clip, bool):
            raise ValueError(
                f"ContractChannel {self.name!r}: runtime_clip must be a bool"
            )
        if not isinstance(self.obs_encoding, ObsEncoding):
            raise ValueError(
                f"ContractChannel {self.name!r}: obs_encoding must be an "
                f"ObsEncoding instance"
            )
        if self.suggested_binding is not None and (
            not isinstance(self.suggested_binding, str)
            or not self.suggested_binding.strip()
        ):
            raise ValueError(
                f"ContractChannel {self.name!r}: suggested_binding must be "
                f"None or a non-empty string"
            )
        if self.latch is not None:
            if self.latch not in _CONTRACT_LATCH_MODES:
                raise ValueError(
                    f"ContractChannel {self.name!r}: latch must be None or one "
                    f"of {list(_CONTRACT_LATCH_MODES)}, got {self.latch!r}"
                )
            if self.kind == CONTRACT_KIND_CONTINUOUS:
                raise ValueError(
                    f"ContractChannel {self.name!r}: latch is only valid for "
                    f"trigger/enum channels, not continuous"
                )
        if self.window_s is not None:
            if isinstance(self.window_s, bool) or not isinstance(self.window_s, (int, float)):
                raise ValueError(
                    f"ContractChannel {self.name!r}: window_s must be None or a "
                    f"number, got {type(self.window_s).__name__}"
                )
            self.window_s = float(self.window_s)
            if self.window_s <= 0.0:
                raise ValueError(
                    f"ContractChannel {self.name!r}: window_s must be > 0, got "
                    f"{self.window_s}"
                )
            if self.kind == CONTRACT_KIND_CONTINUOUS:
                raise ValueError(
                    f"ContractChannel {self.name!r}: window_s is only valid for "
                    f"trigger/enum channels, not continuous"
                )
        if self.supported_latch is not None:
            if not isinstance(self.supported_latch, (list, tuple)):
                raise ValueError(
                    f"ContractChannel {self.name!r}: supported_latch must be a list "
                    f"of latch modes, got {type(self.supported_latch).__name__}"
                )
            modes = [str(m).strip().lower() for m in self.supported_latch]
            for m in modes:
                if m not in _CONTRACT_LATCH_MODES:
                    raise ValueError(
                        f"ContractChannel {self.name!r}: supported_latch has unknown "
                        f"mode {m!r} (known: {list(_CONTRACT_LATCH_MODES)})"
                    )
            if not modes:
                raise ValueError(
                    f"ContractChannel {self.name!r}: supported_latch must be non-empty "
                    f"when set"
                )
            if self.kind == CONTRACT_KIND_CONTINUOUS:
                raise ValueError(
                    f"ContractChannel {self.name!r}: supported_latch is only valid for "
                    f"trigger/enum channels, not continuous"
                )
            self.supported_latch = modes

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "name": self.name,
            "kind": self.kind,
            "low": float(self.low),
            "high": float(self.high),
            "unit": self.unit,
            "default": float(self.default),
            "runtime_clip": bool(self.runtime_clip),
            "source": self.source,
            "obs_encoding": self.obs_encoding.to_dict(),
            "suggested_binding": self.suggested_binding,
        }
        # Presence-gated: emit ``latch`` / ``window_s`` only for trigger/enum
        # channels, so continuous/gait channels serialize byte-identically.
        if self.latch is not None:
            d["latch"] = self.latch
        if self.window_s is not None:
            d["window_s"] = float(self.window_s)
        if self.supported_latch is not None:
            d["supported_latch"] = list(self.supported_latch)
        return d

    @classmethod
    def from_dict(cls, d: Any) -> "ContractChannel":
        if not isinstance(d, dict):
            raise ValueError(
                f"CommandContract channel: expected dict, got {type(d).__name__}"
            )
        name = str(_req(d, "name", "CommandContract channel"))
        ctx = f"CommandContract channel {name!r}"
        runtime_clip = _req(d, "runtime_clip", ctx)
        if not isinstance(runtime_clip, bool):
            raise ValueError(f"{ctx}.runtime_clip: expected bool, got {runtime_clip!r}")
        sb = d.get("suggested_binding")
        return cls(
            name=name,
            kind=str(_req(d, "kind", ctx)),
            low=_req_number(d, "low", ctx),
            high=_req_number(d, "high", ctx),
            unit=str(_req(d, "unit", ctx)),
            default=_req_number(d, "default", ctx),
            runtime_clip=runtime_clip,
            source=str(_req(d, "source", ctx)),
            obs_encoding=ObsEncoding.from_dict(_req(d, "obs_encoding", ctx)),
            suggested_binding=None if sb is None else str(sb),
            latch=(lambda lv: None if lv is None else str(lv))(d.get("latch")),
            window_s=(lambda w: None if w is None else float(w))(d.get("window_s")),
            supported_latch=(
                (lambda s: list(s) if isinstance(s, (list, tuple)) else None)(
                    d.get("supported_latch")
                )
            ),
        )


@dataclass
class CommandContract:
    """The v1 policy command contract (``deploy_contract.commands``).

    Channel ORDER is load-bearing twice over: the velocity trio comes
    first (the runtime feeds a positional ``[vx, vy, vyaw]`` vector into
    ``CommandMapper.map``), and gait channel order mirrors the registry
    ``phase_names`` tuple (the policy obs sub-vector layout).
    """

    contract_version: int
    channels: List[ContractChannel]
    mapping_mode: str
    deadzone: float
    curve_exponent: float

    def __post_init__(self) -> None:
        if isinstance(self.contract_version, bool) or not isinstance(
            self.contract_version, int
        ):
            raise ValueError(
                f"CommandContract.contract_version must be an int, got "
                f"{self.contract_version!r}"
            )
        if self.contract_version != CONTRACT_VERSION:
            raise ValueError(
                f"CommandContract: contract_version "
                f"{self.contract_version} is not supported (this build "
                f"reads v{CONTRACT_VERSION}). The bundle was produced by a "
                f"newer UnitPort — update the app or re-export the bundle."
            )
        if not isinstance(self.channels, list) or not all(
            isinstance(c, ContractChannel) for c in self.channels
        ):
            raise ValueError(
                "CommandContract.channels must be a list of ContractChannel"
            )
        seen: set = set()
        for c in self.channels:
            if c.name in seen:
                raise ValueError(
                    f"CommandContract: duplicate channel name {c.name!r}"
                )
            seen.add(c.name)
        mode = str(self.mapping_mode)
        if mode not in SUPPORTED_MAPPING_MODES:
            raise ValueError(
                f"CommandContract: mapping_mode {self.mapping_mode!r} not in "
                f"{sorted(SUPPORTED_MAPPING_MODES)}"
            )
        for fld_name in ("deadzone", "curve_exponent"):
            v = getattr(self, fld_name)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(
                    f"CommandContract.{fld_name}: expected a number, got {v!r}"
                )
        self.deadzone = float(self.deadzone)
        self.curve_exponent = float(self.curve_exponent)
        if not (0.0 <= self.deadzone < 1.0):
            raise ValueError(
                f"CommandContract.deadzone must be in [0, 1), got {self.deadzone}"
            )
        if self.curve_exponent <= 0.0:
            raise ValueError(
                f"CommandContract.curve_exponent must be > 0, got "
                f"{self.curve_exponent}"
            )

    # -- render-only queries (UI consumers read these; never re-derive) ----

    def channel_names(self) -> List[str]:
        return [c.name for c in self.channels]

    def total_command_dim(self) -> int:
        """Number of command-space scalars (one per channel)."""
        return len(self.channels)

    def total_obs_dim(self) -> int:
        """Obs dims contributed across ALL channels after encoding."""
        return sum(c.obs_encoding.obs_dims for c in self.channels)

    def channels_for_group(self, group: str) -> List[ContractChannel]:
        """Channels whose obs term is compiler-injected into ``group``."""
        return [c for c in self.channels if group in c.obs_encoding.groups]

    def obs_dim_for_group(self, group: str) -> int:
        return sum(c.obs_encoding.obs_dims for c in self.channels_for_group(group))

    def layouts_present(self) -> List[str]:
        """Ordered unique non-identity encodings (Inspector footnote)."""
        out: List[str] = []
        for c in self.channels:
            lay = c.obs_encoding.layout
            if lay != OBS_LAYOUT_IDENTITY and lay not in out:
                out.append(lay)
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_version": int(self.contract_version),
            "channels": [c.to_dict() for c in self.channels],
            "mapping_mode": str(self.mapping_mode),
            "deadzone": float(self.deadzone),
            "curve_exponent": float(self.curve_exponent),
        }

    @classmethod
    def from_dict(cls, d: Any) -> "CommandContract":
        if not isinstance(d, dict):
            raise ValueError(
                f"CommandContract.from_dict: expected dict, got {type(d).__name__}"
            )
        if "contract_version" not in d:
            raise ValueError(
                "CommandContract.from_dict: 'contract_version' is missing — "
                "this is a legacy velocity-only commands block; the caller "
                "must route it through its §8(c) legacy path instead of "
                "parsing it as a v1 contract."
            )
        channels_raw = _req(d, "channels", "CommandContract")
        if not isinstance(channels_raw, list):
            raise ValueError("CommandContract.channels: expected list")
        return cls(
            contract_version=d["contract_version"],
            channels=[ContractChannel.from_dict(c) for c in channels_raw],
            mapping_mode=str(_req(d, "mapping_mode", "CommandContract")),
            deadzone=_req_number(d, "deadzone", "CommandContract"),
            curve_exponent=_req_number(d, "curve_exponent", "CommandContract"),
        )


@dataclass
class ContractResult:
    """Return shape of :func:`derive_command_contract`.

    ``unresolved`` is empty on the strict path (strict raises instead);
    on the preview path every entry MUST surface in the UI.
    """

    contract: CommandContract
    unresolved: List[Unresolved] = field(default_factory=list)


def _gait_obs_encodings(spec: Any) -> List[ObsEncoding]:
    """Per-channel :class:`ObsEncoding` list for the gait channels of
    ``spec`` (a ``registers.gait_commands.GaitCommandSpec``).

    Aligned with the channel order produced by :func:`_build_gait_channels`
    (frequency, phase..., body_height, step_height). Widths derive from the
    registry ``obs_layout`` pins — never re-declared constants (acceptance
    A3). Groups are ``[policy, critic]`` because
    ``env_cfg_compiler._observations_cfg`` appends the gait obs terms to
    BOTH groups (critic = policy ∪ privileged).
    """
    terms: List[Tuple[str, int]] = []
    for pin in spec.obs_layout:
        term, sep, width_raw = str(pin).partition(":")
        width: Optional[int] = None
        if sep:
            try:
                width = int(width_raw)
            except ValueError:
                width = None
        if not sep or width is None or width < 0:
            raise RuntimeError(
                f"gait_commands obs_layout pin {pin!r} (family="
                f"{spec.family!r}) is not '<term>:<width>'"
            )
        terms.append((term, width))
    term_names = tuple(t for t, _ in terms)
    if term_names != _GAIT_OBS_PIN_ORDER:
        raise RuntimeError(
            f"gait_commands obs_layout for family={spec.family!r} is "
            f"{list(term_names)!r}; this encoder maps exactly "
            f"{list(_GAIT_OBS_PIN_ORDER)!r} onto contract channels — extend "
            f"_gait_obs_encodings before shipping a new layout"
        )
    widths = dict(terms)
    if spec.phase_count <= 0 or widths["phase_sin_cos"] != 2 * spec.phase_count:
        raise RuntimeError(
            f"gait_commands obs_layout phase_sin_cos width "
            f"{widths['phase_sin_cos']} != 2 * phase_count "
            f"({spec.phase_count}) for family={spec.family!r}"
        )
    per_phase = widths["phase_sin_cos"] // spec.phase_count
    groups = [OBS_GROUP_POLICY, OBS_GROUP_CRITIC]
    encodings: List[ObsEncoding] = [
        ObsEncoding(OBS_LAYOUT_IDENTITY, widths["frequency"], list(groups))
    ]
    encodings += [
        ObsEncoding(OBS_LAYOUT_SIN_COS, per_phase, list(groups))
        for _ in range(spec.phase_count)
    ]
    encodings.append(
        ObsEncoding(OBS_LAYOUT_IDENTITY, widths["body_height"], list(groups))
    )
    encodings.append(
        ObsEncoding(OBS_LAYOUT_IDENTITY, widths["step_height"], list(groups))
    )
    return encodings


def derive_command_contract(
    motion_params: Dict[str, Any],
    family: Optional[str],
    *,
    strict: bool,
) -> ContractResult:
    """Derive the v1 :class:`CommandContract` from a ``training_motion`` node.

    Parameters
    ----------
    motion_params:
        The node param dict in canvas form (values may be
        ``{"value": ...}``-wrapped).
    family:
        Robot family resolved from the connected Robot node (preview) /
        ``spec.robot.families[0]`` (export). ``None`` / ``""`` collects
        ``FAMILY_UNKNOWN``.
    strict:
        True = export boundary: any unresolved condition raises
        :class:`CommandContractError`. False = preview boundary: unresolved
        conditions are returned and MUST be rendered by the UI.

    Velocity channel ranges are the per-channel union over enabled training
    items through ``registers.commands.derive_command_ranges`` — the SAME
    SSOT the SB3 sampler and the IsaacLab item builder use. Item ordering
    has no effect on the derivation. An entry without an ``"enabled"`` key
    counts as enabled, matching the SB3 sampler
    (``generic_mujoco_env`` ``payload.get("enabled", True)``) — NOT the
    retired deploy block, which wrongly defaulted to disabled.

    Empty ``training_items`` falls back to
    ``registers.commands.default_items_for_families`` — the SAME defaulting
    ``spec_compiler._resolve_training_items`` applies, so the exported
    envelope matches what the policy actually trained on (the retired block
    exported ``{}`` here: train ≠ deploy).
    """
    import json as _json

    from registers import commands as _commands_registry

    unresolved: List[Unresolved] = []

    def _val(key: str, default: Any = None) -> Any:
        v = motion_params.get(key)
        if v is None:
            return default
        if isinstance(v, dict) and "value" in v:
            return v["value"]
        return v

    def _bool_param(key: str, default: bool) -> bool:
        raw_v = _val(key)
        if raw_v is None or (isinstance(raw_v, str) and not raw_v.strip()):
            return default
        if isinstance(raw_v, bool):
            return raw_v
        s = str(raw_v).strip().lower()
        if s == "true":
            return True
        if s == "false":
            return False
        unresolved.append(Unresolved(
            UNRESOLVED_MALFORMED_PARAM,
            f"training_motion.{key}: expected true/false, got {raw_v!r}; "
            f"using default {default}",
        ))
        return default

    def _float_param(key: str, default: float) -> float:
        raw_v = _val(key)
        if raw_v is None or (isinstance(raw_v, str) and not raw_v.strip()):
            return default
        try:
            return float(raw_v)
        except (TypeError, ValueError):
            unresolved.append(Unresolved(
                UNRESOLVED_MALFORMED_PARAM,
                f"training_motion.{key}: expected a number, got {raw_v!r}; "
                f"using default {default}",
            ))
            return default

    def _range_param(
        key: str, default: Tuple[float, float]
    ) -> Tuple[float, float]:
        raw_v = _val(key)
        if raw_v is None or (isinstance(raw_v, str) and not raw_v.strip()):
            return (float(default[0]), float(default[1]))
        v = raw_v
        if isinstance(raw_v, str):
            try:
                v = _json.loads(raw_v)
            except ValueError:
                v = None
        if isinstance(v, (list, tuple)) and len(v) == 2:
            try:
                return (float(v[0]), float(v[1]))
            except (TypeError, ValueError):
                pass
        unresolved.append(Unresolved(
            UNRESOLVED_MALFORMED_PARAM,
            f"training_motion.{key}: expected [lo, hi], got {raw_v!r}; "
            f"using default {list(default)}",
        ))
        return (float(default[0]), float(default[1]))

    fam = str(family or "").strip()
    if not fam:
        unresolved.append(Unresolved(
            UNRESOLVED_FAMILY_UNKNOWN,
            "no robot family resolved — connect a Robot node to the actor "
            "pipe (canvas) / declare families on the robot's registry entry "
            "(export)",
        ))

    # --- training items → velocity channel union -------------------------
    raw_items = _val("training_items")
    items: Dict[str, Any] = {}
    if isinstance(raw_items, str) and raw_items.strip():
        try:
            parsed = _json.loads(raw_items)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            items = parsed
        else:
            unresolved.append(Unresolved(
                UNRESOLVED_MALFORMED_PARAM,
                "training_motion.training_items: expected a JSON object of "
                "item entries",
            ))
    elif isinstance(raw_items, dict):
        items = raw_items
    elif raw_items is not None:
        unresolved.append(Unresolved(
            UNRESOLVED_MALFORMED_PARAM,
            f"training_motion.training_items: expected dict, got "
            f"{type(raw_items).__name__}",
        ))
    if not items:
        # Mirror spec_compiler._resolve_training_items: an empty canvas dict
        # trains on the registry default item set, so the contract must
        # derive from the same set.
        items = _commands_registry.default_items_for_families(
            [fam] if fam else []
        )

    runtime_clip = _bool_param("runtime_clip", True)

    union: Dict[str, Tuple[float, float]] = {}
    enabled_count = 0
    for item_id, entry in items.items():
        if not isinstance(entry, dict):
            unresolved.append(Unresolved(
                UNRESOLVED_MALFORMED_PARAM,
                f"training_motion.training_items[{str(item_id)!r}]: expected "
                f"an entry dict, got {type(entry).__name__}",
            ))
            continue
        if not bool(entry.get("enabled", True)):
            continue
        item_def = _commands_registry.get_item(str(item_id))
        if item_def is None:
            unresolved.append(Unresolved(
                UNRESOLVED_UNKNOWN_ITEM,
                f"training item {str(item_id)!r} is not in the commands "
                f"registry (builtin + user overlay) — re-register it or "
                f"remove it from the Training Motion node",
            ))
            continue
        adv = entry.get("advanced")
        overrides = adv.get("command_overrides") if isinstance(adv, dict) else None
        eff = _commands_registry.derive_command_ranges(
            item_def, entry.get("speed"), overrides
        )
        enabled_count += 1
        for ch_name, rng in eff.items():
            lo, hi = float(rng[0]), float(rng[1])
            if ch_name in union:
                p_lo, p_hi = union[ch_name]
                union[ch_name] = (min(p_lo, lo), max(p_hi, hi))
            else:
                union[ch_name] = (lo, hi)

    if enabled_count == 0:
        unresolved.append(Unresolved(
            UNRESOLVED_NO_ENABLED_ITEMS,
            "no enabled training item resolved — enable at least one item "
            "on the Training Motion node",
        ))

    preferred = ["lin_vel_x", "lin_vel_y", "ang_vel_z"]
    ordered = [c for c in preferred if c in union]
    ordered += sorted(c for c in union if c not in preferred)

    def _unit_for(ch: str) -> str:
        if ch.startswith("lin_vel"):
            return "m/s"
        if ch.startswith("ang_vel"):
            return "rad/s"
        return ""

    channels: List[ContractChannel] = []
    for ch_name in ordered:
        lo, hi = union[ch_name]
        channels.append(ContractChannel(
            name=ch_name,
            kind=CONTRACT_KIND_CONTINUOUS,
            low=lo,
            high=hi,
            unit=_unit_for(ch_name),
            default=0.0,
            runtime_clip=runtime_clip,
            source=CONTRACT_SOURCE_VELOCITY,
            # groups=[] — velocity command obs is NOT compiler-injected; it
            # enters through the user's ``velocity_command`` obs term (see
            # ObsEncoding docstring).
            obs_encoding=ObsEncoding(OBS_LAYOUT_IDENTITY, 1, []),
            suggested_binding=_STANDARD_VELOCITY_BINDINGS.get(ch_name),
        ))

    # --- gait channels (family-keyed via registers.gait_commands) --------
    gait_enabled = _bool_param("gait_enabled", False)
    if gait_enabled and fam:
        from registers.gait_commands import get_gait_command, has_gait_command

        if not has_gait_command(fam):
            unresolved.append(Unresolved(
                UNRESOLVED_FAMILY_NO_GAIT,
                f"family {fam!r} has no entry in gait_commands_catalog.json "
                f"— gait channels cannot derive; disable gait or extend the "
                f"registry via gait_commands_custom.json",
            ))
        else:
            from application.training.isaac_lab.gait_presets import (
                default_ranges_for_family,
            )

            gait_spec = get_gait_command(fam)
            fam_ranges = default_ranges_for_family(fam)
            freq_range = _range_param(
                "gait_frequency_range", fam_ranges["frequency"]
            )
            bh_range = _range_param("body_height_range", fam_ranges["body_height"])
            sh_range = _range_param("step_height_range", fam_ranges["step_height"])
            base = _build_gait_channels(
                fam,
                freq_range=freq_range,
                body_height_range=bh_range,
                step_height_range=sh_range,
                runtime_clip=runtime_clip,
            )
            encodings = _gait_obs_encodings(gait_spec)
            if len(encodings) != len(base):
                raise RuntimeError(
                    f"_gait_obs_encodings produced {len(encodings)} encodings "
                    f"for {len(base)} gait channels (family={fam!r}); the two "
                    f"derivations must stay aligned"
                )
            for ch, enc in zip(base, encodings):
                channels.append(ContractChannel(
                    name=ch.name,
                    kind=CONTRACT_KIND_CONTINUOUS,
                    low=ch.low,
                    high=ch.high,
                    unit=ch.unit,
                    default=ch.default,
                    runtime_clip=ch.runtime_clip,
                    source=CONTRACT_SOURCE_GAIT,
                    obs_encoding=enc,
                    suggested_binding=None,
                ))

    # --- skill items → trigger channels (presence-gated; Slice 2) ---------
    # A skill is a training package whose command interface is a trigger channel
    # (skill_command_path_design.md §1). ``skill_items`` is a sibling json param
    # on training_motion: {skill_id: {name?, kind?, latch?, enabled?, suggested_binding?}}.
    # Trigger-only for now (design §6: enum reserved, vocabulary already in v1).
    # A trigger's obs is COMPILER-injected into BOTH groups (like gait) since there
    # is no user-authored skill obs term — Slice 3 injects it; Slice 2 declares it.
    raw_skills = _val("skill_items")
    skills: Dict[str, Any] = {}
    if isinstance(raw_skills, str) and raw_skills.strip():
        try:
            parsed_sk = _json.loads(raw_skills)
        except ValueError:
            parsed_sk = None
        if isinstance(parsed_sk, dict):
            skills = parsed_sk
        else:
            unresolved.append(Unresolved(
                UNRESOLVED_MALFORMED_PARAM,
                "training_motion.skill_items: expected a JSON object of skill entries",
            ))
    elif isinstance(raw_skills, dict):
        skills = raw_skills
    elif raw_skills is not None:
        unresolved.append(Unresolved(
            UNRESOLVED_MALFORMED_PARAM,
            f"training_motion.skill_items: expected dict, got "
            f"{type(raw_skills).__name__}",
        ))

    existing_names = {c.name for c in channels}
    for skill_id, entry in skills.items():
        sid = str(skill_id).strip()
        if not sid:
            continue
        if not isinstance(entry, dict):
            unresolved.append(Unresolved(
                UNRESOLVED_MALFORMED_PARAM,
                f"training_motion.skill_items[{skill_id!r}]: expected an entry "
                f"dict, got {type(entry).__name__}",
            ))
            continue
        if not bool(entry.get("enabled", True)):
            continue
        kind = str(entry.get("kind", CONTRACT_KIND_TRIGGER)).strip().lower()
        if kind != CONTRACT_KIND_TRIGGER:
            unresolved.append(Unresolved(
                UNRESOLVED_MALFORMED_PARAM,
                f"training_motion.skill_items[{skill_id!r}]: kind {kind!r} is not "
                f"supported yet — only {CONTRACT_KIND_TRIGGER!r} (enum is reserved "
                f"for a later slice)",
            ))
            continue
        latch = str(entry.get("latch", CONTRACT_LATCH_PULSE)).strip().lower()
        if latch not in _CONTRACT_LATCH_MODES:
            unresolved.append(Unresolved(
                UNRESOLVED_MALFORMED_PARAM,
                f"training_motion.skill_items[{skill_id!r}]: latch {latch!r} not in "
                f"{list(_CONTRACT_LATCH_MODES)}; using {CONTRACT_LATCH_PULSE!r}",
            ))
            latch = CONTRACT_LATCH_PULSE
        if sid in existing_names:
            unresolved.append(Unresolved(
                UNRESOLVED_MALFORMED_PARAM,
                f"training_motion.skill_items[{skill_id!r}]: channel name collides "
                f"with an existing command channel — rename the skill",
            ))
            continue
        existing_names.add(sid)
        sug = entry.get("suggested_binding")
        # window_s = the training post-pulse decay window; carried on the deploy
        # channel so the runtime rebuilds the same envelope (train=deploy, Slice 4).
        try:
            window_s = float(entry.get("window_s", 1.0))
        except (TypeError, ValueError):
            window_s = 1.0
        if window_s <= 0.0:
            window_s = 1.0
        channels.append(ContractChannel(
            name=sid,
            kind=CONTRACT_KIND_TRIGGER,
            low=0.0,
            high=1.0,
            unit="",
            default=0.0,
            runtime_clip=runtime_clip,
            source=CONTRACT_SOURCE_SKILL,
            obs_encoding=ObsEncoding(
                OBS_LAYOUT_IDENTITY, 1, [OBS_GROUP_POLICY, OBS_GROUP_CRITIC],
            ),
            suggested_binding=(str(sug) if sug else None),
            latch=latch,
            window_s=window_s,
            supported_latch=list(_SKILL_TRAINED_LATCH_MODES),
        ))

    # --- deployment-side mapping shape ------------------------------------
    mode = str(_val("mapping_mode", MAPPING_LINEAR) or MAPPING_LINEAR).strip().lower()
    if mode not in SUPPORTED_MAPPING_MODES:
        unresolved.append(Unresolved(
            UNRESOLVED_MALFORMED_PARAM,
            f"training_motion.mapping_mode: {mode!r} not in "
            f"{sorted(SUPPORTED_MAPPING_MODES)}; using {MAPPING_LINEAR!r}",
        ))
        mode = MAPPING_LINEAR
    deadzone = _float_param("deadzone", 0.0)
    if not (0.0 <= deadzone < 1.0):
        unresolved.append(Unresolved(
            UNRESOLVED_MALFORMED_PARAM,
            f"training_motion.deadzone: {deadzone} out of [0, 1); using 0.0",
        ))
        deadzone = 0.0
    curve_exponent = _float_param("curve_exponent", 1.0)
    if curve_exponent <= 0.0:
        unresolved.append(Unresolved(
            UNRESOLVED_MALFORMED_PARAM,
            f"training_motion.curve_exponent: {curve_exponent} must be > 0; "
            f"using 1.0",
        ))
        curve_exponent = 1.0

    contract = CommandContract(
        contract_version=CONTRACT_VERSION,
        channels=channels,
        mapping_mode=mode,
        deadzone=deadzone,
        curve_exponent=curve_exponent,
    )
    if strict and unresolved:
        raise CommandContractError(unresolved)
    return ContractResult(contract=contract, unresolved=unresolved)


def validate_per_item_obs_subset(
    commands_block: Dict[str, Any],
    per_item_obs: Dict[str, Any],
) -> None:
    """Cross-check: every channel named in ``per_item_obs.command_ranges``
    must be one of the contract's channel names.

    The two blocks serve different consumers (CommandMapper vs the deploy
    ObsBuilder tail) and stay separate, but they may no longer drift
    silently. Raises ``ValueError`` on a violation. No-op when either block
    is absent/empty (both are presence-gated).
    """
    if not commands_block or not per_item_obs:
        return
    channel_names = {
        str(c.get("name", ""))
        for c in (commands_block.get("channels") or [])
        if isinstance(c, dict)
    }
    ranges = per_item_obs.get("command_ranges") or {}
    if not isinstance(ranges, dict):
        return  # shape is PerItemObsSpec.from_dict's job, not ours
    for item_id, item_ranges in ranges.items():
        if not isinstance(item_ranges, dict):
            continue
        unknown = sorted(set(map(str, item_ranges.keys())) - channel_names)
        if unknown:
            raise ValueError(
                f"per_item_obs.command_ranges[{str(item_id)!r}] references "
                f"channels {unknown!r} that are not in the command contract "
                f"({sorted(channel_names)!r}) — the per-item obs tail and "
                f"the command contract have drifted; both derive from "
                f"registers.commands, so this is an export pipeline bug"
            )


def deploy_command_block(
    canvas_dict: Dict[str, Any], *, family: str
) -> Dict[str, Any]:
    """Producer half of ``deploy_contract.commands`` — thin strict adapter.

    Locates the ``training_motion`` node in *canvas_dict* and serializes
    ``derive_command_contract(strict=True)`` — the SAME derivation the
    canvas preview renders (WYSIWYG byte equality, acceptance A1).

    Returns ``{}`` ONLY when the canvas dict is absent or carries no
    ``training_motion`` node: the bundle then legitimately ships without a
    command contract (presence-gated additive block) and the load-time
    mapper treats the controller stick as an already-real command. Every
    other unresolved condition RAISES :class:`CommandContractError` (§8 at
    the artifact boundary — the retired "best-effort, never abort export"
    behaviour shipped bundles whose stick mapping silently disagreed with
    the trained envelope).
    """
    if not isinstance(canvas_dict, dict):
        from unitport_sdk import log_warning

        log_warning(
            "[command_schema] deploy_command_block: no canvas dict on the "
            "spec — bundle ships without a command contract (controller "
            "stick will be used as a raw command at deploy)."
        )
        return {}
    node = next(
        (
            n for n in (canvas_dict.get("nodes") or [])
            if isinstance(n, dict) and n.get("schema_id") == "training_motion"
        ),
        None,
    )
    if node is None:
        return {}
    result = derive_command_contract(
        node.get("params") or {}, family, strict=True
    )
    return result.contract.to_dict()


__all__ = [
    "CHANNEL_CONTINUOUS",
    "CHANNEL_DISCRETE",
    "CHANNEL_BUTTON",
    "MAPPING_LINEAR",
    "MAPPING_DEADZONE",
    "MAPPING_EXPONENTIAL",
    "SUPPORTED_MAPPING_MODES",
    "CommandChannel",
    "TaskItem",
    "CommandSchema",
    # -- CommandContract v1 --
    "CONTRACT_VERSION",
    "CONTRACT_KIND_CONTINUOUS",
    "CONTRACT_KIND_TRIGGER",
    "CONTRACT_KIND_ENUM",
    "CONTRACT_KINDS",
    "CONTRACT_SOURCE_VELOCITY",
    "CONTRACT_SOURCE_GAIT",
    "CONTRACT_SOURCE_SKILL",
    "CONTRACT_SOURCES",
    "CONTRACT_LATCH_PULSE",
    "CONTRACT_LATCH_HOLD",
    "OBS_LAYOUT_IDENTITY",
    "OBS_LAYOUT_SIN_COS",
    "OBS_GROUP_POLICY",
    "OBS_GROUP_CRITIC",
    "UNRESOLVED_FAMILY_UNKNOWN",
    "UNRESOLVED_FAMILY_NO_GAIT",
    "UNRESOLVED_NO_ENABLED_ITEMS",
    "UNRESOLVED_UNKNOWN_ITEM",
    "UNRESOLVED_MALFORMED_PARAM",
    "Unresolved",
    "CommandContractError",
    "ObsEncoding",
    "ContractChannel",
    "CommandContract",
    "ContractResult",
    "derive_command_contract",
    "validate_per_item_obs_subset",
    "deploy_command_block",
]
