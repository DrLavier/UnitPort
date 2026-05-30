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

The PR-1 shape is velocity-only (3 continuous channels). PR-2 will
add gait parameterisation (frequency, phase offsets, body height,
step height) and preset tables. PR-3 adds arbitrary discrete /
button channels plus the consistency-check hooks.

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

                smin, smax = _pair(entry.get("speed"), (0.0, 1.0))
                template = getattr(item_def, "command_template", {}) or {}
                zero_channels = set(getattr(item_def, "zero_channels", []) or [])

                advanced = entry.get("advanced")
                overrides: Dict[str, Tuple[float, float]] = {}
                if isinstance(advanced, dict):
                    raw_ov = advanced.get("command_overrides")
                    if isinstance(raw_ov, dict):
                        for k, v in raw_ov.items():
                            if isinstance(v, (list, tuple)) and len(v) == 2:
                                try:
                                    overrides[str(k)] = (float(v[0]), float(v[1]))
                                except (TypeError, ValueError):
                                    continue

                eff_ranges: Dict[str, Tuple[float, float]] = {}
                for ch_name, rng in template.items():
                    if not isinstance(rng, (list, tuple)) or len(rng) != 2:
                        continue
                    try:
                        lo = float(rng[0])
                        hi = float(rng[1])
                    except (TypeError, ValueError):
                        continue
                    if ch_name in zero_channels:
                        eff_ranges[str(ch_name)] = (0.0, 0.0)
                        continue
                    eff_ranges[str(ch_name)] = (lo * smax, hi * smax)

                # per-entry override wins
                for k, v in overrides.items():
                    eff_ranges[k] = v
                # zero_channels always force (0, 0)
                for k in zero_channels:
                    if k in eff_ranges:
                        eff_ranges[str(k)] = (0.0, 0.0)

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
            mapping_mode=raw_mode,
            deadzone=_f("deadzone", 0.0),
            curve_exponent=_f("curve_exponent", 1.0),
            gait_enabled=gait_on,
            gait_frequency_range=(freq_lo, freq_hi),
            body_height_range=(bh_lo, bh_hi),
            step_height_range=(sh_lo, sh_hi),
            gait_presets=presets_list,
        )


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
]
