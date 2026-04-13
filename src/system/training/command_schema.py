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
    required_motion_tag:
        Reference motion tags whose presence is required when this
        channel is active (for Conditional AMP). PR-3 hook.
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
    required_motion_tag: List[str] = field(default_factory=list)

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
            "required_motion_tag": list(self.required_motion_tag),
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
            required_motion_tag=list(d.get("required_motion_tag") or []),
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

    # Episode sampler controls (moved here from ILVelocityCommandNode)
    resampling_time_range: Tuple[float, float] = (10.0, 10.0)
    """How often (seconds) the sampler draws a fresh command during an
    episode. ``(lo, hi)`` = uniform sample of resample interval."""

    zero_command_probability: float = 0.0
    """Probability that a freshly-sampled episode uses the all-zero
    command instead of a random point in the channel ranges. Forces the
    policy to learn stationary stance."""

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
            "resampling_time_range": [
                float(self.resampling_time_range[0]),
                float(self.resampling_time_range[1]),
            ],
            "zero_command_probability": float(self.zero_command_probability),
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
            resampling_time_range=resample,
            zero_command_probability=float(d.get("zero_command_probability", 0.0) or 0.0),
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
    def from_node_dict(cls, params: Dict[str, Any]) -> "CommandSchema":
        """Build a CommandSchema from a :class:`TrainingCommandsNode`
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
        """
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
        gait_on = _b("gait_enabled", False)
        freq_lo, freq_hi = _range("gait_frequency_range", 1.5, 3.5)
        bh_lo, bh_hi = _range("body_height_range", 0.28, 0.40)
        sh_lo, sh_hi = _range("step_height_range", 0.03, 0.15)

        if gait_on:
            # Extend channel list with the 7 gait dimensions. Ordering
            # is STABLE because the policy obs reads channels in order
            # — add new channels at the tail, never insert in the middle.
            channels.extend([
                CommandChannel(
                    name="gait_frequency", kind=CHANNEL_CONTINUOUS,
                    low=freq_lo, high=freq_hi, unit="Hz",
                    default=(freq_lo + freq_hi) * 0.5,
                    runtime_clip=runtime_clip,
                    binding="",
                ),
                CommandChannel(
                    name="gait_phase_fl", kind=CHANNEL_CONTINUOUS,
                    low=0.0, high=1.0, unit="",
                    default=0.0, runtime_clip=runtime_clip, binding="",
                ),
                CommandChannel(
                    name="gait_phase_fr", kind=CHANNEL_CONTINUOUS,
                    low=0.0, high=1.0, unit="",
                    default=0.5, runtime_clip=runtime_clip, binding="",
                ),
                CommandChannel(
                    name="gait_phase_rl", kind=CHANNEL_CONTINUOUS,
                    low=0.0, high=1.0, unit="",
                    default=0.5, runtime_clip=runtime_clip, binding="",
                ),
                CommandChannel(
                    name="gait_phase_rr", kind=CHANNEL_CONTINUOUS,
                    low=0.0, high=1.0, unit="",
                    default=0.0, runtime_clip=runtime_clip, binding="",
                ),
                CommandChannel(
                    name="body_height", kind=CHANNEL_CONTINUOUS,
                    low=bh_lo, high=bh_hi, unit="m",
                    default=(bh_lo + bh_hi) * 0.5,
                    runtime_clip=runtime_clip,
                    binding="",
                ),
                CommandChannel(
                    name="step_height", kind=CHANNEL_CONTINUOUS,
                    low=sh_lo, high=sh_hi, unit="m",
                    default=(sh_lo + sh_hi) * 0.5,
                    runtime_clip=runtime_clip,
                    binding="",
                ),
            ])

        # Parse gait presets (JSON string or list). Empty ⇒ fall back
        # to the bundled default preset table.
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
            from src.system.training.gait_presets import default_presets_json
            presets_list = default_presets_json()

        return cls(
            channels=channels,
            resampling_time_range=_range("resampling_time_range", 10.0, 10.0),
            zero_command_probability=_f("zero_command_probability", 0.0),
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
    "CommandSchema",
]
