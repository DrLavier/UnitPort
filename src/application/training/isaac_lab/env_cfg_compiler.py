"""IsaacLabConfigCompiler — compiles a serialized IL node graph into a
Python @configclass file that Isaac Lab's training CLI can consume.

Input: serialized graph dict from TrainingGraphScene.serialize_training_graph()
Output: Python source string (or written to file)
"""
from __future__ import annotations

import json
import logging
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Deploy-metadata sidecar — Module A authoritative record
# ---------------------------------------------------------------------------
DEPLOY_META_FILENAME = "deploy_meta.json"
DEPLOY_META_SCHEMA_VERSION = 1


# Per-term obs dim resolution (producer-side authority).
#
# Isaac Lab's ``ObservationTermCfg`` does not persist ``dim`` — it's
# determined at runtime by the wrapped function's tensor output shape.
# To make the deploy_meta.json sidecar self-describing without requiring
# the parser to maintain its own dim table (which would silently
# corrupt for non-quadrupeds whenever it drifted from this side), we
# resolve each term's dim HERE — at emit time, against the bound
# RobotSpecRef — and write the integer into the sidecar.
#
# Values:
#   * int N         — fixed dimension (e.g. base_lin_vel is always 3)
#   * "num_joints"  — resolved against ``self._robot.num_joints``
#   * None          — unknown / scanner-dependent (skip in sidecar)
_OBS_TERM_DIM_TABLE: Dict[str, Any] = {
    "base_lin_vel": 3,
    "base_ang_vel": 3,
    "projected_gravity": 3,
    "velocity_command": 3,
    "velocity_commands": 3,
    "base_velocity": 3,
    "commands": 3,
    "joint_pos": "num_joints",
    "joint_vel": "num_joints",
    "last_action": "num_joints",
    # ``height_scan`` dim depends on the ray-caster pattern declared on
    # the Play Ground Setting node; left as None so the parser falls
    # back to env.yaml or skips lenient.
    "height_scan": None,
}


# ---------------------------------------------------------------------------
# Compile-time obs-term contract (Module A)
# ---------------------------------------------------------------------------
# This dataclass + ``_normalize_obs_terms`` define the strongly-typed
# contract the compiler enforces on every observation term, regardless of
# what shape the upstream graph dict arrives in. The framework purpose is
# to make the *compiler* the single authoritative point where obs-term
# metadata is decided: Canvas node parameter formats can evolve freely,
# but every flow through this compiler converges to ``_PerTermObsConfig``
# and then both:
#   (a) writes the resolved fields into the generated ObsTerm Python
#       string (so Isaac Lab sees the exact scale/clip/history that
#       training was meant to use), and
#   (b) serialises the same resolved fields into ``deploy_meta.json``
#       alongside ``unitport_env_cfg.py`` (so the export-side
#       ``manifest_parser`` can recover compile-time decisions that the
#       Isaac Lab YAML dumper would otherwise drop, e.g. ``scale: null``
#       when the ObsTerm was emitted without an explicit ``scale=``).
#
# All fields are ``Optional`` so the contract can faithfully carry
# "not specified by upstream" (None) through to the sidecar. The compiler
# does NOT invent defaults for None fields — it just leaves them off the
# emitted ObsTerm so Isaac Lab's ObservationManager applies its own
# default. The sidecar records exactly what was decided, including the
# Nones. The export side (``manifest_parser._extract_obs_term_meta``)
# treats sidecar-None as a hard error (Phase A's whole point: no silent
# defaulting); the env.yaml-only path is back-compat only.
@dataclass(frozen=True)
class _PerTermObsConfig:
    """Resolved compile-time metadata for one observation term.

    Fields:
      * ``scale`` — scalar or per-component list; None means the compiler
        did not emit ``scale=`` on the ObsTerm (Isaac Lab default applies).
      * ``clip`` — ``(lo, hi)`` 2-tuple; None means the compiler did not
        emit ``clip=`` (caller may still set it to a default elsewhere).
      * ``history_length`` — integer >= 1; None means not emitted.
    """
    scale: Optional[Any] = None             # float | List[float] | None
    clip: Optional[Tuple[float, float]] = None
    history_length: Optional[int] = None


_PER_TERM_OBS_FIELDS = frozenset({"scale", "clip", "history_length"})


def _normalize_obs_terms(
    raw: Any,
    *,
    nid: str,
    schema_id: str = "",
) -> Dict[str, _PerTermObsConfig]:
    """Coerce an upstream ``obs_terms`` value into ``Dict[str, _PerTermObsConfig]``.

    Strict contract — ``obs_terms[term].value`` represents the Isaac Lab
    ``ObservationTermCfg.scale``. There is no "weight" / "enable" /
    legacy-number form; a term not present in the dict is disabled, a term
    present declares its scale explicitly. Accepted forms (per term value):

      1. ``int`` / ``float`` — shorthand for ``{"scale": value}``.
      2. ``list`` / ``tuple`` of numbers — shorthand for ``{"scale": value}``
         (per-component scale, length validated against ``dim`` downstream).
      3. ``dict`` with fields in ``{scale, clip, history_length}`` —
         strongly-typed config. Unknown sub-keys raise.
      4. ``str`` that parses as JSON dict — recursive case (some serialisers
         double-wrap nested JSON); numeric strings also accepted as form (1).

    Anything else — including ``None``, ``bool``, and the historical
    ``{name: number}`` shape where the number was silently dropped — raises
    :class:`CanvasConfigError` pointing at the migrator
    ``bootstrap/migrate_il_observation_obs_terms.py``.

    The function never invents fields — every dict value's ``scale``,
    ``clip``, ``history_length`` is carried verbatim into the sidecar.
    """
    if raw is None or raw == "" or raw == {}:
        raise CanvasConfigError(
            nid=nid, key="obs_terms", schema_id=schema_id,
            reason="obs_terms is empty — at least one observation term is "
                   "required for a trainable policy."
        )
    if not isinstance(raw, dict):
        raise CanvasConfigError(
            nid=nid, key="obs_terms", schema_id=schema_id,
            reason=f"obs_terms must be a dict, got {type(raw).__name__}."
        )

    out: Dict[str, _PerTermObsConfig] = {}
    for term_key, value in raw.items():
        if not isinstance(term_key, str) or not term_key.strip():
            raise CanvasConfigError(
                nid=nid, key="obs_terms", schema_id=schema_id,
                reason=f"obs_terms key {term_key!r} is not a non-empty string."
            )
        term_key = term_key.strip()

        # (4) string-wrapped JSON dict or numeric-string shorthand
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                try:
                    value = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise CanvasConfigError(
                        nid=nid, key="obs_terms", schema_id=schema_id,
                        reason=(
                            f"obs_terms[{term_key!r}] value is a "
                            f"JSON-shaped string but parsing failed: {exc}"
                        ),
                    ) from exc
            else:
                # numeric-like string → treat as scalar scale shorthand
                try:
                    value = float(stripped)
                except ValueError as exc:
                    raise CanvasConfigError(
                        nid=nid, key="obs_terms", schema_id=schema_id,
                        reason=(
                            f"obs_terms[{term_key!r}] string value "
                            f"{stripped!r} is neither a JSON object nor "
                            f"a number."
                        ),
                    ) from exc

        # Reject legacy shapes that used to be silently dropped.
        # ``bool`` is excluded first because ``isinstance(True, int)`` is True.
        if value is None or isinstance(value, bool):
            raise CanvasConfigError(
                nid=nid, key="obs_terms", schema_id=schema_id,
                reason=(
                    f"obs_terms[{term_key!r}] value is "
                    f"{type(value).__name__} — the legacy enable/weight "
                    f"shape was removed. Run "
                    f"bootstrap/migrate_il_observation_obs_terms.py to "
                    f"upgrade this canvas to the strict scale form."
                ),
            )

        # (1)(2) scalar / list shorthand — value IS the scale
        if isinstance(value, (int, float)):
            out[term_key] = _PerTermObsConfig(scale=float(value))
            continue
        if isinstance(value, (list, tuple)):
            try:
                scale_list = [float(v) for v in value]
            except (TypeError, ValueError) as exc:
                raise CanvasConfigError(
                    nid=nid, key="obs_terms", schema_id=schema_id,
                    reason=(
                        f"obs_terms[{term_key!r}] list shorthand contains "
                        f"a non-numeric entry: {exc}"
                    ),
                ) from exc
            out[term_key] = _PerTermObsConfig(scale=scale_list)
            continue

        # (3) strongly-typed dict
        if isinstance(value, dict):
            unknown = set(value.keys()) - _PER_TERM_OBS_FIELDS
            if unknown:
                raise CanvasConfigError(
                    nid=nid, key="obs_terms", schema_id=schema_id,
                    reason=(
                        f"obs_terms[{term_key!r}] has unknown sub-key(s) "
                        f"{sorted(unknown)}; allowed = "
                        f"{sorted(_PER_TERM_OBS_FIELDS)}."
                    ),
                )

            scale_raw = value.get("scale")
            scale: Optional[Any]
            if scale_raw is None:
                scale = None
            elif isinstance(scale_raw, (int, float)):
                scale = float(scale_raw)
            elif isinstance(scale_raw, (list, tuple)):
                try:
                    scale = [float(v) for v in scale_raw]
                except (TypeError, ValueError) as exc:
                    raise CanvasConfigError(
                        nid=nid, key="obs_terms", schema_id=schema_id,
                        reason=(
                            f"obs_terms[{term_key!r}].scale list contains "
                            f"a non-numeric entry: {exc}"
                        ),
                    ) from exc
            else:
                raise CanvasConfigError(
                    nid=nid, key="obs_terms", schema_id=schema_id,
                    reason=(
                        f"obs_terms[{term_key!r}].scale must be null, "
                        f"a number, or a list; got "
                        f"{type(scale_raw).__name__}."
                    ),
                )

            clip_raw = value.get("clip")
            clip: Optional[Tuple[float, float]]
            if clip_raw is None:
                clip = None
            elif (
                isinstance(clip_raw, (list, tuple))
                and len(clip_raw) == 2
                and all(isinstance(v, (int, float)) for v in clip_raw)
            ):
                lo, hi = float(clip_raw[0]), float(clip_raw[1])
                if lo > hi:
                    raise CanvasConfigError(
                        nid=nid, key="obs_terms", schema_id=schema_id,
                        reason=(
                            f"obs_terms[{term_key!r}].clip lo {lo} > hi "
                            f"{hi}."
                        ),
                    )
                clip = (lo, hi)
            else:
                raise CanvasConfigError(
                    nid=nid, key="obs_terms", schema_id=schema_id,
                    reason=(
                        f"obs_terms[{term_key!r}].clip must be null or "
                        f"[lo, hi]; got {clip_raw!r}."
                    ),
                )

            hl_raw = value.get("history_length")
            history_length: Optional[int]
            if hl_raw is None:
                history_length = None
            else:
                try:
                    history_length = int(hl_raw)
                except (TypeError, ValueError) as exc:
                    raise CanvasConfigError(
                        nid=nid, key="obs_terms", schema_id=schema_id,
                        reason=(
                            f"obs_terms[{term_key!r}].history_length must "
                            f"be an integer or null; got {hl_raw!r}."
                        ),
                    ) from exc
                if history_length < 1:
                    raise CanvasConfigError(
                        nid=nid, key="obs_terms", schema_id=schema_id,
                        reason=(
                            f"obs_terms[{term_key!r}].history_length must "
                            f"be >= 1; got {history_length}."
                        ),
                    )

            out[term_key] = _PerTermObsConfig(
                scale=scale, clip=clip, history_length=history_length,
            )
            continue

        if isinstance(value, _PerTermObsConfig):
            out[term_key] = value
            continue

        raise CanvasConfigError(
            nid=nid, key="obs_terms", schema_id=schema_id,
            reason=(
                f"obs_terms[{term_key!r}] value type "
                f"{type(value).__name__} not supported. Allowed: number "
                f"(scalar scale), list of numbers (per-component scale), "
                f"dict {{scale, clip, history_length}}, or JSON-string "
                f"of either."
            ),
        )

    if not out:
        raise CanvasConfigError(
            nid=nid, key="obs_terms", schema_id=schema_id,
            reason="obs_terms normalised to an empty dict.",
        )
    return out


def _format_scale_literal(scale: Any) -> str:
    """Format a resolved scale value as a Python literal for the generated
    ObsTerm string. Scalar → ``"1.0"``; list → ``"(0.5, 0.5, 0.5)"``."""
    if isinstance(scale, (list, tuple)):
        return "(" + ", ".join(f"{float(v)}" for v in scale) + ")"
    return f"{float(scale)}"


class CanvasConfigError(ValueError):
    """Raised when canvas content cannot be faithfully translated to env_cfg.

    Strict contract: config must equal canvas. Any time the compiler would
    otherwise fill in a default, silently coerce a bad value, or skip a
    missing node, it must raise this instead so the user is told which
    canvas node needs editing rather than getting a quietly-wrong run.
    """

    def __init__(
        self,
        *,
        nid: str = "",
        key: str = "",
        schema_id: str = "",
        reason: str,
    ) -> None:
        self.nid = nid
        self.key = key
        self.schema_id = schema_id
        self.reason = reason
        loc_parts: List[str] = []
        if nid:
            loc_parts.append(f"node {nid!r}")
        if schema_id:
            loc_parts.append(f"schema_id={schema_id!r}")
        if key:
            loc_parts.append(f"key={key!r}")
        loc = ", ".join(loc_parts)
        msg = f"[canvas {loc}] {reason}" if loc else f"[canvas] {reason}"
        super().__init__(msg)


# ---------------------------------------------------------------------------
# P2.1 — Walk These Ways gait command term (emitted inline)
# ---------------------------------------------------------------------------
# Emitted verbatim into the generated Isaac Lab config file when the
# canvas Training Commands node has gait_enabled. Defines a 7-dim per-
# env command term (frequency + 4 phase offsets + body_height +
# step_height), an internal phase clock that ticks at ``dt * freq``
# each step, and four obs helper functions that read from the term via
# ``env.command_manager.get_term(command_name)``.
#
# The matching RewardTerm functions (track_gait_phase /
# track_body_height_cmd / track_swing_height_cmd) live in
# task_module_registry.py as IL_REWARD_REGISTRY entries with their own
# ``il_inline`` blocks — they are emitted through the existing custom-
# reward path in ``_custom_reward_funcs`` so this module only needs to
# handle the command-term half of the plumbing.
#
# No runtime testing is possible in this environment; the Isaac Lab
# CommandTerm API (get_term, command, step_dt, num_envs, device) is
# assumed stable. Import errors from a diverging Isaac Lab version
# surface at the first ``training`` launch with a clear stack trace
# pointing at this block — that is the intended "fail loud" behaviour.

# ---------------------------------------------------------------------------
# PhysX GPU buffer sizing constants
# ---------------------------------------------------------------------------
# Per-env contact / patch budgets, scene_type-keyed. Derived from observed
# peak demand on Unitree Go2 quadruped at 8192 envs:
#   - rough terrain (stairs/boxes/slopes): ~150K patches → 18.3/env actual,
#     +75% margin → 32/env. Contacts scale ~2× patches → 64/env.
#   - flat plane: foot contacts only, ~4 patches/env actual; 12/env covers
#     transients (e.g. a robot landing flat with all 4 feet at once on the
#     same physics tick, plus knee skids during reset). Contacts: 24/env.
# These factors are intentional over-allocation against the actual mean —
# PhysX dropping contacts on overflow corrupts physics silently, so the
# cost of generosity (a few MB GPU RAM) buys correctness. Going much
# higher wastes VRAM AND adds a small per-step cost from PhysX walking
# the larger buffer to check for overflow.
_PER_ENV_PHYSX_BUDGET = {
    "flat":  {"patches": 12, "contacts": 24},
    "rough": {"patches": 32, "contacts": 64},
}
# Floors prevent micro-runs (n_envs=8 unit tests, single-robot review)
# from sliding under PhysX's own internal minimums and crashing.
_PHYSX_PATCH_FLOOR = 32768
_PHYSX_CONTACT_FLOOR = 131072

_GAIT_COMMAND_INLINE = '''
# =======================================================================
# UnitPort — Walk These Ways gait command term (UnitPort P2.1)
# =======================================================================
import torch as _gait_torch
import math as _gait_math

from isaaclab.managers import CommandTerm as _UnitportCommandTerm
from isaaclab.managers import CommandTermCfg as _UnitportCommandTermCfg


class UniformGaitCommand(_UnitportCommandTerm):
    """Parameterised gait command — Walk These Ways §3.

    Seven per-env dimensions: ``[frequency, phase_fl, phase_fr,
    phase_rl, phase_rr, body_height, step_height]``.

    * Frequency, body height and step height are sampled uniformly
      from the configured ranges on every resample.
    * Phase offsets are sampled either uniformly in ``[0, 1)^4`` or
      snapped to one of the bundled presets (``phase_mode ==
      "preset"``) depending on the config.
    * A per-env phase clock ticks forward at ``dt * frequency`` each
      step so the reward / obs layer can read the live foot phases
      without keeping their own state (see :meth:`per_foot_phase`).
    """

    cfg: "UniformGaitCommandCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        n = env.num_envs
        device = env.device
        self._command = _gait_torch.zeros((n, 7), device=device)
        self._phase_clock = _gait_torch.zeros(n, device=device)
        if cfg.preset_phases:
            self._preset_phases = _gait_torch.tensor(
                cfg.preset_phases, dtype=_gait_torch.float32, device=device
            )
        else:
            self._preset_phases = None

    @property
    def command(self):
        return self._command

    @property
    def phase_clock(self):
        return self._phase_clock

    def per_foot_phase(self):
        """(num_envs, 4) local phase = (phase_clock + phase_offset) mod 1."""
        return (self._phase_clock.unsqueeze(-1) + self._command[:, 1:5]) % 1.0

    def _resample_command(self, env_ids):
        cfg = self.cfg
        device = self._command.device
        n = env_ids.numel() if hasattr(env_ids, "numel") else len(env_ids)

        def _u(lo, hi):
            return _gait_torch.rand(n, device=device) * (hi - lo) + lo

        freq = _u(cfg.freq_range[0], cfg.freq_range[1])
        body_h = _u(cfg.body_height_range[0], cfg.body_height_range[1])
        step_h = _u(cfg.step_height_range[0], cfg.step_height_range[1])

        if self._preset_phases is not None and cfg.phase_mode == "preset":
            k = self._preset_phases.shape[0]
            idx = _gait_torch.randint(0, k, (n,), device=device)
            phase = self._preset_phases[idx]
        else:
            phase = _gait_torch.rand((n, 4), device=device)

        self._command[env_ids, 0] = freq
        self._command[env_ids, 1:5] = phase
        self._command[env_ids, 5] = body_h
        self._command[env_ids, 6] = step_h
        self._phase_clock[env_ids] = 0.0

    def _update_command(self):
        dt = self._env.step_dt
        freq = self._command[:, 0]
        self._phase_clock = (self._phase_clock + dt * freq) % 1.0

    def _update_metrics(self):
        pass


@configclass
class UniformGaitCommandCfg(_UnitportCommandTermCfg):
    class_type: type = UniformGaitCommand

    asset_name: str = "robot"
    freq_range: tuple = (1.5, 3.5)
    body_height_range: tuple = (0.28, 0.40)
    step_height_range: tuple = (0.03, 0.15)
    phase_mode: str = "uniform"          # "uniform" | "preset"
    preset_phases: list = None           # filled by the compiler


# ── gait obs helpers — read from the command manager ──

def _unitport_gait_frequency_obs(env, command_name="gait_command"):
    return env.command_manager.get_term(command_name).command[:, 0:1]


def _unitport_gait_phase_sin_cos_obs(env, command_name="gait_command"):
    """Sin/cos encoding of (phase_clock + phase_offset) per foot — 8 dims.

    Sin/cos avoids the 1.0 → 0.0 wrap-around discontinuity that would
    confuse a regression head reading the raw phase directly.
    """
    term = env.command_manager.get_term(command_name)
    per_foot = term.per_foot_phase()
    angles = per_foot * (2.0 * _gait_math.pi)
    return _gait_torch.cat(
        [_gait_torch.sin(angles), _gait_torch.cos(angles)], dim=-1
    )


def _unitport_gait_body_height_cmd_obs(env, command_name="gait_command"):
    return env.command_manager.get_term(command_name).command[:, 5:6]


def _unitport_gait_step_height_cmd_obs(env, command_name="gait_command"):
    return env.command_manager.get_term(command_name).command[:, 6:7]
# =======================================================================
'''


_WEIGHTED_VELOCITY_COMMAND_INLINE = '''
# =======================================================================
# UnitPort — Weighted multi-template velocity command (UnitPort §1A)
#
# Drop-in replacement for mdp.UniformVelocityCommandCfg's command term.
# Instead of one union range over [lin_vel_x, lin_vel_y, ang_vel_z],
# holds a list of N sub-templates (one per enabled training_item) and
# samples by multinomial(weights) on each resample, then uniform within
# the picked item's per-channel ranges. Output shape (num_envs, 3) =
# [lin_vel_x, lin_vel_y, ang_vel_z] — same as UniformVelocityCommandCfg
# with heading_command=False, so policy obs sees the same 3D vector.
# Weights are mutable: trainer calls set_weights(w) every N iters to
# re-bias sampling toward low-reward items (§2A-2C adaptive sampling).
# =======================================================================
import torch as _wv_torch

from isaaclab.managers import CommandTerm as _UnitportWVCommandTerm
from isaaclab.managers import CommandTermCfg as _UnitportWVCommandTermCfg


class UnitportWeightedVelocityCommand(_UnitportWVCommandTerm):
    cfg: "UnitportWeightedVelocityCommandCfg"

    _CHANNELS = ("lin_vel_x", "lin_vel_y", "ang_vel_z")

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        n = env.num_envs
        device = env.device
        items = list(cfg.items or [])
        m = len(items)
        if m == 0:
            raise ValueError(
                "UnitportWeightedVelocityCommandCfg.items is empty — "
                "canvas must enable at least one training_item."
            )

        lo_t = _wv_torch.zeros((m, 3), device=device)
        hi_t = _wv_torch.zeros((m, 3), device=device)
        bidir_t = _wv_torch.zeros((m, 3), dtype=_wv_torch.bool, device=device)
        for i, item in enumerate(items):
            r = item.get("ranges", {})
            bd = item.get("bidirectional", {})
            for j, ch in enumerate(self._CHANNELS):
                rng = r.get(ch, (0.0, 0.0))
                lo_t[i, j] = float(rng[0])
                hi_t[i, j] = float(rng[1])
                bidir_t[i, j] = bool(bd.get(ch, False))
        self._ranges_lo = lo_t
        self._ranges_hi = hi_t
        self._bidirectional = bidir_t
        self._item_ids_list = [str(item.get("id", f"item_{i}")) for i, item in enumerate(items)]
        # Phase ids resolved at compile time by env_cfg_compiler (see
        # _build_training_item_dict in the compiler). Indexed identically
        # to _item_ids_list. Empty string = motion_tag has no registered
        # phase, mask treats it as unmatched. The runtime mask helper
        # reads this list directly so it does NOT have to import
        # registers (RELEASE/src is not on the Isaac Lab worker's
        # sys.path; the import silently failed and zeroed every
        # phase-aware reward in the previous training run).
        self._item_phase_ids = [str(item.get("phase_id", "") or "") for item in items]

        # Initial weights: cfg list (validated) → uniform fallback.
        init_w = list(cfg.initial_weights) if cfg.initial_weights else []
        if len(init_w) != m:
            init_w = [1.0 / m] * m
        w_t = _wv_torch.tensor(init_w, dtype=_wv_torch.float32, device=device)
        s = w_t.sum()
        if s.item() <= 0:
            w_t = _wv_torch.full((m,), 1.0 / m, dtype=_wv_torch.float32, device=device)
        else:
            w_t = w_t / s
        self._weights = w_t

        self._command = _wv_torch.zeros((n, 3), device=device)
        self._current_item_id = _wv_torch.full((n,), -1, dtype=_wv_torch.long, device=device)
        self._is_standing = _wv_torch.zeros(n, dtype=_wv_torch.bool, device=device)

        self._weight_floor = float(cfg.weight_floor)
        self._weight_ceil = float(cfg.weight_ceil)

    @property
    def command(self):
        return self._command

    @property
    def weights(self):
        return self._weights

    @property
    def item_ids(self):
        return list(self._item_ids_list)

    def current_item_id(self):
        return self._current_item_id

    def set_weights(self, w):
        if w is None:
            return False
        try:
            w = w.to(self._weights.device).to(self._weights.dtype)
        except Exception:
            return False
        if w.numel() != self._weights.numel():
            return False
        w = _wv_torch.clamp(w, min=self._weight_floor, max=self._weight_ceil)
        s = w.sum()
        if s.item() <= 0:
            return False
        self._weights = w / s
        return True

    def _resample_command(self, env_ids):
        cfg = self.cfg
        device = self._command.device
        if hasattr(env_ids, "numel"):
            n = int(env_ids.numel())
        else:
            n = len(env_ids)
        if n == 0:
            return

        standing = (
            _wv_torch.rand(n, device=device) < float(cfg.rel_standing_envs)
        )
        self._is_standing[env_ids] = standing

        item_ids = _wv_torch.multinomial(
            self._weights, num_samples=n, replacement=True
        )

        lo = self._ranges_lo[item_ids]
        hi = self._ranges_hi[item_ids]
        bd = self._bidirectional[item_ids]
        u = _wv_torch.rand((n, 3), device=device)
        vals = lo + u * (hi - lo)

        flip_mask = (
            _wv_torch.rand((n, 3), device=device) < 0.5
        ) & bd
        sign = _wv_torch.where(
            flip_mask, -_wv_torch.ones_like(vals), _wv_torch.ones_like(vals)
        )
        vals = vals * sign

        vals = _wv_torch.where(
            standing.unsqueeze(-1), _wv_torch.zeros_like(vals), vals
        )
        self._command[env_ids] = vals
        self._current_item_id[env_ids] = _wv_torch.where(
            standing, _wv_torch.full_like(item_ids, -1), item_ids
        )

    def _update_command(self):
        # Velocity commands don't evolve between resamples — except when
        # cmd_step_change_prob > 0, in which case each env independently
        # re-samples its own command at the configured per-step probability.
        # This sits on top of the standard resampling_time_range cadence
        # (CommandTerm already calls _resample_command at those boundaries);
        # the per-step probability adds short-burst command churn for envs
        # the canvas wants more agility on.
        p = float(getattr(self.cfg, "cmd_step_change_prob", 0.0) or 0.0)
        if p > 0.0:
            n = self._command.shape[0]
            device = self._command.device
            flip = _wv_torch.rand(n, device=device) < p
            env_ids = flip.nonzero(as_tuple=False).squeeze(-1)
            if env_ids.numel() > 0:
                self._resample_command(env_ids)

    def _update_metrics(self):
        pass


@configclass
class UnitportWeightedVelocityCommandCfg(_UnitportWVCommandTermCfg):
    class_type: type = UnitportWeightedVelocityCommand

    asset_name: str = "robot"
    # items / initial_weights are emitted by the compiler — None defaults
    # exist only so a stand-alone Python import of this generated file
    # doesn't blow up on dataclass evaluation order.
    items: list = None
    initial_weights: list = None
    rel_standing_envs: float = 0.05
    weight_floor: float = 0.03
    weight_ceil: float = 0.30
    adaptive_enabled: bool = False
    # Per-step command resample probability — canvas training_motion.
    # 0 keeps the legacy "only resample at resampling_time_range" behavior.
    cmd_step_change_prob: float = 0.0
# =======================================================================
'''


# =======================================================================
# Phase mask helper — emitted in compile() when reward terms declared
# applies_to. Wraps a reward function so it only contributes on envs
# whose currently-active velocity-command item's motion_tag resolves to
# one of the phase ids in ``phases_to_match``. The helper resolves the
# motion_tag → phase_id table at first call via registers.motion_phases,
# and caches an item_index → phase_id mapping on the command term so
# subsequent steps are O(1) per env. See the Reward × MotionPhase
# isolation plan (custom-mods-canvas-issaclab-go2-ppo) for the rationale.
# =======================================================================
_PHASE_MASK_HELPER_INLINE = '''
# =======================================================================
# UnitPort reward × motion-phase mask helper
# (emitted by env_cfg_compiler when any reward term sets applies_to)
# =======================================================================

# Module-level flag so the diagnostic dump only fires once per training
# run instead of every reward function call. Resets to False every time
# this generated module is re-imported (= new training run).
_UNITPORT_MASK_DIAGNOSTIC_PRINTED = False


def _unitport_resolve_item_phases(cmd):
    """Return item_index -> phase_id list, pre-compiled into the cfg.

    Phase ids are baked in at compile time: env_cfg_compiler's
    _build_training_item_dict resolves each item's motion_tag through
    registers.motion_phases.resolve_phase and writes the result into the
    cfg.items[i]["phase_id"] field. UnitportWeightedVelocityCommand
    then copies that list into self._item_phase_ids on __init__.

    We deliberately do NOT import registers at runtime: the Isaac Lab
    worker process runs from a separate venv without RELEASE/src on
    sys.path, so any `from registers import ...` here raises
    ModuleNotFoundError. The previous implementation caught the
    exception and fell back to a list of empty strings, which silently
    zeroed every phase-aware reward — the policy then had no signal for
    locomotion / agile rewards at all, which is exactly the
    "stand command yet limbs flailing" bug we are fixing.
    """
    return list(getattr(cmd, "_item_phase_ids", []) or [])


def _unitport_mask_diagnostic(cmd, phases_set):
    """One-shot diagnostic dump on the first reward-mask invocation.

    Prints the (item_id, phase_id) mapping so the user can verify in
    training log that compile-time phase injection actually landed.
    Empty phase_ids here mean motion_tags fell through resolve_phase —
    look at registers.motion_phases.motion_tags coverage.
    """
    global _UNITPORT_MASK_DIAGNOSTIC_PRINTED
    if _UNITPORT_MASK_DIAGNOSTIC_PRINTED:
        return
    _UNITPORT_MASK_DIAGNOSTIC_PRINTED = True
    item_ids = list(getattr(cmd, "_item_ids_list", []) or [])
    item_phases = list(getattr(cmd, "_item_phase_ids", []) or [])
    pairs = list(zip(item_ids, item_phases))
    print(
        "[unitport_phase_mask] activated. items_to_phase={} "
        "first_phases_set={}".format(pairs, sorted(phases_set))
    )


def unitport_phase_mask(base_fn, phases_to_match):
    """Wrap a reward func so it contributes only when the env's active
    task item's motion_tag falls into ``phases_to_match``. Standing envs
    (item_id == -1) are treated as the static phase.

    The wrapper degrades gracefully: if the command manager / term is
    unavailable, the base func passes through unchanged so the reward
    is still computed (rather than zeroed and silently breaking PPO).

    Signature preservation: Isaac Lab's RewardManager validates each term
    via ``inspect.signature(term_cfg.func).parameters`` and rejects the
    term when the signature doesn't match the ``params=`` keys declared
    on the RewTerm. We dynamically rebuild a wrapper whose signature is
    *exactly* the base function's — including parameter names + defaults
    — so inspect sees ``(env, asset_cfg)`` (or whatever the base reward
    takes) rather than ``(env, *args, **kwargs)``. Falls back to the
    generic wrapper if signature introspection fails on the base.
    """
    import functools as _functools
    import inspect as _inspect
    import torch as _torch
    phases_set = frozenset(phases_to_match or ())

    def _core(env, *args, **kwargs):
        base = base_fn(env, *args, **kwargs)
        try:
            cmd = env.command_manager.get_term("base_velocity")
        except Exception:
            return base
        item_id = getattr(cmd, "_current_item_id", None)
        if item_id is None:
            return base
        _unitport_mask_diagnostic(cmd, phases_set)
        item_phases = _unitport_resolve_item_phases(cmd)
        if not item_phases:
            return base
        device = base.device if hasattr(base, "device") else item_id.device
        mask = _torch.zeros(item_id.shape[0], dtype=base.dtype, device=device)
        for idx, phase_id in enumerate(item_phases):
            if phase_id and phase_id in phases_set:
                mask = mask + (item_id == idx).to(base.dtype)
        if "static" in phases_set:
            mask = mask + (item_id == -1).to(base.dtype)
        mask = _torch.clamp(mask, max=1.0)
        return base * mask

    # Try to rebuild the exact base signature so Isaac Lab's parameter
    # introspection (manager_base._resolve_common_term_cfg) sees the
    # real keyword names. We can't rely on functools.wraps alone here —
    # Isaac Lab calls inspect.signature() directly on term_cfg.func and
    # in some code paths does not follow __wrapped__, so the wrapper
    # itself must carry the right Parameter list.
    try:
        sig = _inspect.signature(base_fn)
        params = list(sig.parameters.values())
        if not params:
            raise ValueError("base reward has no parameters (need at least env)")

        positional_names = [
            p.name for p in params
            if p.kind in (
                _inspect.Parameter.POSITIONAL_ONLY,
                _inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        keyword_only_names = [
            p.name for p in params
            if p.kind is _inspect.Parameter.KEYWORD_ONLY
        ]

        def wrapped(*call_args, **call_kwargs):
            bound = sig.bind(*call_args, **call_kwargs)
            bound.apply_defaults()
            # The first positional argument is always env by Isaac Lab
            # contract. Forward everything to _core which has access to
            # base_fn via closure.
            return _core(*bound.args, **bound.kwargs)

        # Replace wrapped's signature so inspect.signature(wrapped) and
        # inspect.signature(wrapped).parameters return the base function's
        # parameter list verbatim.
        wrapped.__signature__ = sig
        wrapped = _functools.wraps(base_fn)(wrapped)
        return wrapped
    except (TypeError, ValueError):
        # Builtin / C-level reward that resists signature introspection.
        # Fall back to the generic wrapper — Isaac Lab will accept it
        # because builtins with no inspectable signature skip the check.
        @_functools.wraps(base_fn)
        def fallback(env, *args, **kwargs):
            return _core(env, *args, **kwargs)
        return fallback
# =======================================================================
'''


# =======================================================================
# UnitPort reward × per-item mask helper
# Emitted when the canvas has multiple ``rewards`` nodes fanning out to
# different training_motion items via ``reward_in__<item_id>`` ports.
# Differs from unitport_phase_mask: that one masks by motion_phase
# (coarse — locomotion / static / agile). This one masks by individual
# item index, allowing different rewards nodes to feed disjoint subsets
# of items that share the same phase (e.g. turn vs walk vs strafe all
# map to phase=locomotion).
# =======================================================================
_ITEM_MASK_HELPER_INLINE = '''
# =======================================================================
# UnitPort reward × per-item mask helper
# (emitted by env_cfg_compiler when the canvas has multiple rewards
# nodes fanning out to different training_motion items)
# =======================================================================

def unitport_item_mask(base_fn, item_indices_to_match):
    """Wrap a reward func so it contributes only when the env's active
    task item index is in ``item_indices_to_match``.

    Signature preservation mirrors unitport_phase_mask: Isaac Lab's
    RewardManager validates each term via inspect.signature, so the
    wrapper must carry the base function's parameter list verbatim.
    """
    import functools as _functools
    import inspect as _inspect
    import torch as _torch
    idx_set = frozenset(int(i) for i in (item_indices_to_match or ()))

    def _core(env, *args, **kwargs):
        base = base_fn(env, *args, **kwargs)
        try:
            cmd = env.command_manager.get_term("base_velocity")
        except Exception:
            return base
        item_id = getattr(cmd, "_current_item_id", None)
        if item_id is None:
            return base
        device = base.device if hasattr(base, "device") else item_id.device
        mask = _torch.zeros(item_id.shape[0], dtype=base.dtype, device=device)
        for idx in idx_set:
            mask = mask + (item_id == idx).to(base.dtype)
        mask = _torch.clamp(mask, max=1.0)
        return base * mask

    try:
        sig = _inspect.signature(base_fn)
        params = list(sig.parameters.values())
        if not params:
            raise ValueError("base reward has no parameters (need at least env)")

        def wrapped(*call_args, **call_kwargs):
            bound = sig.bind(*call_args, **call_kwargs)
            bound.apply_defaults()
            return _core(*bound.args, **bound.kwargs)

        wrapped.__signature__ = sig
        wrapped = _functools.wraps(base_fn)(wrapped)
        return wrapped
    except (TypeError, ValueError):
        @_functools.wraps(base_fn)
        def fallback(env, *args, **kwargs):
            return _core(env, *args, **kwargs)
        return fallback
# =======================================================================
'''


class IsaacLabConfigCompiler:
    """Compiles a UnitPort IL node graph into an Isaac Lab Python config file.

    Phase 5: ``robot`` is the bound :class:`RobotSpecRef` snapshot of the
    canvas's Robot node.  It is the *only* source of IR-role ↔ physical-joint
    translation; the substrate-emit sites (init_state.joint_pos, action
    JointPositionActionCfg.joint_names) build a :class:`JointIRResolver`
    from it and translate IR keys at the last mile.  When ``robot`` is None
    the compiler skips IR validation (legacy / unit-test path); production
    callers (IsaacLabTrainingTask) always pass a robot.
    """

    def __init__(
        self,
        graph: Dict[str, Any],
        *,
        robot: Optional["RobotSpecRef"] = None,
    ) -> None:
        self._graph = graph
        self._robot: Optional["RobotSpecRef"] = robot
        self._nodes: Dict[str, Dict[str, Any]] = {}       # id → node dict
        self._params: Dict[str, Dict[str, str]] = {}      # id → parameters
        self._types: Dict[str, str] = {}                   # id → node_type
        self._edges: List[Dict[str, str]] = []
        self._downstream: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
        self._upstream: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
        # Set by ``_rewards_cfg`` when at least one reward term carried
        # a non-empty ``applies_to`` field — drives ``compile()`` to emit
        # the phase-mask helper preamble below the custom reward funcs.
        self._needs_phase_mask_helper: bool = False
        # Set by ``_rewards_cfg`` when the canvas has >1 rewards node or
        # any per-item ``reward_in__<item_id>`` edge — drives ``compile()``
        # to emit the item-mask helper preamble.
        self._needs_item_mask_helper: bool = False
        self._parse()

    @property
    def _joint_ir_resolver(self):
        """Lazy :class:`JointIRResolver` for the bound robot.

        Returns None when ``self._robot`` is None — emit sites detect this
        and either raise (if user-set joint dict needs translation) or fall
        through to a regex-passthrough path.
        """
        if self._robot is None:
            return None
        cache = getattr(self, "_joint_ir_resolver_cache", None)
        if cache is None:
            from application.training.joint_ir import JointIRResolver
            cache = JointIRResolver(self._robot)
            self._joint_ir_resolver_cache = cache
        return cache

    def _parse(self) -> None:
        # RELEASE canvas dict shape (CanvasPage.to_workflow_dict / canvas .json):
        #   nodes:  [{id, schema_id, kind, params: {key: {name, value, param_type}}, ui, opaque_code}]
        #   edges:  [{source_node, source_port, target_node, target_port}]
        # Compiler internal shape (kept identical to DEMO so all _p / _pf / _pi
        # callers untouched):
        #   self._params[nid][key] -> raw VALUE (str | int | float | bool | json-string)
        for n in self._graph.get("nodes", []):
            nid = str(n["id"])
            self._nodes[nid] = n
            # RELEASE wraps each param as {name, value, param_type}; unwrap
            # to a flat {key: value_string} dict so _p / _pf / _pi see the
            # same shape DEMO had ({key: stringified_value}).
            raw_params = n.get("params") or n.get("parameters") or {}
            unwrapped: Dict[str, Any] = {}
            for key, val in raw_params.items():
                if isinstance(val, dict) and "value" in val:
                    unwrapped[key] = val["value"]
                else:
                    unwrapped[key] = val
            self._params[nid] = unwrapped
            # RELEASE uses schema_id; DEMO used node_type. Accept both.
            self._types[nid] = str(n.get("schema_id") or n.get("node_type") or "")
        self._edges = self._graph.get("edges", [])
        for e in self._edges:
            # RELEASE edge keys: source_node / source_port / target_node / target_port
            # DEMO edge keys:    src_id / src_slot / dst_id / dst_slot
            src = str(e.get("source_node") or e.get("src_id") or "")
            dst = str(e.get("target_node") or e.get("dst_id") or "")
            ss = str(e.get("source_port") or e.get("src_slot") or "")
            ds = str(e.get("target_port") or e.get("dst_slot") or "")
            if not src or not dst:
                raise CanvasConfigError(
                    nid="",
                    key="edges",
                    schema_id="",
                    reason=(
                        f"Edge with missing endpoint: src={src!r} dst={dst!r} "
                        f"(source_port={ss!r} target_port={ds!r}). Strict mode "
                        f"rejects dangling edges — repair the connection in "
                        f"the canvas or delete it."
                    ),
                )
            self._downstream[src].append((dst, ss, ds))
            self._upstream[dst].append((src, ss, ds))

    def _find_by_type(self, node_type: str) -> List[str]:
        return [nid for nid, nt in self._types.items() if nt == node_type]

    # ------------------------------------------------------------------
    # P2.1 gait helpers
    # ------------------------------------------------------------------

    def _gait_enabled(self) -> bool:
        """True when the Training Commands node has gait_enabled = true."""
        cmd_ids = self._find_by_type("training_motion")
        if not cmd_ids:
            return False
        raw = self._p(cmd_ids[0], "gait_enabled")
        return str(raw).strip().lower() == "true"

    def _gait_range_tuple(self, key: str, default_lo: float, default_hi: float) -> str:
        """Parse a ``[lo, hi]`` JSON param into a Python tuple literal."""
        cmd_ids = self._find_by_type("training_motion")
        if not cmd_ids:
            return f"({default_lo}, {default_hi})"
        raw = self._p(cmd_ids[0], key, f"[{default_lo}, {default_hi}]")
        try:
            import json as _json
            v = _json.loads(raw)
            if isinstance(v, (list, tuple)) and len(v) == 2:
                return f"({float(v[0])}, {float(v[1])})"
        except Exception:
            pass
        return f"({default_lo}, {default_hi})"

    def _gait_preset_phase_literal(self) -> str:
        """Extract the 4-tuple phase vectors from the gait_presets JSON
        param and return a Python list-of-lists literal the generated
        config can hand straight to ``UniformGaitCommandCfg.preset_phases``.
        Falls back to the bundled default set on parse failure.
        """
        cmd_ids = self._find_by_type("training_motion")
        raw = self._p(cmd_ids[0], "gait_presets") if cmd_ids else ""
        presets: List[List[float]] = []
        if raw:
            try:
                import json as _json
                parsed = _json.loads(str(raw))
                if isinstance(parsed, list):
                    for p in parsed:
                        if not isinstance(p, dict):
                            continue
                        phase = p.get("phase")
                        if (
                            isinstance(phase, (list, tuple))
                            and len(phase) == 4
                        ):
                            presets.append([
                                float(phase[0]), float(phase[1]),
                                float(phase[2]), float(phase[3]),
                            ])
            except Exception:
                pass
        if not presets:
            try:
                from application.training.isaac_lab.gait_presets import DEFAULT_PRESETS
                for pr in DEFAULT_PRESETS:
                    presets.append([float(x) for x in pr.phase])
            except Exception:
                pass
        if not presets:
            return "[]"
        # Render as literal: [[0.0, 0.5, 0.5, 0.0], ...]
        return "[" + ", ".join(
            "[" + ", ".join(f"{x}" for x in ph) + "]" for ph in presets
        ) + "]"

    # ------------------------------------------------------------------
    # UnitPort §1B — Per-item velocity command range builders
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_speed_to_range(template_range, speed_lo, speed_hi):
        """Map a training_item's speed=[lo, hi] onto its speed_channel.

        Convention (per the contract in ``registers/commands.py`` docstring
        — "ranges at speed=1 that the UI scales by the Speed slider"):

          * template_range all-positive (walk.lin_vel_x = [0, 1]):
              → (speed_lo, speed_hi), bidirectional=False
          * template_range all-negative (backward.lin_vel_x = [-1.2, 0]):
              → (-speed_hi, -speed_lo), bidirectional=False  (sign-mirrored)
          * template_range crosses zero (turn.ang_vel_z = [-1.5, 1.5]):
              → (abs(speed_lo), abs(speed_hi)) as magnitude,
                bidirectional=True (the runtime CommandTerm flips sign
                with prob 0.5 per resample).
        """
        t_lo = float(template_range[0])
        t_hi = float(template_range[1])
        s_lo = float(speed_lo)
        s_hi = float(speed_hi)
        if t_lo >= 0 and t_hi >= 0:
            return (s_lo, s_hi), False
        if t_hi <= 0 and t_lo <= 0:
            return (-s_hi, -s_lo), False
        a, b = abs(s_lo), abs(s_hi)
        if a > b:
            a, b = b, a
        return (a, b), True

    def _build_training_item_dict(self, cid: str, item_id: str, item_cfg: dict) -> dict:
        """Build the per-item dict consumed by the inline weighted
        CommandTerm. Pulls the template from registers.commands, then
        applies the speed range on speed_channel and overrides on any
        channel from advanced.command_overrides.

        ``phase_id`` is resolved at compile time via
        :mod:`registers.motion_phases` and baked into the emitted dict so
        the runtime ``unitport_phase_mask`` helper does *not* have to
        import ``registers`` (the Isaac Lab worker process runs from a
        separate venv without RELEASE/src on sys.path; that import
        silently failed and the mask zeroed every phase-aware reward —
        see plan custom-mods-canvas-issaclab-go2-ppo).
        """
        from registers import commands as _commands_reg
        from registers import motion_phases as _motion_phases
        reg_item = _commands_reg.get_item(item_id)
        if reg_item is None:
            raise CanvasConfigError(
                nid=cid,
                key="training_items",
                schema_id=self._types.get(cid, ""),
                reason=(
                    f"training_item {item_id!r} not found in registers.commands. "
                    f"Either fix the id, or add a custom item to "
                    f"~/UnitPort/registers/commands_custom.json."
                ),
            )

        # Speed default: full template range on speed_channel (≡ speed=1).
        raw_speed = item_cfg.get("speed", None)
        if isinstance(raw_speed, (list, tuple)) and len(raw_speed) == 2:
            speed_lo, speed_hi = float(raw_speed[0]), float(raw_speed[1])
        else:
            tch_rng = reg_item.command_template.get(reg_item.speed_channel, (0.0, 0.0))
            speed_lo, speed_hi = abs(float(tch_rng[0])), abs(float(tch_rng[1]))
        if speed_lo > speed_hi:
            speed_lo, speed_hi = speed_hi, speed_lo

        ranges: Dict[str, Tuple[float, float]] = {}
        bidir: Dict[str, bool] = {}
        for ch in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
            tpl = reg_item.command_template.get(ch, (0.0, 0.0))
            if ch in reg_item.zero_channels:
                ranges[ch] = (0.0, 0.0)
                bidir[ch] = False
            elif ch == reg_item.speed_channel:
                resolved, is_bd = self._resolve_speed_to_range(tpl, speed_lo, speed_hi)
                ranges[ch] = resolved
                bidir[ch] = is_bd
            else:
                t_lo, t_hi = float(tpl[0]), float(tpl[1])
                ranges[ch] = (t_lo, t_hi)
                bidir[ch] = (t_lo < 0 and t_hi > 0)

        advanced = item_cfg.get("advanced") or {}
        overrides = advanced.get("command_overrides") or {}
        if isinstance(overrides, dict):
            for ch, rng in overrides.items():
                if ch in ("lin_vel_x", "lin_vel_y", "ang_vel_z") and \
                   isinstance(rng, (list, tuple)) and len(rng) == 2:
                    o_lo, o_hi = float(rng[0]), float(rng[1])
                    ranges[ch] = (o_lo, o_hi)
                    bidir[ch] = (o_lo < 0 and o_hi > 0)

        motion_tag = reg_item.default_motion_tag
        phase_id = _motion_phases.resolve_phase(motion_tag) or ""
        return {
            "id": str(item_id),
            "ranges": ranges,
            "bidirectional": bidir,
            "motion_tag": motion_tag,
            "phase_id": phase_id,
        }

    def _build_training_items_list(self, cid: str) -> List[dict]:
        """Build the items literal for the weighted CommandTerm.
        Raises :class:`CanvasConfigError` if no items are enabled."""
        training_items = self._parse_json_param(cid, "training_items")
        enabled: List[dict] = []
        for item_id, raw_cfg in (training_items or {}).items():
            if not isinstance(raw_cfg, dict):
                continue
            if not raw_cfg.get("enabled"):
                continue
            enabled.append(self._build_training_item_dict(cid, item_id, raw_cfg))
        if not enabled:
            raise CanvasConfigError(
                nid=cid,
                key="training_items",
                schema_id=self._types.get(cid, ""),
                reason=(
                    "No training_item is enabled. At least one item must "
                    "have enabled=true so the weighted CommandTerm has a "
                    "template to draw from."
                ),
            )
        return enabled

    def _has_training_motion(self) -> bool:
        return bool(self._find_by_type("training_motion"))

    # ------------------------------------------------------------------
    # Param readers — _p / _pf / _pi
    # ------------------------------------------------------------------
    # Strict contract (2026-05-11, hardened in follow-up):
    #
    #   * Canvas has a value           → return it (coerced/parsed).
    #   * Canvas omits the key         → fall back to the **manifest**
    #     schema default (single source of truth — never a compiler-
    #     internal default).
    #   * Manifest also omits the key  → compiler↔manifest drift; raise
    #     CanvasConfigError pointing at the offending (schema_id, key).
    #   * Canvas value is None or ""   → raise (canvas wrote a blank,
    #     which masks intent).
    #   * Canvas value fails to parse  → raise.
    #
    # No caller-provided default. The historical signature accepted a
    # ``default`` argument; it has been removed and all ~110 call sites
    # updated. The manifest is now the authoritative source for "what
    # to use when canvas omits a key". This prevents the entire class
    # of bugs where compiler and manifest disagree on a default value.

    def _schema_default(self, nid: str, key: str) -> Any:
        """Return the manifest-declared default for a (node, field) pair.

        Used by :meth:`_p` / :meth:`_pf` / :meth:`_pi` when the canvas
        doesn't carry the key. The manifest is the **single source of
        truth** for parameter defaults — the compiler does not invent
        them.

        Raises :class:`CanvasConfigError` when:
          - the node id has no schema_id on this canvas (malformed);
          - the schema_id isn't registered with ``nodes_registry``
            (stale canvas referencing a removed node);
          - the manifest doesn't declare a parameter for ``key`` —
            this is compiler↔manifest drift, the exact bug class this
            strict path exists to eliminate.
        """
        schema_id = self._types.get(nid, "")
        if not schema_id:
            raise CanvasConfigError(
                nid=nid,
                key=key,
                reason="canvas node has no schema_id — malformed canvas",
            )
        from registers import nodes as nodes_registry
        manifest = nodes_registry.get_manifest(schema_id)
        if manifest is None:
            raise CanvasConfigError(
                nid=nid,
                key=key,
                schema_id=schema_id,
                reason=(
                    f"no registered manifest for schema_id {schema_id!r} — "
                    f"canvas references an unknown node type"
                ),
            )
        for p in manifest.parameters:
            if p.key == key:
                return p.default
        raise CanvasConfigError(
            nid=nid,
            key=key,
            schema_id=schema_id,
            reason=(
                f"compiler reads key {key!r} but the manifest for "
                f"{schema_id!r} does not declare it — compiler↔manifest "
                f"drift. Fix by adding a [[parameters]] block in "
                f"src/nodes/{schema_id}/manifest.toml, or remove the "
                f"compiler read at the call site."
            ),
        )

    def _p(self, nid: str, key: str) -> str:
        """Read a string param. Canvas → manifest default → raise.

        Coerces native int/float/bool to str so callers that ``.strip()`` /
        ``.lower()`` etc. don't blow up on a numeric.

        - Canvas omits ``key`` → falls back to manifest schema default
          (via :meth:`_schema_default`).
        - Canvas writes ``None`` or empty string for ``key`` → raise
          (a blank is "tried to set but left empty", which masks intent).
        - Manifest also omits ``key`` → raise (compiler↔manifest drift).
        """
        params = self._params.get(nid)
        if params is None:
            raise CanvasConfigError(
                nid=nid,
                key=key,
                schema_id=self._types.get(nid, ""),
                reason="canvas has no node with this id",
            )
        if key in params:
            v = params[key]
        else:
            v = self._schema_default(nid, key)
        if v is None:
            raise CanvasConfigError(
                nid=nid,
                key=key,
                schema_id=self._types.get(nid, ""),
                reason=(
                    "value is None — neither canvas nor manifest "
                    "provides a real value"
                ),
            )
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, str):
            if v == "":
                raise CanvasConfigError(
                    nid=nid,
                    key=key,
                    schema_id=self._types.get(nid, ""),
                    reason="value is empty string — canvas wrote a blank",
                )
            return v
        return str(v)

    def _pf(self, nid: str, key: str) -> float:
        """Read a float param. Canvas → manifest default → raise.

        Raises :class:`CanvasConfigError` when the raw value (whether
        from canvas or manifest default) cannot be parsed as float.
        """
        raw = self._p(nid, key)
        try:
            return float(raw)
        except (ValueError, TypeError) as exc:
            raise CanvasConfigError(
                nid=nid,
                key=key,
                schema_id=self._types.get(nid, ""),
                reason=f"value {raw!r} is not a valid float",
            ) from exc

    def _pi(self, nid: str, key: str) -> int:
        """Read an int param. Canvas → manifest default → raise.

        Accepts integer strings as well as float strings (rounds toward
        zero via int(float(...))). Raises :class:`CanvasConfigError`
        when parsing fails.
        """
        raw = self._p(nid, key)
        try:
            return int(float(raw))
        except (ValueError, TypeError) as exc:
            raise CanvasConfigError(
                nid=nid,
                key=key,
                schema_id=self._types.get(nid, ""),
                reason=f"value {raw!r} is not a valid int",
            ) from exc

    def _fan_in_order(self, dst_nid: str, dst_slot: str) -> List[str]:
        """Return source node IDs connected to a fan-in port, in edge order."""
        return [
            src for src, _ss, ds in self._upstream.get(dst_nid, [])
            if ds == dst_slot
        ]

    def _parse_json_param(self, nid: str, key: str) -> dict:
        """Parse a JSON-shaped parameter (dict). Strict — never returns {} on error.

        Tolerates two on-disk encodings that historical canvas files
        produced via :meth:`IRParam.to_dict` (which passes ``value``
        through as-is — Python value or pre-serialised JSON string):

          - native ``dict`` literal  (e.g. ``reward_terms``,
            ``termination_conditions`` in newer canvases)
          - JSON-encoded ``str``     (e.g. ``obs_terms``,
            ``training_items`` in older canvases)

        Anything else — missing key, ``None``, list, malformed JSON,
        non-dict JSON — raises :class:`CanvasConfigError` pointing at
        the offending node so the user can fix the canvas instead of
        getting a silently-empty Cfg downstream (the 2026-05-11
        ``TerminationsCfg`` empty-class incident).
        """
        params = self._params.get(nid, {})
        if key not in params:
            raise CanvasConfigError(
                nid=nid,
                key=key,
                schema_id=self._types.get(nid, ""),
                reason="JSON dict param is missing from canvas (no value on node)",
            )
        raw = params[key]
        if isinstance(raw, dict):
            return raw
        if raw is None:
            raise CanvasConfigError(
                nid=nid,
                key=key,
                schema_id=self._types.get(nid, ""),
                reason="JSON dict param is None — canvas wrote a null value",
            )
        if not isinstance(raw, str):
            raise CanvasConfigError(
                nid=nid,
                key=key,
                schema_id=self._types.get(nid, ""),
                reason=(
                    f"JSON dict param has unsupported type "
                    f"{type(raw).__name__!r} (expected dict or JSON string)"
                ),
            )
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CanvasConfigError(
                nid=nid,
                key=key,
                schema_id=self._types.get(nid, ""),
                reason=f"JSON dict param failed to parse: {exc}",
            ) from exc
        if not isinstance(result, dict):
            raise CanvasConfigError(
                nid=nid,
                key=key,
                schema_id=self._types.get(nid, ""),
                reason=(
                    f"JSON dict param parsed to {type(result).__name__!r}, "
                    f"expected dict"
                ),
            )
        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_body_mapping(self) -> list:
        """Return a list of error strings if body mapping is incomplete.

        Returns an empty list when all *user-required* IR roles are resolved.

        "User-required" excludes the ``AUTO_FROM_ASSET_CATEGORIES`` set
        (``base``, ``feet``) — these are marked ``required=True`` in
        ``ir_canonical.json`` because the runtime needs them, but they
        are NOT something the canvas author has to set manually. The
        compiler resolves them via :meth:`_resolve_body` (joints-mapping
        IR → asset metadata → canonical-name default), and falls back
        to canonical names (``base`` / ``trunk``) only when no signal
        exists at all. Treating these as user-required would force every
        canvas to hand-pick a base body even though the asset registry
        and USD convention already determine it unambiguously.
        """
        from application.training.body_ir import AUTO_FROM_ASSET_CATEGORIES

        mapper = self._get_body_ir_mapper()
        errors = []
        for role in mapper.unresolved_roles(required_only=True):
            m = mapper.get(role)
            if m is not None and m.category in AUTO_FROM_ASSET_CATEGORIES:
                # Auto-from-asset slot — not a canvas obligation.
                continue
            errors.append(
                f"Body mapping: required role '{m.label}' ({role}) is unresolved. "
                f"Open the Policy Exporter node and assign a body."
            )
        return errors

    def compile(self) -> str:
        """Return the full Python source as a string."""
        # Pre-flight: validate body mapping. Strict — required roles
        # left unresolved used to print WARNING-level log lines that
        # nobody read, and the training would silently launch with a
        # broken Robot/Scene cfg (the 2026-05-11 PPO incident). Now we
        # raise CanvasConfigError so the user is forced to fix the
        # Robot node body_mapping before the run is even submitted.
        body_errors = self.validate_body_mapping()
        if body_errors:
            raise CanvasConfigError(
                reason=(
                    "Robot body_mapping has unresolved required roles "
                    "— open the Robot node and assign a body for each. "
                    "Details:\n  - "
                    + "\n  - ".join(body_errors)
                ),
            )

        # Pre-flight: cross-node consistency between rewards and terminations.
        # Caught the silent reward-deadlock that broke the IsaacLab Phase 5
        # smoke run — base_height reward target=0.27 with termination
        # minimum=0.35 means the policy must crouch to claim the reward but
        # immediately gets terminated when it does, producing flat-line
        # reward curves indistinguishable from "training is broken".
        self._validate_reward_termination_consistency()

        lines = self._header()
        lines += self._custom_reward_funcs()
        # Reward × MotionPhase isolation — emit unitport_phase_mask helper
        # when at least one reward term declared applies_to. Wrapping a
        # reward func in unitport_phase_mask(base, phases) zeroes its
        # contribution on envs whose currently-active task item's
        # motion_tag does not fall into one of `phases`. The helper looks
        # up registers.motion_phases at runtime so the mapping stays
        # consistent with the canvas-side Coverage Badge.
        if self._scan_rewards_for_phase_aware():
            self._needs_phase_mask_helper = True
            lines.extend(_PHASE_MASK_HELPER_INLINE.splitlines())
            lines.append("")
        # Item-level mask helper — emitted whenever the canvas has more
        # than one rewards node OR any per-item ``reward_in__<item_id>``
        # edge. Required because phase-level masking (locomotion/static/
        # agile) is too coarse when e.g. turn/strafe/walk all map to
        # phase=locomotion but the user wants different reward profiles.
        if self._scan_rewards_for_item_masking():
            self._needs_item_mask_helper = True
            lines.extend(_ITEM_MASK_HELPER_INLINE.splitlines())
            lines.append("")
        # P2.1 — emit the Walk These Ways gait command term + helpers
        # only when the canvas has gait_enabled on Training Commands.
        # This keeps the generated config minimal for classic velocity
        # tracking runs and avoids importing CommandTerm unnecessarily.
        # §1A — emit the weighted-velocity CommandTerm inline whenever the
        # canvas has a training_motion node. The class is a drop-in
        # replacement for mdp.UniformVelocityCommandCfg's term — 3D output
        # via .command, multinomial per-item sampling driven by
        # self._weights (set_weights API used by the adaptive sampler).
        if self._has_training_motion():
            lines.extend(_WEIGHTED_VELOCITY_COMMAND_INLINE.splitlines())
            lines.append("")
        if self._gait_enabled():
            lines.extend(_GAIT_COMMAND_INLINE.splitlines())
            lines.append("")
        lines += self._scene_cfg()
        lines += self._observations_cfg()
        lines += self._actions_cfg()
        lines += self._commands_cfg()
        lines += self._rewards_cfg()
        lines += self._terminations_cfg()
        lines += self._events_cfg()
        lines += self._root_env_cfg()
        lines += self._ppo_runner_cfg()
        lines += self._unitport_curriculum_cfg()
        lines += self._export_cfg()
        return "\n".join(lines) + "\n"

    def compile_to_file(self, path: Optional[str] = None) -> Path:
        """Write compiled config to a file. Returns the file path.

        Side-effect: also writes ``deploy_meta.json`` next to the compiled
        ``.py`` file. The sidecar is the **single authoritative record** of
        per-term obs metadata (``scale`` / ``clip`` / ``history_length``)
        decided at compile time. Isaac Lab's ``dump_yaml`` would emit
        ``scale: null`` whenever the compiler omitted ``scale=`` on an
        ObsTerm — which makes the trained env.yaml ambiguous from the
        outside (was it 1.0 by intent, or just not written?). The sidecar
        removes that ambiguity: if a term's scale is None here, the
        compiler **did not write it** and Isaac Lab's
        ObservationManager default applies (currently 1.0). The export
        side (``manifest_parser._extract_obs_term_meta``) treats sidecar
        None as a hard error per Module A's strict-contract design.
        """
        source = self.compile()
        if path is None:
            fd, path = tempfile.mkstemp(suffix=".py", prefix="unitport_il_cfg_")
            import os
            os.close(fd)
        p = Path(path)
        p.write_text(source, encoding="utf-8")
        log.info("IsaacLabConfigCompiler wrote %d bytes to %s", len(source), p)

        # Sidecar — atomic write via temp + os.replace on the same volume.
        try:
            meta = self._build_obs_metadata_sidecar()
        except Exception:
            log.exception(
                "IsaacLabConfigCompiler: failed to build deploy_meta.json "
                "payload; sidecar will NOT be written. Bundle will still be "
                "valid for back-compat (env.yaml-only) path."
            )
            return p
        meta_path = p.parent / DEPLOY_META_FILENAME
        try:
            tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(meta, indent=2, sort_keys=False),
                encoding="utf-8",
            )
            import os as _os
            _os.replace(str(tmp), str(meta_path))
            log.info(
                "IsaacLabConfigCompiler wrote sidecar %s (%d obs term(s))",
                meta_path,
                sum(
                    len(grp) for grp in meta.get("obs_groups", {}).values()
                ),
            )
        except Exception:
            log.exception(
                "IsaacLabConfigCompiler: writing %s failed; bundle remains "
                "valid on the env.yaml back-compat path.",
                meta_path,
            )
        return p

    def _build_obs_metadata_sidecar(self) -> Dict[str, Any]:
        """Assemble the ``deploy_meta.json`` payload.

        Structure::

            {
              "schema_version": 1,
              "generated_by": "IsaacLabConfigCompiler",
              "generated_utc": "<ISO-8601>",
              "obs_groups": {
                  "<group_name>": {
                      "<term_name>": {
                          "scale": <number|list|null>,
                          "clip":  [lo, hi] | null,
                          "history_length": <int>|null
                      },
                      ...
                  },
                  ...
              }
            }

        Only obs metadata lives here today. Actuator / sim / action data
        is faithfully preserved by Isaac Lab's ``dump_yaml`` already
        (those fields aren't dropped to null), so the sidecar's purpose
        is strictly: recover what ``dump_yaml`` cannot.

        Empty/missing resolved obs terms (e.g. the graph has no
        ``il_observation`` node) yields ``obs_groups: {}`` — caller treats
        that as "compiler had nothing to record" rather than a structural
        bug.
        """
        groups: Dict[str, Dict[str, Dict[str, Any]]] = {}
        resolved_by_group: Dict[str, Dict[str, _PerTermObsConfig]] = (
            getattr(self, "_resolved_obs_terms_by_group", {}) or {}
        )
        # Resolve num_joints once for the "num_joints" dim sentinel.
        num_joints: Optional[int] = None
        if self._robot is not None:
            nj = getattr(self._robot, "num_joints", None)
            if isinstance(nj, int) and nj > 0:
                num_joints = int(nj)
        for gname, terms in resolved_by_group.items():
            groups[gname] = {}
            for term_name, cfg in terms.items():
                # Resolve dim from the producer-side table. The parser
                # reads this directly so it doesn't need its own dim
                # table (which historically had a 12-DoF default that
                # silently corrupted non-quadruped bundles).
                dim_spec = _OBS_TERM_DIM_TABLE.get(
                    term_name.split(".")[-1] if "." in term_name else term_name
                )
                dim_resolved: Optional[int]
                if isinstance(dim_spec, int):
                    dim_resolved = dim_spec
                elif dim_spec == "num_joints" and num_joints is not None:
                    dim_resolved = num_joints
                else:
                    dim_resolved = None
                groups[gname][term_name] = {
                    "dim": dim_resolved,
                    "scale": (
                        list(cfg.scale)
                        if isinstance(cfg.scale, (list, tuple))
                        else cfg.scale
                    ),
                    "clip": (
                        [float(cfg.clip[0]), float(cfg.clip[1])]
                        if cfg.clip is not None else None
                    ),
                    "history_length": (
                        int(cfg.history_length)
                        if cfg.history_length is not None else None
                    ),
                }
        return {
            "schema_version": DEPLOY_META_SCHEMA_VERSION,
            "generated_by": "IsaacLabConfigCompiler",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "num_joints": num_joints,
            "obs_groups": groups,
        }

    # ------------------------------------------------------------------
    # Code generation sections
    # ------------------------------------------------------------------

    def _header(self) -> List[str]:
        base_imports = [
            "# AUTO-GENERATED by UnitPort IsaacLabConfigCompiler",
            "# Do not edit manually — regenerate from the node graph.",
            "",
            "import math",
            "",
            "import isaaclab.sim as sim_utils",
            "import isaaclab.envs.mdp as mdp",
            "import isaaclab_tasks.manager_based.locomotion.velocity.mdp as velocity_mdp",
            "from isaaclab.envs import ManagerBasedRLEnvCfg",
            "from isaaclab.managers import EventTermCfg as EventTerm",
            "from isaaclab.managers import ObservationGroupCfg as ObsGroup",
            "from isaaclab.managers import ObservationTermCfg as ObsTerm",
            "from isaaclab.managers import RewardTermCfg as RewTerm",
            "from isaaclab.managers import SceneEntityCfg",
            "from isaaclab.managers import TerminationTermCfg as DoneTerm",
            "from isaaclab.actuators import ImplicitActuatorCfg, DCMotorCfg, ActuatorNetMLPCfg, ActuatorNetLSTMCfg, IdealPDActuatorCfg",
            "from isaaclab.scene import InteractiveSceneCfg",
            "from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns",
            "from isaaclab.sim import PhysxCfg, RenderCfg, SimulationCfg",
            "from isaaclab.terrains import TerrainImporterCfg",
            "from isaaclab.utils import configclass",
            "",
            "from isaaclab.assets import ArticulationCfg",
            "from isaaclab.sim.spawners import UsdFileCfg",
            # Per-term observation noise — see _observations_cfg below.
            # Unoise(n_min, n_max) is uniform additive noise; the canvas
            # exposes a single ``corruption_noise_std`` knob which we
            # translate to ``Unoise(n_min=-std, n_max=std)``.
            "from isaaclab.utils.noise import UniformNoiseCfg as Unoise",
            # Phase_1 of AMP_design.yaml §3 — needed when the canvas
            # picked a menagerie robot from the asset_dropdown. The
            # discovery layer stores the asset under a ``nucleus:<rel>``
            # marker that the robot section below expands to
            # ``f"{ISAAC_NUCLEUS_DIR}/<rel>"`` at config-load time. Import
            # is unconditional — it's free if unused, and conditional
            # imports add fragility for negligible savings.
            "from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR",
            # Pre-built rough-terrain generator used when the IL Terrain
            # Config node picks anything other than "flat" — see
            # _scene_cfg below for the dispatch.
            "from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG",
            "",
            "",
        ]
        # Only import the AMP RSI event module when AMP_PPO is the
        # active algorithm. A pure PPO run's generated config has no
        # reason to pull the AMP tree.
        trainer_ids = self._find_by_type("il_ppo_trainer")
        if trainer_ids and self._p(trainer_ids[0], "training_mode").upper() == "AMP_PPO":
            base_imports.append(
                "from application.training.amp import mdp_events as _amp_mdp_events"
            )
            base_imports.append("")
        return base_imports

    # ------------------------------------------------------------------
    # Registry-driven reward compilation
    # ------------------------------------------------------------------
    # ALL reward metadata (function names, module routing, extra params,
    # inline implementations) lives in the kind-namespaced reward
    # sub-registries. The compiler reads from there — no parallel
    # hardcoded dicts.

    def _scan_rewards_for_phase_aware(self) -> bool:
        """Return True when any reward term in the canvas declares applies_to.

        Drives whether ``compile()`` emits the ``unitport_phase_mask``
        helper preamble. Scanning here (separately from ``_rewards_cfg``)
        decouples emit-order: the helper has to be defined *before* the
        ``RewardsCfg`` line that references it.
        """
        rew_ids = self._find_by_type("rewards")
        if not rew_ids:
            return False
        for rid in rew_ids:
            try:
                reward_terms = self._parse_json_param(rid, "reward_terms")
            except Exception:
                continue
            if not isinstance(reward_terms, dict):
                continue
            for entry in reward_terms.values():
                if isinstance(entry, dict):
                    applies = entry.get("applies_to") or []
                    if isinstance(applies, str):
                        applies = [s.strip() for s in applies.split(",") if s.strip()]
                    if applies:
                        return True
        return False

    def _scan_rewards_for_item_masking(self) -> bool:
        """Return True when the canvas needs per-item reward masking.

        Triggers on either (a) more than one ``rewards`` node, or (b) any
        edge from ``rewards.reward_pipe`` into ``training_motion.reward_in__*``.
        Both conditions imply the user is differentiating reward profiles
        per training item — emit the ``unitport_item_mask`` helper.
        """
        rew_ids = self._find_by_type("rewards")
        if len(rew_ids) > 1:
            return True
        for rid in rew_ids:
            for (_dst, src_port, dst_port) in self._downstream.get(rid, []):
                if src_port == "reward_pipe" and dst_port.startswith("reward_in__"):
                    return True
        return False

    def _custom_reward_funcs(self) -> List[str]:
        """Emit inline implementations for custom reward functions.

        Reads ``il_inline`` from the registry for each reward the user
        selected. Only emits functions actually used in the current canvas.
        """
        from application.training.isaac_lab.task_module_registry import lookup, BACKEND_ISAAC

        rew_ids = self._find_by_type("rewards")
        if not rew_ids:
            return []

        # Collect unique inline implementations across ALL rewards nodes
        # (multi-rewards canvas with per-item fanout). Kind+backend scoped
        # lookup — the old flat UNIFIED_REGISTRY was removed because
        # cross-kind key collisions (e.g. reward and termination both
        # naming "base_height") silently dropped inline blocks.
        emitted_funcs: set = set()
        inline_blocks: list = []
        for rid in rew_ids:
            try:
                reward_terms = self._parse_json_param(rid, "reward_terms")
            except Exception:
                continue
            if not isinstance(reward_terms, dict):
                continue
            for func_key in reward_terms:
                item = lookup(func_key, kind="reward", backend=BACKEND_ISAAC)
                if item is None or not item.il_inline:
                    continue
                if item.il_func in emitted_funcs:
                    continue
                emitted_funcs.add(item.il_func)
                inline_blocks.append(item.il_inline)

        if not inline_blocks:
            return []

        lines = [
            "# " + "=" * 70,
            "# UnitPort inline reward functions (not in standard Isaac Lab)",
            "# " + "=" * 70,
        ]
        for block in sorted(inline_blocks):
            lines.append(block)
        lines.append("")
        return lines

    def _simulation_cfg_literal(self) -> str:
        """Return a one-line ``SimulationCfg(...)`` literal for *root* env_cfg.

        Belongs on ``UnitPortEnvCfg`` as ``sim: SimulationCfg = ...``,
        NOT inside ``SceneCfg`` — ``InteractiveScene`` iterates SceneCfg
        fields trying to spawn each as a scene entity (articulation,
        sensor, terrain, light) and ValueErrors on the SimulationCfg
        because it doesn't recognise the type. Standard Isaac Lab pattern
        keeps sim at the env-cfg level alongside scene/observations/etc.

        ``render_interval`` is pinned to ``decimation`` so the viewport
        renders **once per env step** instead of every sim step. Default
        ``render_interval=1`` would render 4× per env step on a typical
        50 Hz Go2 setup (decimation=4) — with 4096 envs + non-headless
        that's a guaranteed GPU melt. Isaac Lab itself logs a warning
        about this mismatch; fixing it at emit time is the right answer.
        """
        # Sim globals (dt / gpu_*) live on the Play Ground Setting node
        # since the ILSimulationConfigNode merge. One source of truth.
        pg_ids = self._find_by_type("play_ground_setting")
        sim_dt = 0.005
        if pg_ids:
            sim_dt = self._pf(pg_ids[0], "sim_dt") or 0.005
        control_dt = 0.02
        decimation = max(1, int(round(control_dt / sim_dt))) if sim_dt > 0 else 4

        # Build the RenderCfg literal: keep Isaac Lab's "balanced" default
        # quality (shadows, AA, ambient occlusion, direct lighting), but
        # explicitly disable the raytracing-heavy features that tank
        # framerate without adding much visual fidelity for our use case.
        # Set to None on a field = "use Kit default"; set to False =
        # "force off".  Reflections and global illumination are the two
        # ray-traced features that consume the most GPU; turning them off
        # gives ~3-4× framerate at non-headless time.  rendering_mode
        # "performance" is the corresponding Kit-side preset and aligns
        # with the rest of the disables.
        render_literal = (
            "RenderCfg("
            "rendering_mode=\"performance\", "
            "enable_reflections=False, "
            "enable_global_illumination=False, "
            "enable_dl_denoiser=False"
            ")"
        )

        if not pg_ids:
            return (
                f"SimulationCfg(dt={sim_dt}, render_interval={decimation}, "
                f"render={render_literal})"
            )
        pid = pg_ids[0]
        dt = self._pf(pid, "sim_dt")
        gz = self._pf(pid, "gravity_z")
        stype = self._p(pid, "scene_type").strip().lower()
        # Auto-compute PhysX GPU buffers from num_envs × terrain density.
        # Removed user-facing sliders 2026-05-10 — both directions of the
        # old slider produced silent failures: too small → "Patch buffer
        # overflow detected" spam every iter and dropped contacts; too
        # large → wasted GPU memory + slower buffer walks per step. The
        # user has no diagnostic to know which side of the cliff they're
        # on. Formula derived from observed peaks at 8192 envs (rough:
        # ~150K patches needed → 18.3/env, +75% margin = 32/env).
        # Flat is much sparser (foot contacts only, ~4 patches/env).
        per_env = _PER_ENV_PHYSX_BUDGET[stype if stype in _PER_ENV_PHYSX_BUDGET else "rough"]
        trainer_ids_pg = self._find_by_type("il_ppo_trainer")
        n_envs_for_phys = (
            self._pi(trainer_ids_pg[0], "num_envs") if trainer_ids_pg else 4096
        )
        # Floors stop pathological micro-runs (e.g. n_envs=8 unit tests)
        # from sliding under PhysX's own internal minimums and crashing.
        gpu_patch = max(_PHYSX_PATCH_FLOOR, n_envs_for_phys * per_env["patches"])
        gpu_contact = max(_PHYSX_CONTACT_FLOOR, n_envs_for_phys * per_env["contacts"])
        return (
            f"SimulationCfg("
            f"dt={dt}, gravity=(0.0, 0.0, {gz}), "
            f"render_interval={decimation}, "
            f"render={render_literal}, "
            f"physx=PhysxCfg("
            f"gpu_max_rigid_contact_count={gpu_contact}, "
            f"gpu_max_rigid_patch_count={gpu_patch}, "
            f"enable_external_forces_every_iteration=True"
            f"))"
        )

    def _scene_cfg(self) -> List[str]:
        lines = ["@configclass", "class SceneCfg(InteractiveSceneCfg):", '    """Scene configuration."""', ""]

        # NB: SimulationCfg lives on UnitPortEnvCfg root (see
        # _root_env_cfg + _simulation_cfg_literal) — InteractiveScene
        # would reject it here as an unknown asset config type.

        # Terrain
        # Isaac Lab's TerrainImporter accepts ONLY three terrain_type values:
        #   "plane"     — infinite flat ground
        #   "generator" — procedural sub-terrain mix via TerrainGeneratorCfg
        #   "usd"       — load a custom USD scene
        # Our IL Terrain Config dropdown offers "flat" / "rough" / "stairs"
        # / "slopes" / "stepping_stones" — which map as follows:
        #
        #   flat                                       → "plane"
        #   rough / stairs / slopes / stepping_stones  → "generator"
        #                                                 + ROUGH_TERRAINS_CFG
        #                                                 (Isaac Lab pre-built
        #                                                 generator that ships
        #                                                 a mix of all four
        #                                                 sub-terrain types)
        #
        # We do NOT yet expose per-sub-terrain weight knobs in the canvas;
        # the user gets the standard rough mix when they pick anything other
        # than flat. Customising the mix requires editing this method or
        # adding a TerrainGeneratorNode to the canvas (future work).
        # §2 Scene — unified Play Ground Setting node is the single
        # source of truth. The old il_terrain_config fallback path has
        # been removed as part of the 6-section migration.
        playground_ids = self._find_by_type("play_ground_setting")
        if playground_ids:
            pid = playground_ids[0]
            stype = self._p(pid, "scene_type").strip().lower()
            mu_s = self._pf(pid, "friction_static")
            mu_d = self._pf(pid, "friction_dynamic")
            # ``restitution`` is intentionally NOT a canvas-level knob —
            # play_ground_setting's manifest does not declare it. The
            # value here is PhysX's physical default (no elastic bounce
            # on terrain contact) emitted into RigidBodyMaterialCfg so
            # the generated env_cfg is complete. Domain-randomised
            # restitution still happens via the domain_rand node's
            # ``restitution_range``.
            restitution = 0.0
            # Explicit dispatch — silent fallback to ROUGH_TERRAINS_CFG used to
            # send canvases with scene_type="custom" (or any non-"flat" value
            # not in the node enum) onto a 6-tile rough generator, dropping
            # per-iter speed by 5-10× before the user knew anything had gone
            # wrong. Match the play_ground_setting node's enum
            # (["flat", "rough"]) and raise loudly on anything else.
            if stype == "flat":
                terrain_type_literal = "plane"
            elif stype == "rough":
                terrain_type_literal = "rough_generator"
            else:
                raise ValueError(
                    f"\n[UnitPort][Compiler] play_ground_setting.scene_type="
                    f"{stype!r} is not a supported terrain type. "
                    f"Valid values: 'flat', 'rough'.\n"
                    f"  Got 'custom'? The picker_scene widget probably wrote "
                    f"this when you selected a custom scene_id; the IsaacLab "
                    f"backend has no custom-USD terrain branch yet. Pick "
                    f"'flat' or 'rough' from the Play Ground Setting node "
                    f"enum to unblock training."
                )
            lines.append(f"    terrain = TerrainImporterCfg(")
            lines.append(f'        prim_path="/World/ground",')
            if terrain_type_literal == "plane":
                lines.append(f'        terrain_type="plane",')
            else:
                lines.append(f'        terrain_type="generator",')
                lines.append(f"        terrain_generator=ROUGH_TERRAINS_CFG,")
                lines.append(f"        max_init_terrain_level=5,")
                lines.append(f"        collision_group=-1,")
            lines.append(f"        physics_material=sim_utils.RigidBodyMaterialCfg(")
            lines.append(
                f"            static_friction={mu_s}, "
                f"dynamic_friction={mu_d}, restitution={restitution},"
            )
            lines.append(f"        ),")
            lines.append(f"        debug_vis=False,")
            lines.append(f"    )")

        # Robot
        # Actuator config now flows from ActorSettingNode — the old
        # ILActuatorConfigNode was a duplicate data source and has
        # been deleted. ActorSetting carries stiffness / damping /
        # effort_limit / velocity_limit as scalar defaults for ALL
        # joints; per-joint-group overrides can be added later via a
        # dedicated "Actuator Override" node if the product needs it.
        actuator_lines = []
        actor_ids_for_actuators = self._find_by_type("actor_setting")
        if actor_ids_for_actuators:
            aid = actor_ids_for_actuators[0]
            kp = self._pf(aid, "stiffness")
            kd = self._pf(aid, "damping")
            eff = self._pf(aid, "effort_limit")
            vel = self._pf(aid, "velocity_limit")
            cfg_cls = self._actuator_cfg_class("implicit_pd")
            actuator_lines.append(
                f'            "legs": {cfg_cls}('
                f'joint_names_expr=[".*"], '
                f'stiffness={kp}, damping={kd}, '
                f'effort_limit={eff}, velocity_limit={vel}),'
            )

        robot_ids = self._find_by_type("robot")
        if robot_ids:
            rid = robot_ids[0]
            # num_envs lives on the IL Trainer node; env_spacing is a
            # scene-level constant (may migrate to PlayGroundSetting later).
            trainer_ids_env = self._find_by_type("il_ppo_trainer")
            if trainer_ids_env:
                n_envs = self._pi(trainer_ids_env[0], "num_envs")
            else:
                n_envs = 4096
            if n_envs < 1:
                n_envs = 4096
            spacing = 2.5

            # Asset_id is the single source of truth; the registry resolves
            # it to a concrete USD source (on-disk or Nucleus URL).
            asset_id = self._p(rid, "asset_id").strip()
            usd_path = ""

            if asset_id:
                from application.service.robot_assets import (
                    get_robot_asset_service,
                )
                from registers.robots import resolve_id as _resolve_robot_id
                sku = _resolve_robot_id(asset_id) or asset_id
                asset = get_robot_asset_service().resolve(sku)
                if asset is None:
                    raise ValueError(
                        f"\n\n[UnitPort][Compiler] Cannot compile Isaac Lab "
                        f"config — Robot node asset_id={asset_id!r} did not "
                        f"resolve to a known SKU (sku={sku!r}).\n\n"
                        f"  Fix: pick a menagerie asset from the Robot node "
                        f"dropdown.\n"
                    )
                if asset.usd_path is not None and asset.usd_path.exists():
                    usd_path = str(asset.usd_path)
                elif asset.usd_url:
                    # RELEASE's RobotAsset.usd_url already carries the
                    # ``nucleus:`` marker prefix (e.g.
                    # ``nucleus:Robots/Unitree/Go2/go2.usd``); the emit-side
                    # path below splits on ``nucleus:`` and joins the remainder
                    # to ``ISAAC_NUCLEUS_DIR``. Don't add another prefix or
                    # the URL ends up as ``<ISAAC_NUCLEUS_DIR>/nucleus:Robots/...``.
                    usd_path = str(asset.usd_url)
                else:
                    raise ValueError(
                        f"\n\n[UnitPort][Compiler] Cannot compile Isaac Lab "
                        f"config — Robot node asset_id={asset_id!r} has no "
                        f"USD source (sku={sku!r}, usd_path=None, "
                        f"usd_url=empty).\n\n"
                        f"  Fix: pick a menagerie asset that ships a Nucleus "
                        f"USD URL (unitree_go2, unitree_a1, unitree_g1, "
                        f"unitree_h1, boston_dynamics_spot, …), or register "
                        f"a local USD via the Robot Assets sidebar panel.\n"
                    )

            # Phase_1: emit-side handling of the ``nucleus:`` marker the
            # discovery layer attaches to menagerie assets. The marker
            # cannot be embedded as a literal string (Isaac Lab would
            # try to open ``nucleus:Robots/...`` as a file path); we
            # convert it to a Python f-string that resolves
            # ``ISAAC_NUCLEUS_DIR`` at config-load time inside the isaac
            # venv. Plain disk paths are emitted as quoted literals.
            if usd_path.startswith("nucleus:"):
                rel = usd_path[len("nucleus:"):]
                # NOTE: this Python expression — NOT a quoted string —
                # is interpolated into the generated config file.
                usd_path_expr = f'f"{{ISAAC_NUCLEUS_DIR}}/{rel}"'
            elif usd_path:
                # repr() emits a valid Python string literal with backslashes
                # escaped — needed on Windows where paths like
                # ``D:\Unitport\...`` would otherwise trigger ``\U`` /``\n`` etc.
                # unicode-escape errors when the generated config is exec'd.
                usd_path_expr = repr(usd_path)
            else:
                # Neither asset_id nor usd_path supplied → emit a
                # placeholder that fails loudly at config load. Better
                # than silently falling back to a hardcoded go2.
                usd_path_expr = (
                    '"<MISSING_USD_PATH — set asset_id on the IL Robot Asset node>"'
                )

            # Init pose and contact settings are now owned by the
            # ActorSetting node (Robot node is display-only).
            actor_ids = self._find_by_type("actor_setting")
            if actor_ids:
                aid_node = actor_ids[0]
                px = self._pf(aid_node, "init_pos_x")
                py = self._pf(aid_node, "init_pos_y")
                pz = self._pf(aid_node, "init_pos_z")
                contact = self._p(aid_node, "contact_track_air_time").lower() == "true"
            else:
                px, py, pz = 0.0, 0.0, 0.4
                contact = True

            # Initial joint pose: user-supplied JSON from ActorSetting,
            # else fall back to the canonical quadruped standing pose.
            init_joint_pos = (
                '{'
                '".*_hip_joint": 0.0, '
                '".*_thigh_joint": 0.9, '
                '".*_calf_joint": -1.8'
                '}'
            )
            import json as _json
            # Priority: JointInitNode (delegate) > actor_setting fallback
            ji_ids = self._find_by_type("joint_init")
            raw_angles = ""
            source_node = ""
            if ji_ids:
                raw_angles = self._p(ji_ids[0], "angles").strip()
                source_node = "joint_init.angles"
            if (not raw_angles or raw_angles == "{}") and actor_ids:
                raw_angles = self._p(actor_ids[0], "init_joint_angles").strip()
                source_node = "actor_setting.init_joint_angles"
            if raw_angles and raw_angles != "{}":
                try:
                    parsed = _json.loads(raw_angles)
                except Exception as exc:
                    raise ValueError(
                        f"\n[UnitPort][Compiler] Failed to parse {source_node} "
                        f"as JSON: {exc!r}\n  Raw value: {raw_angles[:200]!r}"
                    )
                if isinstance(parsed, dict) and parsed:
                    # Substrate-emit boundary (Phase 5 §IR-only contract):
                    # canvas dict carries IR roles; Isaac Lab Articulation
                    # ._process_cfg matches keys against the USD's actual joint
                    # names (physical) → translate IR → physical here, at the
                    # last mile, via the bound JointIRResolver. Validation
                    # raises with a self-documenting error if any key is not
                    # an IR role.
                    resolver = self._joint_ir_resolver
                    if resolver is None:
                        raise ValueError(
                            f"\n[UnitPort][Compiler] {source_node} has joint "
                            f"entries but no robot is bound on the compiler. "
                            f"This means IsaacLabConfigCompiler was constructed "
                            f"without a RobotSpecRef — the IR→physical "
                            f"translation cannot run. Pass robot=spec.robot to "
                            f"IsaacLabConfigCompiler / compile_env_cfg_to_file."
                        )
                    physical = resolver.to_physical_dict(
                        {str(k): float(v) for k, v in parsed.items()},
                        where=source_node,
                    )
                    log.info(
                        "[env_cfg_compiler] %s: IR→physical translation = %s",
                        source_node,
                        {ir: (resolver.to_physical(ir), physical[resolver.to_physical(ir)])
                         for ir in parsed.keys()},
                    )
                    pairs = ", ".join(
                        f'"{k}": {v}' for k, v in physical.items()
                    )
                    init_joint_pos = "{" + pairs + "}"

            lines.append(f"    num_envs = {n_envs}")
            lines.append(f"    env_spacing = {spacing}")
            lines.append(f"    robot = ArticulationCfg(")
            lines.append(f"        prim_path=\"{{ENV_REGEX_NS}}/Robot\",")
            lines.append(f'        spawn=UsdFileCfg(usd_path={usd_path_expr},')
            lines.append(f"            activate_contact_sensors={contact},")
            lines.append(f"        ),")
            lines.append(f"        init_state=ArticulationCfg.InitialStateCfg(")
            lines.append(f"            pos=({px}, {py}, {pz}),")
            lines.append(f"            joint_pos={init_joint_pos},")
            lines.append(f'            joint_vel={{".*": 0.0}},')
            lines.append(f"        ),")
            if actuator_lines:
                lines.append(f"        actuators={{")
                lines.extend(actuator_lines)
                lines.append(f"        }},")
            lines.append(f"    )")

        # Contact sensor — settings owned by ActorSetting node.
        contact_actor = self._find_by_type("actor_setting")
        if contact_actor:
            aid_node = contact_actor[0]
            hist = self._pi(aid_node, "contact_history_length")
            air = self._p(aid_node, "contact_track_air_time").lower() == "true"
            lines.append(f"    contact_forces = ContactSensorCfg(")
            lines.append(f'        prim_path="{{ENV_REGEX_NS}}/Robot/.*",')
            lines.append(f"        history_length={hist},")
            lines.append(f"        track_air_time={air},")
            lines.append(f"    )")

        # Height scanner (ray-caster) — now gated on the Play Ground
        # Setting node's height_scan_enabled flag. Falls back to the
        # legacy IL Terrain Config if the new node is absent.
        hs_source = None
        if playground_ids:
            pid = playground_ids[0]
            if self._p(pid, "height_scan_enabled").lower() == "true":
                hs_source = pid
        elif terrain_ids:
            tid = terrain_ids[0]
            if self._p(tid, "enable_height_scan").lower() == "true":
                hs_source = tid

        if hs_source is not None:
            if True:
                res = self._pf(hs_source, "scan_resolution")
                sx = self._pf(hs_source, "scan_size_x")
                sy = self._pf(hs_source, "scan_size_y")
                # Resolve the actual base link name via the joints_mapping
                # IR — A1 has ``trunk``, Go2/Anymal have ``base``, etc.
                # Strict: unresolved → CanvasConfigError (was silent
                # fallback="base" which masked Anymal/A1 misconfig).
                base_body = self._resolve_body("base")
                lines.append(f"    height_scanner = RayCasterCfg(")
                lines.append(f'        prim_path="{{ENV_REGEX_NS}}/Robot/{base_body}",')
                lines.append(f"        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),")
                lines.append(f'        ray_alignment="yaw",')
                lines.append(f"        pattern_cfg=patterns.GridPatternCfg(")
                lines.append(f"            resolution={res}, size=({sx}, {sy}),")
                lines.append(f"        ),")
                lines.append(f"        debug_vis=False,")
                lines.append(f"        mesh_prim_paths=[\"/World/ground\"],")
                lines.append(f"    )")

        lines += ["", ""]
        return lines

    def _observations_cfg(self) -> List[str]:
        lines = ["@configclass", "class ObservationsCfg:", '    """Observation configuration."""', ""]

        obs_ids = self._find_by_type("il_observation")

        # ``height_scan`` needs a SceneEntityCfg("height_scanner") backing.
        # §2 Scene — prefer the new Play Ground Setting node; fall back
        # to the legacy IL Terrain Config node during the transition.
        has_height_scanner = False
        pg_ids = self._find_by_type("play_ground_setting")
        if pg_ids:
            has_height_scanner = (
                self._p(pg_ids[0], "height_scan_enabled").lower() == "true"
            )

        for oid in obs_ids:
            gname = self._p(oid, "group_name")
            noise_std = self._pf(oid, "corruption_noise_std")
            enable_noise = self._p(oid, "enable_corruption").lower() == "true"

            # Parse obs_terms JSON, then normalise to the strong contract
            # ``Dict[str, _PerTermObsConfig]``. This is the single place that
            # decides per-term scale / clip / history_length for both the
            # emitted ObsTerm Python string AND the deploy_meta.json sidecar
            # (compile_to_file → _build_obs_metadata_sidecar). Anything wrong
            # with the upstream shape raises CanvasConfigError here so the
            # user is pointed at the canvas node instead of getting a
            # mysteriously-bad bundle downstream.
            obs_terms_raw = self._parse_json_param(oid, "obs_terms")
            obs_terms_resolved: Dict[str, _PerTermObsConfig] = (
                _normalize_obs_terms(
                    obs_terms_raw, nid=oid, schema_id=self._types.get(oid, ""),
                )
            )
            obs_terms = obs_terms_resolved  # keep the existing iteration var
            # Stash for compile_to_file's sidecar writer.
            self._resolved_obs_terms_by_group: Dict[str, Dict[str, _PerTermObsConfig]] = (
                getattr(self, "_resolved_obs_terms_by_group", {}) or {}
            )
            self._resolved_obs_terms_by_group[gname] = dict(obs_terms_resolved)

            lines.append("    @configclass")
            lines.append(f"    class {gname.capitalize()}Cfg(ObsGroup):")
            # ``concatenate_terms = True`` makes ``ObservationManager.compute()``
            # return a single flat tensor of shape ``(num_envs, sum_of_term_dims)``
            # for this group instead of a dict-of-tensors. The Isaac Lab
            # ObsGroup default IS True, but we set it explicitly so the
            # behaviour can't drift across versions and so the AMP runner's
            # actor MLP always sees a real torch.Tensor (not a dict / not a
            # TensorDict subclass with a __torch_function__ that breaks
            # nn.Linear's dispatch).
            lines.append(f"        concatenate_terms = True")
            # ``enable_corruption`` is a real ObsGroup class-level marker
            # that ObservationManager honours. ``corruption_noise_std``
            # is NOT — Isaac Lab models noise as a per-term ``noise=``
            # field on each ObsTerm (Unoise / Gnoise instances), not as
            # a group-level scalar. Emitting it as a class attribute
            # makes ObservationManager._prepare_terms() raise:
            #   TypeError: Configuration for the term 'corruption_noise_std'
            #   is not of type ObservationTermCfg. Received: '<class 'float'>'.
            # Translate the canvas's single std value into a per-term
            # Unoise(n_min=-std, n_max=std) attached to each ObsTerm.
            apply_noise = enable_noise and noise_std > 0
            if enable_noise:
                lines.append(f"        enable_corruption = True")

            for term_type in obs_terms:
                if term_type == "height_scan" and not has_height_scanner:
                    lines.append(
                        f"        # height_scan SKIPPED: no height_scanner node "
                        f"on the canvas. Add a height-scanner sensor before "
                        f"re-enabling this term, or remove it from obs_terms."
                    )
                    continue
                func_name = self._obs_term_func(term_type)

                # ── Term-specific params ──
                # Most observation funcs in isaaclab.envs.mdp.observations
                # work with no params (they read defaults from the env's
                # standard "robot" / "contact_sensor" SceneEntityCfgs).
                # The exceptions are functions that operate on a *named*
                # subsystem and need a string handle:
                #
                #   generated_commands  → command_name  (which command to fetch)
                #
                # The only command we currently emit is ``base_velocity``
                # from il_velocity_cmd → CommandsCfg.base_velocity. When
                # the user adds more command nodes / sources we'll need
                # a richer registry, but ``base_velocity`` covers every
                # current canvas.
                params_str = ""
                if func_name == "generated_commands":
                    params_str = ', params={"command_name": "base_velocity"}'
                elif term_type == "height_scan":
                    # mdp.height_scan requires a SceneEntityCfg pointing
                    # at the ray-caster sensor emitted by _scene_cfg
                    # when enable_height_scan=true. The Isaac Lab stock
                    # offset convention subtracts 0.5m from the raw
                    # raycasts so values cluster around 0 for flat
                    # ground (matches legged_gym / robot_lab defaults).
                    params_str = (
                        ', params={"sensor_cfg": SceneEntityCfg('
                        '"height_scanner"), "offset": 0.5}'
                    )

                # Compose the per-term args. Each piece is appended only
                # when the resolved ``_PerTermObsConfig`` carries a non-None
                # value for that field — None passes through to the sidecar
                # unchanged so the downstream contract knows exactly which
                # fields the compiler decided vs. left to Isaac Lab.
                term_cfg = obs_terms_resolved[term_type]
                args: List[str] = [f"func=mdp.{func_name}"]
                if apply_noise:
                    args.append(
                        f"noise=Unoise(n_min=-{noise_std}, n_max={noise_std})"
                    )
                if term_cfg.scale is not None:
                    args.append(f"scale={_format_scale_literal(term_cfg.scale)}")
                # ``clip=(-100, 100)`` (the historical default) caps every
                # policy/critic obs before the MLP sees it — protects the
                # value function from physics blow-ups. If the user
                # explicitly sets clip on this term, theirs wins; otherwise
                # we keep the historical safety guard. (See
                # amp_obs_terms.py::_AMP_OBS_CLIP for the matching guard on
                # the discriminator's input side.)
                clip_tuple = term_cfg.clip if term_cfg.clip is not None else (-100.0, 100.0)
                args.append(f"clip=({clip_tuple[0]}, {clip_tuple[1]})")
                if term_cfg.history_length is not None:
                    args.append(f"history_length={int(term_cfg.history_length)}")
                lines.append(
                    f"        {term_type} = ObsTerm({', '.join(args)}{params_str})"
                )

            # P2.1 — append 4 Walk These Ways gait observation terms to
            # the policy group (only to "policy" by convention; private
            # obs groups like "critic" are free to add their own if the
            # user configures a second obs node later). Ordering is
            # stable (freq → phase sin/cos → body_h → step_h) so the
            # trained policy's input permutation can't drift.
            if self._gait_enabled() and gname == "policy":
                lines.append(
                    '        gait_frequency = ObsTerm('
                    'func=_unitport_gait_frequency_obs, '
                    'params={"command_name": "gait_command"})'
                )
                lines.append(
                    '        gait_phase = ObsTerm('
                    'func=_unitport_gait_phase_sin_cos_obs, '
                    'params={"command_name": "gait_command"})'
                )
                lines.append(
                    '        body_height_cmd = ObsTerm('
                    'func=_unitport_gait_body_height_cmd_obs, '
                    'params={"command_name": "gait_command"})'
                )
                lines.append(
                    '        step_height_cmd = ObsTerm('
                    'func=_unitport_gait_step_height_cmd_obs, '
                    'params={"command_name": "gait_command"})'
                )

            lines.append("")
            lines.append(f"    {gname}: {gname.capitalize()}Cfg = {gname.capitalize()}Cfg()")

        lines += ["", ""]
        return lines

    def _actions_cfg(self) -> List[str]:
        lines = ["@configclass", "class ActionsCfg:", '    """Action configuration."""', ""]
        # §1 Actor — prefer the unified ActorSetting node; fall back to
        # Action space flows from ActorSettingNode — the legacy
        # ILActionConfigNode was a duplicate data source and has been
        # deleted.
        actor_ids = self._find_by_type("actor_setting")
        if actor_ids:
            aid = actor_ids[0]
            expr_raw = self._p(aid, "action_joint_names_expr").strip()
            scale = self._pf(aid, "action_scale")
            use_offset = (
                self._p(aid, "action_use_default_offset").lower() == "true"
            )
            # Substrate-emit boundary (Phase 5 §IR-only contract):
            # Isaac Lab JointPositionActionCfg.joint_names is a regex list
            # matched against USD physical joint names. Two acceptable
            # canvas inputs:
            #   (a) Single regex catchall (default ".*") — passthrough; the
            #       launcher will match every USD joint, no IR translation
            #       needed.
            #   (b) JSON-encoded list of IR roles — translate each to its
            #       physical name before emitting. (User must use IR roles;
            #       physical names / vendor abbrev are rejected.)
            joint_names_literal = '".*"'
            try:
                import json as _json
                parsed = _json.loads(expr_raw) if expr_raw else None
            except Exception:
                parsed = None
            if isinstance(parsed, list) and parsed:
                # Explicit list — items are either regex patterns (passed
                # through) or IR roles (translated to physical names).
                # Mixed lists are supported (e.g. ["hip_.*", "thigh_FL"]).
                _regex_metachars = set(".^$*+?()[]{}|\\")
                literal_items: List[str] = []
                regex_items: List[str] = []
                indices: List[int] = []  # type per slot: 0=regex, 1=literal
                for x in parsed:
                    s = str(x)
                    if any(ch in _regex_metachars for ch in s):
                        regex_items.append(s)
                        indices.append(0)
                    else:
                        literal_items.append(s)
                        indices.append(1)
                phys_literals: List[str] = []
                if literal_items:
                    resolver = self._joint_ir_resolver
                    if resolver is None:
                        raise ValueError(
                            "\n[UnitPort][Compiler] actor_setting.action_joint_names_expr "
                            "contains literal joint roles but no robot is bound "
                            "on the compiler. Pass robot=spec.robot to "
                            "compile_env_cfg_to_file."
                        )
                    phys_literals = resolver.to_physical_list(
                        literal_items,
                        where="actor_setting.action_joint_names_expr",
                    )
                # Reassemble in original order: regex passthrough + translated IR roles.
                emitted: List[str] = []
                ri = li = 0
                for kind in indices:
                    if kind == 0:
                        emitted.append(regex_items[ri])
                        ri += 1
                    else:
                        emitted.append(phys_literals[li])
                        li += 1
                joint_names_literal = ", ".join(f'"{e}"' for e in emitted)
            elif isinstance(parsed, list):
                # Empty list ("[]") → user accepted the default; emit ".*"
                # catchall regex (Isaac Lab matches every USD joint).
                joint_names_literal = '".*"'
            else:
                # Raw value is a single regex string (legacy non-JSON path).
                regex = expr_raw or ".*"
                joint_names_literal = f'"{regex}"'

            lines.append(f"    joint_pos = mdp.JointPositionActionCfg(")
            lines.append(f'        asset_name="robot",')
            lines.append(f'        joint_names=[{joint_names_literal}],')
            lines.append(f"        scale={scale},")
            lines.append(f"        use_default_offset={use_offset},")
            lines.append(f"    )")
        lines += ["", ""]
        return lines

    def _commands_cfg(self) -> List[str]:
        lines = ["@configclass", "class CommandsCfg:", '    """Command configuration."""', ""]
        # Training Commands is the single source for velocity + gait
        # command schemas — legacy il_velocity_cmd / il_gamepad_input
        # fallback paths removed.
        cmd_ids = self._find_by_type("training_motion")
        if cmd_ids:
            cid = cmd_ids[0]
            # §1B — per-item weighted sampling. Each enabled training_item
            # contributes one sub-template (ranges = template ⊕ speed ⊕
            # advanced.command_overrides). The inline CommandTerm picks
            # one item per env-reset by self.weights (uniform until the
            # adaptive sampler updates it). gain_lin_* are reserved for
            # the deployment-side CommandBus envelope and no longer
            # define the training command space.
            items = self._build_training_items_list(cid)
            initial_weights = [1.0 / len(items)] * len(items)
            resample = self._p(cid, "resampling_time_range")
            rel_standing = self._pf(cid, "zero_command_probability")
            cmd_step_change_prob = self._pf(cid, "cmd_step_change_prob")
            lines.append("    base_velocity = UnitportWeightedVelocityCommandCfg(")
            lines.append('        asset_name="robot",')
            lines.append(f"        items={items!r},")
            lines.append(f"        initial_weights={initial_weights!r},")
            lines.append(f"        rel_standing_envs={rel_standing},")
            # Adaptive-sampling fields. Phase C wires these to canvas
            # params (adaptive_motion_enabled/weight_floor/weight_ceil).
            # Static defaults below keep the term in pure-uniform mode.
            lines.append(f"        weight_floor=0.03,")
            lines.append(f"        weight_ceil=0.30,")
            lines.append(f"        adaptive_enabled=False,")
            lines.append(f"        cmd_step_change_prob={cmd_step_change_prob},")
            lines.append(f"        resampling_time_range={resample},")
            lines.append(f"        debug_vis=False,")
            lines.append(f"    )")
            # P2.1 — Walk These Ways parameterised gait command term.
            # Emitted as a SECOND entry in CommandsCfg alongside
            # base_velocity, so the policy obs sees both the velocity
            # command and the gait command stacked in the order the
            # command manager iterates them.
            if self._gait_enabled():
                freq_lit = self._gait_range_tuple("gait_frequency_range", 1.5, 3.5)
                bh_lit = self._gait_range_tuple("body_height_range", 0.28, 0.40)
                sh_lit = self._gait_range_tuple("step_height_range", 0.03, 0.15)
                preset_lit = self._gait_preset_phase_literal()
                lines.append("    gait_command = UniformGaitCommandCfg(")
                lines.append('        asset_name="robot",')
                lines.append(f"        freq_range={freq_lit},")
                lines.append(f"        body_height_range={bh_lit},")
                lines.append(f"        step_height_range={sh_lit},")
                lines.append(f'        phase_mode="uniform",')
                lines.append(f"        preset_phases={preset_lit},")
                lines.append(f"        resampling_time_range={resample},")
                lines.append(f"        debug_vis=False,")
                lines.append(f"    )")
        lines += ["", ""]
        return lines

    def _rewards_cfg(self) -> List[str]:
        lines = ["@configclass", "class RewardsCfg:", '    """Reward configuration."""', ""]

        rew_ids = self._find_by_type("rewards")
        if not rew_ids:
            raise CanvasConfigError(
                reason=(
                    "canvas must contain at least one 'rewards' node — "
                    "compiler refuses to emit an empty RewardsCfg, which "
                    "would silently produce a zero-reward training run "
                    "(the 2026-05-11 PPO 'Mean reward never prints' "
                    "incident's other half)."
                ),
            )

        # Per-item fanout: walk each rewards node's downstream edges and
        # collect target item ids reached via ``reward_in__<item_id>``
        # ports on training_motion. Multi-rewards canvas = each node
        # supplies its terms only to its connected items.
        REWARD_IN_PREFIX = "reward_in__"
        fanout: Dict[str, List[str]] = {}
        for rid in rew_ids:
            items: List[str] = []
            for (_dst, src_port, dst_port) in self._downstream.get(rid, []):
                if src_port == "reward_pipe" and dst_port.startswith(REWARD_IN_PREFIX):
                    items.append(dst_port[len(REWARD_IN_PREFIX):])
            fanout[rid] = items

        any_per_item = any(fanout[rid] for rid in rew_ids)
        multi_mode = len(rew_ids) > 1 or any_per_item

        if not multi_mode:
            # Legacy single-rewards-node path: unconditional or phase-masked
            # via the ``applies_to`` field on individual reward terms.
            return self._emit_single_rewards_node(lines, rew_ids[0])

        # Multi-rewards mode: build item_id → index map from the
        # training_motion node's enabled-item iteration order (matches
        # _build_training_items_list, used downstream by the command term).
        tm_ids = self._find_by_type("training_motion")
        item_index: Dict[str, int] = {}
        if tm_ids:
            items_list = self._build_training_items_list(tm_ids[0])
            for idx, it in enumerate(items_list):
                item_index[str(it.get("id", f"item_{idx}"))] = idx

        seen_field_names: set = set()
        any_term_emitted = False
        for rid in rew_ids:
            connected_items = fanout.get(rid, [])
            if not connected_items:
                # Multi-rewards canvas with an unconnected rewards node:
                # its terms are skipped (no items to apply to). Surface
                # a warning so the user sees it in the cmd log.
                log.warning(
                    "[il-compile] rewards node %r has no reward_in__* "
                    "connections in a multi-rewards canvas — its terms "
                    "are skipped.", rid,
                )
                continue
            item_indices = sorted({
                item_index[it] for it in connected_items if it in item_index
            })
            if not item_indices:
                continue
            try:
                reward_terms = self._parse_json_param(rid, "reward_terms")
            except Exception:
                reward_terms = {}
            if not reward_terms:
                continue
            indices_literal = ", ".join(str(i) for i in item_indices)
            mask_args = (
                f"({indices_literal},)"
                if len(item_indices) == 1
                else f"({indices_literal})"
            )
            for func, entry in reward_terms.items():
                if isinstance(entry, dict):
                    weight = entry.get("weight", 0.0)
                else:
                    weight = entry
                try:
                    w = float(weight)
                except (ValueError, TypeError) as exc:
                    raise CanvasConfigError(
                        nid=rid,
                        key="reward_terms",
                        schema_id=self._types.get(rid, ""),
                        reason=(
                            f"reward weight for {func!r} is not a valid float: "
                            f"{weight!r} ({type(weight).__name__})"
                        ),
                    ) from exc
                mdp_module, mdp_func = self._reward_func_ref(func)
                params_str = self._reward_extra_params_from_node(rid, func)
                func_ref = f"{mdp_module}.{mdp_func}" if mdp_module else mdp_func
                wrapped = f"unitport_item_mask({func_ref}, {mask_args})"
                # Field-name disambiguation: same func across multiple
                # rewards nodes → suffix subsequent ones with ``__n<rid>``.
                field = func if func not in seen_field_names else f"{func}__n{rid}"
                seen_field_names.add(field)
                lines.append(
                    f"    {field} = RewTerm(func={wrapped}, weight={w}{params_str})"
                )
                any_term_emitted = True

        if not any_term_emitted:
            raise CanvasConfigError(
                reason=(
                    f"canvas has {len(rew_ids)} 'rewards' nodes but no "
                    "reward terms were emitted — all nodes either lack "
                    "reward_in__* connections or have empty reward_terms."
                ),
            )

        self._needs_item_mask_helper = True
        lines += ["", ""]
        return lines

    def _emit_single_rewards_node(
        self, lines: List[str], rid: str
    ) -> List[str]:
        """Legacy single-rewards-node emit path (no per-item fanout).

        Terms are unconditional unless they carry a non-empty
        ``applies_to`` field — in which case they're wrapped in
        ``unitport_phase_mask`` so they only contribute on envs whose
        active task item's motion_tag falls into one of the declared
        phases. Empty applies_to ⇒ unconditional.
        """
        reward_terms = self._parse_json_param(rid, "reward_terms")
        if not reward_terms:
            raise CanvasConfigError(
                nid=rid,
                key="reward_terms",
                schema_id=self._types.get(rid, ""),
                reason="reward_terms is empty — at least one reward term required.",
            )
        phase_masked = False
        for func, entry in reward_terms.items():
            applies_to: List[str] = []
            if isinstance(entry, dict):
                weight = entry.get("weight", 0.0)
                raw_applies = entry.get("applies_to") or []
                if isinstance(raw_applies, str):
                    applies_to = [s.strip() for s in raw_applies.split(",") if s.strip()]
                elif isinstance(raw_applies, (list, tuple)):
                    applies_to = [str(s).strip() for s in raw_applies if str(s).strip()]
            else:
                weight = entry
            try:
                w = float(weight)
            except (ValueError, TypeError) as exc:
                raise CanvasConfigError(
                    nid=rid,
                    key="reward_terms",
                    schema_id=self._types.get(rid, ""),
                    reason=(
                        f"reward weight for {func!r} is not a valid float: "
                        f"{weight!r} ({type(weight).__name__})"
                    ),
                ) from exc
            mdp_module, mdp_func = self._reward_func_ref(func)
            params_str = self._reward_extra_params_from_node(rid, func)
            func_ref = f"{mdp_module}.{mdp_func}" if mdp_module else mdp_func
            if applies_to:
                phases_literal = ", ".join(repr(p) for p in applies_to)
                wrapped = (
                    f"unitport_phase_mask({func_ref}, ({phases_literal},))"
                    if len(applies_to) == 1
                    else f"unitport_phase_mask({func_ref}, ({phases_literal}))"
                )
                lines.append(
                    f"    {func} = RewTerm(func={wrapped}, weight={w}{params_str})"
                )
                phase_masked = True
            else:
                lines.append(
                    f"    {func} = RewTerm(func={func_ref}, weight={w}{params_str})"
                )
        if phase_masked:
            self._needs_phase_mask_helper = True
        lines += ["", ""]
        return lines

    def _terminations_cfg(self) -> List[str]:
        """Emit @configclass TerminationsCfg from the unified items list.

        The Terminations node now stores every IL termination knob inside
        ``termination_conditions`` (a JSON dict of {item_key: threshold})
        instead of standalone toggle params. We map each item key the IL
        registry knows about to its corresponding ``DoneTerm`` line.

        For ``illegal_contact`` the contact body regex list is hardcoded
        here to a sensible quadruped default; loosening that knob requires
        editing the asset's ``task_template.json`` rather than the canvas,
        keeping the frontend identical between SB3 and IL.
        """
        lines = ["@configclass", "class TerminationsCfg:", '    """Termination configuration."""', ""]
        term_ids = self._find_by_type("terminations")
        if not term_ids:
            raise CanvasConfigError(
                reason=(
                    "canvas must contain exactly one 'terminations' node — "
                    "compiler refuses to emit an empty TerminationsCfg, which "
                    "would silently produce a training run where episodes "
                    "never end and rsl_rl's rewbuffer never fills (the "
                    "2026-05-11 'Mean reward never prints' incident's root)."
                ),
            )
        if len(term_ids) > 1:
            raise CanvasConfigError(
                reason=(
                    f"canvas has {len(term_ids)} 'terminations' nodes "
                    f"({term_ids!r}); exactly one is required."
                ),
            )

        tid = term_ids[0]
        items = self._parse_json_param(tid, "termination_conditions")
        if not items:
            raise CanvasConfigError(
                nid=tid,
                key="termination_conditions",
                schema_id=self._types.get(tid, ""),
                reason="termination_conditions is empty — at least one termination required (time_out is the canonical minimum).",
            )

        def _f(key: str) -> float:
            """Strict float read from items[key] — raises if missing or unparseable."""
            if key not in items:
                raise CanvasConfigError(
                    nid=tid,
                    key="termination_conditions",
                    schema_id=self._types.get(tid, ""),
                    reason=f"missing required sub-key {key!r}",
                )
            raw = items[key]
            try:
                return float(raw)
            except (ValueError, TypeError) as exc:
                raise CanvasConfigError(
                    nid=tid,
                    key="termination_conditions",
                    schema_id=self._types.get(tid, ""),
                    reason=(
                        f"sub-key {key!r} value {raw!r} ({type(raw).__name__}) "
                        f"is not a valid float"
                    ),
                ) from exc

        if "time_out" in items:
            # ``time_out`` on DoneTerm is a bool flag (Isaac Lab
            # TerminationTermCfg.time_out: bool = False) — when truthy, the
            # term feeds _truncated_buf instead of _terminated_buf. The
            # actual timeout *duration* lives on env_cfg.episode_length_s
            # (emitted at the UnitPortEnvCfg level via _ppo_runner_cfg's
            # caller). Passing the canvas float here works by truthy
            # coercion but is API-incorrect — emit the explicit bool to
            # match Isaac Lab's documented signature and align with the
            # sibling DoneTerm emissions below (base_height /
            # bad_orientation use params={} + default time_out=False).
            lines.append(f"    time_out = DoneTerm(func=mdp.time_out, time_out=True)")

        if "illegal_contact" in items:
            thresh = _f("illegal_contact")
            # Two corrections vs the original hardcoding:
            #
            #  (a) Sensor entity name = ``contact_forces`` (matches what
            #      _scene_cfg actually emits — see line ~393). The old
            #      string ``contact_sensor`` was a stale convention that
            #      never got updated when SceneCfg changed and shows up
            #      at runtime as:
            #         ValueError: The scene entity 'contact_sensor' does
            #         not exist. Available entities: ['terrain', 'robot',
            #         'contact_forces'].
            #
            #  (b) Body regex includes both ``base`` (Go2/H1/G1 USD root
            #      naming) and ``trunk`` (A1 / Anymal USD root naming),
            #      mirroring the same fix already applied to
            #      add_base_mass in _events_cfg.
            lines.append(f"    illegal_contact = DoneTerm(")
            lines.append(f"        func=mdp.illegal_contact,")
            # Resolve via joints_mapping IR — base + thigh + calf bodies are
            # the canonical "illegal contact" set for quadrupeds.  Calf is
            # included so the "elbow drop" failure mode (lower leg used as a
            # ground crutch to fake stability reward) terminates the episode
            # instead of just incurring a small undesired_contacts penalty.
            # If the mapper returns nothing (no Robot node body_mapping and
            # no asset_id), fall back to a permissive regex covering both
            # A1-style ("trunk") and Go2-style ("base") root naming.
            ic_bodies = self._resolve_bodies("base", "thighs", "calves")
            if ic_bodies:
                ic_expr = "[" + ", ".join(f'"{b}"' for b in ic_bodies) + "]"
            else:
                ic_expr = '["(base|trunk)", ".*thigh", ".*calf"]'
            lines.append(
                f"        params={{\"sensor_cfg\": SceneEntityCfg("
                f"\"contact_forces\", body_names={ic_expr}), "
                f"\"threshold\": {thresh}}},"
            )
            lines.append(f"    )")

        if "base_height" in items:
            min_h = _f("base_height")
            lines.append(
                f"    base_height = DoneTerm(func=mdp.root_height_below_minimum, "
                f"params={{\"minimum_height\": {min_h}}})"
            )

        # Roll/pitch termination. Without this, a quadruped can flip and
        # ragdoll without triggering base_height (torso still > min_h while
        # inverted), producing 900+ step "dead" episodes that blow up the
        # value function. ``bad_orientation`` uses projected gravity to
        # detect tilt, so ``limit_angle`` is the max allowed deviation from
        # upright in radians (0.7 ≈ 40°, a safe margin for AMP gaits).
        if "bad_orientation" in items:
            limit = _f("bad_orientation")
            lines.append(
                f"    bad_orientation = DoneTerm(func=mdp.bad_orientation, "
                f"params={{\"limit_angle\": {limit}}})"
            )

        lines += ["", ""]
        return lines

    def _events_cfg(self) -> List[str]:
        lines = ["@configclass", "class EventCfg:", '    """Domain randomization events."""', ""]
        dr_ids = self._find_by_type("domain_rand")
        if dr_ids:
            did = dr_ids[0]
            if self._p(did, "enable_friction_rand").lower() == "true":
                mode = self._p(did, "friction_mode")
                sf = self._p(did, "static_friction_range")
                df = self._p(did, "dynamic_friction_range")
                # mdp.randomize_rigid_body_material requires 5 mandatory
                # params: static_friction_range, dynamic_friction_range,
                # restitution_range, num_buckets, asset_cfg. The previous
                # emit only passed the first two and Isaac Lab refused
                # the term at EventManager prepare time. Defaults below
                # match Isaac Lab's stock Go2 velocity task config so
                # canvases that don't expose finer knobs still produce
                # a runnable env.
                rr = self._p(did, "restitution_range")
                # ``num_buckets`` is PhysX-internal pre-generated material
                # quantization (N material variants sampled per env reset).
                # 64 is Isaac Lab's stock value for legged-robot tasks and
                # has no user-facing semantics; deliberately not exposed in
                # the domain_rand manifest.
                lines.append(f"    physics_material = EventTerm(")
                lines.append(f'        func=mdp.randomize_rigid_body_material,')
                lines.append(f'        mode="{mode}",')
                lines.append(f"        params={{")
                lines.append(f'            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),')
                lines.append(f'            "static_friction_range": {sf},')
                lines.append(f'            "dynamic_friction_range": {df},')
                lines.append(f'            "restitution_range": {rr},')
                lines.append(f'            "num_buckets": 64,')
                lines.append(f"        }},")
                lines.append(f"    )")
            if self._p(did, "enable_mass_rand").lower() == "true":
                mode = self._p(did, "mass_mode")
                body = self._p(did, "mass_target_body")
                mr = self._p(did, "mass_offset_range")
                # mdp.randomize_rigid_body_mass requires 3 mandatory params:
                # asset_cfg, mass_distribution_params, operation. Operation
                # specifies how the random sample modifies the existing
                # mass: "add" (additive offset), "scale" (multiplicative),
                # or "abs" (replace). Canvas's "mass_offset_range" name
                # implies additive offsets, so default to "add". Future
                # work can expose mass_operation as a UI knob.
                op = self._p(did, "mass_operation")

                # ── body name resolution ──
                # If the user picked a canonical role name ("base",
                # "trunk", "torso"), resolve through the joints_mapping
                # IR so the compiled config uses the correct USD link
                # name for the active robot family.  A custom literal
                # body name (e.g. "FR_hip") is passed through verbatim.
                _canonical_roles = {"base", "trunk", "torso", "pelvis", "body"}
                if body in _canonical_roles:
                    # Strict: canonical role must resolve through the
                    # joints_mapping IR. The historical regex fallback
                    # ``"(base|trunk)"`` silently coexisted with a
                    # missing body_mapping and is exactly the kind of
                    # default the 2026-05-11 contract bans.
                    body_expr = f'"{self._resolve_body("base")}"'
                else:
                    body_expr = f'"{body}"'

                lines.append(f"    add_base_mass = EventTerm(")
                lines.append(f'        func=mdp.randomize_rigid_body_mass,')
                lines.append(f'        mode="{mode}",')
                lines.append(f"        params={{")
                lines.append(f'            "asset_cfg": SceneEntityCfg("robot", body_names=[{body_expr}]),')
                lines.append(f'            "mass_distribution_params": {mr},')
                lines.append(f'            "operation": "{op}",')
                lines.append(f"        }},")
                lines.append(f"    )")
            if self._p(did, "enable_external_push").lower() == "true":
                mode = self._p(did, "push_mode")
                vx = self._p(did, "push_velocity_x_range")
                vy = self._p(did, "push_velocity_y_range")
                interval = self._p(did, "push_interval_range_s")
                lines.append(f"    push_robot = EventTerm(")
                lines.append(f'        func=mdp.push_by_setting_velocity,')
                lines.append(f'        mode="{mode}",')
                lines.append(f"        interval_range_s={interval},")
                lines.append(f"        params={{\"velocity_range\": {{\"x\": {vx}, \"y\": {vy}}}}},")
                lines.append(f"    )")
            if self._p(did, "enable_init_pose_rand").lower() == "true":
                mode = self._p(did, "init_pose_mode")
                px = self._p(did, "init_pos_x_range")
                py = self._p(did, "init_pos_y_range")
                yaw = self._p(did, "init_yaw_range")
                lines.append(f"    reset_base = EventTerm(")
                lines.append(f'        func=mdp.reset_root_state_uniform,')
                lines.append(f'        mode="{mode}",')
                lines.append(f"        params={{")
                lines.append(f'            "pose_range": {{"x": {px}, "y": {py}}},')
                lines.append(f'            "velocity_range": {{"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": {yaw}}},')
                lines.append(f"        }},")
                lines.append(f"    )")
            if self._p(did, "enable_joint_noise").lower() == "true":
                mode = self._p(did, "joint_noise_mode")
                pos_noise = self._pf(did, "joint_pos_noise")
                vel_noise = self._pf(did, "joint_vel_noise")
                lines.append(f"    reset_joints = EventTerm(")
                lines.append(f'        func=mdp.reset_joints_by_offset,')
                lines.append(f'        mode="{mode}",')
                lines.append(f"        params={{")
                lines.append(f'            "position_range": (-{pos_noise}, {pos_noise}),')
                lines.append(f'            "velocity_range": (-{vel_noise}, {vel_noise}),')
                lines.append(f"        }},")
                lines.append(f"    )")

        # actor_setting.init_pose_noise_scale — episode-start joint noise on
        # top of init_joint_angles. Independent of domain_rand's enable_joint_noise
        # (which is a domain-randomization knob); this one is the user-facing
        # "ε around init pose" tunable that the Init Pose section advertises.
        # Only emit when DR has not already laid down its own reset_joints (to
        # avoid Isaac Lab raising a duplicate-event-key error); the field's
        # role overlaps with DR joint noise, so DR wins when both are on.
        actor_ids_for_pose_noise = self._find_by_type("actor_setting")
        dr_joint_noise_on = bool(dr_ids) and (
            self._p(dr_ids[0], "enable_joint_noise").lower() == "true"
        )
        if actor_ids_for_pose_noise and not dr_joint_noise_on:
            aid_noise = actor_ids_for_pose_noise[0]
            pose_noise = self._pf(aid_noise, "init_pose_noise_scale")
            if pose_noise > 0.0:
                lines.append(f"    actor_init_joint_noise = EventTerm(")
                lines.append(f'        func=mdp.reset_joints_by_offset,')
                lines.append(f'        mode="reset",')
                lines.append(f"        params={{")
                lines.append(f'            "position_range": (-{pose_noise}, {pose_noise}),')
                lines.append(f'            "velocity_range": (0.0, 0.0),')
                lines.append(f"        }},")
                lines.append(f"    )")

        # RSI — fires AFTER reset_base / reset_joints so its reference-motion
        # writes become the final reset state for the Bernoulli(rsi_prob)
        # fraction of envs. Prior resets still apply to the (1-rsi_prob)
        # complement because the event fn subsets env_ids before writing
        # rather than writing all then undoing.
        trainer_ids = self._find_by_type("il_ppo_trainer")
        motion_ids = self._find_by_type("training_motion")
        _amp_active = bool(trainer_ids) and (
            self._p(trainer_ids[0], "training_mode").upper() == "AMP_PPO"
        )
        _rsi_on = bool(motion_ids) and (
            self._p(motion_ids[0], "reference_state_init_enabled").lower()
            == "true"
        )
        if _amp_active and _rsi_on:
            mid = motion_ids[0]
            rsi_prob = self._pf(mid, "rsi_prob")
            rsi_joint_noise = self._pf(mid, "rsi_joint_noise")
            lines.append(f"    reset_from_reference_motion = EventTerm(")
            lines.append(f"        func=_amp_mdp_events.reset_from_reference_motion,")
            lines.append(f'        mode="reset",')
            lines.append(f"        params={{")
            lines.append(f'            "asset_cfg": SceneEntityCfg("robot"),')
            lines.append(f'            "pool_id": "default",')
            lines.append(f'            "rsi_prob": {rsi_prob},')
            lines.append(f'            "joint_noise": {rsi_joint_noise},')
            lines.append(f"        }},")
            lines.append(f"    )")
        lines += ["", ""]
        return lines

    def _root_env_cfg(self) -> List[str]:
        # ManagerBasedRLEnvCfg has TWO required top-level fields that
        # have no canvas-side equivalent:
        #
        #   decimation       : sim steps per control step
        #                      = round(control_dt / sim_dt)
        #   episode_length_s : max episode duration in seconds
        #                      = the time_out termination value when set
        #
        # Without these, ManagerBasedRLEnv.__init__ raises:
        #   "Missing values detected in object UnitPortEnvCfg for the
        #    following fields: - decimation - episode_length_s"
        # which is exactly the crash phase_3 was hitting end-to-end.
        #
        # Resolve from the canvas:
        #   sim_dt          : play_ground_setting.sim_dt (default 0.005)
        #   control_dt      : 0.02 (50 Hz Go2 default — no canvas knob)
        #   episode_length  : terminations.termination_conditions["time_out"]
        #                     when present, else 20.0s default

        # Sim dt → decimation
        sim_dt = 0.005
        pg_ids = self._find_by_type("play_ground_setting")
        if pg_ids:
            sim_dt = self._pf(pg_ids[0], "sim_dt") or 0.005
        control_dt = 0.02   # 50 Hz — matches Isaac Lab Go2 default
        decimation = max(1, int(round(control_dt / sim_dt))) if sim_dt > 0 else 4

        # Episode length → from the time_out termination value. Strict:
        # ``terminations`` node is mandatory (enforced upstream in
        # ``_terminations_cfg``) and must declare ``time_out`` as a
        # float — anything else is a canvas error, not a fall-back-to-
        # 20s situation (the 2026-05-11 silent-fallback regression).
        term_ids = self._find_by_type("terminations")
        if not term_ids:
            raise CanvasConfigError(
                reason=(
                    "canvas is missing the 'terminations' node — "
                    "episode_length_s is derived from its time_out "
                    "value and has no default."
                ),
            )
        tcs = self._parse_json_param(term_ids[0], "termination_conditions")
        if "time_out" not in tcs:
            raise CanvasConfigError(
                nid=term_ids[0],
                key="termination_conditions",
                schema_id=self._types.get(term_ids[0], ""),
                reason=(
                    "missing required sub-key 'time_out' — Isaac Lab "
                    "episode_length_s is derived from it; canvas must "
                    "set the desired episode duration in seconds."
                ),
            )
        try:
            episode_length_s = float(tcs["time_out"])
        except (TypeError, ValueError) as exc:
            raise CanvasConfigError(
                nid=term_ids[0],
                key="termination_conditions",
                schema_id=self._types.get(term_ids[0], ""),
                reason=(
                    f"sub-key 'time_out' value {tcs['time_out']!r} "
                    f"({type(tcs['time_out']).__name__}) is not a "
                    f"valid float (seconds)"
                ),
            ) from exc

        sim_literal = self._simulation_cfg_literal()
        return [
            "@configclass",
            "class UnitPortEnvCfg(ManagerBasedRLEnvCfg):",
            '    """Auto-generated environment configuration."""',
            "",
            f"    sim: SimulationCfg = {sim_literal}",
            "    scene: SceneCfg = SceneCfg()",
            "    observations: ObservationsCfg = ObservationsCfg()",
            "    actions: ActionsCfg = ActionsCfg()",
            "    commands: CommandsCfg = CommandsCfg()",
            "    rewards: RewardsCfg = RewardsCfg()",
            "    terminations: TerminationsCfg = TerminationsCfg()",
            "    events: EventCfg = EventCfg()",
            "",
            "    # Required ManagerBasedRLEnvCfg root fields — no canvas",
            "    # equivalents today, derived from sim_config.dt + the",
            "    # time_out termination at compile time.",
            f"    decimation: int = {decimation}",
            f"    episode_length_s: float = {episode_length_s}",
            "",
            "",
        ]

    def _ppo_runner_cfg(self) -> List[str]:
        lines = ["@configclass", "class PPORunnerCfg:", '    """PPO training runner configuration."""', ""]
        # H2 (AMP fix plan): emit ``algorithm_class`` as a plain string
        # field so the compiled config self-documents which runner it
        # targets. The launcher already knows the algorithm from the
        # ``--unitport_algorithm`` CLI arg, but surfacing it on the
        # compiled cfg lets canvas → compiled-config traceability
        # checks work without a subprocess round-trip. See also
        # ``src/system/training/run_meta.py`` for the load-bearing
        # run-directory marker that downstream tools consume.
        # Unified IL trainer — algorithm is chosen via the trainer's
        # training_mode parameter.
        trainer_ids = self._find_by_type("il_ppo_trainer")
        has_amp_trainer = False
        if trainer_ids:
            _mode = str(self._p(trainer_ids[0], "training_mode") or "PPO").strip()
            has_amp_trainer = (_mode == "AMP_PPO")
        algorithm_class = "AMP_PPO" if has_amp_trainer else "PPO"
        lines.append(f'    algorithm_class: str = "{algorithm_class}"')
        lines.append("")


        if trainer_ids:
            tid = trainer_ids[0]
            lines.append(f"    max_iterations = {self._pi(tid, 'max_iterations')}")
            lines.append(f"    num_steps_per_env = {self._pi(tid, 'num_steps_per_env')}")
            lines.append(f"    num_learning_epochs = {self._pi(tid, 'num_learning_epochs')}")
            lines.append(f"    num_minibatches = {self._pi(tid, 'num_minibatches')}")
            lines.append(f"    learning_rate = {self._pf(tid, 'learning_rate')}")
            lines.append(f"    discount_factor = {self._pf(tid, 'discount_factor')}")
            lines.append(f"    gae_lambda = {self._pf(tid, 'gae_lambda')}")
            lines.append(f"    clip_param = {self._pf(tid, 'clip_param')}")
            lines.append(f"    entropy_coef = {self._pf(tid, 'entropy_coef')}")
            lines.append(f"    value_loss_coef = {self._pf(tid, 'value_loss_coef')}")
            lines.append(f"    max_grad_norm = {self._pf(tid, 'max_grad_norm')}")
            schedule = self._p(tid, "schedule")
            lines.append(f'    schedule = "{schedule}"')
            if schedule == "adaptive":
                lines.append(f"    desired_kl = {self._pf(tid, 'desired_kl')}")
            lines.append(f"    save_interval = {self._pi(tid, 'save_interval')}")
            lines.append(f"    seed = {self._pi(tid, 'seed')}")

        # Policy network
        net_ids = self._find_by_type("il_policy_network")
        if net_ids:
            nid = net_ids[0]
            lines.append("")
            lines.append("    # Actor-Critic Network")
            lines.append(f"    actor_hidden_dims = {self._p(nid, 'actor_hidden_dims')}")
            lines.append(f"    critic_hidden_dims = {self._p(nid, 'critic_hidden_dims')}")
            lines.append(f'    activation = "{self._p(nid, "activation")}"')
            # Canvas stores log_std (community convention, e.g. -1.0 → std ≈ 0.37).
            # The compiled config emits the direct std that rsl_rl / ActorCritic expects.
            import math as _math
            _log_std = self._pf(nid, 'init_noise_std')
            lines.append(f"    init_noise_std = {_math.exp(_log_std):.6f}  # exp({_log_std})")
            # AMP discriminator hidden dims — only emitted when an
            # explicit DiscriminatorNode is wired on the canvas. The
            # legacy fallback to ``il_policy_network.disc_hidden_dims``
            # was removed in the strict-canvas migration (the field
            # itself is gone from il_policy_network's manifest).
            disc_ids = self._find_by_type("discriminator")
            if disc_ids:
                _disc_raw = self._p(disc_ids[0], "disc_hidden_dims")
                import json as _jd
                try:
                    _disc_list = [int(x) for x in _jd.loads(_disc_raw) if int(x) > 0]
                except Exception as _exc:
                    raise CanvasConfigError(
                        nid=disc_ids[0],
                        key="disc_hidden_dims",
                        schema_id="discriminator",
                        reason=(
                            f"disc_hidden_dims is not a JSON list of "
                            f"positive ints (got {_disc_raw!r}): {_exc}"
                        ),
                    )
                if not _disc_list:
                    raise CanvasConfigError(
                        nid=disc_ids[0],
                        key="disc_hidden_dims",
                        schema_id="discriminator",
                        reason="disc_hidden_dims resolved to an empty list",
                    )
                lines.append(f"    disc_hidden_dims = {_disc_list}")

        lines += ["", ""]
        return lines

    def _unitport_curriculum_cfg(self) -> List[str]:
        """Emit a module-level ``UNITPORT_CURRICULUM`` dict.

        Aggregates curriculum_* parameters scattered across five host
        nodes (training_motion, actor_setting, il_ppo_trainer,
        terminations, il_observation) into a single dict that the
        launcher reads and hands to
        :class:`AMPOnPolicyRunner` for per-iteration dispatch.

        Schema (keys only emitted when the owning *enabled* toggle is
        on; disabled dimensions are omitted entirely so the runner
        treats them as no-op)::

            UNITPORT_CURRICULUM = {
                "command_velocity":        {"start": 0.25, "end": 1.0, "ramp_iters": 800},
                "action_scale":            {"start": 0.15, "end": 0.25, "ramp_iters": 300},
                "entropy_coef":            {"start": 0.01, "end": 0.003, "ramp_iters": 1500},
                "termination_base_height": {"start": 0.18, "end": 0.22, "ramp_iters": 500},
                "obs_noise_scale":         {"start": 0.25, "end": 1.0, "ramp_iters": 500},
            }

        When no dimension is enabled, emits ``UNITPORT_CURRICULUM = {}``
        so the launcher's hasattr check stays truthy but the runner's
        update loop short-circuits immediately.
        """
        dims: List[Tuple[str, str, str, float, float, int]] = []

        # Each tuple: (canvas_node_type, ui_prefix, dict_key, default_start, default_end, default_ramp)
        specs = [
            ("training_motion",  "command_curriculum",        "command_velocity",         0.25, 1.0,  800),
            ("actor_setting",    "action_scale_curriculum",   "action_scale",             0.15, 0.25, 300),
            ("il_ppo_trainer",   "entropy_schedule",          "entropy_coef",             0.01, 0.003, 1500),
            ("terminations",     "termination_curriculum",    "termination_base_height",  0.18, 0.22, 500),
            ("il_observation",   "obs_noise_curriculum",      "obs_noise_scale",          0.25, 1.0,  500),
        ]

        emitted: List[str] = []
        for node_type, prefix, key, d_start, d_end, d_ramp in specs:
            nids = self._find_by_type(node_type)
            if not nids:
                continue
            nid = nids[0]
            enabled_raw = str(self._p(nid, f"{prefix}_enabled") or "false").strip().lower()
            if enabled_raw not in ("true", "1", "yes", "on"):
                continue
            start = self._pf(nid, f"{prefix}_start")
            end = self._pf(nid, f"{prefix}_end")
            ramp = self._pi(nid, f"{prefix}_ramp_iters")
            emitted.append(
                f'    "{key}": {{"start": {start}, "end": {end}, "ramp_iters": {max(1, int(ramp))}}},'
            )

        lines = [
            "# " + "=" * 70,
            "# UnitPort curriculum schedules — dispatched by AMPOnPolicyRunner",
            "# " + "=" * 70,
            "# Per-iteration schedules for cold-start RL training. Each key",
            "# holds a linear ramp (start → end over ramp_iters outer PPO",
            "# iterations). Omitted keys = dimension disabled on the canvas.",
            "# The runner reads this dict via getattr(cfg_module,",
            "# 'UNITPORT_CURRICULUM', {}) and mutates alg/env state at the",
            "# top of each outer iteration.",
        ]
        if emitted:
            lines.append("UNITPORT_CURRICULUM = {")
            lines.extend(emitted)
            lines.append("}")
        else:
            lines.append("UNITPORT_CURRICULUM = {}")
        lines += ["", ""]
        return lines

    def _export_cfg(self) -> List[str]:
        """Emit export metadata as a dict constant (consumed by IsaacLabBackend).

        Reads from the unified ``export`` node. Field names align
        with the ExportConfig dataclass.
        """
        exp_ids = self._find_by_type("export")
        if not exp_ids:
            return []
        eid = exp_ids[0]
        onnx = self._p(eid, "include_onnx").lower() == "true"
        ts = self._p(eid, "include_torchscript").lower() == "true"
        name = self._p(eid, "bundle_name")
        auto_imp = self._p(eid, "auto_import").lower() == "true"
        lines = [
            "# Post-training export configuration (read by UnitPort backend)",
            "EXPORT_CFG = {",
            f'    "export_onnx": {onnx},',
            f'    "export_torchscript": {ts},',
            f'    "bundle_name": "{name}",',
            f'    "auto_import": {auto_imp},',
            "}",
            "",
            "",
        ]
        return lines

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _actuator_cfg_class(model: str) -> str:
        _map = {
            "implicit_pd":    "ImplicitActuatorCfg",
            "dc_motor":       "DCMotorCfg",
            "actuator_net":   "ActuatorNetMLPCfg",
            "ideal_torque":   "IdealPDActuatorCfg",
        }
        return _map.get(model, "ImplicitActuatorCfg")

    @staticmethod
    def _obs_term_func(term_type: str) -> str:
        _map = {
            "base_lin_vel": "base_lin_vel",
            "base_ang_vel": "base_ang_vel",
            "projected_gravity": "projected_gravity",
            "velocity_command": "generated_commands",
            "joint_pos": "joint_pos_rel",
            "joint_vel": "joint_vel_rel",
            "last_action": "last_action",
            "height_scan": "height_scan",
        }
        return _map.get(term_type, term_type)

    # ------------------------------------------------------------------
    # Registry-driven reward helpers
    # ------------------------------------------------------------------
    # All reward metadata is read from the reward sub-registries via
    # kind-scoped lookup at compile time.
    # No hardcoded function maps or param dicts in this file.

    @staticmethod
    def _reward_func_ref(func_key: str) -> tuple:
        """Return (module_alias_or_empty, function_name) for a reward key.

        Reads ``il_module`` and ``il_func`` from the registry entry.
        Kind-aware lookup guarantees we never pick up a same-named
        termination entry (e.g. ``base_height`` exists in both kinds).
        """
        from application.training.isaac_lab.task_module_registry import lookup, BACKEND_ISAAC
        item = lookup(func_key, kind="reward", backend=BACKEND_ISAAC)
        if item is None or not item.il_func:
            return "mdp", func_key
        return item.il_module, item.il_func

    @staticmethod
    def _reward_func_name(func_key: str) -> str:
        from application.training.isaac_lab.task_module_registry import lookup, BACKEND_ISAAC
        item = lookup(func_key, kind="reward", backend=BACKEND_ISAAC)
        return item.il_func if (item and item.il_func) else func_key

    def _reward_extra_params(self, nid: str, func_key: str) -> str:
        """Build the ``params={...}`` kwarg string for a RewTerm.

        Reads the ``il_params`` template from the registry, substitutes
        node-level values (``{node_std}``, ``{node_threshold}``), then
        resolves any ``{ir:role}`` body-name placeholders via BodyIR.

        Uses reward-scoped registries explicitly — see ``_reward_func_ref``
        for the collision rationale (same key, reward vs termination).
        """
        from application.training.isaac_lab.task_module_registry import (
            REWARD_REGISTRY, IL_REWARD_REGISTRY,
        )
        item = IL_REWARD_REGISTRY.get(func_key) or REWARD_REGISTRY.get(func_key)
        if item is None or not item.il_params:
            return ""
        # Phase 1: resolve {ir:role} body-name placeholders FIRST
        # (before .format(), because {ir:feet} is not a valid format key)
        params_str = item.il_params
        if "{ir:" in params_str:
            mapper = self._get_body_ir_mapper()
            from application.training.body_ir import resolve_body_params
            params_str = resolve_body_params(params_str, mapper)
        # Phase 2: substitute node-level values ({node_std}, {node_threshold})
        # plus robot-level scalars pulled from the canvas Robot node so
        # reward params can vary per machine without hardcoding (e.g.
        # base_height target).
        params_str = params_str.format(
            node_std=self._pf(nid, "std"),
            node_threshold=self._pf(nid, "threshold"),
            robot_target_height=self._resolve_robot_target_height(),
        )
        return ", params={" + params_str + "}"

    def _resolve_robot_target_height(self) -> float:
        """Return the effective base_height target (m), or 0.0 sentinel.

        Resolution:
          1. Robot node ``target_height`` param — explicit per-project
             override entered on the canvas slider. Returned verbatim
             when > 0.
          2. ``0.0`` sentinel — signals to ``_unitport_base_height_l2``
             that no canvas override was set; the reward function then
             auto-resolves the target to ``asset.data.default_root_state[:, 2]``
             (the asset's spawn z) at the first call. This is brand-neutral
             and avoids the old ``0.34`` Go2-only magic that silently
             broke training for any robot with a different standing height.
        """
        robot_ids = self._find_by_type("robot")
        if robot_ids:
            override = self._pf(robot_ids[0], "target_height")
            if override > 0.0:
                return override
        return 0.0

    # ------------------------------------------------------------------
    # Cross-node consistency validation
    # ------------------------------------------------------------------

    def _validate_reward_termination_consistency(self) -> None:
        """Raise ValueError on cross-node configurations that destroy training.

        Currently checks one rule (more can be added as we hit them):

        **base_height conflict**: when the canvas has both a ``base_height``
        reward (which pulls root z toward ``target_height``) AND a
        ``base_height`` termination (which kills the episode when root z
        drops below ``minimum_height``), the two thresholds must not invert
        — if ``target_height < minimum_height`` the policy gets terminated
        every time it tries to claim the reward, producing flat reward
        curves that look like "training is broken" but are actually
        "rewarded behavior is illegal".
        """
        rew_ids = self._find_by_type("rewards")
        term_ids = self._find_by_type("terminations")
        if not rew_ids or not term_ids:
            # Upstream emit (_rewards_cfg / _terminations_cfg) already
            # raises CanvasConfigError when either node is missing.
            # Reaching here with one absent means the caller bypassed
            # those — still skip the cross-check rather than mis-fire.
            return

        # Multi-rewards canvas: ``base_height`` may live in any rewards
        # node. Aggregate keys across all rewards nodes for the
        # cross-node check.
        rewards_keys: set = set()
        for rid in rew_ids:
            try:
                rt = self._parse_json_param(rid, "reward_terms")
            except Exception:
                continue
            if isinstance(rt, dict):
                rewards_keys.update(rt.keys())
        terminations = self._parse_json_param(term_ids[0], "termination_conditions")
        if "base_height" not in rewards_keys or "base_height" not in terminations:
            return

        target = self._resolve_robot_target_height()
        try:
            minimum = float(terminations["base_height"])
        except (ValueError, TypeError) as exc:
            raise CanvasConfigError(
                nid=term_ids[0],
                key="termination_conditions",
                schema_id=self._types.get(term_ids[0], ""),
                reason=(
                    f"sub-key 'base_height' value {terminations['base_height']!r} "
                    f"is not a valid float — cannot validate "
                    f"reward/termination conflict; canvas must fix."
                ),
            ) from exc

        if target < minimum:
            raise ValueError(
                f"\n[UnitPort][Compiler] Reward/termination conflict: "
                f"base_height reward target={target} m is BELOW the "
                f"base_height termination minimum={minimum} m.\n"
                f"  This means the policy must drop the robot below "
                f"{minimum} m to claim the reward, but the env will "
                f"terminate the episode the instant it does — reward "
                f"signal is unsatisfiable, training will not converge.\n"
                f"  Fix one of:\n"
                f"    (a) Robot node target_height ≥ {minimum} m "
                f"(currently {target} m).\n"
                f"    (b) Terminations node base_height ≤ {target} m "
                f"(currently {minimum} m).\n"
                f"    (c) Remove base_height from the Rewards or "
                f"Terminations node if you only want one of the two."
            )

    def _get_body_ir_mapper(self):
        """Build a BodyIRMapper from the canvas's Joints Mapping node.

        The ``joints_mapping`` node is the authoritative IR layer: it
        owns the USD→IR-role mapping that the user has visually
        validated on the canvas.  **Every** body-name / link-path the
        compiler emits must trace back to it.

        Priority:
          1. Robot node ``body_mapping`` param — AUTHORITATIVE when set
          2. Robot asset registry auto-resolve — when the body_mapping
             param is empty, fall back to the asset's structural
             metadata (base_link, foot_link_names, etc.)
          3. Empty mapper — last resort (unusable canvas)
        """
        if hasattr(self, "_body_ir_mapper_cache"):
            return self._body_ir_mapper_cache

        from application.training.body_ir import BodyIRMapper

        _log = logging.getLogger(__name__)

        # 1. Robot node body_mapping param — AUTHORITATIVE SOURCE
        #    (absorbed from the deprecated JointsMappingNode in the 6-section refactor)
        mapping_ids = self._find_by_type("robot")
        if mapping_ids:
            import json
            raw = self._p(mapping_ids[0], "body_mapping")
            if raw:
                try:
                    d = json.loads(raw)
                    mapper = BodyIRMapper.from_dict(d)
                    # Use the user's mapping unconditionally.  If it's
                    # partially resolved, warn so the user can see
                    # which roles still need attention — but never
                    # silently bypass the mapping to fall back to
                    # asset metadata (previous behaviour, which hid
                    # the user's intent behind opaque regex defaults).
                    if not mapper.all_resolved(required_only=True):
                        # ``unresolved_roles()`` returns ``List[str]``
                        # (role ids only), not IRRole objects.
                        unresolved = mapper.unresolved_roles(required_only=True)
                        _log.warning(
                            "[IR] Robot node body_mapping has unresolved roles: "
                            "%s — compiler will use the partial mapping as-is. "
                            "Open the Joints Mapping node on the canvas to "
                            "resolve these.",
                            unresolved,
                        )
                    base_body = None
                    base_role = mapper.get("base")
                    if base_role and base_role.body:
                        base_body = base_role.body
                    # ``roles`` is a @property on BodyIRMapper, NOT a
                    # method.  Calling it raises "'list' object is not
                    # callable" — accessing it as an attribute is the
                    # correct usage.
                    _log.info(
                        "[IR] body mapper source = Robot node body_mapping "
                        "(base=%s, %d roles resolved)",
                        base_body,
                        sum(1 for r in mapper.roles if r.resolved),
                    )
                    self._body_ir_mapper_cache = mapper
                    return mapper
                except Exception as exc:
                    _log.error(
                        "[IR] failed to parse joints_mapping body_mapping "
                        "JSON: %s — falling back to robot asset auto-resolve",
                        exc,
                    )

        # 2. Auto-resolve from the Robot node's asset_id (when its
        #    body_mapping param was empty / failed to parse)
        robot_ids = self._find_by_type("robot")
        if robot_ids:
            asset_id = self._p(robot_ids[0], "asset_id").strip()
            if asset_id:
                try:
                    from application.service.robot_assets import (
                        get_robot_asset_service,
                    )
                    from registers.robots import resolve_id as _resolve_robot_id
                    sku = _resolve_robot_id(asset_id) or asset_id
                    asset = get_robot_asset_service().resolve(sku)
                    if asset is None:
                        raise RuntimeError(
                            f"asset_id={asset_id!r} did not resolve "
                            f"(sku={sku!r})"
                        )
                    mapper = BodyIRMapper.from_robot_asset(asset)
                    _log.info(
                        "[IR] body mapper source = robot_asset "
                        "auto-resolve (asset_id=%s, sku=%s) — no "
                        "joints_mapping node on the canvas",
                        asset_id, sku,
                    )
                    self._body_ir_mapper_cache = mapper
                    return mapper
                except Exception as exc:
                    _log.error(
                        "[IR] robot asset auto-resolve failed for "
                        "asset_id=%s: %s",
                        asset_id, exc,
                    )

        # 3. Last resort: empty mapper
        _log.warning(
            "[IR] body mapper source = EMPTY — no Robot node body_mapping "
            "and no resolvable robot asset. Compiler will emit fallback "
            "regexes that may not match the active robot family."
        )
        mapper = BodyIRMapper([])
        self._body_ir_mapper_cache = mapper
        return mapper

    # ------------------------------------------------------------------
    # Body-name resolution helpers
    # ------------------------------------------------------------------
    #
    # Every site in the compiler that needs to emit a body name (scene
    # prim paths, body_names filters, SceneEntityCfg params, …) MUST go
    # through one of these helpers so the Robot node body_mapping's IR
    # layer is the single source of truth.  Hardcoded regexes like
    # ``"(base|trunk)"`` or ``".*_foot"`` cause subtle failures when
    # the user swaps robot families (A1 "trunk" vs Go2 "base",
    # quadruped "_foot" vs humanoid "_ankle", etc.) — resolving via
    # the mapper keeps the compiler robot-agnostic.

    # Compiler-internal canonical defaults for body roles the canvas
    # author is not expected to set by hand (the auto-from-asset set in
    # body_ir.AUTO_FROM_ASSET_CATEGORIES). These are USD/MJCF naming
    # conventions, not silent canvas fallbacks — they're emitted into
    # the generated config the same way "PhysX gravity = -9.81" is, not
    # as a substitute for missing canvas data.
    _CANONICAL_BODY_DEFAULTS: Dict[str, str] = {
        "base": "base",
    }

    def _resolve_body(self, role_or_category: str) -> str:
        """Return the first body name matching a role or category.

        Tries, in order:
          1. Exact role_id match (``mapper.get("base").body``)
          2. Category bodies (``mapper.get_category_bodies("base")[0]``)
          3. For an auto-from-asset role (``base`` / ``feet``) only:
             a canonical default name (``base`` for quadruped /
             humanoid USDs by convention).

        Raises :class:`CanvasConfigError` when neither resolves AND the
        role is not in ``AUTO_FROM_ASSET_CATEGORIES`` — i.e. the canvas
        was supposed to declare it. ``base`` and ``feet`` are exempted
        because the IR catalog marks them required-but-derivable: the
        runtime needs the name, but the value lives in asset metadata
        (or, for unitree_go2 etc., is the USD convention ``"base"``).
        """
        from application.training.body_ir import AUTO_FROM_ASSET_CATEGORIES

        mapper = self._get_body_ir_mapper()
        # Role id first (exact match like "base")
        role = mapper.get(role_or_category)
        if role and role.body:
            return str(role.body)
        # Fall back to category (e.g. "feet" → all feet bodies)
        bodies = mapper.get_category_bodies(role_or_category)
        if bodies:
            return str(bodies[0])
        # Compiler-internal canonical default for auto-from-asset slots
        # — only kicks in when the asset path also returned nothing.
        category = role.category if role is not None else role_or_category
        if category in AUTO_FROM_ASSET_CATEGORIES:
            canonical = self._CANONICAL_BODY_DEFAULTS.get(category)
            if canonical is not None:
                return canonical
        raise CanvasConfigError(
            reason=(
                f"Robot body_mapping has no body for role/category "
                f"{role_or_category!r} — open the Robot node and "
                f"resolve the missing role, or change the caller to "
                f"use a robot that defines this role."
            ),
        )

    def _resolve_bodies(self, *role_or_categories: str) -> List[str]:
        """Return all body names for the given roles/categories.

        Combines results from multiple categories (e.g.
        ``_resolve_bodies("base", "thighs")`` for illegal_contact).
        Deduplicates while preserving order.
        """
        try:
            mapper = self._get_body_ir_mapper()
        except Exception:
            return []
        out: List[str] = []
        seen: set = set()
        for key in role_or_categories:
            # Try as a role id first
            role = mapper.get(key)
            if role and role.body and role.body not in seen:
                out.append(str(role.body))
                seen.add(role.body)
                continue
            # Fall back to category
            try:
                for b in mapper.get_category_bodies(key):
                    if b and b not in seen:
                        out.append(str(b))
                        seen.add(b)
            except Exception:
                pass
        return out

    def _reward_extra_params_from_node(self, nid: str, func_key: str) -> str:
        return self._reward_extra_params(nid, func_key)


# ---------------------------------------------------------------------------
# Top-level helpers (RELEASE wiring layer)
# ---------------------------------------------------------------------------

def compile_env_cfg_to_file(
    canvas_dict: Dict[str, Any],
    *,
    out_dir: Path,
    file_name: str = "unitport_env_cfg.py",
    robot: Optional["RobotSpecRef"] = None,
) -> Path:
    """Compile a canvas IR-shape dict into ``<out_dir>/unitport_env_cfg.py``.

    The output file declares ``UnitPortEnvCfg`` (an Isaac Lab @configclass)
    + ``PPORunnerCfg`` (an RSL-RL OnPolicyRunnerCfg). The launcher loads it
    via ``--unitport_config <path>`` (`il_train_launcher.py:376-434`) and
    `gym.register("UnitPort-Custom-v0", env_cfg=UnitPortEnvCfg, ...)`.

    Args:
        canvas_dict: ``CanvasPage.to_workflow_dict()`` shape — the IR with
            ``nodes`` (each ``{id, schema_id, params, ...}``) and ``edges``
            (each ``{source_node, source_port, target_node, target_port}``).
        out_dir:    target directory; must exist (typically ``<run_dir>``).
        file_name:  output filename — keep default unless a test needs to
            differentiate.
        robot:      Phase 5 — bound :class:`RobotSpecRef` for IR-role ↔
            physical-joint translation at the substrate-emit boundary
            (init_state.joint_pos and JointPositionActionCfg.joint_names).
            Production callers (IsaacLabTrainingTask) always pass this;
            unit tests with no joint dict in the canvas may omit it.

    Returns:
        Absolute :class:`Path` of the written file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / file_name
    compiler = IsaacLabConfigCompiler(
        canvas_dict if isinstance(canvas_dict, dict) else {},
        robot=robot,
    )
    return compiler.compile_to_file(str(target))


__all__ = [
    "IsaacLabConfigCompiler",
    "compile_env_cfg_to_file",
    "DEPLOY_META_FILENAME",
    "DEPLOY_META_SCHEMA_VERSION",
]
