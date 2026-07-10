# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

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

# UI-only metadata an obs term dict may carry that the compiler accepts but
# IGNORES (no effect on the emitted ObservationTermCfg). ``data_type`` records
# which scale input shape the canvas editor offers for the term (scalar / list /
# both) — purely an authoring affordance; the runtime obs uses ``scale`` only.
_OBS_UI_META_FIELDS = frozenset({"data_type"})


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
            unknown = set(value.keys()) - _PER_TERM_OBS_FIELDS - _OBS_UI_META_FIELDS
            if unknown:
                raise CanvasConfigError(
                    nid=nid, key="obs_terms", schema_id=schema_id,
                    reason=(
                        f"obs_terms[{term_key!r}] has unknown sub-key(s) "
                        f"{sorted(unknown)}; allowed = "
                        f"{sorted(_PER_TERM_OBS_FIELDS | _OBS_UI_META_FIELDS)}."
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

_UNIFORM_GAIT_COMMAND_INLINE = '''
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

    def body_height_cmd(self):
        """(num_envs,) commanded base height in metres."""
        return self._command[:, 5]

    def step_height_cmd(self):
        """(num_envs,) commanded swing foot apex height in metres."""
        return self._command[:, 6]

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


# ---------------------------------------------------------------------------
# Gait command emit dispatch tables (family-keyed via registry.gait_commands).
#
# Three parallel tables keyed by ``GaitCommandSpec.class_name`` describe
# how to emit one gait CommandTerm flavour into the generated config:
#
#   * ``_GAIT_INLINE_BY_CLASS`` — the verbatim Python source block defining
#     the CommandTerm class + its Cfg + per-class obs helpers, appended to
#     the top of the generated config file.
#   * ``_GAIT_OBS_TERMS_BY_CLASS`` — the ordered list of policy-group
#     ObsTerm emit lines that reference the per-class obs helpers.
#   * ``_GAIT_CFG_EMITTERS`` — callable that takes the compiler instance
#     plus the resampling tuple literal and returns the indented Cfg
#     instantiation lines for ``CommandsCfg``.
#
# The lookup key is the registry-supplied ``class_name`` -- the compiler
# never branches on a literal family string. Adding a new flavour
# requires (1) one new entry in each table and (2) one new entry in
# ``gait_commands_catalog.json``; the dispatch path is untouched.
# Missing entries (registry references a class the emit side has not
# implemented yet) raise via :meth:`_resolve_gait_spec` so a half-wired
# flavour cannot silently emit the wrong block.
# ---------------------------------------------------------------------------


def _emit_uniform_gait_command_cfg(compiler, resample: str) -> List[str]:
    """Quadruped CommandsCfg emit for UniformGaitCommandCfg.

    Returns the indented config lines (4-space indent matching the
    @configclass body) the caller appends to its growing ``lines`` list.

    Range defaults sourced from gait_presets.default_ranges_for_family
    (single source of truth shared with the training-side CommandSchema
    gait-channel emitter); per-canvas overrides win as before via
    _gait_range_tuple.
    """
    from application.training.isaac_lab.gait_presets import (
        default_ranges_for_family,
    )
    ranges = default_ranges_for_family("quadruped")
    freq_lit = compiler._gait_range_tuple(
        "gait_frequency_range", *ranges["frequency"]
    )
    bh_lit = compiler._gait_range_tuple(
        "body_height_range", *ranges["body_height"]
    )
    sh_lit = compiler._gait_range_tuple(
        "step_height_range", *ranges["step_height"]
    )
    preset_lit = compiler._gait_preset_phase_literal()
    return [
        "    gait_command = UniformGaitCommandCfg(",
        '        asset_name="robot",',
        f"        freq_range={freq_lit},",
        f"        body_height_range={bh_lit},",
        f"        step_height_range={sh_lit},",
        f'        phase_mode="uniform",',
        f"        preset_phases={preset_lit},",
        f"        resampling_time_range={resample},",
        f"        debug_vis=False,",
        f"    )",
    ]


_GAIT_INLINE_BY_CLASS: Dict[str, str] = {
    "UniformGaitCommand": _UNIFORM_GAIT_COMMAND_INLINE,
}


# Each entry is the verbatim list of ObsTerm emit lines for the policy
# obs group, appended whenever gait is enabled. Stable ordering is
# load-bearing -- the trained policy's input permutation cannot drift.
_GAIT_OBS_TERMS_BY_CLASS: Dict[str, List[str]] = {
    "UniformGaitCommand": [
        '        gait_frequency = ObsTerm('
        'func=_unitport_gait_frequency_obs, '
        'params={"command_name": "gait_command"})',
        '        gait_phase = ObsTerm('
        'func=_unitport_gait_phase_sin_cos_obs, '
        'params={"command_name": "gait_command"})',
        '        body_height_cmd = ObsTerm('
        'func=_unitport_gait_body_height_cmd_obs, '
        'params={"command_name": "gait_command"})',
        '        step_height_cmd = ObsTerm('
        'func=_unitport_gait_step_height_cmd_obs, '
        'params={"command_name": "gait_command"})',
    ],
}


_GAIT_CFG_EMITTERS: Dict[str, Any] = {
    "UniformGaitCommand": _emit_uniform_gait_command_cfg,
}


_BIPED_GAIT_COMMAND_INLINE = '''
# =======================================================================
# UnitPort — Biped Walk These Ways gait command term (UnitPort P2.1 / 7-gamma)
# =======================================================================
# Five per-env dimensions: ``[frequency, phase_L, phase_R, body_height,
# step_height]``. Two phases vs the quadruped's four. Otherwise the
# semantic shape is identical: per-foot phase clock, sampled freq /
# heights, optional preset_phases snap on resample.
import torch as _biped_gait_torch
import math as _biped_gait_math

from isaaclab.managers import CommandTerm as _UnitportBipedCommandTerm
from isaaclab.managers import CommandTermCfg as _UnitportBipedCommandTermCfg


class BipedGaitCommand(_UnitportBipedCommandTerm):
    """Parameterised biped gait command — Walk These Ways §3 (biped).

    Five per-env dimensions: ``[frequency, phase_L, phase_R, body_height,
    step_height]``. Each foot has its own clock-offset phase in [0, 1);
    in stance when local phase < 0.5, swing otherwise.

    Defense-in-depth assert at construction time that the bound robot
    family is in the biped/humanoid set -- the dispatch table guarantees
    this already (only biped/humanoid resolve BipedGaitCommand via the
    registry), but the assert documents the contract at the leaf.
    """

    cfg: "BipedGaitCommandCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        n = env.num_envs
        device = env.device
        self._command = _biped_gait_torch.zeros((n, 5), device=device)
        self._phase_clock = _biped_gait_torch.zeros(n, device=device)
        if cfg.preset_phases:
            self._preset_phases = _biped_gait_torch.tensor(
                cfg.preset_phases, dtype=_biped_gait_torch.float32, device=device
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
        """(num_envs, 2) local phase = (phase_clock + phase_offset) mod 1.

        Same semantic as UniformGaitCommand.per_foot_phase but width 2
        (L, R). Reward consumers that read per_foot_phase() get the
        family-correct width without family-keyed branching."""
        return (self._phase_clock.unsqueeze(-1) + self._command[:, 1:3]) % 1.0

    def body_height_cmd(self):
        """(num_envs,) commanded base height in metres."""
        return self._command[:, 3]

    def step_height_cmd(self):
        """(num_envs,) commanded swing foot apex height in metres."""
        return self._command[:, 4]

    def _resample_command(self, env_ids):
        cfg = self.cfg
        device = self._command.device
        n = env_ids.numel() if hasattr(env_ids, "numel") else len(env_ids)

        def _u(lo, hi):
            return _biped_gait_torch.rand(n, device=device) * (hi - lo) + lo

        freq = _u(cfg.freq_range[0], cfg.freq_range[1])
        body_h = _u(cfg.body_height_range[0], cfg.body_height_range[1])
        step_h = _u(cfg.step_height_range[0], cfg.step_height_range[1])

        if self._preset_phases is not None and cfg.phase_mode == "preset":
            k = self._preset_phases.shape[0]
            idx = _biped_gait_torch.randint(0, k, (n,), device=device)
            phase = self._preset_phases[idx]
        else:
            phase = _biped_gait_torch.rand((n, 2), device=device)

        self._command[env_ids, 0] = freq
        self._command[env_ids, 1:3] = phase
        self._command[env_ids, 3] = body_h
        self._command[env_ids, 4] = step_h
        self._phase_clock[env_ids] = 0.0

    def _update_command(self):
        dt = self._env.step_dt
        freq = self._command[:, 0]
        self._phase_clock = (self._phase_clock + dt * freq) % 1.0

    def _update_metrics(self):
        pass


@configclass
class BipedGaitCommandCfg(_UnitportBipedCommandTermCfg):
    class_type: type = BipedGaitCommand

    asset_name: str = "robot"
    freq_range: tuple = (1.0, 2.0)
    body_height_range: tuple = (0.65, 0.75)
    step_height_range: tuple = (0.05, 0.10)
    phase_mode: str = "uniform"          # "uniform" | "preset"
    preset_phases: list = None           # filled by the compiler


# -- biped gait obs helpers -- read from the command manager --

def _unitport_biped_gait_frequency_obs(env, command_name="gait_command"):
    return env.command_manager.get_term(command_name).command[:, 0:1]


def _unitport_biped_gait_phase_sin_cos_obs(env, command_name="gait_command"):
    """Sin/cos encoding of (phase_clock + phase_offset) per foot -- 4 dims.

    Two phases (L, R) x sin/cos = 4 dims. Sin/cos avoids the 1.0 -> 0.0
    wrap-around discontinuity for the regression head."""
    term = env.command_manager.get_term(command_name)
    per_foot = term.per_foot_phase()
    angles = per_foot * (2.0 * _biped_gait_math.pi)
    return _biped_gait_torch.cat(
        [_biped_gait_torch.sin(angles), _biped_gait_torch.cos(angles)], dim=-1
    )


def _unitport_biped_gait_body_height_cmd_obs(env, command_name="gait_command"):
    return env.command_manager.get_term(command_name).command[:, 3:4]


def _unitport_biped_gait_step_height_cmd_obs(env, command_name="gait_command"):
    return env.command_manager.get_term(command_name).command[:, 4:5]
# =======================================================================
'''


def _emit_biped_gait_command_cfg(compiler, resample: str) -> List[str]:
    """Biped CommandsCfg emit for BipedGaitCommandCfg.

    Same line shape as the quadruped emitter (10 indented lines, 4-space
    indent matching the @configclass body) but uses biped class names and
    biped-class fallback range defaults (G1/H1-class body height ~0.70 m,
    step ~0.07 m, frequency ~1.5 Hz).

    Range defaults sourced from gait_presets.default_ranges_for_family
    (single source of truth shared with the training-side CommandSchema
    gait-channel emitter); per-canvas overrides win as before via
    _gait_range_tuple."""
    from application.training.isaac_lab.gait_presets import (
        default_ranges_for_family,
    )
    ranges = default_ranges_for_family("biped")
    freq_lit = compiler._gait_range_tuple(
        "gait_frequency_range", *ranges["frequency"]
    )
    bh_lit = compiler._gait_range_tuple(
        "body_height_range", *ranges["body_height"]
    )
    sh_lit = compiler._gait_range_tuple(
        "step_height_range", *ranges["step_height"]
    )
    preset_lit = compiler._gait_preset_phase_literal()
    return [
        "    gait_command = BipedGaitCommandCfg(",
        '        asset_name="robot",',
        f"        freq_range={freq_lit},",
        f"        body_height_range={bh_lit},",
        f"        step_height_range={sh_lit},",
        f'        phase_mode="uniform",',
        f"        preset_phases={preset_lit},",
        f"        resampling_time_range={resample},",
        f"        debug_vis=False,",
        f"    )",
    ]


# Register BipedGaitCommand in the dispatch tables. Adding biped here
# (a) un-traps the 7-beta R5 raise that fired when the registry resolved
# class_name="BipedGaitCommand" without an emit entry, and (b) does NOT
# disturb the quadruped emit path: the lookup is dict-by-class-name, so
# the UniformGaitCommand path picks the same object it picked before.
_GAIT_INLINE_BY_CLASS["BipedGaitCommand"] = _BIPED_GAIT_COMMAND_INLINE
_GAIT_OBS_TERMS_BY_CLASS["BipedGaitCommand"] = [
    '        gait_frequency = ObsTerm('
    'func=_unitport_biped_gait_frequency_obs, '
    'params={"command_name": "gait_command"})',
    '        gait_phase = ObsTerm('
    'func=_unitport_biped_gait_phase_sin_cos_obs, '
    'params={"command_name": "gait_command"})',
    '        body_height_cmd = ObsTerm('
    'func=_unitport_biped_gait_body_height_cmd_obs, '
    'params={"command_name": "gait_command"})',
    '        step_height_cmd = ObsTerm('
    'func=_unitport_biped_gait_step_height_cmd_obs, '
    'params={"command_name": "gait_command"})',
]
_GAIT_CFG_EMITTERS["BipedGaitCommand"] = _emit_biped_gait_command_cfg


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

        # ── Adaptive item sampling state (Phase C, §2A-2C). When
        # cfg.adaptive_enabled, the term accumulates each step's per-env
        # reward into per-item buckets (keyed by _current_item_id) and every
        # adaptive_update_interval_steps re-biases self._weights toward
        # under-performing (low mean-reward) items, bounded by
        # [weight_floor, weight_ceil] via set_weights. Lives env-side (here)
        # rather than runner-side because the vanilla rsl_rl OnPolicyRunner
        # used by plain-PPO exposes no per-iteration hook — this way plain
        # PPO and AMP get identical behaviour with no runner changes.
        self._num_items = m
        self._adaptive_enabled = bool(getattr(cfg, "adaptive_enabled", False))
        self._adaptive_interval = int(
            getattr(cfg, "adaptive_update_interval_steps", 0) or 0
        )
        self._adaptive_smoothing = float(getattr(cfg, "adaptive_smoothing", 0.5))
        self._item_reward_sum = _wv_torch.zeros(m, dtype=_wv_torch.float32, device=device)
        self._item_step_count = _wv_torch.zeros(m, dtype=_wv_torch.float32, device=device)
        self._adaptive_step_counter = 0

        # ── Per-command reward telemetry (Mission Control chart). ALWAYS ON,
        # independent of adaptive sampling: every step we scatter the per-env
        # reward into per-item buckets keyed by _current_item_id, and every
        # _metric_emit_interval steps print a machine-parseable
        # [UnitPort][CMDMETRICS] line (backend.py turns each item into a
        # cmd_reward_<id> chart series) then reset the buckets. Kept on its own
        # accumulators + cadence so toggling adaptive sampling never perturbs
        # the chart. Emits env-side — like adaptive sampling — because the
        # vanilla rsl_rl OnPolicyRunner used by plain PPO exposes no
        # per-iteration hook; this way plain PPO and AMP behave identically.
        self._metric_emit_interval = int(
            getattr(cfg, "metric_emit_interval_steps", 0) or 0
        )
        self._metric_reward_sum = _wv_torch.zeros(m, dtype=_wv_torch.float32, device=device)
        self._metric_step_count = _wv_torch.zeros(m, dtype=_wv_torch.float32, device=device)
        self._metric_emit_counter = 0

        # Heading-command mode (legged_gym parity). When on, the yaw
        # channel is closed-loop: a target world-heading is sampled per
        # resample and ang_vel_z is recomputed each step from the heading
        # error (see _update_command). The per-env clip bounds are the
        # selected item's ang_vel_z range (so the recomputed rate stays
        # inside the command envelope the policy was trained on).
        self._heading_command = bool(getattr(cfg, "heading_command", False))
        self._heading_stiffness = float(getattr(cfg, "heading_control_stiffness", 0.5))
        self._heading_lo = float(getattr(cfg, "heading_lo", -3.141592653589793))
        self._heading_hi = float(getattr(cfg, "heading_hi", 3.141592653589793))
        self._heading_target = _wv_torch.zeros(n, device=device)
        self._angz_lo = _wv_torch.zeros(n, device=device)
        self._angz_hi = _wv_torch.zeros(n, device=device)

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
        # Standing (zero-command) envs KEEP their sampled item identity. A
        # standing "walk" env is still running the walk item, just handed a
        # zero velocity command for this window, and MUST still receive that
        # item's rewards — legged_gym rewards standing with the tracking term
        # (error 0 -> exp(0) = 1). The standing state is tracked separately in
        # self._is_standing, which the reward masks read for phase routing
        # (see unitport_item_mask / unitport_phase_mask). The previous
        # ``where(standing, -1, item_ids)`` folded "standing" into the item id,
        # which made unitport_item_mask (item_id == idx) evaluate False and
        # ZERO every reward on ~rel_standing_envs of the fleet — a CLAUDE.md §8
        # illusion (the canvas configured a tracking reward that silently never
        # ran on the standing envs) and a cross-engine divergence (the SB3 env
        # always rewards its standing envs).
        self._current_item_id[env_ids] = item_ids

        if self._heading_command:
            # Sample a fresh target heading and latch the picked item's
            # ang_vel_z range as this env's yaw-rate clip bounds. The
            # open-loop vals[:, 2] just sampled above is overwritten every
            # step by the heading controller in _update_command.
            self._heading_target[env_ids] = self._heading_lo + _wv_torch.rand(
                n, device=device
            ) * (self._heading_hi - self._heading_lo)
            self._angz_lo[env_ids] = self._ranges_lo[item_ids, 2]
            self._angz_hi[env_ids] = self._ranges_hi[item_ids, 2]

    def _recompute_heading_yaw(self):
        # Closed-loop yaw: ang_vel_z = clip(stiffness * wrap_to_pi(target -
        # yaw_world), item ang_vel_z range). Mirrors legged_gym's
        # _post_physics_step_callback heading branch. Reads the base world
        # yaw from the articulation (IsaacLab ArticulationData.heading_w),
        # the same source IsaacLab's own UniformVelocityCommand uses.
        asset = self._env.scene[self.cfg.asset_name]
        yaw = asset.data.heading_w
        err = self._heading_target - yaw
        # wrap_to_pi — legged_gym convention (π maps to +π, not −π).
        two_pi = 2.0 * _wv_torch.pi
        err = err % two_pi
        err = err - two_pi * (err > _wv_torch.pi).to(err.dtype)
        raw = self._heading_stiffness * err
        angz = _wv_torch.maximum(
            _wv_torch.minimum(raw, self._angz_hi), self._angz_lo
        )
        self._command[:, 2] = _wv_torch.where(
            self._is_standing, _wv_torch.zeros_like(angz), angz
        )

    def _adaptive_step(self):
        # Phase C adaptive sampling. Every step: scatter the current per-env
        # reward into per-item buckets keyed by the env's active item. Every
        # _adaptive_interval steps: turn the per-item mean reward into new
        # sampling weights (under-performers up-weighted), EMA-blend with the
        # current weights, hand to set_weights (which clamps to
        # [floor, ceil] and renormalises), then reset the accumulators.
        if self._num_items < 2 or self._adaptive_interval <= 0:
            return
        # command_manager.compute runs AFTER reward_manager.compute in the env
        # step, so reward_buf holds THIS step's reward when we're called.
        reward = getattr(self._env, "reward_buf", None)
        if reward is None:
            return
        ids = self._current_item_id
        valid = ids >= 0  # envs not yet resampled carry the -1 sentinel
        if bool(valid.any()):
            v_ids = ids[valid]
            v_r = reward[valid].to(self._item_reward_sum.dtype)
            self._item_reward_sum.index_add_(0, v_ids, v_r)
            self._item_step_count.index_add_(0, v_ids, _wv_torch.ones_like(v_r))

        self._adaptive_step_counter += 1
        if self._adaptive_step_counter < self._adaptive_interval:
            return
        self._adaptive_step_counter = 0

        seen = self._item_step_count > 0
        if int(seen.sum().item()) < 2:
            # <2 items sampled this window — can't compare; wait for coverage.
            self._item_reward_sum.zero_()
            self._item_step_count.zero_()
            return
        mean_r = self._item_reward_sum / _wv_torch.clamp(self._item_step_count, min=1.0)
        r_seen = mean_r[seen]
        r_min = r_seen.min()
        spread = r_seen.max() - r_min
        scale = _wv_torch.clamp(r_seen.abs().mean(), min=1e-6)
        # Deadband: ignore noise-level reward spread, keep current weights.
        if float(spread.item()) <= 1e-3 * float(scale.item()):
            self._item_reward_sum.zero_()
            self._item_step_count.zero_()
            return
        # Priority ∝ (1 − normalised_reward) + 0.1: worst seen item → ~1.1,
        # best → ~0.1. Redistribute the seen items' current weight mass by
        # priority; unseen items keep their current weight. set_weights does
        # the floor/ceil clamp + renormalise.
        norm = (mean_r - r_min) / spread
        priority = (1.0 - norm) + 0.1
        target = self._weights.clone()
        seen_mass = self._weights[seen].sum()
        pr = priority[seen]
        pr = pr / pr.sum() * seen_mass
        target[seen] = pr
        a = self._adaptive_smoothing
        new_w = (1.0 - a) * self._weights + a * target
        self.set_weights(new_w)
        self._item_reward_sum.zero_()
        self._item_step_count.zero_()

    def _metric_step(self):
        # Per-command reward telemetry. Mirrors _adaptive_step's bucketing but
        # with independent accumulators and an independent (rollout-length)
        # cadence, so enabling/disabling adaptive sampling never moves the
        # chart series. command_manager.compute runs AFTER reward_manager.compute
        # in the env step, so reward_buf holds THIS step's reward here.
        if self._metric_emit_interval <= 0:
            return
        reward = getattr(self._env, "reward_buf", None)
        if reward is not None:
            ids = self._current_item_id
            valid = ids >= 0  # envs not yet resampled carry the -1 sentinel
            if bool(valid.any()):
                v_ids = ids[valid]
                v_r = reward[valid].to(self._metric_reward_sum.dtype)
                self._metric_reward_sum.index_add_(0, v_ids, v_r)
                self._metric_step_count.index_add_(0, v_ids, _wv_torch.ones_like(v_r))
        self._metric_emit_counter += 1
        if self._metric_emit_counter < self._metric_emit_interval:
            return
        self._metric_emit_counter = 0
        self._emit_command_metrics()

    def _emit_command_metrics(self):
        # Print mean per-item reward over the window as a single machine-
        # parseable line. Items NOT sampled this window are omitted — never a
        # fake zero (CLAUDE.md §8). backend.py's parse_cmdmetrics_line turns
        # each "<item_id>=<mean>" token into a cmd_reward_<item_id> series.
        counts = self._metric_step_count
        means = self._metric_reward_sum / _wv_torch.clamp(counts, min=1.0)
        parts = []
        for i, item_id in enumerate(self._item_ids_list):
            if float(counts[i].item()) <= 0.0:
                continue
            safe = "".join(
                c if (c.isalnum() or c == "_") else "_" for c in str(item_id)
            )
            parts.append(safe + "=" + format(float(means[i].item()), ".5f"))
        self._metric_reward_sum.zero_()
        self._metric_step_count.zero_()
        if parts:
            print("[UnitPort][CMDMETRICS] " + " ".join(parts), flush=True)

    def _update_command(self):
        # Per-command reward telemetry first — runs every step regardless of
        # whether adaptive sampling is enabled.
        self._metric_step()
        if self._adaptive_enabled:
            self._adaptive_step()
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
        # Heading controller runs every step (after any step-change resample
        # above), so the yaw command tracks the target heading continuously.
        if self._heading_command:
            self._recompute_heading_yaw()

    def _update_metrics(self):
        pass


@configclass
class UnitportWeightedVelocityCommandCfg(_UnitportWVCommandTermCfg):
    class_type: type = UnitportWeightedVelocityCommand

    asset_name: str = "robot"
    # Articulation root body name — recorded in env.yaml so the deploy
    # manifest_parser can resolve the base link without a silent fallback
    # (CLAUDE.md §1.8). Compiler emits this from
    # RobotSpec.bodies_role_map_for(active_format) by finding the body
    # whose ir_role == "base".
    body_name: str = ""
    # items / initial_weights are emitted by the compiler — None defaults
    # exist only so a stand-alone Python import of this generated file
    # doesn't blow up on dataclass evaluation order.
    items: list = None
    initial_weights: list = None
    rel_standing_envs: float = 0.05
    weight_floor: float = 0.03
    weight_ceil: float = 0.30
    adaptive_enabled: bool = False
    # Phase C adaptive sampling: re-weight cadence in ENV STEPS (the compiler
    # converts the canvas's iteration-count via num_steps_per_env) and the
    # EMA blend factor applied to each weight update (0 = frozen, 1 = no
    # smoothing). 0 interval keeps the term in pure-uniform mode.
    adaptive_update_interval_steps: int = 0
    adaptive_smoothing: float = 0.5
    # Per-step command resample probability — canvas training_motion.
    # 0 keeps the legacy "only resample at resampling_time_range" behavior.
    cmd_step_change_prob: float = 0.0
    # Heading-command mode (legged_gym parity). False = open-loop yaw
    # sampling (legacy). heading_lo/hi are the world-heading target sample
    # range (rad); stiffness maps heading error → yaw rate.
    heading_command: bool = False
    heading_control_stiffness: float = 0.5
    heading_lo: float = -3.141592653589793
    heading_hi: float = 3.141592653589793
    # Per-command reward telemetry emit cadence in ENV STEPS. The compiler
    # sets it to the trainer's num_steps_per_env so one [UnitPort][CMDMETRICS]
    # emit ~ one PPO/AMP iteration; 0 disables the telemetry line entirely.
    metric_emit_interval_steps: int = 0
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
        # Standing (zero-command) envs route to the "static" phase regardless
        # of which item they belong to. They now KEEP their item id in
        # _current_item_id (so unitport_item_mask still rewards them), so the
        # standing signal is read from _is_standing instead of the old
        # ``item_id == -1`` sentinel. A None flag (foreign command term)
        # degrades to "nothing is standing" (matches the pre-decoupling path
        # when no env was standing).
        is_standing = getattr(cmd, "_is_standing", None)
        if is_standing is not None:
            moving = (~is_standing).to(base.dtype)
            standing_f = is_standing.to(base.dtype)
        else:
            moving = _torch.ones(item_id.shape[0], dtype=base.dtype, device=device)
            standing_f = _torch.zeros(item_id.shape[0], dtype=base.dtype, device=device)
        mask = _torch.zeros(item_id.shape[0], dtype=base.dtype, device=device)
        for idx, phase_id in enumerate(item_phases):
            if phase_id and phase_id in phases_set:
                mask = mask + (item_id == idx).to(base.dtype) * moving
        if "static" in phases_set:
            mask = mask + standing_f
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
        # Standing (zero-command) envs retain their sampled item id (the
        # command term no longer overwrites it with a -1 sentinel), so they
        # match here exactly like a moving env of the same item and receive
        # the item's rewards — that is what rewards a standing env for holding
        # the commanded zero velocity (legged_gym parity, CLAUDE.md §8).
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


# skill_command_path_design.md Slice 3 — one trigger command term per skill.
# A single decaying scalar in [0, 1]: on a randomised resample it fires to 1.0 with
# probability trigger_prob, then decays linearly to 0 over window_s seconds. The SAME
# scalar serves the policy obs (the skill-window phase) AND the reward gate (post-pulse
# window with decay). Randomised resampling_time_range IS the curriculum — the trigger
# fires at varied times so the policy learns the conditional (trigger -> skill) mapping.
_SKILL_TRIGGER_COMMAND_INLINE = '''
# =======================================================================
# UnitPort — skill trigger command term (skill_command_path_design.md Slice 3)
# =======================================================================
import torch as _skill_trig_torch

from isaaclab.managers import CommandTerm as _UnitportSkillCommandTerm
from isaaclab.managers import CommandTermCfg as _UnitportSkillCommandTermCfg


class SkillTriggerCommand(_UnitportSkillCommandTerm):
    """One trigger channel: a decaying pulse in [0, 1] (obs + reward-gate envelope)."""

    cfg: "SkillTriggerCommandCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._command = _skill_trig_torch.zeros((env.num_envs, 1), device=env.device)

    @property
    def command(self):
        return self._command

    def _resample_command(self, env_ids):
        cfg = self.cfg
        device = self._command.device
        n = env_ids.numel() if hasattr(env_ids, "numel") else len(env_ids)
        fire = (_skill_trig_torch.rand(n, device=device) < float(cfg.trigger_prob)).float()
        # 1.0 fires the skill window; 0.0 = a no-skill window (learn to stay grounded).
        self._command[env_ids, 0] = fire

    def _update_command(self):
        # Linear decay 1 -> 0 over window_s seconds (the post-pulse window + decay).
        dt = self._env.step_dt
        w = max(float(self.cfg.window_s), 1e-6)
        self._command[:, 0] = _skill_trig_torch.clamp(
            self._command[:, 0] - dt / w, min=0.0
        )

    def _update_metrics(self):
        pass


def _unitport_skill_trigger_obs(env, command_name="skill_trigger"):
    return env.command_manager.get_term(command_name).command[:, 0:1]


@configclass
class SkillTriggerCommandCfg(_UnitportSkillCommandTermCfg):
    class_type: type = SkillTriggerCommand

    asset_name: str = "robot"
    window_s: float = 1.0        # post-pulse window length (obs + reward-gate decay), s
    trigger_prob: float = 0.5    # P(fire) at each randomised resample
# =======================================================================
'''


# skill_command_path_design.md Slice 3 — reward gate keyed on a skill trigger.
# Multiply a reward by the trigger's decaying envelope ``command[:, 0]`` in [0, 1]
# (the post-pulse window with decay) so a skill package's reward is on only within the
# window. Signature-preserved (Isaac Lab's RewardManager inspects the func); PASSTHROUGH
# (return base, NOT zero) if the command term is missing — never silently kill the
# reward (CLAUDE.md §8). Composes inside unitport_item_mask at the emit site.
_TRIGGER_GATE_HELPER_INLINE = '''
# =======================================================================
# UnitPort — skill trigger reward gate (skill_command_path_design.md Slice 3)
# =======================================================================
def unitport_trigger_gate(base_fn, command_name):
    import functools as _tg_functools
    import inspect as _tg_inspect

    def _core(env, *args, **kwargs):
        base = base_fn(env, *args, **kwargs)
        try:
            envelope = env.command_manager.get_term(command_name).command[:, 0]
        except Exception:
            return base
        return base * envelope

    try:
        sig = _tg_inspect.signature(base_fn)
        if not list(sig.parameters.values()):
            raise ValueError("base reward has no parameters (need at least env)")

        def wrapped(*call_args, **call_kwargs):
            bound = sig.bind(*call_args, **call_kwargs)
            bound.apply_defaults()
            return _core(*bound.args, **bound.kwargs)

        wrapped.__signature__ = sig
        wrapped = _tg_functools.wraps(base_fn)(wrapped)
        return wrapped
    except (TypeError, ValueError):
        @_tg_functools.wraps(base_fn)
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
        # Stashed by ``_compile_pd_payload_for_emit`` when an ActuatorPDNode
        # is wired; surfaced into deploy_meta.json so the bundle finalizer
        # can stamp pd_param + derived PhysX gains into deploy_contract
        # without re-deriving from env.yaml. Shape:
        #     {"pd_param": {...}, "physx_gains": {...}}  (see Stage D)
        self._stashed_pd_meta: Optional[Dict[str, Any]] = None
        # Stashed by ``_terminations_cfg`` → emitted as ``unitport_terminations``
        # in the deploy_meta.json sidecar (per-condition grace_period_s audit).
        self._stashed_termination_meta: Optional[Dict[str, Any]] = None
        # Training-package (Method A) reward lowering: synthesise a virtual
        # ``rewards`` node + ``reward_in__`` edges per inline-reward package into
        # the graph BEFORE _parse so _rewards_cfg reads packages via the existing
        # fanout path (package_weight folded once, at synthesis). Returns the
        # graph unchanged when no package carries inline rewards (byte-identical).
        from application.training.package_synthesis import synthesize_into_graph
        self._graph = synthesize_into_graph(self._graph)
        self._parse()

    @property
    def _active_format(self) -> str:
        """The asset format this compile run targets (MJCF / USD / URDF).

        Comes from ``RobotSpecRef.active_format`` (set by
        ``spec_compiler._populate_robot`` from the canvas Robot node's
        ``active_override`` + canvas backend preference). Empty string
        when no spec is bound — emit sites fall back to the registry's
        preferred_format via BodyIRMapper's auto-pick.
        """
        if self._robot is None:
            return ""
        return str(getattr(self._robot, "active_format", "") or "")

    @property
    def _joint_ir_resolver(self):
        """Lazy :class:`JointIRResolver` for the bound robot + active format.

        Returns None when ``self._robot`` is None — emit sites detect this
        and either raise (if user-set joint dict needs translation) or fall
        through to a regex-passthrough path.
        """
        if self._robot is None:
            return None
        cache = getattr(self, "_joint_ir_resolver_cache", None)
        if cache is None:
            from application.training.joint_ir import JointIRResolver
            cache = JointIRResolver(self._robot, active_format=self._active_format or None)
            self._joint_ir_resolver_cache = cache
        return cache

    def _resolve_base_body_name(self) -> str:
        """Look up the articulation root link name for the bound robot.

        Reads :class:`RobotSpec`'s ``bodies_role_map_for(active_format)``
        and returns the body whose ``ir_role`` matches an articulation-root
        alias. Tries ``base`` first (the conventional quadruped/wheeled
        slot), then falls back through the biped/humanoid root aliases
        ``pelvis`` / ``torso`` / ``trunk`` / ``body`` — same set the
        compiler already treats as canonical articulation roots in
        ``_canonical_roles`` (§3193). The auto-suggester
        (``body_ir._suggest_role_id``) deliberately routes biped pelvis
        links to ``ir_role='pelvis'`` rather than collapsing them to
        ``'base'``, so this method must understand the per-family
        articulation-root vocabulary instead of demanding the literal
        ``'base'`` string. Raises when no candidate matches.
        """
        if self._robot is None:
            raise RuntimeError(
                "[env_cfg_compiler] cannot resolve base body name — no "
                "RobotSpec bound. Pass robot=spec.robot to "
                "compile_env_cfg_to_file (CLAUDE.md §1.8)."
            )
        fmt = self._active_format or "MJCF"
        bodies = self._robot.bodies_role_map_for(fmt)
        # Quadruped / wheeled: literal 'base'. Biped/humanoid: 'pelvis'
        # is the natural articulation root (G1, H1). Other community
        # robots use 'torso' (trunk-rooted humanoids) or 'body' (Spot).
        root_aliases = ("base", "pelvis", "torso", "trunk", "body")
        by_role: Dict[str, str] = {}
        for body_name, ir_role in bodies.items():
            by_role.setdefault(str(ir_role), str(body_name))
        for alias in root_aliases:
            if alias in by_role:
                return by_role[alias]
        raise RuntimeError(
            f"[env_cfg_compiler] robot {self._robot.sku!r} has no body "
            f"mapped to any articulation-root ir_role "
            f"({', '.join(root_aliases)}) in format {fmt!r}. Fix the "
            f"robot entry in registers/data/robots_canonical.json — "
            f"exactly one body must declare an articulation-root ir_role."
        )

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

    def _resolve_gait_spec(self):
        """Resolve the family-keyed gait command spec for the bound robot.

        Returns ``None`` when gait is not enabled on this canvas (the legit
        "no gait emit" path -- callers short-circuit on None).

        When gait IS enabled, every downstream prerequisite MUST resolve.
        Raises with a three-part directive (the offending field, the source
        of the bad config, the fix-up step) on any of:

          * ``self._robot is None`` -- gait_enabled=true but no RobotSpecRef
            bound (RobotNode missing or unwired at compile time);
          * ``robot.families`` empty -- the bound robot has no declared
            family, so the family-keyed dispatch table has no key to look
            up;
          * the primary family has no entry in
            ``gait_commands_catalog.json`` -- the family is declared but
            not gait-trainable (non-legged families intentionally have no
            entry, so a canvas requesting gait on them is a user error,
            not a silent ``False`` candidate);
          * the registry-supplied ``class_name`` has no corresponding
            entry in ``_GAIT_INLINE_BY_CLASS`` -- the registry references
            a CommandTerm flavour the emit side has not implemented yet
            (e.g. before the BipedGaitCommand emit block lands). This
            raise prevents a silently-wrong block from shipping during
            the registry-then-emit segmented rollout.
        """
        if not self._gait_enabled():
            return None
        if self._robot is None:
            raise RuntimeError(
                "[env_cfg_compiler] gait_enabled=true on Training "
                "Commands but no RobotSpecRef is bound; wire a Robot "
                "node and bind it before training, or set gait_enabled="
                "false on the Training Commands node"
            )
        families = list(getattr(self._robot, "families", []) or [])
        if not families:
            raise RuntimeError(
                f"[env_cfg_compiler] gait_enabled=true but robot "
                f"sku={getattr(self._robot, 'sku', '?')!r} has no "
                f"declared families; declare at least one family on the "
                f"Robot Asset card so gait_commands dispatch has a key, "
                f"or set gait_enabled=false"
            )
        family = families[0]
        from registers.gait_commands import (
            get_gait_command,
            has_gait_command,
        )
        if not has_gait_command(family):
            raise RuntimeError(
                f"[env_cfg_compiler] gait_enabled=true but family="
                f"{family!r} has no entry in gait_commands_catalog.json "
                f"(non-legged families are intentionally omitted); "
                f"either set gait_enabled=false on Training Commands or "
                f"add a gait command entry for {family!r} via a user "
                f"overlay at <USER_CONFIG_DIR>/registers/"
                f"gait_commands_custom.json"
            )
        spec = get_gait_command(family)
        if spec.class_name not in _GAIT_INLINE_BY_CLASS:
            raise RuntimeError(
                f"[env_cfg_compiler] gait_commands registry resolves "
                f"family={family!r} to class_name={spec.class_name!r} "
                f"but the env_cfg_compiler emit side has no INLINE "
                f"block / obs-term table / Cfg emitter for that class "
                f"(known emit classes: "
                f"{sorted(_GAIT_INLINE_BY_CLASS.keys())!r}); add the "
                f"three matching entries in _GAIT_INLINE_BY_CLASS / "
                f"_GAIT_OBS_TERMS_BY_CLASS / _GAIT_CFG_EMITTERS, or "
                f"correct the registry entry"
            )
        return spec

    def _gait_range_tuple(self, key: str, default_lo: float, default_hi: float) -> str:
        """Parse a ``[lo, hi]`` JSON param into a Python tuple literal."""
        cmd_ids = self._find_by_type("training_motion")
        if not cmd_ids:
            return f"({default_lo}, {default_hi})"
        params = self._params.get(cmd_ids[0]) or {}
        raw = params.get(key)
        if raw is None or str(raw) == "":
            return f"({default_lo}, {default_hi})"
        try:
            import json as _json
            v = _json.loads(str(raw))
            if isinstance(v, (list, tuple)) and len(v) == 2:
                return f"({float(v[0])}, {float(v[1])})"
        except Exception:
            pass
        return f"({default_lo}, {default_hi})"

    def _gait_preset_phase_literal(self) -> str:
        """Extract the phase vectors from the gait_presets JSON param and
        return a Python list-of-lists literal the generated config can
        hand straight to the resolved gait command class's preset_phases.

        Phase-tuple width is family-keyed via the resolved GaitCommandSpec
        (quadruped phase_count=4, biped/humanoid phase_count=2). Presets
        whose phase tuple width does not match the resolved family count
        are silently dropped (the canvas widget may carry leftover
        cross-family presets from a previous robot swap; harvest only the
        family-matching ones).

        Falls back to the bundled family-matching default set when no
        valid presets are provided. Returns ``"[]"`` only if no defaults
        exist for the family (shouldn't happen in practice -- DEFAULT
        and biped presets ship in gait_presets.py).
        """
        spec = self._resolve_gait_spec()
        # _gait_preset_phase_literal should only be called from the gait
        # Cfg emitters, which themselves short-circuit on spec is None.
        # Defense in depth: if somehow called when gait is disabled, fall
        # back to the historical quadruped 4-phase shape (preserves R6
        # byte-identical for the quadruped emit path when external
        # callers re-use this helper).
        phase_count = spec.phase_count if spec is not None else 4

        cmd_ids = self._find_by_type("training_motion")
        raw = ""
        if cmd_ids:
            params = self._params.get(cmd_ids[0]) or {}
            raw = str(params.get("gait_presets") or "")
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
                            and len(phase) == phase_count
                        ):
                            presets.append([float(x) for x in phase])
            except Exception:
                pass
        if not presets:
            try:
                from application.training.isaac_lab.gait_presets import (
                    default_presets_for_phase_count,
                )
                for pr in default_presets_for_phase_count(phase_count):
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
                    f"<USER_CONFIG_DIR>/registers/commands_custom.json."
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

    def _resolve_initial_item_weights(
        self, cid: str, items: List[dict]
    ) -> List[float]:
        """Build the ``initial_weights`` list aligned to ``items`` order.

        Reads the optional per-item ``weight`` from the canvas
        ``training_items`` dict (the Weight Set pie editor's data of record),
        keyed by item id. A missing weight defaults to the uniform share
        ``1/N`` so an untouched canvas (no ``weight`` keys) stays uniform.
        The result is normalised to sum 1; if every weight is non-positive
        the function falls back to a uniform vector.

        Negative / NaN weights are not silently coerced — they are a malformed
        canvas input that ``spec_validator._check_item_weights`` rejects loudly
        (CLAUDE.md §8) before compilation reaches here; the ``float()`` below
        would also raise on a non-numeric value rather than paper over it.
        """
        n = len(items)
        if n == 0:
            return []
        uniform = 1.0 / n
        raw_ti = self._parse_json_param(cid, "training_items") or {}
        if not isinstance(raw_ti, dict):
            raw_ti = {}
        weights: List[float] = []
        for it in items:
            raw = raw_ti.get(str(it.get("id", "")), {})
            wv = raw.get("weight") if isinstance(raw, dict) else None
            weights.append(uniform if wv is None else float(wv))
        total = sum(weights)
        if total <= 0.0:
            return [uniform] * n
        return [w / total for w in weights]

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
        ``.lower()`` etc. don't blow up on a numeric. JSON-typed params
        (dict / list values that canvas widgets wrote as native Python
        objects, e.g. ``joint_pose_table`` widget writes a dict) serialise
        through ``json.dumps`` so downstream ``json.loads`` callers get
        valid JSON — never Python repr with single quotes.

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
        if isinstance(v, (dict, list, tuple)):
            # JSON-typed canvas param (joint_pose_table writes dict,
            # range_list writes list of [lo, hi] pairs, etc.). The previous
            # ``str(v)`` path produced Python repr ('single quotes') which
            # broke every downstream ``json.loads`` — surfaced as
            # JSONDecodeError on Spot's init_joint_angles dict written by
            # the new JointPoseEditorDialog. Fixed 2026-05-19.
            import json as _json
            return _json.dumps(v)
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

    def _resolve_play_ground_dt(self) -> tuple[float, float, int]:
        """Resolve (sim_dt, control_dt, decimation) from canvas play_ground_setting.

        Single source of truth for the IL timing triple — replaces two
        callsites that previously each had:

            sim_dt = 0.005
            if pg_ids:
                sim_dt = self._pf(pg_ids[0], "sim_dt") or 0.005
            control_dt = 0.02
            decimation = max(1, int(round(control_dt/sim_dt))) if sim_dt>0 else 4

        That pattern silently substituted 5 ms + decimation 4 whenever the
        canvas had no play_ground_setting node OR the user set sim_dt to
        exactly 0 (CLAUDE.md §1.8 — a wrong sim_dt in env.yaml means the
        exported bundle declares a control frequency the trained policy
        never saw, producing twitching at deploy time).

        Raises ``CanvasConfigError`` when:
          * No play_ground_setting node on the canvas.
          * Its sim_dt parameter is missing / non-positive.
          * control_dt / sim_dt doesn't yield a positive integer
            decimation (control_dt is the hardcoded IL 50 Hz target;
            unusual sim_dt values like 0.025 produce decimation < 1).

        Returns ``(sim_dt, control_dt, decimation)`` as ``(float, float, int)``.
        ``control_dt`` is 0.02 (50 Hz) — pinned here, not canvas-configurable
        yet; downstream emitters can override if a future canvas knob lands.
        """
        pg_ids = self._find_by_type("play_ground_setting")
        if not pg_ids:
            raise CanvasConfigError(
                nid="",
                key="sim_dt",
                schema_id="play_ground_setting",
                reason=(
                    "play_ground_setting node is required on the canvas "
                    "but is missing. The IL training pipeline needs its "
                    "sim_dt to compute the env.yaml decimation; refusing "
                    "to substitute 5 ms + decimation 4 defaults "
                    "(CLAUDE.md §1.8)."
                ),
            )
        sim_dt = self._pf(pg_ids[0], "sim_dt")
        if sim_dt <= 0:
            raise CanvasConfigError(
                nid=pg_ids[0],
                key="sim_dt",
                schema_id="play_ground_setting",
                reason=(
                    f"sim_dt={sim_dt} must be > 0. The simulation timestep "
                    f"drives env.yaml decimation; non-positive values would "
                    f"silently substitute the 5 ms default."
                ),
            )
        control_dt = 0.02  # IL pipeline-wide 50 Hz target (CLAUDE.md §1.10)
        decimation = max(1, int(round(control_dt / sim_dt)))
        if decimation < 1:
            raise CanvasConfigError(
                nid=pg_ids[0],
                key="sim_dt",
                schema_id="play_ground_setting",
                reason=(
                    f"sim_dt={sim_dt} too coarse: control_dt/sim_dt = "
                    f"{control_dt / sim_dt} rounds to decimation < 1. "
                    f"Pick sim_dt ≤ {control_dt}."
                ),
            )
        return sim_dt, control_dt, decimation

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
        # Stage 4: variant-aware termination + observation inline funcs.
        # No-op when the canvas's termination / observation terms use
        # only preset behaviour (the resolver returns no source). When a
        # term carries a ``variant`` tag, the variant Python body is
        # emitted here in place of the preset's il_inline, exactly
        # parallel to the reward path. Family filter rejection
        # (e.g. biped variant on quadruped robot) silently falls back
        # to preset via _collect_variant_sources.
        lines += self._custom_termination_funcs()
        lines += self._custom_observation_funcs()
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
        gait_spec = self._resolve_gait_spec()
        if gait_spec is not None:
            lines.extend(
                _GAIT_INLINE_BY_CLASS[gait_spec.class_name].splitlines()
            )
            lines.append("")
        # Slice 3 — the skill trigger command term class (once) when any skill exists,
        # and the reward-gate helper when any package is trigger-gated.
        if self._has_skill_triggers():
            lines.extend(_SKILL_TRIGGER_COMMAND_INLINE.splitlines())
            lines.append("")
        if self._scan_rewards_for_trigger_gate():
            lines.extend(_TRIGGER_GATE_HELPER_INLINE.splitlines())
            lines.append("")
        # Module-level UNITPORT_TERRAIN_CFG must precede SceneCfg, which
        # references it (rough canvases only; flat emits nothing).
        lines += self._terrain_generator_literal()
        # Custom heightfield generator (scene_type='custom'); emits nothing
        # otherwise. Also precedes SceneCfg.
        lines += self._custom_terrain_generator_literal()
        lines += self._scene_cfg()
        lines += self._observations_cfg()
        lines += self._actions_cfg()
        lines += self._commands_cfg()
        lines += self._rewards_cfg()
        lines += self._terminations_cfg()
        lines += self._events_cfg()
        # CurriculumCfg must precede UnitPortEnvCfg, which references it.
        lines += self._curriculum_manager_cfg()
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

        v2: also carries the canonical ``pd_param`` (omega_n, zeta per
        group) and the compile-time-derived PhysX gains, when an
        ActuatorPDNode is wired. The bundle finalizer (Stage F) reads
        this to stamp ``pd_param`` + ``mujoco_pd_gains`` into
        deploy_contract WITHOUT re-deriving from env.yaml (env.yaml
        carries the PhysX dict only — its provenance source is here).

        Otherwise: obs metadata. Actuator / sim / action data is
        faithfully preserved by Isaac Lab's ``dump_yaml`` already
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
        # Slice 4 — record the compiler-injected skill trigger obs terms (dim 1,
        # appended after the user terms in _emit_obs_group) so _compute_obs_layout
        # keeps them (they carry no _OBS_TERM_DIM_TABLE entry and would otherwise be
        # silently skipped → short deploy obs). The deploy ObsBuilder computes each
        # {sid}_trigger from the command vector at that channel's index (command_slice).
        # NOTE: gait obs terms are still NOT recorded here (pre-existing gap); a
        # canvas with BOTH gait and a skill trigger would misalign the trigger obs
        # offset, so that combination is fail-loud at validation (R_SKILL2), not
        # silently emitted.
        skill_names = [sid for sid, _ in self._skill_trigger_items()]
        if skill_names:
            for gname in groups:
                for sid in skill_names:
                    groups[gname][f"{sid}_trigger"] = {
                        "dim": 1,
                        "scale": None,
                        "clip": None,
                        "history_length": None,
                    }
        out: Dict[str, Any] = {
            "schema_version": DEPLOY_META_SCHEMA_VERSION,
            "generated_by": "IsaacLabConfigCompiler",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "num_joints": num_joints,
            "obs_groups": groups,
        }
        # Stage D: pd_param + PhysX-derived gains for the bundle finalizer.
        # The bundle finalizer (Stage F) reads ``unitport_pd_param`` from
        # this sidecar and stamps the canonical (omega_n, zeta) source +
        # both engines' derived arrays into deploy_contract.
        if self._stashed_pd_meta is not None:
            out["unitport_pd_param"] = self._stashed_pd_meta
        # Termination grace audit (per-condition grace_period_s → grace_steps).
        if self._stashed_termination_meta is not None:
            out["unitport_terminations"] = self._stashed_termination_meta
        return out

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
            "from isaaclab.managers import CurriculumTermCfg as CurrTerm",
            "from isaaclab.managers import TerminationTermCfg as DoneTerm",
            "from isaaclab.actuators import ImplicitActuatorCfg, DCMotorCfg, ActuatorNetMLPCfg, ActuatorNetLSTMCfg, IdealPDActuatorCfg, RemotizedPDActuatorCfg",
            "from isaaclab.scene import InteractiveSceneCfg",
            "from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns",
            "from isaaclab.sim import PhysxCfg, RenderCfg, SimulationCfg",
            "from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg",
            "import isaaclab.terrains as terrain_gen",
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
            # 缺口③ — reward_terms is paged; scan term payloads across all pages.
            from application.compiler.term_payload import iter_reward_pages
            for _pid, page_terms in iter_reward_pages(reward_terms):
                for entry in page_terms.values():
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
        from scripts import lookup, BACKEND_ISAAC

        rew_ids = self._find_by_type("rewards")
        if not rew_ids:
            return []

        # Collect unique inline implementations across ALL rewards nodes
        # (multi-rewards canvas with per-item fanout). Kind+backend scoped
        # lookup — the old flat UNIFIED_REGISTRY was removed because
        # cross-kind key collisions (e.g. reward and termination both
        # naming "base_height") silently dropped inline blocks.
        #
        # Variant injection (Stage 4): when a term payload carries a
        # ``variant`` tag (see application.compiler.term_payload) the
        # resolver's user variant source replaces the registry preset's
        # ``il_inline``. The function name inside the variant must match
        # the preset's ``il_func`` — otherwise the RewTerm references in
        # the generated env_cfg still point at the preset name and would
        # call the wrong (or missing) function.
        try:
            from application.compiler.term_payload import parse_term_payload
            from application.service.scripts import resolver as _resolver
        except Exception:                                         # noqa: BLE001
            parse_term_payload = None                              # type: ignore[assignment]
            _resolver = None                                       # type: ignore[assignment]
        emitted_funcs: set = set()
        inline_blocks: list = []
        for rid in rew_ids:
            try:
                reward_terms = self._parse_json_param(rid, "reward_terms")
            except Exception:
                continue
            if not isinstance(reward_terms, dict):
                continue
            # 缺口③ — reward_terms is paged; emit inline for funcs across all pages.
            from application.compiler.term_payload import iter_reward_pages
            _flat_terms = [
                (fk, pl)
                for _pid, page_terms in iter_reward_pages(reward_terms)
                for fk, pl in page_terms.items()
            ]
            for func_key, payload in _flat_terms:
                item = lookup(func_key, kind="reward", backend=BACKEND_ISAAC)
                if item is None or not item.il_inline:
                    continue
                if item.il_func in emitted_funcs:
                    continue
                # Pull variant source when one is selected; else preset.
                source = item.il_inline
                if parse_term_payload is not None and _resolver is not None:
                    try:
                        _, variant, _ = parse_term_payload(payload)
                    except Exception:                             # noqa: BLE001
                        variant = None
                    if variant:
                        resolved = _resolver.resolve(
                            "reward", func_key,
                            variant=variant, backend=BACKEND_ISAAC,
                            robot_sku=(self._robot.sku if self._robot is not None else None),
                        )
                        if (
                            resolved is not None
                            and resolved.origin in ("user_variant", "system_variant")
                            and resolved.source
                        ):
                            source = resolved.source
                emitted_funcs.add(item.il_func)
                inline_blocks.append(source)

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

    def _custom_funcs_for_kind(
        self, *, kind: str, node_type: str, terms_param_key: str,
        section_title: str,
    ) -> List[str]:
        """Generic emit pass shared by termination + observation variants.

        Mirrors :meth:`_custom_reward_funcs` but for an arbitrary kind.
        Walks every node of ``node_type`` on the canvas; for each entry
        in its ``terms_param_key`` dict, looks up the preset's
        ``il_inline`` and (if a variant tag is present) swaps in the
        variant source. Deduplicates by ``item.il_func`` so multiple
        canvas nodes referencing the same key don't double-emit.

        ``kind`` is the resolver kind tag ("termination" / "observation"
        — never "reward"; reward has its own method to keep its hot
        path inlined). Returns empty list when no node carries inline
        funcs (preset-only canvas).
        """
        from scripts import lookup, BACKEND_ISAAC

        node_ids = self._find_by_type(node_type)
        if not node_ids:
            return []
        try:
            from application.compiler.term_payload import parse_term_payload
            from application.service.scripts import resolver as _resolver
        except Exception:                                         # noqa: BLE001
            parse_term_payload = None                              # type: ignore[assignment]
            _resolver = None                                       # type: ignore[assignment]
        emitted_funcs: set = set()
        inline_blocks: list = []
        for nid in node_ids:
            try:
                terms = self._parse_json_param(nid, terms_param_key)
            except Exception:
                continue
            if not isinstance(terms, dict):
                continue
            for func_key, payload in terms.items():
                item = lookup(func_key, kind=kind, backend=BACKEND_ISAAC)
                if item is None or not item.il_inline:
                    continue
                if item.il_func in emitted_funcs:
                    continue
                source = item.il_inline
                if parse_term_payload is not None and _resolver is not None:
                    try:
                        _, variant, _ = parse_term_payload(payload)
                    except Exception:                             # noqa: BLE001
                        variant = None
                    if variant:
                        resolved = _resolver.resolve(
                            kind, func_key,
                            variant=variant, backend=BACKEND_ISAAC,
                            robot_sku=(self._robot.sku if self._robot is not None else None),
                        )
                        if (
                            resolved is not None
                            and resolved.origin in ("user_variant", "system_variant")
                            and resolved.source
                        ):
                            source = resolved.source
                emitted_funcs.add(item.il_func)
                inline_blocks.append(source)

        if not inline_blocks:
            return []
        lines = [
            "# " + "=" * 70,
            f"# UnitPort inline {section_title} (not in standard Isaac Lab)",
            "# " + "=" * 70,
        ]
        for block in sorted(inline_blocks):
            lines.append(block)
        lines.append("")
        return lines

    def _custom_termination_funcs(self) -> List[str]:
        """Variant-aware termination function emit pass (Stage 4)."""
        return self._custom_funcs_for_kind(
            kind="termination",
            node_type="terminations",
            terms_param_key="termination_conditions",
            section_title="termination functions",
        )

    def _custom_observation_funcs(self) -> List[str]:
        """Variant-aware observation function emit pass (Stage 4)."""
        return self._custom_funcs_for_kind(
            kind="observation",
            node_type="il_observation",
            terms_param_key="obs_terms",
            section_title="observation functions",
        )

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
        # since the ILSimulationConfigNode merge. One source of truth —
        # missing / non-positive sim_dt is a hard error (see
        # ``_resolve_play_ground_dt`` for the rationale + ban on 5 ms +
        # decimation 4 silent defaults). The helper guarantees
        # ``play_ground_setting`` exists, so the rest of this function
        # can safely index ``pg_ids[0]``.
        sim_dt, control_dt, decimation = self._resolve_play_ground_dt()
        pg_ids = self._find_by_type("play_ground_setting")

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

    def _rough_playground_pid(self) -> Optional[str]:
        """Return the play_ground_setting node id iff scene_type == 'rough'.

        Centralises the "is this a rough/generator canvas" predicate shared
        by the terrain generator literal, the scene cfg, and the curriculum
        manager cfg. flat canvases (or no playground node) → None.
        """
        pids = self._find_by_type("play_ground_setting")
        if not pids:
            return None
        pid = pids[0]
        if self._p(pid, "scene_type").strip().lower() == "rough":
            return pid
        return None

    def _terrain_generator_literal(self) -> List[str]:
        """Emit module-level ``UNITPORT_TERRAIN_CFG`` for rough canvases.

        Structurally identical to Isaac Lab's ``ROUGH_TERRAINS_CFG``
        (isaaclab/terrains/config/rough.py) — only ``num_rows`` (=
        difficulty_levels) and each sub-terrain ``proportion`` are
        canvas-driven; the inner ranges (step heights, slope/noise ranges)
        stay at the stock values so a default canvas is byte-equivalent to
        the prior ``ROUGH_TERRAINS_CFG`` path. ``curriculum`` is left to the
        env ``__post_init__`` (driven by whether a terrain_levels curriculum
        term is present). flat canvases emit nothing.
        """
        pid = self._rough_playground_pid()
        if pid is None:
            return []
        num_rows = max(1, self._pi(pid, "difficulty_levels"))
        p_stairs = self._pf(pid, "prop_pyramid_stairs")
        p_stairs_inv = self._pf(pid, "prop_pyramid_stairs_inv")
        p_boxes = self._pf(pid, "prop_boxes")
        p_rough = self._pf(pid, "prop_random_rough")
        p_slope = self._pf(pid, "prop_slope")
        p_slope_inv = self._pf(pid, "prop_slope_inv")
        return [
            "# " + "=" * 70,
            "# UnitPort rough-terrain generator — canvas-driven sub-terrain mix",
            "# (proportions) + difficulty rows (num_rows). Inner ranges mirror",
            "# Isaac Lab's ROUGH_TERRAINS_CFG. curriculum flag set by",
            "# UnitPortEnvCfg.__post_init__ from the terrain_levels curriculum term.",
            "# " + "=" * 70,
            "UNITPORT_TERRAIN_CFG = TerrainGeneratorCfg(",
            "    size=(8.0, 8.0),",
            "    border_width=20.0,",
            f"    num_rows={num_rows},",
            "    num_cols=20,",
            "    horizontal_scale=0.1,",
            "    vertical_scale=0.005,",
            "    slope_threshold=0.75,",
            "    use_cache=False,",
            "    sub_terrains={",
            f'        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(',
            f"            proportion={p_stairs}, step_height_range=(0.05, 0.23), step_width=0.3,",
            "            platform_width=3.0, border_width=1.0, holes=False,",
            "        ),",
            f'        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(',
            f"            proportion={p_stairs_inv}, step_height_range=(0.05, 0.23), step_width=0.3,",
            "            platform_width=3.0, border_width=1.0, holes=False,",
            "        ),",
            f'        "boxes": terrain_gen.MeshRandomGridTerrainCfg(',
            f"            proportion={p_boxes}, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0,",
            "        ),",
            f'        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(',
            f"            proportion={p_rough}, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25,",
            "        ),",
            f'        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(',
            f"            proportion={p_slope}, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25,",
            "        ),",
            f'        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(',
            f"            proportion={p_slope_inv}, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25,",
            "        ),",
            "    },",
            ")",
            "",
            "",
        ]

    def _curriculum_manager_cfg(self) -> List[str]:
        """Emit the IsaacLab-native ``CurriculumCfg`` manager group.

        Carries the ``terrain_levels`` curriculum term (Isaac Lab's stock
        distance-based difficulty promote/demote, ``terrain_levels_vel``,
        keyed to our ``base_velocity`` command) ONLY when the canvas has a
        rough playground with curriculum_enabled. Otherwise an empty group
        — the env ``__post_init__`` then forces the generator's curriculum
        flag off. This is distinct from the custom ``UNITPORT_CURRICULUM``
        dict (per-iteration schedules read by the runner); this is the
        engine-side CurriculumManager.
        """
        lines = ["@configclass", "class CurriculumCfg:", '    """Curriculum terms for the MDP."""', ""]
        pid = self._rough_playground_pid()
        enabled = False
        if pid is not None:
            enabled = str(self._p(pid, "curriculum_enabled")).strip().lower() in ("true", "1", "yes", "on")
        if enabled:
            lines.append("    terrain_levels = CurrTerm(func=velocity_mdp.terrain_levels_vel)")
        else:
            lines.append("    pass")
        lines += ["", ""]
        return lines

    def _custom_terrain_generator_literal(self) -> List[str]:
        """Emit module-level ``UNITPORT_CUSTOM_TERRAIN_CFG`` for
        ``scene_type='custom'``.

        Loads the canvas's canonical heightfield ``.npz`` and code-gens a
        ``TerrainGeneratorCfg`` with one custom sub-terrain whose function
        (``application.training.terrain.isaaclab_runtime``) rebuilds the tile
        in the Kit worker from that same ``.npz`` — no USD. Emits nothing for
        non-custom canvases; fail-loud (§8) if custom is selected but the
        heightfield is missing/invalid.
        """
        pids = self._find_by_type("play_ground_setting")
        if not pids:
            return []
        pid = pids[0]
        if self._p(pid, "scene_type").strip().lower() != "custom":
            return []
        src = self._p(pid, "custom_terrain_path").strip()
        if not src:
            raise ValueError(
                "[UnitPort][Compiler] scene_type='custom' but "
                "custom_terrain_path is empty — import a heightfield on the "
                "Play Ground Setting node (§8)."
            )
        vscale = self._pf(pid, "custom_terrain_vertical_scale")
        if not (vscale > 0.0):
            vscale = 0.005
        from application.training.terrain.isaaclab_lowering import (
            emit_custom_terrain_generator_cfg,
        )
        from application.training.terrain.loaders.npz import NpzHeightFieldLoader

        contract = NpzHeightFieldLoader().load(src)  # validates + sha256
        return emit_custom_terrain_generator_cfg(
            src, contract.height_field, vertical_scale=vscale
        )

    def _terrain_importer_lines(
        self, pid: str, terrain_type_literal: str,
        mu_s: float, mu_d: float, restitution: float,
    ) -> List[str]:
        """Emit the ``terrain = TerrainImporterCfg(...)`` block for SceneCfg.

        Split out of :meth:`_scene_cfg` (which also derives robot/actuator
        gains and so cannot run without an MJCF) so the terrain wiring is
        unit-testable on its own. ``plane`` → flat ground; ``rough_generator``
        → the canvas-driven ``UNITPORT_TERRAIN_CFG`` + ``max_init_terrain_level``
        from the node (curriculum start row).
        """
        lines = [
            "    terrain = TerrainImporterCfg(",
            '        prim_path="/World/ground",',
        ]
        if terrain_type_literal == "plane":
            lines.append('        terrain_type="plane",')
        elif terrain_type_literal == "custom_generator":
            # User-imported heightfield — a single fixed tile (no curriculum
            # this phase), built by the module-level UNITPORT_CUSTOM_TERRAIN_CFG
            # (_custom_terrain_generator_literal). No max_init_terrain_level:
            # there is no difficulty grid to start partway up.
            lines.append('        terrain_type="generator",')
            lines.append("        terrain_generator=UNITPORT_CUSTOM_TERRAIN_CFG,")
            lines.append("        collision_group=-1,")
        else:
            # The rough generator's sub-terrain inner ranges (noise/slope/step)
            # stay at ROUGH_TERRAINS_CFG values; only proportions + num_rows are
            # canvas-driven (the removed roughness_amplitude/slope_max sliders
            # never fed these ranges). Parameterising inner geometry is a
            # separate future feature.
            max_init = self._pi(pid, "max_init_terrain_level")
            lines.append('        terrain_type="generator",')
            lines.append("        terrain_generator=UNITPORT_TERRAIN_CFG,")
            lines.append(f"        max_init_terrain_level={max_init},")
            lines.append("        collision_group=-1,")
        lines.append("        physics_material=sim_utils.RigidBodyMaterialCfg(")
        lines.append(
            f"            static_friction={mu_s}, "
            f"dynamic_friction={mu_d}, restitution={restitution},"
        )
        lines.append("        ),")
        lines.append("        debug_vis=False,")
        lines.append("    )")
        return lines

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
        #   flat   → "plane"
        #   rough  → "generator" + UNITPORT_TERRAIN_CFG (emitted module-level
        #            by _terrain_generator_literal; the canvas drives the
        #            per-sub-terrain proportions + num_rows, and — when
        #            curriculum_enabled — a terrain_levels CurriculumTerm
        #            promotes/demotes difficulty per the robot's walked
        #            distance).
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
            # send canvases with an unexpected scene_type onto a 6-tile rough
            # generator, dropping per-iter speed by 5-10× before the user knew
            # anything had gone wrong. Match the play_ground_setting node's enum
            # (["flat", "rough", "custom"]) and raise loudly on anything else.
            if stype == "flat":
                terrain_type_literal = "plane"
            elif stype == "rough":
                terrain_type_literal = "rough_generator"
            elif stype == "custom":
                # User-imported heightfield → custom TerrainGenerator (no USD).
                terrain_type_literal = "custom_generator"
            else:
                raise ValueError(
                    f"\n[UnitPort][Compiler] play_ground_setting.scene_type="
                    f"{stype!r} is not a supported terrain type. "
                    f"Valid values: 'flat', 'rough', 'custom'."
                )
            lines += self._terrain_importer_lines(
                pid, terrain_type_literal, mu_s, mu_d, restitution
            )

        # Robot
        # Actuator config — canonical (omega_n, zeta) PD parameterization
        # via the ActuatorPDNode, with PhysX-side gains derived at compile
        # time mass-weighted off the MJCF mass matrix (kp = m_eff*omega_n^2,
        # kd = 2*zeta*sqrt(kp*m_eff)) — the SAME formula+m_eff the MuJoCo
        # bundle finalizer uses, so the emitted stiffness/damping equal the
        # bundle's mujoco_pd_gains (PhysX stiffness is real torque units, NOT
        # mass-normalized — CLAUDE.md §10). Falls back to the legacy
        # ActorSettingNode scalar path when no actuator_pd node is wired.
        # The fallback is a one-release back-compat bridge — RELEASE/CLAUDE.md
        # §1.8 (c) on-disk legacy compat — and emits a WARN at compile time.
        actuator_lines = []
        actor_ids_for_actuators = self._find_by_type("actor_setting")
        # PD config now lives on RobotNode (merged from the short-lived
        # ActuatorPDNode in May 2026 — the pd_groups / pd_param_mode /
        # pd_effort_limit / ... fields all live on robot now).
        robot_ids_for_pd = self._find_by_type("robot")
        if actor_ids_for_actuators:
            aid = actor_ids_for_actuators[0]

            # Canonical PD path. Returns the fully-rendered actuator-dict
            # lines: one "legs" ImplicitActuatorCfg, plus one
            # RemotizedPDActuatorCfg per remotized joint group when the
            # robot's manifest declares any (see remotized_emit.py). Returns
            # None only when the canvas predates the PD merge → legacy scalar
            # path below.
            pd_lines = self._compile_pd_payload_for_emit(
                actor_setting_node_id=aid,
                actuator_pd_node_id=(robot_ids_for_pd[0] if robot_ids_for_pd else None),
            )
            if pd_lines is not None:
                actuator_lines.extend(pd_lines)
            else:
                # The canonical (omega_n, zeta) path returned None — the robot
                # is unbound, has no declared families, the MJCF is missing, or
                # a physics import failed (each logged a specific reason above).
                # The legacy scalar stiffness/damping fields were REMOVED from
                # ActorSetting (2026-05); both engines now derive gains from
                # (omega_n, zeta). There is nothing valid to fall back to, so we
                # fail loud (§1.8) rather than emit fabricated gains.
                raise RuntimeError(
                    "[env_cfg_compiler] could not derive PD gains from the "
                    "RobotNode (omega_n, zeta). The robot is likely unbound, "
                    "missing a family declaration, or missing its MJCF (needed "
                    "for the mass-weighted solver). Complete the Robot Asset "
                    "card (family + Dump MJCF) and ensure the RobotNode is "
                    "wired. Legacy scalar stiffness/damping fallback was "
                    "removed — PD is parameterized only by (omega_n, zeta) "
                    "(CLAUDE.md §10)."
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

            # Initial joint pose: user-supplied JSON from ActorSetting or
            # JointInitNode. CLAUDE.md §1.8: NO silent Go2-shaped fallback —
            # an unspecified standing pose silently produces a buggy bundle
            # the moment the robot isn't a 12-DoF hip/thigh/calf quadruped
            # (Spot's 16-DoF ank-equipped legs being the canonical failure).
            # Either the canvas explicitly declares angles for every IR
            # role the robot exposes, or compile raises here.
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

            if not raw_angles or raw_angles == "{}":
                raise ValueError(
                    "\n[UnitPort][Compiler] No initial joint angles declared on "
                    "the canvas — add a JointInit node (preferred) or fill "
                    "ActorSetting.init_joint_angles with one entry per IR role "
                    "the selected robot exposes. The previous Go2-shaped "
                    "regex default (`.*_hip_joint`/`.*_thigh_joint`/`.*_calf_joint`) "
                    "was removed (CLAUDE.md §1.8) because it silently produced "
                    "broken env.yaml for any non-Go2 morphology."
                )

            try:
                parsed = _json.loads(raw_angles)
            except Exception as exc:
                raise ValueError(
                    f"\n[UnitPort][Compiler] Failed to parse {source_node} "
                    f"as JSON: {exc!r}\n  Raw value: {raw_angles[:200]!r}"
                )
            if not isinstance(parsed, dict) or not parsed:
                raise ValueError(
                    f"\n[UnitPort][Compiler] {source_node} must be a non-empty "
                    f"JSON object mapping IR roles to angles (radians). Got: "
                    f"{type(parsed).__name__!r}"
                )

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
            # CLAUDE.md §1.8 + canvas-derived-keys: validate that the
            # canvas dict's IR-role keys match the authoritative set
            # declared by the upstream Robot Node's SKU for the active
            # format. Mismatch here means the canvas-side reconcile hooks
            # (NodeItem.on_param_changed("asset_id") /
            # page._connect_raw / page._disconnect_edge_by_id_raw) all
            # failed to fire for this canvas. As a last-resort training
            # gate we RECONCILE here (drop extra + fill missing with 0.0)
            # with a WARN so the user sees what happened — better to let
            # training run with a correct joint set than to block them
            # on a canvas sync glitch.
            # Exclude bucket roles (``misc`` / ``sensor*``) from init-pose
            # reconcile: these are catch-all categories for cosmetic /
            # not-actuated joints (Head_upper, base_white, IMU mounts,
            # lidar mounts, ...) which (a) don't have a single canonical
            # IR target for ``to_physical`` and (b) aren't part of the
            # policy's action space. Including them here would force a
            # bogus 0.0 entry that ``to_physical_dict`` then rightly
            # rejects as a non-IR key.
            required_ir_roles = [
                r for r in self._robot.joint_ir_roles_for(self._active_format or "USD")
                if r != "misc" and not str(r).startswith("sensor")
            ]
            required_set = set(required_ir_roles)
            provided_set = set(str(k) for k in parsed.keys())
            missing_ir = sorted(required_set - provided_set)
            extra_ir = sorted(provided_set - required_set)
            if missing_ir or extra_ir:
                log.warning(
                    "[env_cfg_compiler] %s out-of-sync with upstream "
                    "Robot Node SKU=%r (format=%r): missing=%s extra=%s. "
                    "Reconciling at compile time (extra keys dropped, "
                    "missing keys filled with 0.0) so training can "
                    "proceed. The canvas-side reconcile hooks should "
                    "have fired earlier — re-open + save the ActorSetting "
                    "init_joint_angles editor to bake the reconcile into "
                    "the canvas file.",
                    source_node,
                    self._robot.sku,
                    self._active_format or "USD",
                    missing_ir,
                    extra_ir,
                )
                reconciled_parsed: Dict[str, float] = {}
                for role in required_ir_roles:
                    try:
                        reconciled_parsed[role] = float(
                            parsed.get(role, 0.0)
                        )
                    except (TypeError, ValueError):
                        reconciled_parsed[role] = 0.0
                parsed = reconciled_parsed
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

            # Validate every init joint angle is within the robot's
            # physical joint limits. Isaac Lab's Articulation._validate_cfg
            # also runs this check at launch and raises with the bare
            # physical name, but by then the user has waited for sim
            # bring-up. Catching it here lets us surface a Spot-specific
            # actionable message ("knee range is [-2.793, -0.247], you
            # have 0.0 — open ActorSetting.init_joint_angles") and avoids
            # paying for an env reset that's doomed to fail.
            # CLAUDE.md §1.8: skip silently only when MJCF is unavailable
            # (no source-of-truth for ranges); otherwise FAIL LOUD.
            self._validate_joint_pose_against_mjcf(
                ir_to_physical={ir: resolver.to_physical(ir) for ir in parsed.keys()},
                ir_to_angle={ir: float(v) for ir, v in parsed.items()},
                source_node=source_node,
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

        # Contact sensor — settings owned by ActorSetting node. WYSIWYG:
        # the sensor's prim_path is built from the IR body roles the user
        # picked in actor_setting.contact_body_names (resolved via the
        # Robot Node's IR→body mapping). Empty picker = no sensor at all
        # (spec_validator R5 _check_contact_sensor_coverage refuses to
        # compile when contact consumers exist but the sensor is empty).
        contact_actor = self._find_by_type("actor_setting")
        if contact_actor:
            aid_node = contact_actor[0]
            hist = self._pi(aid_node, "contact_history_length")
            air = self._p(aid_node, "contact_track_air_time").lower() == "true"
            raw_cbn = self._p(aid_node, "contact_body_names") or "[]"
            try:
                cbn_roles = json.loads(raw_cbn) if isinstance(raw_cbn, str) else list(raw_cbn)
            except (ValueError, TypeError):
                cbn_roles = []
            if not isinstance(cbn_roles, list):
                cbn_roles = []
            # strict=True: an explicit contact body not on this robot fails loud
            # (§1.8) instead of being silently dropped. Wildcards ('.*'/'*') are
            # skipped by _resolve_bodies, so the "all" default is unaffected.
            cbn_bodies = self._resolve_bodies(
                *[str(r) for r in cbn_roles], strict=True
            )
            if cbn_bodies:
                # Alternation regex: prim_path matches /Robot/<body> for
                # any picked body. IsaacLab's prim_path regex engine
                # supports (a|b|c) alternation.
                bodies_alt = "|".join(cbn_bodies)
                lines.append(f"    contact_forces = ContactSensorCfg(")
                lines.append(f'        prim_path="{{ENV_REGEX_NS}}/Robot/({bodies_alt})",')
                lines.append(f"        history_length={hist},")
                lines.append(f"        track_air_time={air},")
                lines.append(f"    )")
            # else: empty picker → no contact_forces entity in scene.
            # R5 _check_contact_sensor_coverage will raise CONTACT_SENSOR_
            # COVERAGE_INSUFFICIENT if any reward/termination consumes
            # contact_forces, surfacing the misconfig at canvas-time
            # rather than at IsaacLab runtime.

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

        # 缺口② — the Observation node is consumed BY EDGE (the one wired to
        # il_policy_network.obs_vector), NOT by type. The policy/critic split is
        # now a SINGLE node (``obs_terms`` = policy; ``critic_privileged_terms`` =
        # the extra terms the value fn additionally sees). An il_observation node
        # whose obs_vector output is unwired has no legitimacy (canvas contract:
        # wiring = use) and must never silently compile. Mirrors the rewards-by-
        # edge pattern (_rewards_cfg). spec_validator._check_il_observation_wired
        # is the primary fail-loud; this raise is defense-in-depth.
        pol_ids = self._find_by_type("il_policy_network")
        wired = [
            oid for oid in self._find_by_type("il_observation")
            if any(
                dst in pol_ids and ss == "obs_vector" and ds == "obs_vector"
                for (dst, ss, ds) in self._downstream.get(oid, [])
            )
        ]
        if not wired:
            raise CanvasConfigError(
                reason=(
                    "il_observation.obs_vector is not wired to il_policy_network "
                    "— an unwired Observation node would silently compile to no "
                    "policy obs. Connect the Observation node's obs_vector output "
                    "to the Policy Network node (canvas contract: a node must be "
                    "wired to be used)."
                ),
            )
        if len(wired) > 1:
            raise CanvasConfigError(
                reason=(
                    f"{len(wired)} il_observation nodes are wired to "
                    f"il_policy_network ({wired!r}); exactly one is supported. The "
                    f"policy/critic split is now a single node "
                    f"(critic_privileged_terms), not a second node — run the "
                    f"obs-merge migration on this canvas."
                ),
            )
        for oid in self._find_by_type("il_observation"):
            if oid not in wired:
                log.warning(
                    "[il-compile] ignoring un-wired il_observation node %r "
                    "(obs is consumed by edge to il_policy_network).", oid,
                )
        oid = wired[0]

        # ``height_scan`` needs a SceneEntityCfg("height_scanner") backing.
        has_height_scanner = False
        pg_ids = self._find_by_type("play_ground_setting")
        if pg_ids:
            has_height_scanner = (
                self._p(pg_ids[0], "height_scan_enabled").lower() == "true"
            )

        noise_std = self._pf(oid, "corruption_noise_std")
        enable_noise = self._p(oid, "enable_corruption").lower() == "true"
        apply_noise = enable_noise and noise_std > 0

        # Gait obs terms (Walk These Ways) — appended to BOTH groups so the
        # critic stays a strict superset of the policy (缺口②: critic = policy ∪
        # privileged). Stable order (freq → phase sin/cos → body_h → step_h).
        gait_terms: Optional[List[str]] = None
        obs_spec = self._resolve_gait_spec()
        if obs_spec is not None:
            gait_terms = list(_GAIT_OBS_TERMS_BY_CLASS[obs_spec.class_name])
        # Slice 3 — skill trigger obs terms, compiler-injected into both groups.
        skill_terms = self._skill_trigger_obs_terms() or None

        self._resolved_obs_terms_by_group: Dict[str, Dict[str, _PerTermObsConfig]] = (
            getattr(self, "_resolved_obs_terms_by_group", {}) or {}
        )

        # POLICY group (the DEPLOYED obs) from ``obs_terms``.
        policy_resolved = _normalize_obs_terms(
            self._parse_json_param(oid, "obs_terms"),
            nid=oid, schema_id=self._types.get(oid, ""),
        )
        self._resolved_obs_terms_by_group["policy"] = dict(policy_resolved)
        self._emit_obs_group(
            lines, "PolicyCfg", "policy", policy_resolved,
            has_height_scanner=has_height_scanner,
            enable_corruption=enable_noise, apply_noise=apply_noise,
            noise_std=noise_std, gait_terms=gait_terms,
            skill_terms=skill_terms,
        )

        # CRITIC group (privileged, training-only) = policy ∪ critic_privileged_terms.
        # Emitted ONLY when privileged terms are declared; empty / absent ⇒
        # PolicyCfg only (symmetric, byte-identical to a single-policy canvas).
        # The privileged critic is NOISELESS (enable_corruption=False, no per-term
        # Unoise) — it is the value fn's clean ground-truth input. NEVER enters
        # the deploy bundle (that reads obs_terms = policy only).
        priv_raw = {}
        if "critic_privileged_terms" in self._params.get(oid, {}):
            priv_raw = self._parse_json_param(oid, "critic_privileged_terms")
        if priv_raw:
            priv_resolved = _normalize_obs_terms(
                priv_raw, nid=oid, schema_id=self._types.get(oid, ""),
            )
            critic_resolved = {**policy_resolved, **priv_resolved}
            self._resolved_obs_terms_by_group["critic"] = dict(critic_resolved)
            self._emit_obs_group(
                lines, "CriticCfg", "critic", critic_resolved,
                has_height_scanner=has_height_scanner,
                enable_corruption=False, apply_noise=False,
                noise_std=noise_std, gait_terms=gait_terms,
                skill_terms=skill_terms,
            )

        lines += ["", ""]
        return lines

    def _emit_obs_group(
        self,
        lines: List[str],
        class_name: str,
        attr_name: str,
        resolved_terms: "Dict[str, _PerTermObsConfig]",
        *,
        has_height_scanner: bool,
        enable_corruption: bool,
        apply_noise: bool,
        noise_std: float,
        gait_terms: Optional[List[str]],
        skill_terms: Optional[List[str]] = None,
    ) -> None:
        """Emit one ``class <class_name>Cfg(ObsGroup)`` + its ``attr: Cfg = Cfg()``.

        Shared by the policy and critic groups (缺口② single-node split). The
        critic passes ``enable_corruption=False`` + ``apply_noise=False`` (clean
        privileged input). ``gait_terms`` (pre-indented source lines) append to
        both groups so the critic stays a strict superset of the policy.
        """
        lines.append("    @configclass")
        lines.append(f"    class {class_name}(ObsGroup):")
        # concatenate_terms → flat (num_envs, sum_dims) tensor, not dict.
        lines.append("        concatenate_terms = True")
        # enable_corruption is a real ObsGroup marker; corruption_noise_std is
        # NOT (it becomes a per-term Unoise below, never a class attribute).
        if enable_corruption:
            lines.append("        enable_corruption = True")

        for term_type in resolved_terms:
            if term_type == "height_scan" and not has_height_scanner:
                lines.append(
                    f"        # height_scan SKIPPED: no height_scanner node "
                    f"on the canvas. Add a height-scanner sensor before "
                    f"re-enabling this term, or remove it from obs_terms."
                )
                continue
            func_name = self._obs_term_func(term_type)
            # Term-specific params: only named-subsystem funcs need a handle.
            params_str = ""
            if func_name == "generated_commands":
                params_str = ', params={"command_name": "base_velocity"}'
            elif term_type == "height_scan":
                params_str = (
                    ', params={"sensor_cfg": SceneEntityCfg('
                    '"height_scanner"), "offset": 0.5}'
                )
            term_cfg = resolved_terms[term_type]
            args: List[str] = [f"func=mdp.{func_name}"]
            if apply_noise:
                args.append(f"noise=Unoise(n_min=-{noise_std}, n_max={noise_std})")
            if term_cfg.scale is not None:
                args.append(f"scale={_format_scale_literal(term_cfg.scale)}")
            # clip=(-100, 100) historical safety guard unless user overrides.
            clip_tuple = term_cfg.clip if term_cfg.clip is not None else (-100.0, 100.0)
            args.append(f"clip=({clip_tuple[0]}, {clip_tuple[1]})")
            if term_cfg.history_length is not None:
                args.append(f"history_length={int(term_cfg.history_length)}")
            lines.append(
                f"        {term_type} = ObsTerm({', '.join(args)}{params_str})"
            )

        if gait_terms is not None:
            lines.extend(gait_terms)
        # Slice 3 — skill trigger obs terms, appended to BOTH groups (critic superset).
        if skill_terms:
            lines.extend(skill_terms)

        lines.append("")
        lines.append(f"    {attr_name}: {class_name} = {class_name}()")

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

    # ----- skill trigger channels (skill_command_path_design.md Slice 3) ---

    def _skill_trigger_items(self) -> List[Tuple[str, dict]]:
        """(skill_id, entry) for each ENABLED trigger skill on the training_motion node.

        ``skill_id`` IS the command-term NAME (== the Slice-2 trigger contract channel
        name), so one string threads the command term, its obs term, and the reward gate.
        Trigger-only for now (enum reserved). Absent skill_items => [] (byte-identical).
        """
        tm_ids = self._find_by_type("training_motion")
        if not tm_ids:
            return []
        raw = (self._params.get(tm_ids[0], {}) or {}).get("skill_items")
        if isinstance(raw, str):
            import json as _json_sk
            raw = _json_sk.loads(raw) if raw.strip() else {}
        if not isinstance(raw, dict):
            return []
        out: List[Tuple[str, dict]] = []
        for sid, sk in raw.items():
            s = str(sid).strip()
            if not s or not isinstance(sk, dict):
                continue
            if not bool(sk.get("enabled", True)):
                continue
            if str(sk.get("kind", "trigger")).strip().lower() != "trigger":
                continue  # enum reserved for a later slice
            out.append((s, sk))
        return out

    def _has_skill_triggers(self) -> bool:
        return bool(self._skill_trigger_items())

    def _skill_trigger_obs_terms(self) -> List[str]:
        """One compiler-injected ObsTerm line per trigger skill — appended to BOTH obs
        groups (like gait), since there is no user-authored skill obs term."""
        return [
            f'        {sid}_trigger = ObsTerm('
            f'func=_unitport_skill_trigger_obs, '
            f'params={{"command_name": "{sid}"}})'
            for sid, _sk in self._skill_trigger_items()
        ]

    def _emit_skill_trigger_command_cfg(self, sid: str, sk: dict, resample: str) -> List[str]:
        """CommandsCfg field lines for one skill trigger command term (attr = NAME = sid)."""
        def _f(key: str, default: float) -> float:
            v = sk.get(key)
            try:
                return float(v)
            except (TypeError, ValueError):
                return float(default)
        window_s = _f("window_s", 1.0)
        trigger_prob = min(1.0, max(0.0, _f("trigger_prob", 0.5)))
        return [
            f"    {sid} = SkillTriggerCommandCfg(",
            '        asset_name="robot",',
            f"        window_s={window_s},",
            f"        trigger_prob={trigger_prob},",
            f"        resampling_time_range={resample},",
            f"        debug_vis=False,",
            f"    )",
        ]

    def _scan_rewards_for_trigger_gate(self) -> bool:
        """True iff any rewards node carries a non-empty ``skill_gate`` param (a skill
        package's reward gated on a trigger) — drives the gate-helper preamble emit."""
        for rid in self._find_by_type("rewards"):
            g = (self._params.get(rid, {}) or {}).get("skill_gate")
            if g and str(g).strip():
                return True
        return False

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
            # Per-item initial sampling weight (Weight Set pie editor). Each
            # enabled item may carry an optional ``weight`` in the canvas
            # ``training_items`` dict; align it to the enabled-item order
            # ``_build_training_items_list`` produced. A missing weight falls
            # back to the uniform share (1/N) — so an untouched canvas stays
            # uniform (legacy behaviour). The vector is normalised here; the
            # command term's ``__init__`` re-normalises, which is harmless.
            initial_weights = self._resolve_initial_item_weights(cid, items)
            resample = self._p(cid, "resampling_time_range")
            # Standing is provided by the explicit 'stand' training item (zero
            # ranges), NOT a separate zero-command probability — the
            # `zero_command_probability` node param was removed (it double-counted
            # standing). The command term keeps rel_standing_envs for API
            # compatibility; we pin it to 0 so nothing zeroes commands behind the
            # item sampler's back.
            rel_standing = 0.0
            cmd_step_change_prob = self._pf(cid, "cmd_step_change_prob")
            # Phase C adaptive item sampling. Canvas exposes the cadence in
            # ITERATIONS (intuitive); convert to env-steps the command term
            # counts via the trainer's rollout length (num_steps_per_env).
            adaptive_enabled = self._pi_bool(cid, "adaptive_motion_enabled", False)
            adaptive_floor = self._pf(cid, "adaptive_weight_floor")
            adaptive_ceil = self._pf(cid, "adaptive_weight_ceil")
            adaptive_iters = self._pi(cid, "adaptive_update_interval")
            _trainer_ids = self._find_by_type("il_ppo_trainer")
            _steps_per_env = (
                self._pi(_trainer_ids[0], "num_steps_per_env") if _trainer_ids else 24
            )
            adaptive_interval_steps = max(1, int(adaptive_iters) * int(_steps_per_env))
            heading_command = self._pi_bool(cid, "heading_command", False)
            heading_stiffness = self._pf(cid, "heading_control_stiffness")
            import json as _json_hdg
            _hr = _json_hdg.loads(self._p(cid, "heading_range"))
            heading_lo, heading_hi = float(_hr[0]), float(_hr[1])
            # CLAUDE.md §1.8: emit body_name so env.yaml records the
            # articulation root link. Without this the deploy manifest_parser
            # used to silently default to "base", which is wrong for any
            # robot whose root body isn't literally named "base" (Spot uses
            # "body"). Resolved from the bound RobotSpec's
            # bodies_role_map_for(active_format) by finding ir_role == "base".
            base_body = self._resolve_base_body_name()
            lines.append("    base_velocity = UnitportWeightedVelocityCommandCfg(")
            lines.append('        asset_name="robot",')
            lines.append(f"        body_name={base_body!r},")
            lines.append(f"        items={items!r},")
            lines.append(f"        initial_weights={initial_weights!r},")
            lines.append(f"        rel_standing_envs={rel_standing},")
            # Adaptive-sampling fields (Phase C) wired to the training_motion
            # canvas params. adaptive_enabled=False keeps initial_weights
            # (uniform) frozen for the whole run — the legacy behaviour.
            lines.append(f"        weight_floor={adaptive_floor},")
            lines.append(f"        weight_ceil={adaptive_ceil},")
            lines.append(f"        adaptive_enabled={adaptive_enabled!r},")
            lines.append(f"        adaptive_update_interval_steps={adaptive_interval_steps},")
            lines.append(f"        cmd_step_change_prob={cmd_step_change_prob},")
            lines.append(f"        heading_command={heading_command!r},")
            lines.append(f"        heading_control_stiffness={heading_stiffness},")
            lines.append(f"        heading_lo={heading_lo},")
            lines.append(f"        heading_hi={heading_hi},")
            # Per-command reward telemetry: emit one [UnitPort][CMDMETRICS]
            # line per rollout (num_steps_per_env), so the Mission Control
            # chart gets ~one point per training iteration per command.
            lines.append(f"        metric_emit_interval_steps={int(_steps_per_env)},")
            lines.append(f"        resampling_time_range={resample},")
            lines.append(f"        debug_vis=False,")
            lines.append(f"    )")
            # P2.1 — Walk These Ways parameterised gait command term.
            # Emitted as a SECOND entry in CommandsCfg alongside
            # base_velocity, so the policy obs sees both the velocity
            # command and the gait command stacked in the order the
            # command manager iterates them.
            cfg_spec = self._resolve_gait_spec()
            if cfg_spec is not None:
                emitter = _GAIT_CFG_EMITTERS[cfg_spec.class_name]
                lines.extend(emitter(self, str(resample)))
            # Slice 3 — one skill trigger command term per enabled skill_item.
            # Attr name = skill_id = command-manager NAME (== the obs term's and the
            # reward gate's command_name). The randomised resample is the curriculum.
            for _sid, _sk in self._skill_trigger_items():
                lines.extend(
                    self._emit_skill_trigger_command_cfg(_sid, _sk, str(resample))
                )
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
            # Slice 3 — a skill-gated package's synthetic rewards node carries a
            # ``skill_gate`` param (= the trigger command NAME). Compose the trigger
            # window gate INSIDE the item mask so the term is on only within the
            # trigger's decaying post-pulse window. Empty => no gate (byte-identical).
            gate_name = str(
                (self._params.get(rid, {}) or {}).get("skill_gate", "") or ""
            ).strip()
            # 缺口③ — iterate ALL reward pages (Global + joint pages). Group-page
            # terms carry their joint filter in params_str + a unique field_base.
            for func, field_base, w, params_str, _applies in self._resolve_reward_emit_terms(rid):
                mdp_module, mdp_func = self._reward_func_ref(func)
                func_ref = f"{mdp_module}.{mdp_func}" if mdp_module else mdp_func
                gated_ref = (
                    f"unitport_trigger_gate({func_ref}, {gate_name!r})"
                    if gate_name else func_ref
                )
                wrapped = f"unitport_item_mask({gated_ref}, {mask_args})"
                # Field-name disambiguation: same field across multiple
                # rewards nodes → suffix subsequent ones with ``__n<rid>``.
                field = field_base if field_base not in seen_field_names else f"{field_base}__n{rid}"
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
        any_emitted = False
        # 缺口③ — iterate ALL reward pages (Global + joint pages). Group-page
        # terms carry their joint filter in params_str + a unique field_base;
        # applies_to (phase mask) still wraps every emitted term.
        for func, field_base, w, params_str, applies_to in self._resolve_reward_emit_terms(rid):
            any_emitted = True
            mdp_module, mdp_func = self._reward_func_ref(func)
            func_ref = f"{mdp_module}.{mdp_func}" if mdp_module else mdp_func
            if applies_to:
                phases_literal = ", ".join(repr(p) for p in applies_to)
                wrapped = (
                    f"unitport_phase_mask({func_ref}, ({phases_literal},))"
                    if len(applies_to) == 1
                    else f"unitport_phase_mask({func_ref}, ({phases_literal}))"
                )
                lines.append(
                    f"    {field_base} = RewTerm(func={wrapped}, weight={w}{params_str})"
                )
                phase_masked = True
            else:
                lines.append(
                    f"    {field_base} = RewTerm(func={func_ref}, weight={w}{params_str})"
                )
        if not any_emitted:
            raise CanvasConfigError(
                nid=rid,
                key="reward_terms",
                schema_id=self._types.get(rid, ""),
                reason="reward_terms has no reward terms across any page — at least one required.",
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

        # Policy control dt (= sim_dt * decimation) — the unit of
        # episode_length_buf. Single source of truth, same call the sim cfg
        # uses. grace_period_s (seconds) -> grace_steps via this.
        import math as _math
        _, _control_dt, _ = self._resolve_play_ground_dt()

        # Provenance for the deploy_meta sidecar (audit trail). Populated by
        # _cond as a side effect so the recorded grace_steps is exactly what
        # is emitted (single parse, no drift). Stashed on self below.
        _prov: Dict[str, Any] = {}

        def _cond(key: str) -> Tuple[float, int]:
            """Strict read of one termination knob -> (threshold, grace_steps).

            Accepts BOTH payload shapes (Design A, co-located grace):
              * legacy scalar  ``items[key] = 0.2``            -> grace 0
              * structured dict ``{"weight": 0.2, "grace_period_s": 0.5}``
                (``weight`` is the shared term-payload numeric = the
                threshold for terminations; ``threshold`` also accepted).
            ``grace_period_s`` (seconds) is converted to whole policy steps
            via ceil(grace_s / control_dt). Fails loud (CLAUDE.md §8) on
            missing key / non-numeric / negative grace.
            """
            if key not in items:
                raise CanvasConfigError(
                    nid=tid,
                    key="termination_conditions",
                    schema_id=self._types.get(tid, ""),
                    reason=f"missing required sub-key {key!r}",
                )
            raw = items[key]
            if isinstance(raw, dict):
                thr_raw = raw.get("weight", raw.get("threshold"))
                grace_raw = raw.get("grace_period_s", 0.0)
            else:
                thr_raw, grace_raw = raw, 0.0
            try:
                threshold = float(thr_raw)
            except (ValueError, TypeError) as exc:
                raise CanvasConfigError(
                    nid=tid,
                    key="termination_conditions",
                    schema_id=self._types.get(tid, ""),
                    reason=(
                        f"sub-key {key!r} threshold {thr_raw!r} "
                        f"({type(thr_raw).__name__}) is not a valid float"
                    ),
                ) from exc
            try:
                grace_s = float(grace_raw)
            except (ValueError, TypeError) as exc:
                raise CanvasConfigError(
                    nid=tid,
                    key="termination_conditions",
                    schema_id=self._types.get(tid, ""),
                    reason=(
                        f"sub-key {key!r} grace_period_s {grace_raw!r} "
                        f"({type(grace_raw).__name__}) is not a valid float"
                    ),
                ) from exc
            if grace_s < 0.0:
                raise CanvasConfigError(
                    nid=tid,
                    key="termination_conditions",
                    schema_id=self._types.get(tid, ""),
                    reason=f"sub-key {key!r} grace_period_s={grace_s} must be >= 0",
                )
            grace_steps = int(_math.ceil(grace_s / _control_dt)) if grace_s > 0.0 else 0
            _prov[key] = {
                "threshold": threshold,
                "grace_period_s": grace_s,
                "grace_steps": grace_steps,
            }
            return threshold, grace_steps

        # Module-level grace-wrapper funcs, emitted *before* the class block.
        # A condition with grace_period_s>0 cannot fire while
        # env.episode_length_buf < grace_steps (the spawn/settle transient),
        # but reward still accrues and actions still output. grace=0 emits the
        # plain DoneTerm — byte-identical to the pre-grace path.
        _wrapper_lines: List[str] = []
        _emitted_wrappers: set = set()

        def _grace_wrapper(name: str, params_sig: str, base_call: str) -> None:
            if name in _emitted_wrappers:
                return
            _emitted_wrappers.add(name)
            _wrapper_lines.extend([
                f"def {name}(env, {params_sig}, grace_steps):",
                f"    fired = {base_call}",
                f"    return fired & (env.episode_length_buf >= grace_steps)",
                "",
            ])

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
            #
            # grace_period_s on time_out is meaningless — time_out IS the
            # episode-length cutoff, not a transient-sensitive failure. Reject
            # it loud (CLAUDE.md §8) rather than silently ignore.
            _to_val = items["time_out"]
            if isinstance(_to_val, dict) and float(_to_val.get("grace_period_s", 0.0) or 0.0) > 0.0:
                raise CanvasConfigError(
                    nid=tid,
                    key="termination_conditions",
                    schema_id=self._types.get(tid, ""),
                    reason=(
                        "time_out does not support grace_period_s — it is the "
                        "episode-length cutoff, not a transient failure. Remove "
                        "grace from time_out (put it on base_height / "
                        "bad_orientation / illegal_contact instead)."
                    ),
                )
            lines.append(f"    time_out = DoneTerm(func=mdp.time_out, time_out=True)")

        if "illegal_contact" in items:
            thresh, ic_grace = _cond("illegal_contact")
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
            # Family-aware illegal-contact body set. The IR-role categories
            # available differ per morphology, so we can't share a single
            # tuple:
            #   * quadruped / wheeled: base + thighs + calves. Calf is
            #     included so the "elbow drop" failure mode (lower leg used
            #     as a ground crutch to fake stability reward) terminates
            #     the episode instead of just incurring a small
            #     undesired_contacts penalty.
            #   * biped / humanoid: upper body should never touch ground.
            #     IL G1/H1 stock uses torso_link only (a tight signal); we
            #     broaden slightly to torso + pelvis + waist + shoulders
            #     so an arm-plant failure also terminates. Hips/knees/
            #     ankles/feet are excluded — feet are designed to contact,
            #     and knee/hip braced kneels are a recoverable state we
            #     don't want to hard-terminate.
            #   * generic: just the articulation root.
            # Fallback regex when no robot is bound is intentionally narrow
            # — favours base/torso/pelvis names that work across the
            # families we ship; quadruped-specific thigh/calf wildcards
            # were a Go2-only assumption that silently broke G1 (#2026-05).
            families = set(getattr(self._robot, "families", []) or []) if self._robot is not None else set()
            if families & {"biped", "humanoid"}:
                ic_categories: Tuple[str, ...] = ("base", "torso", "pelvis", "waist", "shoulders")
                fallback_re = '["torso.*", "pelvis", ".*shoulder.*"]'
            elif families & {"quadruped", "wheeled"}:
                ic_categories = ("base", "thighs", "calves")
                fallback_re = '["(base|trunk)", ".*thigh", ".*calf"]'
            else:
                ic_categories = ("base",)
                fallback_re = '["(base|trunk|torso|pelvis|body)"]'
            ic_bodies = self._resolve_bodies(*ic_categories)
            if ic_bodies:
                ic_expr = "[" + ", ".join(f'"{b}"' for b in ic_bodies) + "]"
            else:
                ic_expr = fallback_re
            ic_sensor = (
                f"SceneEntityCfg(\"contact_forces\", body_names={ic_expr})"
            )
            if ic_grace > 0:
                _grace_wrapper(
                    "_unitport_term_illegal_contact_graced",
                    "sensor_cfg, threshold",
                    "mdp.illegal_contact(env, sensor_cfg=sensor_cfg, threshold=threshold)",
                )
                lines.append(f"    illegal_contact = DoneTerm(")
                lines.append(f"        func=_unitport_term_illegal_contact_graced,")
                lines.append(
                    f"        params={{\"sensor_cfg\": {ic_sensor}, "
                    f"\"threshold\": {thresh}, \"grace_steps\": {ic_grace}}},"
                )
                lines.append(f"    )")
            else:
                lines.append(f"    illegal_contact = DoneTerm(")
                lines.append(f"        func=mdp.illegal_contact,")
                lines.append(
                    f"        params={{\"sensor_cfg\": {ic_sensor}, "
                    f"\"threshold\": {thresh}}},"
                )
                lines.append(f"    )")

        if "base_height" in items:
            min_h, bh_grace = _cond("base_height")
            if bh_grace > 0:
                _grace_wrapper(
                    "_unitport_term_base_height_graced",
                    "minimum_height",
                    "mdp.root_height_below_minimum(env, minimum_height=minimum_height)",
                )
                lines.append(
                    f"    base_height = DoneTerm(func=_unitport_term_base_height_graced, "
                    f"params={{\"minimum_height\": {min_h}, \"grace_steps\": {bh_grace}}})"
                )
            else:
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
            limit, bo_grace = _cond("bad_orientation")
            if bo_grace > 0:
                _grace_wrapper(
                    "_unitport_term_bad_orientation_graced",
                    "limit_angle",
                    "mdp.bad_orientation(env, limit_angle=limit_angle)",
                )
                lines.append(
                    f"    bad_orientation = DoneTerm(func=_unitport_term_bad_orientation_graced, "
                    f"params={{\"limit_angle\": {limit}, \"grace_steps\": {bo_grace}}})"
                )
            else:
                lines.append(
                    f"    bad_orientation = DoneTerm(func=mdp.bad_orientation, "
                    f"params={{\"limit_angle\": {limit}}})"
                )

        # Per-axis roll / pitch termination (legged_gym asymmetric tilt). Unlike
        # the symmetric ``bad_orientation`` cone, these gate ONE axis each so a
        # canvas can tolerate fore/aft lean (pitch) more than sideways tip-over
        # (roll). Inline DoneTerm reads projected_gravity_b directly (no native
        # mdp). ONE wrapper covers graced + non-graced: the episode_length_buf
        # gate is a no-op when grace_steps == 0. ``_cond`` records provenance
        # into ``_prov`` as a side effect, same as the siblings above.
        _axis_tilt = {
            # key -> (fn name, numerator gravity component): roll=g_y, pitch=g_x
            "fall_threshold_roll": ("_unitport_term_fall_roll", "g[:, 1]"),
            "fall_threshold_pitch": ("_unitport_term_fall_pitch", "g[:, 0]"),
        }
        for _akey, (_fname, _gcomp) in _axis_tilt.items():
            if _akey not in items:
                continue
            ax_limit, ax_grace = _cond(_akey)
            if _fname not in _emitted_wrappers:
                _emitted_wrappers.add(_fname)
                _wrapper_lines.extend([
                    f"def {_fname}(env, limit, grace_steps):",
                    f"    import torch",
                    f'    g = env.scene["robot"].data.projected_gravity_b',
                    f"    angle = torch.atan2({_gcomp}, -g[:, 2]).abs()",
                    f"    return (angle > limit) & (env.episode_length_buf >= grace_steps)",
                    "",
                ])
            lines.append(
                f"    {_akey} = DoneTerm(func={_fname}, "
                f"params={{\"limit\": {ax_limit}, \"grace_steps\": {ax_grace}}})"
            )

        # Command-follow timeout (survival-side pressure to actually move). Past
        # the HALFWAY point of the episode, if a non-trivial velocity command is
        # given but the robot's command-relative velocity error is still above
        # ``threshold``, terminate. Self-gates on command magnitude so the stand
        # item (cmd ~= 0) is exempt; emitted as a real failure termination (NOT
        # time_out) so it feeds the terminated buffer and incurs the
        # termination penalty. Inline DoneTerm reads the command + body velocity
        # + episode length directly (no native mdp).
        if "command_follow_timeout" in items:
            cft_threshold, cft_grace = _cond("command_follow_timeout")
            _cft_fn = "_unitport_term_command_follow_timeout"
            if _cft_fn not in _emitted_wrappers:
                _emitted_wrappers.add(_cft_fn)
                _wrapper_lines.extend([
                    f"def {_cft_fn}(env, threshold, grace_steps):",
                    f"    import torch",
                    f'    asset = env.scene["robot"]',
                    f'    cmd = env.command_manager.get_command("base_velocity")',
                    f"    # only enforce in the SECOND HALF of the episode",
                    f"    half = float(env.max_episode_length) * 0.5",
                    f"    in_second_half = env.episode_length_buf.float() >= half",
                    f"    # self-gate on command magnitude: a near-zero (stand) command is exempt",
                    f"    commanded = torch.norm(cmd, dim=1) > 0.15",
                    f"    lin_err = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2] - cmd[:, :2]), dim=1)",
                    f"    ang_err = torch.square(asset.data.root_ang_vel_b[:, 2] - cmd[:, 2])",
                    f"    err = torch.sqrt(lin_err + ang_err)",
                    f"    fired = in_second_half & commanded & (err > threshold)",
                    f"    return fired & (env.episode_length_buf >= grace_steps)",
                    "",
                ])
            lines.append(
                f"    command_follow_timeout = DoneTerm(func={_cft_fn}, "
                f"params={{\"threshold\": {cft_threshold}, \"grace_steps\": {cft_grace}}})"
            )

        lines += ["", ""]
        # Stash termination provenance for the deploy_meta.json sidecar
        # (audit trail of every threshold + grace_period_s → grace_steps).
        self._stashed_termination_meta = {
            "schema_version": 1,
            "step_dt": float(_control_dt),
            "conditions": _prov,
        }
        # Module-level grace wrappers (if any) must be defined before the
        # class block that references them.
        if _wrapper_lines:
            header = [
                "# " + "=" * 70,
                "# UnitPort termination grace wrappers (time-gated via "
                "env.episode_length_buf)",
                "# " + "=" * 70,
            ]
            return header + _wrapper_lines + lines
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

                # ── target-body resolution (canvas mass_target_scope) ──
                # Canvas-authoritative, user-selectable (NOT hardcoded):
                #   root       — the robot's ACTUAL articulation-root link via
                #                _resolve_base_body_name() (base→pelvis→torso→
                #                trunk→body); 1:1 legged_gym props[0] base-mass.
                #   all_bodies — body_names=".*"; perturb every link's mass.
                #   custom     — mass_target_body, resolved as an IR role when
                #                possible (e.g. "torso"→"torso_link"), else
                #                passed verbatim as a USD link name / regex.
                # The historical path hardcoded _resolve_body("base"), which on
                # a biped/humanoid (root="pelvis", "base" role unmapped) fell
                # through to the literal "base" → IsaacLab matched zero bodies
                # → the (swallowed) timeline-callback error left the class-based
                # event term un-instantiated → crash at apply (CLAUDE.md §8).
                scope = self._p(did, "mass_target_scope")
                if scope == "all_bodies":
                    body_names_literal = '".*"'
                elif scope == "custom":
                    # Resolve via the bound RobotSpec (same source as the root
                    # path), so an IR role maps to its USD link; a literal link
                    # name / regex passes verbatim. self._robot is always bound
                    # (compiler requires robot=, CLAUDE.md §1.8).
                    fmt = self._active_format or "MJCF"
                    role_to_body: Dict[str, str] = {}
                    try:
                        for _bn, _role in self._robot.bodies_role_map_for(fmt).items():
                            role_to_body.setdefault(str(_role), str(_bn))
                    except Exception:
                        pass
                    resolved_body = role_to_body.get(body, body)
                    body_names_literal = f'["{resolved_body}"]'
                else:  # "root" (default — legged parity)
                    body_names_literal = f'["{self._resolve_base_body_name()}"]'

                lines.append(f"    add_base_mass = EventTerm(")
                lines.append(f'        func=mdp.randomize_rigid_body_mass,')
                lines.append(f'        mode="{mode}",')
                lines.append(f"        params={{")
                lines.append(f'            "asset_cfg": SceneEntityCfg("robot", body_names={body_names_literal}),')
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

        # Sim dt → decimation. Single source of truth via
        # ``_resolve_play_ground_dt`` — refuses silently substituting
        # 5 ms + decimation 4 when sim_dt is missing / non-positive.
        sim_dt, control_dt, decimation = self._resolve_play_ground_dt()

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
        # time_out's value is the episode DURATION (seconds), not a
        # threshold. Accept the legacy scalar and the structured-dict form
        # (``{"weight": 15.0}`` — ``weight`` is the shared term-payload
        # numeric, ``threshold`` also accepted) so a canvas that stored
        # time_out structurally still yields a valid episode_length_s.
        _to_raw = tcs["time_out"]
        if isinstance(_to_raw, dict):
            _to_num = _to_raw.get("weight", _to_raw.get("threshold"))
        else:
            _to_num = _to_raw
        try:
            episode_length_s = float(_to_num)
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
            "    curriculum: CurriculumCfg = CurriculumCfg()",
            "",
            "    # Required ManagerBasedRLEnvCfg root fields — no canvas",
            "    # equivalents today, derived from sim_config.dt + the",
            "    # time_out termination at compile time.",
            f"    decimation: int = {decimation}",
            f"    episode_length_s: float = {episode_length_s}",
            "",
            "    def __post_init__(self):",
            "        # Terrain curriculum: enable the generator's difficulty",
            "        # progression iff a terrain_levels CurriculumTerm is present",
            "        # (canvas curriculum_enabled + rough/generator terrain).",
            "        # getattr-guarded so flat / no-terrain canvases are safe.",
            "        _terrain = getattr(self.scene, \"terrain\", None)",
            "        _gen = getattr(_terrain, \"terrain_generator\", None) if _terrain is not None else None",
            "        if _gen is not None:",
            "            _gen.curriculum = getattr(self.curriculum, \"terrain_levels\", None) is not None",
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
            # init_noise_std is the DIRECT initial action std (legged_gym
            # convention: 1.0 == std 1.0), emitted verbatim. The framework
            # honours the user's typed value — it does NOT exp() it. (The
            # pre-2026-06 build treated this as log_std and applied exp(),
            # silently inflating a user's 1.0 into std e≈2.718; that was a bug.)
            # A non-positive std is invalid input → fail loud (§8); legacy
            # log-space canvases (value ≤ 0) must be run through
            # bootstrap/migrate_canvas_init_noise_std_direct.py first.
            _std = self._pf(nid, 'init_noise_std')
            if _std <= 0.0:
                raise CanvasConfigError(
                    nid=nid,
                    key="init_noise_std",
                    schema_id="il_policy_network",
                    reason=(
                        f"init_noise_std must be a positive direct std "
                        f"(legged_gym: 1.0 = std 1.0), got {_std}. Non-positive "
                        f"values were the old log-space convention — re-author "
                        f"the node or run "
                        f"bootstrap/migrate_canvas_init_noise_std_direct.py."
                    ),
                )
            lines.append(f"    init_noise_std = {_std:.6f}")
            # 缺口① — recurrent policy (GRU/LSTM). These three fields are read
            # back by the launcher (_p("rnn_type"/"rnn_hidden_size"/
            # "rnn_num_layers")) to select rsl_rl's ActorCriticRecurrent vs the
            # feed-forward ActorCritic. They MUST be emitted here — without
            # them _p() fell back to "none" and silently trained an MLP while
            # the bundle's deploy_contract.recurrent (built from this same node)
            # said GRU, so deploy crashed with "ONNX graph has no 'h_in'"
            # (CLAUDE.md §8/§11 — producer and consumer must agree). "none" =
            # feed-forward (byte-identical to the pre-recurrent path).
            lines.append(f'    rnn_type = "{self._p(nid, "rnn_type")}"')
            lines.append(f"    rnn_hidden_size = {self._pi(nid, 'rnn_hidden_size')}")
            lines.append(f"    rnn_num_layers = {self._pi(nid, 'rnn_num_layers')}")
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

    def _validate_joint_pose_against_mjcf(
        self,
        *,
        ir_to_physical: Dict[str, str],
        ir_to_angle: Dict[str, float],
        source_node: str,
    ) -> None:
        """Raise if any IR-role init angle is outside the MJCF joint range.

        Only fires when the bound robot has an on-disk MJCF (the
        canonical source of physical joint limits today; the registry
        doesn't carry ranges yet). For USD-only robots, validation is
        skipped with a WARN — Isaac Lab's _validate_cfg still catches
        the violation at launch, just with a less actionable error.

        CLAUDE.md §1.8 conformance: this is fail-loud, not warn-and-fill.
        The "0.0 fallback for missing IR roles" upstream produces the
        most common violation (Spot's knee range [-2.793, -0.247] doesn't
        contain 0); this validator turns the silent corruption into an
        actionable compile-time error pointing at ActorSetting.
        """
        robot = self._robot
        if robot is None:
            return
        sku = getattr(robot, "sku", "") or ""
        if not sku:
            return

        try:
            from application.service.robot_assets.service import (
                get_robot_asset_service,
            )
            asset = get_robot_asset_service().resolve(sku)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "[env_cfg_compiler] joint-range validator: failed to "
                "resolve RobotAsset for sku=%r (%s); skipping. Isaac "
                "Lab's _validate_cfg will catch violations at launch.",
                sku, exc,
            )
            return

        mjcf = getattr(asset, "mjcf_path", None) if asset else None
        if mjcf is None or not mjcf.is_file():
            log.warning(
                "[env_cfg_compiler] joint-range validator: robot sku=%r "
                "has no on-disk MJCF (mjcf_path=%r); skipping init-pose "
                "range check. Isaac Lab's _validate_cfg will surface any "
                "limit violation at launch with the bare physical joint "
                "name. Run 'Dump MJCF' from the Robot Asset card to "
                "enable the compile-time check.",
                sku, str(mjcf) if mjcf else None,
            )
            return

        try:
            import mujoco  # type: ignore
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "[env_cfg_compiler] joint-range validator: mujoco import "
                "failed (%s); skipping init-pose check.", exc,
            )
            return

        try:
            m = mujoco.MjModel.from_xml_path(str(mjcf))
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "[env_cfg_compiler] joint-range validator: MJCF parse "
                "failed for sku=%r at %r (%s); skipping check.",
                sku, str(mjcf), exc,
            )
            return

        # Index MJCF joints by name with their limits (only joints with
        # limited=True carry a meaningful range; the rest accept any qpos).
        mj_ranges: Dict[str, Tuple[float, float]] = {}
        for ji in range(m.njnt):
            jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, ji)
            if not jname:
                continue
            if not bool(m.jnt_limited[ji]):
                continue
            lo, hi = float(m.jnt_range[ji][0]), float(m.jnt_range[ji][1])
            mj_ranges[jname] = (lo, hi)

        violations: List[str] = []
        for ir, phys in ir_to_physical.items():
            limits = mj_ranges.get(phys)
            if limits is None:
                continue  # joint is unlimited in MJCF, any qpos is legal
            lo, hi = limits
            angle = float(ir_to_angle.get(ir, 0.0))
            if angle < lo or angle > hi:
                violations.append(
                    f"  • IR role {ir!r} (physical {phys!r}): "
                    f"angle={angle:.4f} rad is outside MJCF range "
                    f"[{lo:.4f}, {hi:.4f}]"
                )

        if violations:
            raise ValueError(
                "\n[UnitPort][Compiler] " + source_node + " has joint "
                "angles outside the robot's MJCF limits:\n"
                + "\n".join(violations) + "\n"
                "\n"
                "This usually means the canvas's ActorSetting.init_joint_angles "
                "was auto-reconciled with the value 0.0 for IR roles the "
                "previous robot didn't expose, and the current robot's "
                "joint range doesn't include 0 (Spot's knee bends only "
                "backward in [-2.793, -0.247] — 0.0 is out of range).\n"
                "\n"
                "Open ActorSetting.init_joint_angles in the canvas and set "
                "a pose inside every joint's physical range, then re-launch. "
                "Reference Spot pose: hip_x=±0.1, hip_y=0.9, knee=-1.55."
            )

    def _compile_pd_payload_for_emit(
        self,
        *,
        actor_setting_node_id: str,
        actuator_pd_node_id: Optional[str],   # historical kwarg name — now refers to robot node (PD merged there)
    ) -> Optional[List[str]]:
        """Build the rendered actuator-dict lines for the emitted env_cfg.

        Returns a ``List[str]`` of actuator-dict entry lines (one ``"legs"``
        ImplicitActuatorCfg, plus one RemotizedPDActuatorCfg per remotized
        joint group declared in the robot's brand manifest) when the robot
        node carries PD params AND the robot is bound with joint info;
        ``None`` when the caller should fall back to the legacy scalar emit
        path. The per-joint stiffness/damping come from the mass-weighted
        solver (World B); remotized groups additionally carry a
        ``joint_parameter_lookup`` and no ``effort_limit``.

        The ``actuator_pd_node_id`` kwarg name is kept for back-compat
        but now points at the RobotNode (where the PD params live after
        the May-2026 consolidation).

        Side-effect: stashes pd_param + per-joint physx gains into
        :attr:`_stashed_pd_meta` for the deploy_meta.json sidecar.
        """
        if actuator_pd_node_id is None:
            return None
        # When the RobotNode predates the PD merge, none of the pd_*
        # params exist — fall back to legacy scalar emit.
        if not any(self._p(actuator_pd_node_id, k) for k in (
            "pd_groups", "pd_param_mode",
        )):
            return None
        if self._robot is None:
            log.warning(
                "[env_cfg_compiler] ActuatorPDNode wired but no RobotSpecRef "
                "is bound; cannot resolve joint physical names. Falling back "
                "to legacy scalar emit."
            )
            return None
        families = list(getattr(self._robot, "families", []) or [])
        if not families:
            log.warning(
                "[env_cfg_compiler] ActuatorPDNode wired but robot has no "
                "declared families; cannot pick PD group defaults. Falling "
                "back to legacy scalar emit."
            )
            return None

        try:
            from application.physics.pd_param import PDParam
            from application.physics.physx_gain_solver import solve as solve_physx
            from application.physics.mujoco_gain_solver import (
                effective_inertia_diag,
            )
            from application.training.joint_ir import JointIRResolver
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "[env_cfg_compiler] failed to import physics helpers (%s); "
                "falling back to legacy scalar emit.",
                exc,
            )
            return None

        primary_family = families[0]
        try:
            base = PDParam.from_family_defaults(primary_family)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "[env_cfg_compiler] family=%r has no PD group defaults "
                "(%s); falling back to legacy scalar emit.",
                primary_family, exc,
            )
            return None

        # Apply canvas overrides.
        overrides_raw = self._p(actuator_pd_node_id, "pd_groups").strip()
        overrides: Dict[str, Dict[str, float]] = {}
        if overrides_raw and overrides_raw not in ("{}", ""):
            try:
                import json as _json
                parsed = _json.loads(overrides_raw)
                if isinstance(parsed, dict):
                    overrides = {
                        str(gid): dict(vals)
                        for gid, vals in parsed.items()
                        if isinstance(vals, dict)
                    }
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "[env_cfg_compiler] actuator_pd.pd_groups JSON parse "
                    "failed (%s); using family defaults only.", exc,
                )
        try:
            pd_param = base.with_overrides(overrides) if overrides else base
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"[env_cfg_compiler] ActuatorPDNode pd_groups override "
                f"rejected: {exc}"
            ) from exc

        # Resolve IR roles → physical names via the bound robot.
        resolver = self._joint_ir_resolver
        if resolver is None:
            log.warning(
                "[env_cfg_compiler] JointIRResolver unavailable; falling "
                "back to legacy scalar emit."
            )
            return None
        ir_roles = list(resolver._ir_to_physical.keys())   # public-ish: see joint_ir.py:115
        if not ir_roles:
            log.warning(
                "[env_cfg_compiler] resolver carries no IR roles; falling "
                "back to legacy scalar emit."
            )
            return None
        physical = [resolver.to_physical(r) for r in ir_roles]

        # PhysX gains are mass-weighted (kp = m_eff·ωn²) off the SAME
        # effective inertia the MuJoCo bundle finalizer uses — both read
        # the robot's MJCF via mj_fullM at the MJCF nominal stance
        # (nominal_qpos=None ⇒ keyframe-0/qpos0). Sharing the m_eff source
        # makes the emitted ImplicitActuatorCfg stiffness/damping equal the
        # bundle's mujoco_pd_gains by construction, so the trained policy
        # meets identical real-unit gains in MuJoCo review (CLAUDE.md §10).
        sku = getattr(self._robot, "sku", "") or ""
        mjcf_path = None
        try:
            from application.service.robot_assets.service import (
                get_robot_asset_service,
            )
            asset = get_robot_asset_service().resolve(sku) if sku else None
            mjcf_path = getattr(asset, "mjcf_path", None) if asset else None
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"[env_cfg_compiler] could not resolve RobotAsset for "
                f"sku={sku!r} to compute PD effective inertia: {exc}"
            ) from exc
        if mjcf_path is None or not mjcf_path.is_file():
            raise RuntimeError(
                f"[env_cfg_compiler] robot sku={sku!r} has an ActuatorPDNode "
                f"wired but no on-disk MJCF (mjcf_path="
                f"{str(mjcf_path) if mjcf_path else None!r}). The (omega_n, "
                f"zeta) PD parameterization derives engine gains as "
                f"kp = m_eff·ωn² from the MJCF mass matrix; without an MJCF "
                f"the PhysX and MuJoCo sides cannot agree. Open the Robot "
                f"Asset card and run 'Dump MJCF' before training."
            )

        # Resolve the USD→MJCF inertia overlay (link inertia → the USD the
        # policy actually trains/deploys on) so m_eff is derived on the same
        # link inertia BOTH engines realize (CLAUDE.md §10/§11). Absent overlay
        # = None = raw MJCF (unchanged behavior). The bundle finalizer resolves
        # the SAME overlay by SKU so its m_eff matches this one (the 1e-6 parity
        # guard there enforces it).
        inertia_overlay = None
        try:
            from application.service.robot_assets.runtime import (
                read_mjcf_inertia_overlay,
            )
            inertia_overlay = read_mjcf_inertia_overlay(sku) if sku else None
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "[env_cfg_compiler] inertia overlay read failed for sku=%r "
                "(%s); deriving m_eff on raw MJCF.", sku, exc,
            )
            inertia_overlay = None

        try:
            inertia = effective_inertia_diag(
                mjcf_path=Path(mjcf_path),
                joint_order_physical=physical,
                nominal_qpos=None,  # MJCF keyframe-0 / qpos0; matches finalizer
                inertia_overlay=inertia_overlay,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"[env_cfg_compiler] effective-inertia (mj_fullM) failed for "
                f"sku={sku!r}: {exc}"
            ) from exc

        try:
            gains = solve_physx(
                joint_order_physical=physical,
                joint_ir_roles=ir_roles,
                pd_param=pd_param,
                m_eff=inertia.m_eff,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"[env_cfg_compiler] physx_gain_solver failed: {exc}"
            ) from exc

        # Per-joint gains keyed by physical NAME (stable across permutations —
        # World B lesson; the emit helpers subset these per actuator group).
        kp_by_name = {name: float(v) for name, v in zip(physical, gains.kp)}
        kd_by_name = {name: float(v) for name, v in zip(physical, gains.kd)}

        # Per-joint armature keyed by physical NAME (CLAUDE.md §10). EXPLICIT
        # by-name alignment, NOT by index: ``inertia.armature`` is parallel to
        # ``physical`` (MJCF joint order) and we key it by name here; the emit
        # helper renders ``armature={name: value}`` which IsaacLab resolves by
        # joint name against the USD articulation (USD joint order ≠ MJCF order).
        # So the MJCF→USD permutation cannot mis-pair armature with the wrong
        # joint — the same name-keyed contract stiffness/damping already use.
        if len(inertia.armature) != len(physical):
            raise RuntimeError(
                f"[env_cfg_compiler] effective_inertia_diag returned "
                f"{len(inertia.armature)} armature entries for {len(physical)} "
                f"joints (sku={sku!r}) — cannot align armature by name."
            )
        armature_by_name = {
            name: float(v) for name, v in zip(physical, inertia.armature)
        }

        # effort_limit / velocity_limit live on the RobotNode (moved from
        # ActorSetting in 2026-05 so all actuator/PD knobs sit on one node next
        # to (omega_n, zeta)). ``actuator_pd_node_id`` IS the robot node id (see
        # the kwarg docstring). Both engines + the SB3 bundle read from here.
        eff = self._pf(actuator_pd_node_id, "effort_limit")
        vel = self._pf(actuator_pd_node_id, "velocity_limit")

        # Per-joint actuator caps (legged_gym parity): clip EACH joint to its
        # own real motor limit, captured by Dump USD into the registry
        # (physics_per_format.USD DriveAPI max_force / max_velocity), keyed by IR
        # role. A flat scalar starves the load-bearing joints of a humanoid
        # (G1 hip/knee need 88–139 N·m; the manifest default 30 N·m made the
        # robot collapse). Joints absent from the capture fall back to the
        # RobotNode scalar; if that scalar is itself the bare manifest default
        # (i.e. the user never set it AND there is no USD capture) we FAIL LOUD
        # (CLAUDE.md §8) — a silent manifest-default 30 N·m torque-starves the
        # robot and "looks like it worked". The user must run Dump USD (captures
        # real per-joint limits) or set the caps in the PD Group panel.
        from application.physics.actuator_caps import resolve_per_joint_caps
        _caps_by_role = resolve_per_joint_caps(sku, fmt="USD")
        _user_set_eff = "effort_limit" in (self._params.get(actuator_pd_node_id) or {})
        _user_set_vel = "velocity_limit" in (self._params.get(actuator_pd_node_id) or {})
        effort_by_name: Dict[str, float] = {}
        velocity_by_name: Dict[str, float] = {}
        _eff_fallback: List[str] = []
        _vel_fallback: List[str] = []
        for _name, _role in zip(physical, ir_roles):
            _cap = _caps_by_role.get(_role) or {}
            _e = _cap.get("effort")
            _v = _cap.get("velocity")
            if _e is None:
                _e = eff
                _eff_fallback.append(_name)
            if _v is None:
                _v = vel
                _vel_fallback.append(_name)
            effort_by_name[_name] = float(_e)
            velocity_by_name[_name] = float(_v)
        if _eff_fallback and not _user_set_eff:
            raise CanvasConfigError(
                nid=actuator_pd_node_id,
                key="effort_limit",
                schema_id=self._types.get(actuator_pd_node_id, ""),
                reason=(
                    f"robot sku={sku!r} has no per-joint actuator torque limits "
                    f"and the RobotNode's effort_limit is unset, so joints "
                    f"{_eff_fallback} would silently fall back to the manifest "
                    f"default {float(eff):.1f} N·m — which torque-starves "
                    f"load-bearing joints (the 2026-06 G1 'limp collapse'). "
                    f"Fix: run 'Dump USD' on the Robot Asset card to capture the "
                    f"real per-joint limits (max_force), OR set the actuator caps "
                    f"explicitly in the RobotNode's PD Group panel. Refusing to "
                    f"train with a guessed torque ceiling (CLAUDE.md §8)."
                ),
            )
        elif _eff_fallback:
            log.warning(
                "[env_cfg_compiler] joints %s have no per-joint USD effort "
                "capture; using the RobotNode scalar effort_limit=%.1f for them "
                "(sku=%r).", _eff_fallback, float(eff), sku,
            )
        if _vel_fallback and not _user_set_vel:
            log.warning(
                "[env_cfg_compiler] joints %s have no per-joint USD velocity "
                "capture; using manifest/RobotNode velocity_limit=%.1f (sku=%r).",
                _vel_fallback, float(vel), sku,
            )

        # ---- Stage 0 audit producer: scene friction + foot-geometry hash ----
        # scene_contract: the single-source ground friction (canvas
        # play_ground_setting). PhysX RigidBodyMaterialCfg, the SB3 MuJoCo env
        # AND the deploy MuJoCo (method A — PolicyRunner applies it) all consume
        # this ONE value. Absent playground node → None (the audit marks the row
        # n/a; never default-to-1, §8).
        _scene_contract = None
        _pg_ids = self._find_by_type("play_ground_setting")
        if _pg_ids:
            _pg = _pg_ids[0]
            _scene_contract = {
                "friction_static": float(self._pf(_pg, "friction_static")),
                "friction_dynamic": float(self._pf(_pg, "friction_dynamic")),
            }

        # foot_geometry: pose-invariant hash of the robot's foot collision geoms
        # (Layer-1 realized==designed). A legged robot whose registry
        # bodies_per_format.MJCF lacks resolvable foot geoms gets a loud WARN +
        # status='unavailable' — NOT a training abort (that would regress
        # existing canvases); the deploy-side self-check only bites on
        # status='ok'. Manipulator (no foot roles) → status='not_applicable'.
        _foot_geometry = None
        try:
            import mujoco as _mj_fg
            from application.physics.foot_geometry import hash_foot_geometry
            from application.training.validation.mjcf_base_calibration import (
                resolve_foot_roles_for_sku,
            )
            from registers import robots as _robots_reg_fg
            _spec_fg = _robots_reg_fg.get_robot(sku) if sku else None
            _bodies_mjcf = (
                (_spec_fg or {}).get("bodies_per_format", {}).get("MJCF", {}) or {}
            )
            _ir_to_body = {}
            for _bspec in _bodies_mjcf.values():
                if isinstance(_bspec, dict):
                    _nm = str(_bspec.get("name", ""))
                    _ir = str(_bspec.get("ir_role", ""))
                    if _nm and _ir:
                        _ir_to_body[_ir] = _nm
            # Resolve the family canonical feet to the roles ACTUALLY present on
            # this SKU (foot_L → ankle_roll_L → ankle_L), so a single-DOF-ankle
            # humanoid (H1, feet mapped to ankle_*) hashes 'ok' instead of the
            # canonical foot_* missing → 'unavailable'.
            _foot_roles = resolve_foot_roles_for_sku(
                primary_family, _ir_to_body.keys())
            _fg_model = _mj_fg.MjModel.from_xml_path(str(mjcf_path))
            _foot_geometry = hash_foot_geometry(
                _fg_model, foot_roles=_foot_roles, ir_to_body_name=_ir_to_body,
            )
        except Exception as exc:  # noqa: BLE001 — FootGeometryError or load failure
            log.warning(
                "[env_cfg_compiler] foot-geometry hash unavailable for sku=%r "
                "(%s); stashing status='unavailable'. The deploy-side "
                "foot-geometry self-check is skipped for this bundle (it only "
                "engages on status='ok').", sku, exc,
            )
            _foot_geometry = {
                "schema": "unitport.foot_geometry/v1",
                "status": "unavailable",
                "per_role": {}, "aggregate": "",
                "notes": [str(exc)],
            }

        # Stash for deploy_meta sidecar.
        self._stashed_pd_meta = {
            "pd_param": pd_param.to_dict(),
            "physx_gains": gains.to_dict(),
            "scene_contract": _scene_contract,
            "foot_geometry": _foot_geometry,
            "m_eff_source": {
                "mjcf_path": str(mjcf_path),
                "nominal_qpos_source": inertia.nominal_qpos_source,
                "qpos_ref": list(inertia.qpos_ref),
                "qpos_sha256": inertia.qpos_sha256(),
                "m_eff": [float(v) for v in inertia.m_eff],
                "joint_order_physical": list(physical),
                # armature provenance (CLAUDE.md §10): the MJCF-authoritative
                # reflected-rotor inertia folded into m_eff, AND the value we
                # actually emit to PhysX ImplicitActuatorCfg.armature below. Both
                # arrays are parallel to joint_order_physical (ir_roles array
                # ordering); the calibration gate matches them by name, never
                # index. ``overlay_applied`` records whether m_eff was derived on
                # the USD-corrected link inertia (so finalizer/gate agree).
                "armature_mjcf": [float(v) for v in inertia.armature],
                "physx_armature_emitted": [float(v) for v in inertia.armature],
                "array_ordering": "ir_roles",
                "overlay_applied": bool(inertia.overlay_applied),
            },
            "primary_family": primary_family,
            "resolve_at_reset": self._pi_bool(actuator_pd_node_id, "pd_resolve_at_reset", True),
            "calibration_blocking": self._pi_bool(actuator_pd_node_id, "pd_calibration_blocking", True),
            "skip_calibration": self._pi_bool(actuator_pd_node_id, "pd_skip_calibration", False),
            "effort_limit": float(eff),
            "velocity_limit": float(vel),
            # Per-joint caps actually emitted (legged_gym parity). Keyed by
            # physical NAME like kp/kd; the scalars above are the fallback used
            # for any joint absent from the USD physics capture.
            "effort_limit_per_joint": dict(effort_by_name),
            "velocity_limit_per_joint": dict(velocity_by_name),
        }

        # Resolve the robot's remotized-actuator manifest (if any) and render
        # the actuator-dict lines. A MISSING manifest is normal (non-remotized
        # robot → groups=[] → single legacy "legs" group). A MALFORMED
        # manifest (pattern matches nothing, joint claimed twice, table fails
        # to load) RAISES — never silently falls back (§8).
        from .remotized_emit import (
            build_actuator_lines,
            build_remotized_provenance,
            match_remotized_groups,
        )

        groups = []
        try:
            from registers import brands as _brands_reg
            from registers import robots as _robots_reg
            spec = _robots_reg.get_robot(sku) if sku else None
            manifest = (
                _brands_reg.remotized_manifest(
                    spec.get("brand", ""), spec.get("model", "")
                )
                if spec else None
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"[env_cfg_compiler] failed to load remotized manifest for "
                f"sku={sku!r}: {exc}"
            ) from exc
        if manifest and manifest.get("remotized_joints"):
            from application.physics.actuators.torque_lookup import (
                TorqueLookupTable,
            )
            groups = match_remotized_groups(
                physical,
                manifest["remotized_joints"],
                lambda p: TorqueLookupTable.from_yaml(Path(p)),
            )

        prov = build_remotized_provenance(
            physical=physical, ir_roles=ir_roles, groups=groups,
        )
        if prov is not None:
            self._stashed_pd_meta["remotized_joints"] = prov

        return build_actuator_lines(
            physical=physical,
            kp=kp_by_name,
            kd=kd_by_name,
            effort_limit=effort_by_name,
            velocity_limit=velocity_by_name,
            groups=groups,
            armature=armature_by_name,
            implicit_cls=self._actuator_cfg_class("implicit_pd"),
        )

    def _pi_bool(self, nid: str, key: str, default: bool) -> bool:
        """Read a bool parameter; tolerant of "true"/"false"/"True" strings."""
        raw = self._p(nid, key)
        if isinstance(raw, bool):
            return raw
        s = str(raw).strip().lower()
        if s in ("true", "1", "yes"):
            return True
        if s in ("false", "0", "no", ""):
            return False
        return default

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
        from scripts import lookup, BACKEND_ISAAC
        item = lookup(func_key, kind="reward", backend=BACKEND_ISAAC)
        if item is None or not item.il_func:
            return "mdp", func_key
        return item.il_module, item.il_func

    @staticmethod
    def _reward_func_name(func_key: str) -> str:
        from scripts import lookup, BACKEND_ISAAC
        item = lookup(func_key, kind="reward", backend=BACKEND_ISAAC)
        return item.il_func if (item and item.il_func) else func_key

    def _lookup_variant_il_params(
        self,
        nid: str,
        terms_param_key: str,
        kind: str,
        func_key: str,
        payload: Any = None,
    ) -> Optional[str]:
        """Return the variant's ``il_params`` template, or None.

        Reads the canvas node's ``terms_param_key`` dict, finds the
        payload for ``func_key``, extracts the ``variant`` tag, and
        consults the resolver. Returns ``meta.il_params_override`` when
        a user/system variant is resolved AND declared one; otherwise
        None (caller falls back to preset il_params).

        Family filter is enforced via ``robot_sku`` — a variant pinned
        to a family that doesn't match the bound robot returns None
        and the caller uses preset, exactly like
        ``_collect_variant_sources`` in spec_compiler.
        """
        try:
            from application.compiler.term_payload import parse_term_payload
            from application.service.scripts import resolver as _resolver
        except Exception:                                         # noqa: BLE001
            return None
        if payload is None:
            # 缺口③ — paged reward_terms: caller normally passes the per-page
            # payload. Fall back to a flat re-read for non-paged callers.
            try:
                terms = self._parse_json_param(nid, terms_param_key)
            except Exception:                                     # noqa: BLE001
                return None
            if not isinstance(terms, dict):
                return None
            payload = terms.get(func_key)
        if payload is None:
            return None
        try:
            _, variant, _ = parse_term_payload(payload)
        except Exception:                                         # noqa: BLE001
            return None
        if not variant:
            return None
        resolved = _resolver.resolve(
            kind, func_key, variant=variant,
            backend="isaac_lab",
            robot_sku=(self._robot.sku if self._robot is not None else None),
        )
        if resolved is None:
            return None
        return resolved.il_params_override or None

    def _reward_extra_params(self, nid: str, func_key: str, payload: Any = None) -> str:
        """Build the ``params={...}`` kwarg string for a RewTerm.

        缺口③ — ``payload`` is the term's per-page payload (reward_terms is
        paged); threaded into variant + item_value resolution so a term on
        different pages resolves its own variant/value.

        Reads the ``il_params`` template from the registry, substitutes
        node-level values (``{node_std}``, ``{node_threshold}``), then
        resolves any ``{ir:role}`` body-name placeholders via BodyIR.

        Uses reward-scoped registries explicitly — see ``_reward_func_ref``
        for the collision rationale (same key, reward vs termination).

        Stage 4 variant override: if the canvas's reward_terms payload
        for this ``func_key`` carries a ``variant`` tag whose
        ``VariantMeta.il_params_override`` is set, that template
        REPLACES the preset's ``item.il_params`` entirely. The variant
        author owns the body / sensor / threshold params; remaining
        ``{ir:role}``, ``{node_std}``, etc. placeholders are still
        resolved by the same two-phase pipeline below.
        """
        from scripts import (
            REWARD_REGISTRY, IL_REWARD_REGISTRY,
        )
        item = IL_REWARD_REGISTRY.get(func_key) or REWARD_REGISTRY.get(func_key)
        if item is None:
            return ""
        params_str = item.il_params or ""
        # Variant-supplied il_params_override (Stage 4): replace the
        # preset template entirely when the canvas-side variant tag
        # declares one in its variants.toml.
        variant_il_params = self._lookup_variant_il_params(
            nid, "reward_terms", "reward", func_key, payload=payload,
        )
        if variant_il_params is not None:
            params_str = variant_il_params
        if not params_str:
            return ""
        # Phase 1: resolve {ir:role} body-name placeholders FIRST
        # (before .format(), because {ir:feet} is not a valid format key)
        if "{ir:" in params_str:
            mapper = self._get_body_ir_mapper()
            from application.training.body_ir import resolve_body_params
            params_str = resolve_body_params(params_str, mapper)
        # Phase 2: substitute node-level values ({node_std}, {node_threshold})
        # plus the per-item ``{item_value}`` (the Rewards node "Value" chip —
        # a reward's function-internal param, e.g. base_height target height).
        # ``{item_value}`` defaults to 0.0 = "auto" (the reward resolves its
        # own brand-neutral target). No reach-back into the Robot node.
        params_str = params_str.format(
            node_std=self._pf(nid, "std"),
            node_threshold=self._pf(nid, "threshold"),
            item_value=self._resolve_reward_item_value(nid, func_key, payload=payload),
        )
        return ", params={" + params_str + "}"

    def _resolve_partition_joint_names(
        self, group_id: str, *, rid: str, func: str
    ) -> List[str]:
        """Physical joint names for a PD-group partition on the bound robot.

        缺口③ — expands one joint-subset partition (a ``pd_groups`` family
        group id, e.g. ``hip_y`` / ``knee`` / ``shoulder_pitch``) into the
        concrete physical joint names a ``SceneEntityCfg`` needs. Reuses the
        family-keyed PD joint-group regex (single source of truth) + the
        compiler's ``JointIRResolver`` for IR-role → physical translation.

        Fail-loud (CLAUDE.md §8): no bound robot, robot without a family, an
        unknown partition for the family, a missing joint table, or a
        partition that matches NONE of the robot's joints all raise — never a
        silent empty joint set (which would emit a no-op reward that looks
        configured but does nothing = "user setting != actual running").
        """
        import re as _re
        from registers import pd_groups

        if self._robot is None:
            raise CanvasConfigError(
                nid=rid, key="reward_terms", schema_id=self._types.get(rid, ""),
                reason=(
                    f"reward {func!r} uses joint partitions but no robot is bound "
                    f"to the compiler — cannot resolve partition {group_id!r} to "
                    f"physical joints. Wire a Robot node on the canvas."
                ),
            )
        families = list(getattr(self._robot, "families", []) or [])
        if not families:
            raise CanvasConfigError(
                nid=rid, key="reward_terms", schema_id=self._types.get(rid, ""),
                reason=(
                    f"reward {func!r} uses joint partitions but robot "
                    f"{self._robot.sku!r} declares no family — partitions are "
                    f"resolved against the family's PD joint groups."
                ),
            )
        family = families[0]
        g = pd_groups.get_group(family, group_id)
        if g is None:
            valid = pd_groups.list_group_ids(family)
            raise CanvasConfigError(
                nid=rid, key="reward_terms", schema_id=self._types.get(rid, ""),
                reason=(
                    f"reward {func!r} references joint partition {group_id!r}, "
                    f"which is not a PD joint group for family {family!r}. "
                    f"Valid partitions for this family: {valid}. Fix: pick a "
                    f"partition that exists for this robot's family on the "
                    f"Rewards node, or remove it."
                ),
            )
        resolver = self._joint_ir_resolver
        if resolver is None:
            raise CanvasConfigError(
                nid=rid, key="reward_terms", schema_id=self._types.get(rid, ""),
                reason=(
                    f"reward {func!r} partition {group_id!r} needs a joint table "
                    f"to resolve IR roles → physical names, but robot "
                    f"{self._robot.sku!r} has none for the active format. Run "
                    f"Dump in the Robot Asset card first."
                ),
            )
        regex = _re.compile(g["ir_role_regex"])
        matched = [r for r in resolver.ir_roles if regex.fullmatch(r)]
        if not matched:
            raise CanvasConfigError(
                nid=rid, key="reward_terms", schema_id=self._types.get(rid, ""),
                reason=(
                    f"reward {func!r} partition {group_id!r} (family {family!r}, "
                    f"regex {g['ir_role_regex']!r}) matches NONE of robot "
                    f"{self._robot.sku!r}'s IR roles {resolver.ir_roles}. The "
                    f"partition has no joints on this robot — pick a partition "
                    f"this robot actually has, or remove it from the Rewards node."
                ),
            )
        return [resolver.to_physical(r) for r in matched]

    def _reward_is_paginable(self, func: str) -> bool:
        """True iff reward ``func`` may live on a joint page (il_partition_source set)."""
        from scripts import lookup, BACKEND_ISAAC
        item = lookup(func, kind="reward", backend=BACKEND_ISAAC)
        return bool(item and getattr(item, "il_partition_source", ""))

    def _resolve_reward_emit_terms(self, rid: str):
        """缺口③ — yield ``(func, field_base, weight, params_str, applies_to)`` for
        EVERY reward across ALL pages of node ``rid``'s paged ``reward_terms``.

        Global page (``PAGE_GLOBAL``): no joint filter, ``field_base=func``,
        params from the registry (variant/value resolved against the page
        payload). Group page (a pd_group id): the reward is restricted to that
        partition's physical joints via ``SceneEntityCfg(joint_names=…)`` and
        ``field_base=f"{func}_{group}"``. Group pages accept ONLY joint-
        paginable rewards (fail-loud §8). Legacy flat reward_terms migrates to
        the global page transparently (R6 byte-identical emit).
        """
        from application.compiler.term_payload import iter_reward_pages, PAGE_GLOBAL
        reward_terms = self._parse_json_param(rid, "reward_terms")
        for page_id, page_terms in iter_reward_pages(reward_terms):
            for func, entry in page_terms.items():
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
                        nid=rid, key="reward_terms", schema_id=self._types.get(rid, ""),
                        reason=(
                            f"reward weight for {func!r} (page {page_id!r}) is not a "
                            f"valid float: {weight!r} ({type(weight).__name__})"
                        ),
                    ) from exc
                if page_id == PAGE_GLOBAL:
                    params_str = self._reward_extra_params_from_node(rid, func, payload=entry)
                    field_base = func
                else:
                    if not self._reward_is_paginable(func):
                        raise CanvasConfigError(
                            nid=rid, key="reward_terms", schema_id=self._types.get(rid, ""),
                            reason=(
                                f"reward {func!r} is on joint page {page_id!r} but is "
                                f"not joint-paginable (no il_partition_source). Only "
                                f"joint-subset rewards (joint_deviation_l1, "
                                f"dof_pos_limits, joint_*_penalty) belong on a joint "
                                f"page; move {func!r} to the Global page."
                            ),
                        )
                    joint_names = self._resolve_partition_joint_names(page_id, rid=rid, func=func)
                    params_str = (
                        ', params={"asset_cfg": SceneEntityCfg("robot", '
                        f"joint_names={joint_names!r})}}"
                    )
                    field_base = f"{func}_{page_id}"
                yield func, field_base, w, params_str, applies_to

    def _resolve_reward_item_value(self, nid: str, func_key: str, payload: Any = None) -> float:
        """Return the per-item ``{item_value}`` for a reward term (or 0.0).

        Decodes the optional ``value`` field (the canvas "Value" chip). ``None``
        / absent → 0.0, the brand-neutral "auto" sentinel the reward function
        interprets itself. Never reaches back into the Robot node.

        缺口③ — ``reward_terms`` is now PAGED, so the caller passes the term's
        ``payload`` directly (a term can appear on multiple pages with different
        values). When ``payload`` is None we fall back to a flat re-read (legacy
        / non-paged callers).
        """
        from application.compiler.term_payload import parse_item_value
        if payload is None:
            try:
                terms = self._parse_json_param(nid, "reward_terms")
            except Exception:                                     # noqa: BLE001
                terms = {}
            payload = terms.get(func_key) if isinstance(terms, dict) else None
        v = parse_item_value(payload)
        return float(v) if v is not None else 0.0

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
        # node. Find the node that carries it (and its per-item Value).
        bh_rid: Optional[str] = None
        for rid in rew_ids:
            try:
                rt = self._parse_json_param(rid, "reward_terms")
            except Exception:
                continue
            if isinstance(rt, dict) and "base_height" in rt:
                bh_rid = rid
                break
        terminations = self._parse_json_param(term_ids[0], "termination_conditions")
        if bh_rid is None or "base_height" not in terminations:
            return

        # The reward target is now the per-item "Value" chip (0.0 = auto →
        # the reward resolves to the asset's nominal spawn z at runtime).
        # When auto, we cannot statically compare against the termination
        # minimum, and the auto target (asset nominal) is by construction a
        # sane standing height ≥ any reasonable minimum — so skip the check.
        target = self._resolve_reward_item_value(bh_rid, "base_height")
        if target <= 0.0:
            return
        # base_height value may be a scalar (legacy) or the structured
        # ``{"weight": <threshold>, "grace_period_s": ...}`` form — extract
        # the threshold from either (``weight``/``threshold`` key).
        _bh_raw = terminations["base_height"]
        if isinstance(_bh_raw, dict):
            _bh_num = _bh_raw.get("weight", _bh_raw.get("threshold"))
        else:
            _bh_num = _bh_raw
        try:
            minimum = float(_bh_num)
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
                f"    (a) Rewards node base_height Value (target height) ≥ "
                f"{minimum} m (currently {target} m).\n"
                f"    (b) Terminations node base_height ≤ {target} m "
                f"(currently {minimum} m).\n"
                f"    (c) Remove base_height from the Rewards or "
                f"Terminations node if you only want one of the two."
            )

    def _get_body_ir_mapper(self):
        """Build the body IR mapper for the bound robot + active format.

        Stage 3 simplification: every site that needs a body name resolves
        through one lookup-based path:

          1. Resolve ``asset_id`` from the canvas Robot node → registry SKU
          2. Resolve the SKU via ``RobotAssetService`` → :class:`RobotAsset`
          3. ``BodyIRMapper.from_robot_asset(asset, active_format=fmt)``
             reads ``bodies_per_format[fmt]`` deterministically — no
             runtime MJCF parsing, no joint-name suffix-strip heuristics
          4. Replay per-format user overrides on top
                (``RobotAssetService.get_body_ir_overrides(sku, fmt=fmt)``)

        Empty mapper is returned when ``asset_id`` is missing or the
        robot has no body table for the active format — emit sites
        either raise (when they need a name) or fall back to a permissive
        regex (legacy default for illegal_contact).
        """
        if hasattr(self, "_body_ir_mapper_cache"):
            return self._body_ir_mapper_cache

        from application.training.body_ir import BodyIRMapper, apply_user_overrides

        _log = logging.getLogger(__name__)
        fmt = self._active_format or None

        robot_ids = self._find_by_type("robot")
        if not robot_ids:
            _log.warning(
                "[IR] body mapper source = EMPTY — no Robot node on canvas; "
                "compiler will emit fallback regexes"
            )
            mapper = BodyIRMapper([])
            self._body_ir_mapper_cache = mapper
            return mapper

        asset_id = self._p(robot_ids[0], "asset_id").strip()
        if not asset_id:
            _log.warning(
                "[IR] body mapper source = EMPTY — Robot node asset_id is empty"
            )
            mapper = BodyIRMapper([])
            self._body_ir_mapper_cache = mapper
            return mapper

        try:
            from application.service.robot_assets import (
                get_robot_asset_service,
            )
            from registers.robots import resolve_id as _resolve_robot_id
            sku = _resolve_robot_id(asset_id) or asset_id
            svc = get_robot_asset_service()
            asset = svc.resolve(sku)
            if asset is None:
                raise RuntimeError(
                    f"asset_id={asset_id!r} did not resolve (sku={sku!r})"
                )
            mapper = BodyIRMapper.from_robot_asset(asset, active_format=fmt)

            # Replay per-format user overrides (canvas Body Mapping table
            # writes through RobotAssetService.set_body_ir_overrides(sku, _, fmt)).
            overrides = svc.get_body_ir_overrides(sku, fmt=fmt) if fmt else svc.get_body_ir_overrides(sku)
            if overrides:
                apply_user_overrides(mapper, overrides)

            base_body = None
            base_role = mapper.get("base")
            if base_role and base_role.body:
                base_body = base_role.body
            _log.info(
                "[IR] body mapper: asset_id=%s sku=%s format=%s base=%s "
                "%d roles resolved",
                asset_id, sku, fmt or "(auto)", base_body,
                sum(1 for r in mapper.roles if r.resolved),
            )
            self._body_ir_mapper_cache = mapper
            return mapper
        except Exception as exc:
            _log.error(
                "[IR] body mapper resolve failed for asset_id=%s format=%s: %s",
                asset_id, fmt or "(auto)", exc,
            )

        # Last resort: empty mapper
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

    def _resolve_bodies(
        self, *role_or_categories: str, strict: bool = False
    ) -> List[str]:
        """Return all body names for the given roles/categories.

        Combines results from multiple categories (e.g.
        ``_resolve_bodies("base", "thighs")`` for illegal_contact).
        Deduplicates while preserving order. Wildcards (``.*`` / ``*``) are
        pass-through "all" tokens — the caller expands them, so they are never
        treated as unresolved.

        ``strict=True`` (used for the user-picked ``contact_body_names``) fails
        LOUD when an explicit, non-wildcard role/category resolves to NO body on
        this robot — such a role would otherwise be silently dropped (§1.8). The
        default (``strict=False``, illegal_contact's family-category path) keeps
        the lenient skip so its own ``fallback_re`` can take over.
        """
        try:
            mapper = self._get_body_ir_mapper()
        except Exception:
            if strict:
                raise
            return []
        out: List[str] = []
        seen: set = set()
        unresolved: List[str] = []
        for key in role_or_categories:
            if str(key) in (".*", "*"):
                continue  # wildcard = "all"; expanded by the caller, never unresolved
            bodies: List[str] = []
            # Try as a role id first
            role = mapper.get(key)
            if role and role.body:
                bodies = [str(role.body)]
            else:
                # Fall back to category
                try:
                    bodies = [str(b) for b in mapper.get_category_bodies(key) if b]
                except Exception:
                    bodies = []
            if not bodies:
                unresolved.append(str(key))
                continue
            for b in bodies:
                if b not in seen:
                    out.append(b)
                    seen.add(b)
        if strict and unresolved:
            raise CanvasConfigError(
                reason=(
                    f"actor_setting.contact_body_names references body role(s) "
                    f"{unresolved} that the selected robot does not expose — an "
                    f"unknown contact body would be silently dropped (§1.8). Pick "
                    f"only body roles the Robot node exposes, or '.*' for all."
                ),
            )
        return out

    def _reward_extra_params_from_node(self, nid: str, func_key: str, payload: Any = None) -> str:
        return self._reward_extra_params(nid, func_key, payload=payload)


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
