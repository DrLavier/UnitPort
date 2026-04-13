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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


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


class IsaacLabConfigCompiler:
    """Compiles a UnitPort IL node graph into an Isaac Lab Python config file."""

    def __init__(self, graph: Dict[str, Any]) -> None:
        self._graph = graph
        self._nodes: Dict[str, Dict[str, Any]] = {}       # id → node dict
        self._params: Dict[str, Dict[str, str]] = {}      # id → parameters
        self._types: Dict[str, str] = {}                   # id → node_type
        self._edges: List[Dict[str, str]] = []
        self._downstream: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
        self._upstream: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
        self._parse()

    def _parse(self) -> None:
        for n in self._graph.get("nodes", []):
            nid = str(n["id"])
            self._nodes[nid] = n
            self._params[nid] = n.get("parameters", {})
            self._types[nid] = n.get("node_type", "")
        self._edges = self._graph.get("edges", [])
        for e in self._edges:
            src, dst = str(e["src_id"]), str(e["dst_id"])
            self._downstream[src].append((dst, e["src_slot"], e["dst_slot"]))
            self._upstream[dst].append((src, e["src_slot"], e["dst_slot"]))

    def _find_by_type(self, node_type: str) -> List[str]:
        return [nid for nid, nt in self._types.items() if nt == node_type]

    # ------------------------------------------------------------------
    # P2.1 gait helpers
    # ------------------------------------------------------------------

    def _gait_enabled(self) -> bool:
        """True when the Training Commands node has gait_enabled = true."""
        cmd_ids = self._find_by_type("training_commands")
        if not cmd_ids:
            return False
        raw = self._p(cmd_ids[0], "gait_enabled", "false")
        return str(raw).strip().lower() == "true"

    def _gait_range_tuple(self, key: str, default_lo: float, default_hi: float) -> str:
        """Parse a ``[lo, hi]`` JSON param into a Python tuple literal."""
        cmd_ids = self._find_by_type("training_commands")
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
        cmd_ids = self._find_by_type("training_commands")
        raw = self._p(cmd_ids[0], "gait_presets", "") if cmd_ids else ""
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
                from src.system.training.gait_presets import DEFAULT_PRESETS
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

    def _p(self, nid: str, key: str, default: str = "") -> str:
        return self._params.get(nid, {}).get(key, default)

    def _pf(self, nid: str, key: str, default: float = 0.0) -> float:
        try:
            return float(self._p(nid, key, str(default)))
        except (ValueError, TypeError):
            return default

    def _pi(self, nid: str, key: str, default: int = 0) -> int:
        try:
            return int(float(self._p(nid, key, str(default))))
        except (ValueError, TypeError):
            return default

    def _fan_in_order(self, dst_nid: str, dst_slot: str) -> List[str]:
        """Return source node IDs connected to a fan-in port, in edge order."""
        return [
            src for src, _ss, ds in self._upstream.get(dst_nid, [])
            if ds == dst_slot
        ]

    def _parse_json_param(self, nid: str, key: str) -> dict:
        """Parse a JSON-encoded parameter dict, e.g. reward_terms or obs_terms."""
        raw = self._p(nid, key, "{}")
        try:
            result = json.loads(raw)
            return result if isinstance(result, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_body_mapping(self) -> list:
        """Return a list of error strings if body mapping is incomplete.

        Returns an empty list when all required IR roles are resolved.
        Called by the training launcher as a pre-flight check.
        """
        mapper = self._get_body_ir_mapper()
        errors = []
        for role in mapper.unresolved_roles(required_only=True):
            m = mapper.get(role)
            errors.append(
                f"Body mapping: required role '{m.label}' ({role}) is unresolved. "
                f"Open the Policy Exporter node and assign a body."
            )
        return errors

    def compile(self) -> str:
        """Return the full Python source as a string."""
        # Pre-flight: validate body mapping
        body_errors = self.validate_body_mapping()
        if body_errors:
            import logging
            log = logging.getLogger(__name__)
            for e in body_errors:
                log.warning("[BodyIR] %s", e)

        lines = self._header()
        lines += self._custom_reward_funcs()
        # P2.1 — emit the Walk These Ways gait command term + helpers
        # only when the canvas has gait_enabled on Training Commands.
        # This keeps the generated config minimal for classic velocity
        # tracking runs and avoids importing CommandTerm unnecessarily.
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
        lines += self._export_cfg()
        return "\n".join(lines) + "\n"

    def compile_to_file(self, path: Optional[str] = None) -> Path:
        """Write compiled config to a file. Returns the file path."""
        source = self.compile()
        if path is None:
            fd, path = tempfile.mkstemp(suffix=".py", prefix="unitport_il_cfg_")
            import os
            os.close(fd)
        p = Path(path)
        p.write_text(source, encoding="utf-8")
        log.info("IsaacLabConfigCompiler wrote %d bytes to %s", len(source), p)
        return p

    # ------------------------------------------------------------------
    # Code generation sections
    # ------------------------------------------------------------------

    def _header(self) -> List[str]:
        return [
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

    # ------------------------------------------------------------------
    # Registry-driven reward compilation
    # ------------------------------------------------------------------
    # ALL reward metadata (function names, module routing, extra params,
    # inline implementations) lives in IL_REWARD_REGISTRY entries.
    # The compiler reads from there — no parallel hardcoded dicts.

    def _custom_reward_funcs(self) -> List[str]:
        """Emit inline implementations for custom reward functions.

        Reads ``il_inline`` from the registry for each reward the user
        selected. Only emits functions actually used in the current canvas.
        """
        from src.system.training.task_module_registry import IL_REWARD_REGISTRY

        rew_ids = self._find_by_type("rewards")
        if not rew_ids:
            return []

        reward_terms = self._parse_json_param(rew_ids[0], "reward_terms")
        # Collect unique inline implementations needed
        emitted_funcs: set = set()
        inline_blocks: list = []
        for func_key in reward_terms:
            item = IL_REWARD_REGISTRY.get(func_key)
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
            sim_dt = self._pf(pg_ids[0], "sim_dt", 0.005) or 0.005
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
        dt = self._pf(pid, "sim_dt", 0.005)
        # Gravity + GPU budgets — all on Play Ground Setting.
        gz = self._pf(pid, "gravity_z", -9.81)
        gpu_contact = self._pi(pid, "gpu_max_rigid_contact_count", 524288)
        gpu_patch = self._pi(pid, "gpu_max_rigid_patch_count", 81920)
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
            stype = self._p(pid, "scene_type", "flat").strip().lower()
            mu_s = self._pf(pid, "friction_static", 1.0)
            mu_d = self._pf(pid, "friction_dynamic", 0.8)
            restitution = self._pf(pid, "restitution", 0.0)
            lines.append(f"    terrain = TerrainImporterCfg(")
            lines.append(f'        prim_path="/World/ground",')
            if stype == "flat":
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
            kp = self._pf(aid, "stiffness", 25.0)
            kd = self._pf(aid, "damping", 0.5)
            eff = self._pf(aid, "effort_limit", 30.0)
            vel = self._pf(aid, "velocity_limit", 30.0)
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
            # num_envs / env_spacing are training-scale knobs — they live
            # on the training / scene side now, not on the Robot node.
            # Kept as compile-time constants until the next section migrates.
            n_envs = 4096
            spacing = 2.5

            # Asset_id is the single source of truth; the registry resolves
            # it to a concrete USD source (on-disk or Nucleus URL).
            asset_id = self._p(rid, "asset_id", "").strip()
            usd_path = ""

            if asset_id:
                from src.system.training.robot_assets import (
                    rescan, resolve_for_training, RobotAssetValidationError,
                )
                rescan()  # ensure menagerie + archives are registered
                try:
                    usd_path = str(resolve_for_training(asset_id))
                except RobotAssetValidationError as exc:
                    raise ValueError(
                        f"\n\n[UnitPort][Compiler] Cannot compile Isaac Lab "
                        f"config — Robot node asset_id={asset_id!r} has no "
                        f"USD source.\n\n"
                        f"  Reason: {exc}\n\n"
                        f"  Fix: pick a menagerie asset from the Robot "
                        f"node dropdown that ships a Nucleus USD URL "
                        f"(unitree_go2, unitree_a1, unitree_g1, unitree_h1, "
                        f"boston_dynamics_spot, …).\n\n"
                        f"  Archive assets like AMP_for_hardware_a1 / "
                        f"MetalHead_a1 only ship URDF/MJCF and CANNOT be "
                        f"used directly for Isaac Lab training without "
                        f"first converting their URDF to USD.\n"
                    ) from exc

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
                usd_path_expr = f'"{usd_path}"'
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
                px = self._pf(aid_node, "init_pos_x", 0.0)
                py = self._pf(aid_node, "init_pos_y", 0.0)
                pz = self._pf(aid_node, "init_pos_z", 0.4)
                contact = self._p(
                    aid_node, "contact_track_air_time", "true"
                ).lower() == "true"
            else:
                px, py, pz = 0.0, 0.0, 0.4
                contact = True

            # Initial joint pose: user-supplied JSON from ActorSetting,
            # else fall back to the canonical quadruped standing pose.
            init_joint_pos = (
                '{'
                '".*L_hip_joint": 0.1, '
                '".*R_hip_joint": -0.1, '
                '"F[LR]_thigh_joint": 0.8, '
                '"R[LR]_thigh_joint": 1.0, '
                '".*_calf_joint": -1.5'
                '}'
            )
            import json as _json
            # Priority: JointInitNode (delegate) > actor_setting fallback
            ji_ids = self._find_by_type("joint_init")
            raw_angles = ""
            if ji_ids:
                raw_angles = self._p(ji_ids[0], "angles", "").strip()
            if (not raw_angles or raw_angles == "{}") and actor_ids:
                raw_angles = self._p(actor_ids[0], "init_joint_angles", "").strip()
            if raw_angles and raw_angles != "{}":
                try:
                    parsed = _json.loads(raw_angles)
                    if isinstance(parsed, dict) and parsed:
                        pairs = ", ".join(
                            f'"{k}": {float(v)}' for k, v in parsed.items()
                        )
                        init_joint_pos = "{" + pairs + "}"
                except Exception:
                    pass

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
            hist = self._pi(aid_node, "contact_history_length", 3)
            air = self._p(aid_node, "contact_track_air_time", "true").lower() == "true"
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
            if self._p(pid, "height_scan_enabled", "false").lower() == "true":
                hs_source = pid
        elif terrain_ids:
            tid = terrain_ids[0]
            if self._p(tid, "enable_height_scan", "false").lower() == "true":
                hs_source = tid

        if hs_source is not None:
            if True:
                res = self._pf(hs_source, "scan_resolution", 0.1)
                sx = self._pf(hs_source, "scan_size_x", 1.6)
                sy = self._pf(hs_source, "scan_size_y", 1.0)
                # Resolve the actual base link name via the joints_mapping
                # IR — A1 has ``trunk``, Go2/Anymal have ``base``, etc.
                base_body = self._resolve_body("base", fallback="base")
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
                self._p(pg_ids[0], "height_scan_enabled", "false").lower() == "true"
            )

        for oid in obs_ids:
            gname = self._p(oid, "group_name", "policy")
            noise_std = self._pf(oid, "corruption_noise_std", 0.0)
            enable_noise = self._p(oid, "enable_corruption", "true").lower() == "true"

            # Parse obs_terms JSON: {"term_key": 1.0, ...}
            obs_terms = self._parse_json_param(oid, "obs_terms")

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

                if apply_noise:
                    lines.append(
                        f"        {term_type} = ObsTerm("
                        f"func=mdp.{func_name}, "
                        f"noise=Unoise(n_min=-{noise_std}, n_max={noise_std})"
                        f"{params_str})"
                    )
                else:
                    lines.append(
                        f"        {term_type} = ObsTerm("
                        f"func=mdp.{func_name}{params_str})"
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
            expr = self._p(aid, "action_joint_names_expr", ".*")
            scale = self._pf(aid, "action_scale", 0.25)
            use_offset = (
                self._p(aid, "action_use_default_offset", "true").lower() == "true"
            )
            lines.append(f"    joint_pos = mdp.JointPositionActionCfg(")
            lines.append(f'        asset_name="robot",')
            lines.append(f'        joint_names=["{expr}"],')
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
        cmd_ids = self._find_by_type("training_commands")
        if cmd_ids:
            cid = cmd_ids[0]
            # Gain-based lever coupling — derive training ranges from
            # the four gain fields (same numbers the deployment
            # CommandBus will use at runtime, so the sim2real envelope
            # is guaranteed by construction).
            gain_fwd = self._pf(cid, "gain_lin_forward", 2.0)
            gain_bwd = self._pf(cid, "gain_lin_backward", 1.0)
            gain_lat = self._pf(cid, "gain_lin_lateral", 0.5)
            gain_yaw = self._pf(cid, "gain_yaw", 1.5)
            lin_vel_x = f"({-abs(gain_bwd)}, {abs(gain_fwd)})"
            lin_vel_y = f"({-abs(gain_lat)}, {abs(gain_lat)})"
            ang_vel_z = f"({-abs(gain_yaw)}, {abs(gain_yaw)})"
            lines.append("    base_velocity = mdp.UniformVelocityCommandCfg(")
            lines.append('        asset_name="robot",')
            lines.append(f"        ranges=mdp.UniformVelocityCommandCfg.Ranges(")
            lines.append(f"            lin_vel_x={lin_vel_x},")
            lines.append(f"            lin_vel_y={lin_vel_y},")
            lines.append(f"            ang_vel_z={ang_vel_z},")
            lines.append(f"        ),")
            resample = self._p(cid, "resampling_time_range", "[10.0, 10.0]")
            rel_standing = self._pf(cid, "zero_command_probability", 0.0)
            lines.append(f"        resampling_time_range={resample},")
            lines.append(f"        rel_standing_envs={rel_standing},")
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
            lines += ["", ""]
            return lines

        rid = rew_ids[0]
        # Parse reward_terms JSON: {"func_key": weight, ...}
        reward_terms = self._parse_json_param(rid, "reward_terms")

        for func, weight in reward_terms.items():
            try:
                w = float(weight)
            except (ValueError, TypeError):
                w = 1.0
            mdp_module, mdp_func = self._reward_func_ref(func)
            params_str = self._reward_extra_params_from_node(rid, func)
            func_ref = f"{mdp_module}.{mdp_func}" if mdp_module else mdp_func
            lines.append(f"    {func} = RewTerm(func={func_ref}, weight={w}{params_str})")

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
            lines += ["", ""]
            return lines

        tid = term_ids[0]
        items = self._parse_json_param(tid, "termination_conditions")

        def _f(key: str, fallback: float) -> float:
            try:
                return float(items[key])
            except (KeyError, ValueError, TypeError):
                return fallback

        if "time_out" in items:
            max_ep = _f("time_out", 20.0)
            lines.append(f"    time_out = DoneTerm(func=mdp.time_out, time_out={max_ep})")

        if "illegal_contact" in items:
            thresh = _f("illegal_contact", 1.0)
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
            # Resolve via joints_mapping IR — base + thigh bodies are
            # the canonical "illegal contact" set for quadrupeds.  If
            # the mapper returns nothing (no Robot node body_mapping and no
            # asset_id), fall back to a permissive regex covering both
            # A1-style ("trunk") and Go2-style ("base") root naming.
            ic_bodies = self._resolve_bodies("base", "thighs")
            if ic_bodies:
                ic_expr = "[" + ", ".join(f'"{b}"' for b in ic_bodies) + "]"
            else:
                ic_expr = '["(base|trunk)", ".*thigh"]'
            lines.append(
                f"        params={{\"sensor_cfg\": SceneEntityCfg("
                f"\"contact_forces\", body_names={ic_expr}), "
                f"\"threshold\": {thresh}}},"
            )
            lines.append(f"    )")

        if "base_height" in items:
            min_h = _f("base_height", 0.2)
            lines.append(
                f"    base_height = DoneTerm(func=mdp.root_height_below_minimum, "
                f"params={{\"minimum_height\": {min_h}}})"
            )

        lines += ["", ""]
        return lines

    def _events_cfg(self) -> List[str]:
        lines = ["@configclass", "class EventCfg:", '    """Domain randomization events."""', ""]
        dr_ids = self._find_by_type("domain_rand")
        if dr_ids:
            did = dr_ids[0]
            if self._p(did, "enable_friction_rand", "true").lower() == "true":
                mode = self._p(did, "friction_mode", "reset")
                sf = self._p(did, "static_friction_range", "[0.5, 1.25]")
                df = self._p(did, "dynamic_friction_range", "[0.5, 1.25]")
                # mdp.randomize_rigid_body_material requires 5 mandatory
                # params: static_friction_range, dynamic_friction_range,
                # restitution_range, num_buckets, asset_cfg. The previous
                # emit only passed the first two and Isaac Lab refused
                # the term at EventManager prepare time. Defaults below
                # match Isaac Lab's stock Go2 velocity task config so
                # canvases that don't expose finer knobs still produce
                # a runnable env.
                rr = self._p(did, "restitution_range", "(0.0, 0.0)")
                nb = self._pi(did, "num_buckets", 64)
                lines.append(f"    physics_material = EventTerm(")
                lines.append(f'        func=mdp.randomize_rigid_body_material,')
                lines.append(f'        mode="{mode}",')
                lines.append(f"        params={{")
                lines.append(f'            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),')
                lines.append(f'            "static_friction_range": {sf},')
                lines.append(f'            "dynamic_friction_range": {df},')
                lines.append(f'            "restitution_range": {rr},')
                lines.append(f'            "num_buckets": {nb},')
                lines.append(f"        }},")
                lines.append(f"    )")
            if self._p(did, "enable_mass_rand", "true").lower() == "true":
                mode = self._p(did, "mass_mode", "reset")
                body = self._p(did, "mass_target_body", "base")
                mr = self._p(did, "mass_offset_range", "[-0.5, 0.5]")
                # mdp.randomize_rigid_body_mass requires 3 mandatory params:
                # asset_cfg, mass_distribution_params, operation. Operation
                # specifies how the random sample modifies the existing
                # mass: "add" (additive offset), "scale" (multiplicative),
                # or "abs" (replace). Canvas's "mass_offset_range" name
                # implies additive offsets, so default to "add". Future
                # work can expose mass_operation as a UI knob.
                op = self._p(did, "mass_operation", "add")

                # ── body name resolution ──
                # If the user picked a canonical role name ("base",
                # "trunk", "torso"), resolve through the joints_mapping
                # IR so the compiled config uses the correct USD link
                # name for the active robot family.  A custom literal
                # body name (e.g. "FR_hip") is passed through verbatim.
                _canonical_roles = {"base", "trunk", "torso", "pelvis", "body"}
                if body in _canonical_roles:
                    resolved = self._resolve_body("base", fallback=None)
                    body_expr = f'"{resolved}"' if resolved else '"(base|trunk)"'
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
            if self._p(did, "enable_external_push", "true").lower() == "true":
                mode = self._p(did, "push_mode", "interval")
                vx = self._p(did, "push_velocity_x_range", "[-0.5, 0.5]")
                vy = self._p(did, "push_velocity_y_range", "[-0.5, 0.5]")
                interval = self._p(did, "push_interval_range_s", "[10.0, 15.0]")
                lines.append(f"    push_robot = EventTerm(")
                lines.append(f'        func=mdp.push_by_setting_velocity,')
                lines.append(f'        mode="{mode}",')
                lines.append(f"        interval_range_s={interval},")
                lines.append(f"        params={{\"velocity_range\": {{\"x\": {vx}, \"y\": {vy}}}}},")
                lines.append(f"    )")
            if self._p(did, "enable_init_pose_rand", "true").lower() == "true":
                mode = self._p(did, "init_pose_mode", "reset")
                px = self._p(did, "init_pos_x_range", "[-0.5, 0.5]")
                py = self._p(did, "init_pos_y_range", "[-0.5, 0.5]")
                yaw = self._p(did, "init_yaw_range", "[-3.14, 3.14]")
                lines.append(f"    reset_base = EventTerm(")
                lines.append(f'        func=mdp.reset_root_state_uniform,')
                lines.append(f'        mode="{mode}",')
                lines.append(f"        params={{")
                lines.append(f'            "pose_range": {{"x": {px}, "y": {py}}},')
                lines.append(f'            "velocity_range": {{"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": {yaw}}},')
                lines.append(f"        }},")
                lines.append(f"    )")
            if self._p(did, "enable_joint_noise", "true").lower() == "true":
                mode = self._p(did, "joint_noise_mode", "reset")
                pos_noise = self._pf(did, "joint_pos_noise", 0.25)
                vel_noise = self._pf(did, "joint_vel_noise", 0.1)
                lines.append(f"    reset_joints = EventTerm(")
                lines.append(f'        func=mdp.reset_joints_by_offset,')
                lines.append(f'        mode="{mode}",')
                lines.append(f"        params={{")
                lines.append(f'            "position_range": (-{pos_noise}, {pos_noise}),')
                lines.append(f'            "velocity_range": (-{vel_noise}, {vel_noise}),')
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
            sim_dt = self._pf(pg_ids[0], "sim_dt", 0.005) or 0.005
        control_dt = 0.02   # 50 Hz — matches Isaac Lab Go2 default
        decimation = max(1, int(round(control_dt / sim_dt))) if sim_dt > 0 else 4

        # Episode length → from time_out termination value
        episode_length_s = 20.0
        term_ids = self._find_by_type("terminations")
        if term_ids:
            tcs = self._parse_json_param(term_ids[0], "termination_conditions")
            try:
                episode_length_s = float(tcs.get("time_out", 20.0))
            except (TypeError, ValueError):
                episode_length_s = 20.0

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
        # training_mode parameter. Legacy amp_ppo_trainer node_type is
        # also recognised for canvases saved before the merge.
        trainer_ids = self._find_by_type("il_ppo_trainer") or self._find_by_type("amp_ppo_trainer")
        has_amp_trainer = False
        if trainer_ids:
            _mode = str(self._p(trainer_ids[0], "training_mode", "PPO") or "PPO").strip()
            has_amp_trainer = (_mode == "AMP_PPO") or bool(self._find_by_type("amp_ppo_trainer"))
        algorithm_class = "AMP_PPO" if has_amp_trainer else "PPO"
        lines.append(f'    algorithm_class: str = "{algorithm_class}"')
        lines.append("")


        if trainer_ids:
            tid = trainer_ids[0]
            lines.append(f"    max_iterations = {self._pi(tid, 'max_iterations', 1500)}")
            lines.append(f"    num_steps_per_env = {self._pi(tid, 'num_steps_per_env', 24)}")
            lines.append(f"    num_learning_epochs = {self._pi(tid, 'num_learning_epochs', 5)}")
            lines.append(f"    num_minibatches = {self._pi(tid, 'num_minibatches', 4)}")
            lines.append(f"    learning_rate = {self._pf(tid, 'learning_rate', 0.001)}")
            lines.append(f"    discount_factor = {self._pf(tid, 'discount_factor', 0.99)}")
            lines.append(f"    gae_lambda = {self._pf(tid, 'gae_lambda', 0.95)}")
            lines.append(f"    clip_param = {self._pf(tid, 'clip_param', 0.2)}")
            lines.append(f"    entropy_coef = {self._pf(tid, 'entropy_coef', 0.01)}")
            lines.append(f"    value_loss_coef = {self._pf(tid, 'value_loss_coef', 1.0)}")
            lines.append(f"    max_grad_norm = {self._pf(tid, 'max_grad_norm', 1.0)}")
            schedule = self._p(tid, "schedule", "adaptive")
            lines.append(f'    schedule = "{schedule}"')
            if schedule == "adaptive":
                lines.append(f"    desired_kl = {self._pf(tid, 'desired_kl', 0.01)}")
            lines.append(f"    save_interval = {self._pi(tid, 'save_interval', 100)}")
            lines.append(f"    seed = {self._pi(tid, 'seed', 42)}")

        # Policy network
        net_ids = self._find_by_type("il_policy_network")
        if net_ids:
            nid = net_ids[0]
            lines.append("")
            lines.append("    # Actor-Critic Network")
            lines.append(f"    actor_hidden_dims = {self._p(nid, 'actor_hidden_dims', '[128, 64, 32]')}")
            lines.append(f"    critic_hidden_dims = {self._p(nid, 'critic_hidden_dims', '[128, 64, 32]')}")
            lines.append(f'    activation = "{self._p(nid, "activation", "elu")}"')
            # Canvas stores log_std (community convention, e.g. -1.0 → std ≈ 0.37).
            # The compiled config emits the direct std that rsl_rl / ActorCritic expects.
            import math as _math
            _log_std = self._pf(nid, 'init_noise_std', -1.0)
            lines.append(f"    init_noise_std = {_math.exp(_log_std):.6f}  # exp({_log_std})")
            # AMP discriminator hidden dims live on the policy network node
            # (same place the actor/critic dims live).  Only consumed by the
            # AMP_PPO launcher branch; harmless for the plain PPO path.
            lines.append(f"    disc_hidden_dims = {self._p(nid, 'disc_hidden_dims', '[1024, 512]')}")

        lines += ["", ""]
        return lines

    def _export_cfg(self) -> List[str]:
        """Emit export metadata as a dict constant (consumed by IsaacLabBackend).

        Reads from the unified ``export`` node (legacy
        ``il_policy_exporter`` has been deleted). Field names align
        with the ExportConfig dataclass — legacy ``export_onnx`` etc.
        are kept on the node for back-compat with bundle_exporter, so
        the compile output does not need to rename them.
        """
        exp_ids = self._find_by_type("export")
        if not exp_ids:
            return []
        eid = exp_ids[0]
        onnx = self._p(eid, "include_onnx", self._p(eid, "export_onnx", "true")).lower() == "true"
        ts = self._p(eid, "include_torchscript", self._p(eid, "export_torchscript", "false")).lower() == "true"
        name = self._p(eid, "bundle_name", "")
        auto_imp = self._p(eid, "auto_import", "true").lower() == "true"
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
    # All reward metadata is read from IL_REWARD_REGISTRY at compile time.
    # No hardcoded function maps or param dicts in this file.

    @staticmethod
    def _reward_func_ref(func_key: str) -> tuple:
        """Return (module_alias_or_empty, function_name) for a reward key.

        Reads ``il_module`` and ``il_func`` from the registry entry.
        """
        from src.system.training.task_module_registry import IL_REWARD_REGISTRY
        item = IL_REWARD_REGISTRY.get(func_key)
        if item is None or not item.il_func:
            return "mdp", func_key
        return item.il_module, item.il_func

    @staticmethod
    def _reward_func_name(func_key: str) -> str:
        from src.system.training.task_module_registry import IL_REWARD_REGISTRY
        item = IL_REWARD_REGISTRY.get(func_key)
        return item.il_func if (item and item.il_func) else func_key

    def _reward_extra_params(self, nid: str, func_key: str) -> str:
        """Build the ``params={...}`` kwarg string for a RewTerm.

        Reads the ``il_params`` template from the registry, substitutes
        node-level values (``{node_std}``, ``{node_threshold}``), then
        resolves any ``{ir:role}`` body-name placeholders via BodyIR.
        """
        from src.system.training.task_module_registry import IL_REWARD_REGISTRY
        item = IL_REWARD_REGISTRY.get(func_key)
        if item is None or not item.il_params:
            return ""
        # Phase 1: resolve {ir:role} body-name placeholders FIRST
        # (before .format(), because {ir:feet} is not a valid format key)
        params_str = item.il_params
        if "{ir:" in params_str:
            mapper = self._get_body_ir_mapper()
            from src.system.training.body_ir import resolve_body_params
            params_str = resolve_body_params(params_str, mapper)
        # Phase 2: substitute node-level values ({node_std}, {node_threshold})
        params_str = params_str.format(
            node_std=self._pf(nid, "std", 0.25),
            node_threshold=self._pf(nid, "threshold", 0.5),
        )
        return ", params={" + params_str + "}"

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

        from src.system.training.body_ir import BodyIRMapper

        _log = logging.getLogger(__name__)

        # 1. Robot node body_mapping param — AUTHORITATIVE SOURCE
        #    (absorbed from the deprecated JointsMappingNode in the 6-section refactor)
        mapping_ids = self._find_by_type("robot")
        if mapping_ids:
            import json
            raw = self._p(mapping_ids[0], "body_mapping", "")
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
            asset_id = self._p(robot_ids[0], "asset_id", "").strip()
            if asset_id:
                try:
                    from src.system.training.robot_assets import get_asset, rescan
                    rescan()
                    asset = get_asset(asset_id)
                    mapper = BodyIRMapper.from_robot_asset(asset)
                    _log.info(
                        "[IR] body mapper source = robot_asset "
                        "auto-resolve (asset_id=%s) — no joints_mapping "
                        "node on the canvas",
                        asset_id,
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

    def _resolve_body(self, role_or_category: str,
                      fallback: Optional[str] = None) -> Optional[str]:
        """Return the first body name matching a role or category.

        Tries, in order:
          1. Exact role_id match (``mapper.get("base").body``)
          2. Category bodies (``mapper.get_category_bodies("base")[0]``)
          3. The ``fallback`` string (may be None)

        Use this when a single body name is needed (e.g. scene prim
        path, mass target).  For lists (e.g. illegal_contact), use
        ``_resolve_bodies`` instead.
        """
        try:
            mapper = self._get_body_ir_mapper()
        except Exception:
            return fallback
        # Role id first (exact match like "base")
        role = mapper.get(role_or_category)
        if role and role.body:
            return str(role.body)
        # Fall back to category (e.g. "feet" → all feet bodies)
        try:
            bodies = mapper.get_category_bodies(role_or_category)
            if bodies:
                return str(bodies[0])
        except Exception:
            pass
        return fallback

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
