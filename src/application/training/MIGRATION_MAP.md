<!--
SPDX-FileCopyrightText: 2026 SU CHANG
SPDX-License-Identifier: Apache-2.0
-->

# MIGRATION_MAP — RELEASE 27 Nodes → TrainingSpec Field Contract

> **Purpose** — freeze the 1:N mapping from RELEASE's 27 fine-grained canvas
> nodes onto the unified `TrainingSpec` dataclass tree. Stage 3 lowering.py
> reads the canvas IR and emits this struct; Stage 7/8/9/10 backend launchers
> consume it. Editing this file changes the canvas → trainer contract — touch
> with care, and re-run `python -m py_compile application/training/training_spec.py`
> after every change.
>
> **Companion code** — `application/training/training_spec.py` (dataclasses),
> `application/training/spec_validator.py` (Stage 3 finalization), `application/compiler/lowering.py`
> (canvas → IR → spec).

## TL;DR

```
TrainingSpec
├── algorithm:    AlgorithmConfig          ← algorithm_config + (il_)ppo_trainer + amp_trainer
├── robot:        RobotSpec  (registers)   ← robot
├── actor:        ActorConfig              ← actor_setting + joint_init + init_pose + base_asset
├── obs_action:   ObsActionContract        ← obs_action_config + il_observation + il_policy_network
├── physics:      PhysicsConfig            ← physics_config + play_ground_setting (sim_dt overlap)
├── scene:        SceneConfig              ← play_ground_setting + env_assembler
├── task:         TaskConfig               ← task_config
├── rewards:      RewardConfig             ← rewards + multigated_reward + (amp_helper overrides)
├── terminations: TerminationConfig        ← terminations
├── motion:       MotionConfig             ← training_motion
├── il:           ImitationLearningConfig  ← (training_mode=="AMP_PPO" only)
│   ├── amp:      AMPConfig                ← amp_helper + discriminator + amp_trainer overrides
│   └── motion_ref:MotionRefConfig         ← training_motion (reference_motion_config slice)
├── domain_rand:  DomainRandConfig         ← domain_rand
├── stage_schedule: StageScheduleConfig    ← stage_switch + multigated_reward (curriculum gate)
├── eval:         EvalConfig               ← eval_config
├── vis:          VisCheckConfig           ← vis_check
└── export:       ExportConfig             ← export
```

Total source nodes: 27 (all builtin). Target dataclasses: 17 (1 root + 16 nested).

---

## Per-node mapping table

Columns:
- **Node id** — RELEASE manifest id
- **Layer** — A/B/C/D/IL (per manifest `layer`)
- **DEMO origin** — closest legacy ancestor (FYI; the contract is RELEASE-side)
- **Target field** — `TrainingSpec.<path>`
- **Notes** — gating, conditional fields, joint with other nodes

| Node id              | Layer | DEMO origin                | Target field                                | Notes |
|---|---|---|---|---|
| `robot`              | A    | `RobotNode`                | `spec.robot` (RobotSpec from registers)     | `asset_id` resolves via `registers.robots.get_robot_spec(asset_id)`; `target_height` overrides RobotSpec.capabilities. |
| `actor_setting`      | A    | `ActorSettingNode`         | `spec.actor` (§1 init_pos / §3 contacts / §4 actuator / §5 action / §6 curriculum) | Reads `robot_pipe`; merges `joint_init` port ⊕ local `init_joint_angles` (port wins). |
| `joint_init`         | A    | `JointInitNode`            | `spec.actor.joint_init` (Dict[str, float])  | Scalar passthrough for ActorSetting. |
| `init_pose`          | A    | `InitPoseNode`             | `spec.actor.init_pose` (mode/noise/RSI)     | Drives RSI sampler (Stage 5 motion loader consumes `rsi_prob`). |
| `base_asset`         | A    | `BaseAssetNode`            | `spec.algorithm.checkpoint`                  | `start_point` token determines `load_mode` ⊕ `checkpoint_file`. |
| `physics_config`     | A    | `PhysicsConfigNode`        | `spec.physics`                               | sim_dt / control_dt / episode_max_steps / action_type. Conflicts with PlayGround.sim_dt resolved by `spec_validator` (PG wins for IL backend, PhysicsConfig wins for SB3). |
| `play_ground_setting`| A    | `SceneConfigNode + ILTerrainConfigNode + ILSimulationConfigNode` | `spec.scene` + `spec.physics` overlap | scene_id / scene_type / arena / friction / rough-curriculum / height_scan / sim_dt / GPU caps. |
| `task_config`        | A    | `TaskConfigNode`           | `spec.task`                                  | velocity_tracking command (vx/vy/wz) + curriculum + truncation. |
| `terminations`       | A    | `TerminationsNode`         | `spec.terminations`                          | backend-keyed registry (`terminations` for sb3, `il_terminations` for isaac_lab). |
| `rewards`            | A    | `RewardsNode`              | `spec.rewards.terms`                         | backend-keyed; std/threshold are global priors. **Two-phase merge** with `multigated_reward`. |
| `multigated_reward`  | A    | `MultiGatedRewardNode`     | `spec.rewards` (stages) + `spec.stage_schedule` | Wraps two RewardsNode outputs into stage_0/stage_1; emits `total_steps` (covers algorithm_config.total_timesteps when wired). |
| `obs_action_config`  | A    | `ObsActionConfigNode`      | `spec.obs_action`                            | obs_components / clip / frame_stack / action_type/scale/clip. SB3 path uses this directly; IL path merges with il_observation. |
| `training_motion`    | B    | `TrainingCommandsNode + ReferenceMotionNode` | `spec.motion` + `spec.il.motion_ref` | Single source for command envelope + reference clip — guarantees train=reference=execute alignment. consumer_mode picks tracking/amp/both. |
| `domain_rand`        | A    | `DomainRandNode (SB3) + ILDomainRandNode` | `spec.domain_rand`                  | backend-gated (sb3 vs isaac_lab); rand_schedule is shared. |
| `il_observation`     | IL   | `ILObservationNode`        | `spec.obs_action.il_terms` + `spec.obs_action.corruption` | Only loaded for `algorithm.training_mode in {"PPO","AMP_PPO"}` AND backend=isaac_lab; SB3 ignores. |
| `il_policy_network`  | IL   | `ILPolicyNetworkNode`      | `spec.algorithm.policy_net`                  | actor/critic hidden dims + activation; required for IL trainers. |
| `algorithm_config`   | C    | `AlgorithmConfigNode`      | `spec.algorithm` (PPO/SAC/TD3 + backend)     | Stage 1 selector (`auto`/`isaac_lab`/`sb3_mujoco`). Maps SB3 hyperparams; AMP fields come from `discriminator` + `amp_trainer`. |
| `il_ppo_trainer`     | IL   | `ILPPOTrainerNode`         | `spec.algorithm.il_ppo` (when training_mode=="PPO") | RSL-RL hyperparam set (rsl_rl-style; Stage 8 SB3 port adapts). |
| `amp_trainer`        | IL   | `AMPTrainerNode`           | `spec.algorithm.il_ppo` + `spec.il.amp` (when training_mode=="AMP_PPO") | Same hyperparam set as il_ppo_trainer; alias defaults to AMP_PPO mode. |
| `discriminator`      | IL   | `DiscriminatorNode`        | `spec.il.amp.disc`                           | Required only in AMP_PPO; disc network + reward shaping + replay + obs. |
| `amp_helper`         | IL   | `AMPHelperNode`            | `spec.il.amp.helper_overrides`               | Optional pass-through that may inject reward-weight overrides. Applies AFTER `rewards` + `multigated_reward` resolution. |
| `env_assembler`      | B    | `EnvAssemblerNode`         | `spec.env` (SB3 VecEnv config)               | Aggregator for SB3 path: VecEnv type/n_envs + wrapper stack. Skipped on IL path. |
| `train`              | C    | `TrainNode`                | (sink — no fields)                            | Pure passthrough; just gates SB3 train submission. Stage 4 execute() routes spec to SB3Task. |
| `stage_switch`       | IL   | `StageSwitchNode`          | `spec.stage_schedule.stages`                  | Up to 6 train_pipe inputs; `stages_config` provides per-stage budget. |
| `eval_config`        | D    | `EvalConfigNode`           | `spec.eval`                                  | Optional; Stage 12 acceptance optional. |
| `vis_check`          | D    | `VisCheckNode`             | `spec.vis`                                   | Optional MuJoCo viewer milestones; Stage 7 SB3 trainer can inject. |
| `export`             | D    | `ExportNode`               | `spec.export`                                | bundle_name / version / targets / review_backend. **Drives BundleExporter (already shipped).** |

---

## Cross-cutting rules

### R1 — Algorithm-mode gating

| `algorithm_config.algorithm` | Required nodes | Optional nodes |
|---|---|---|
| `PPO` (sb3)                  | algorithm_config, robot, actor_setting, physics_config, task_config, rewards, terminations, obs_action_config, env_assembler, train | base_asset, eval_config, export, domain_rand, vis_check |
| `SAC`/`TD3` (sb3)            | same as PPO + (sb3 PER fields)                              | same |
| `il_ppo_trainer.training_mode == "PPO"` (isaac_lab) | il_ppo_trainer, robot, actor_setting, physics_config, task_config, rewards, terminations, il_observation, il_policy_network, play_ground_setting | base_asset, eval_config, export, domain_rand, vis_check, training_motion |
| `il_ppo_trainer.training_mode == "AMP_PPO"` | above + training_motion + discriminator | amp_helper |

`spec_validator` enforces missing required ports as `MissingRequiredPort` errors; missing required nodes as `MissingRequiredNode`.

### R2 — Backend-keyed registries (rewards / terminations / domain_rand)

These three nodes carry a `backend` enum that toggles which registry the
`registry_module` widget queries:

```
rewards.backend == "sb3"        → REWARD_REGISTRY      → spec.rewards.terms (sb3 keys)
rewards.backend == "isaac_lab"  → IL_REWARD_REGISTRY   → spec.rewards.terms (il keys)
```

Lowering passes both keysets through; `spec_validator` flags cross-backend
contamination (e.g. SB3 algorithm_config but rewards.backend == isaac_lab).

### R3 — Two-stage rewards merge (rewards × multigated_reward)

`MultiGatedRewardNode` takes **two** `RewardsNode` outputs as `stage_0` and
`stage_1`. Stage 3 lowering must:

1. resolve each upstream `rewards` independently (per backend)
2. emit `spec.rewards.stages = [stage0_terms, stage1_terms]`
3. emit `spec.stage_schedule.stages = [{ ratio, behavior, blend_steps, ... }, ...]`
4. emit `spec.rewards.terms = stage_0_terms` (active stage at training start)

When multigated_reward is **not** wired, `spec.rewards.terms` comes directly
from the single `rewards` node; `stage_schedule.stages = []`.

### R4 — sim_dt conflict (physics_config × play_ground_setting)

Both nodes carry `sim_dt`. Resolution rule:

- `algorithm_config.backend in {"isaac_lab"}`: `play_ground_setting.sim_dt` wins
- `algorithm_config.backend in {"sb3_mujoco","auto"}`: `physics_config.sim_dt` wins

`spec_validator` warns (not errors) on mismatch; lowering picks per the rule.

### R5 — total_timesteps wiring (multigated_reward → algorithm_config)

When `multigated_reward.total_steps` is wired into `algorithm_config.total_steps`
(hidden port; merged in NodeRow UI), it overrides
`algorithm_config.total_timesteps`. Lowering reads the **wired value** when
present, else the parameter.

### R6 — Discriminator reference-motion routing

```
training_motion ──reference_motion_config──► discriminator ──discriminator_config──► il_ppo_trainer
```

The discriminator ALWAYS sits between training_motion and the trainer in
AMP_PPO mode. Lowering hoists discriminator's `reference_motion_config` out
of the trainer's input dict and attaches it to `spec.il.motion_ref`.

### R7 — Conditional inputs (manifest meta)

Several nodes declare `conditional_on` on inputs (Stage 0 added port-side
support). When the gate is false, the port is hidden in the canvas AND
ignored by lowering even if a stale connection exists.

| Node                | Port                          | Gate                                   |
|---|---|---|
| `il_ppo_trainer`    | reference_motion_config       | `training_mode == "AMP_PPO"` |
| `il_ppo_trainer`    | discriminator_config          | `training_mode == "AMP_PPO"` |
| `amp_trainer`       | (same two)                    | (same — default mode is AMP_PPO)       |

(Stage 4 may extend this list to other nodes; update this table when it does.)

---

## Field-level notes (selected nodes only — the rest are mechanical)

### `algorithm_config.backend` (Stage 1)

`auto` / `isaac_lab` / `sb3_mujoco`. Lowering passes the literal through;
`select_backend(spec.algorithm.backend)` resolves at submit time
(`application.training.backend`). When `algorithm` is `SAC`/`TD3` and `backend`
resolves to `isaac_lab`, validator raises `BackendAlgorithmMismatch` (Isaac Lab
is PPO-only in Stage 1; SAC/TD3 land via Stage 7 SB3 trainer).

### `il_ppo_trainer` vs `algorithm_config` co-existence

Both nodes carry hyperparams; they do NOT mirror. `algorithm_config` covers
SB3 PPO/SAC/TD3 + AMP backend selector. `il_ppo_trainer` covers RSL-RL-style
PPO + AMP_PPO. Lowering picks ONE based on whether the canvas wires
`il_ppo_trainer` (IL path) or `algorithm_config + train` (SB3 path) downstream.
`spec_validator` rejects canvases that wire BOTH simultaneously
(`AmbiguousAlgorithmSource`).

### `actor_setting.action_joint_names_expr` (regex / list)

Resolved against `RobotSpec.joint_order`; missing names become
`UnmappedActionJoints` errors at validate time.

### `training_motion.training_items` (per-item dict)

Each item carries `enabled / speed / clip / advanced`. Lowering filters
disabled items, projects to `spec.motion.task_items` (ordered list), and
indexes clips into `spec.il.motion_ref.clip_paths` keyed by item id.

---

## What lowering.py needs to do (Stage 3.B)

`canvas_to_ir(workflow_dict) -> WorkflowIR` is currently passthrough
(`application/compiler/lowering.py:189-226`). Stage 3.B will:

1. **Topology pass** — graph-level checks (cycles, unconnected required ports
   per `R1`, dangling outputs).
2. **Manifest validation** — every node's params/ports match its manifest
   schema (type / range / choices). Reuse `nodes._registry` + `ParamSpec`
   accessors.
3. **Family classification** — decide algorithm path (sb3 vs il, ppo vs amp_ppo)
   from algorithm_config + il_ppo_trainer presence.
4. **Spec assembly** — walk nodes per `MIGRATION_MAP.md` table above; emit
   `TrainingSpec`.
5. **Cross-cutting checks** — `R1`–`R7` (validator hooks).

Output: `(WorkflowIR, TrainingSpec)` pair. WorkflowIR keeps the canvas-
debug-friendly node graph; TrainingSpec is the runtime contract.

`canvas_to_ir` MAY return `WorkflowIR` only when `spec` resolution fails
(soft fail) — but Stage 7+ submit paths always require the spec, so the soft-
fail mode exists only for canvas-time linting.

---

## Authoring rules

- **No new fields here without a corresponding manifest field.** If you need
  a new spec field that no node sources, add the manifest entry first, run
  `register_nodes()`, then update this map.
- **No silent defaults.** Every dataclass field has a default in the manifest;
  the spec dataclass mirrors that default. If the manifest changes, this map
  changes.
- **Backwards compat on rename:** if you rename a manifest key, add it to
  `_LEGACY_KEY_MAP` in `training_spec.py` and KEEP both spellings working
  for two minor versions.
- **No brand strings.** This file lives under `application/training/` —
  same red line as the rest of core. Robot specifics come from
  `registers.robots.get_robot_spec()` only.
