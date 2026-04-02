# Loco-MuJoCo Agent Vector Index

This file is the human-readable companion to `AGENT_VECTOR_INDEX.jsonl`.

## Purpose

This index is optimized for vector retrieval by other agents that need to:

- find the correct entrypoint in `loco-mujoco/`
- locate robot environment definitions quickly
- identify where datasets, trajectories, rewards, goals, observations, and training examples live
- reduce repo-wide blind search during integration work

## Recommended Retrieval Strategy

Use this order:

1. Search `AGENT_VECTOR_INDEX.jsonl` by robot name plus task keyword.
2. Open the referenced file from the best-matching chunk.
3. If the task is environment creation, inspect factory files first.
4. If the task is robot wiring, inspect the concrete env class next.
5. If the task is dataset mismatch or motion import, inspect `trajectory/` and `smpl/`.
6. If the task is reward, goal, reset, terminal, domain randomization, or terrain behavior, inspect `core/`.

## High-Value Query Patterns

- `UnitreeH1 observation_spec actuation_spec`
- `UnitreeG1 imitation factory`
- `custom trajectory trajectoryhandler`
- `AMASS retargeting robot_confs`
- `GoalTrajMimic MimicReward`
- `GoalRandomRootVelocity TargetVelocityGoalReward`
- `domain randomization terrain tutorial`
- `MjxUnitreeH1 vectorized env`
- `Gymnasium wrapper loco-mujoco`

## Fast Path By Intent

If the intent is "create env":
- start at `loco_mujoco/task_factories/imitation_factory.py`
- or `loco_mujoco/task_factories/rl_factory.py`

If the intent is "add or adapt robot":
- start at `loco_mujoco/environments/humanoids/` or `loco_mujoco/environments/quadrupeds/`
- use an existing robot with similar morphology as template

If the intent is "import community motion/action dataset":
- start at `loco_mujoco/task_factories/dataset_confs.py`
- then `loco_mujoco/trajectory/handler.py`
- then `loco_mujoco/smpl/` only if human-motion retargeting is required

If the intent is "modify task behavior":
- start at `loco_mujoco/core/`

If the intent is "reuse training script":
- start at `examples/training_examples/`

## Notes

- The JSONL file is the primary machine-readable artifact.
- Each line is a standalone retrieval chunk with `id`, `type`, `path`, `tags`, `symbols`, `summary`, `query_hints`, and `integration_notes`.
- Factory APIs are the preferred modern entrypoints; deprecated `LocoEnv.make()` and `LocoEnv.generate()` should not be the default integration path.
