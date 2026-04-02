---
tags:
  - reinforcement-learning
  - robotics
  - mujoco
  - locomotion
  - unitree
  - go2
  - quadruped
  - sac
  - stable-baselines3
  - strands-robots
library_name: stable-baselines3
model-index:
  - name: SAC-Unitree-Go2-MuJoCo
    results:
      - task:
          type: reinforcement-learning
          name: Quadruped Locomotion
        dataset:
          type: custom
          name: MuJoCo LocomotionEnv
        metrics:
          - type: mean_reward
            value: 4912
            name: Best Mean Reward
          - type: mean_distance
            value: 21.0
            name: Mean Forward Distance (m)
---

# SAC Unitree Go2 — MuJoCo Locomotion Policy

A **Soft Actor-Critic (SAC)** policy trained to make the Unitree Go2 quadruped **walk forward** in MuJoCo simulation.

Trained entirely on a MacBook (CPU, no GPU, no Isaac Gym) using [strands-robots](https://github.com/cagataycali/strands-gtc-nvidia).

## Results

| Metric | Value |
|--------|-------|
| Algorithm | SAC (Soft Actor-Critic) |
| Training steps | 1.74M |
| Training time | ~40 min (MacBook M-series, CPU) |
| Parallel envs | 8 |
| Network | MLP [256, 256] |
| Best reward | **4,912** |
| Mean distance | **21 meters** per episode |
| Forward velocity | ~1 m/s |
| Episode length | 1,000/1,000 (full episodes) |

## Demo Video

<video src="https://huggingface.co/cagataydev/sac-unitree-go2-mujoco/resolve/main/go2_walking.mp4" controls autoplay loop muted></video>

## Usage

```python
from stable_baselines3 import SAC

model = SAC.load("best/best_model")

# In a MuJoCo Go2 environment:
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
```

## Reward Function

```
reward = forward_vel × 5.0       # primary: move forward
       + alive_bonus × 1.0       # stay upright
       + upright_reward × 0.3    # orientation bonus
       - ctrl_cost × 0.001       # minimize energy
       - lateral_penalty × 0.3   # don't drift sideways
       - smoothness × 0.0001     # discourage jerky motion
```

## Why SAC > PPO

PPO (500K steps): Go2 learned to stand still. Reward = 615, distance = 0.02m.
SAC (1.74M steps): Go2 walks 21 meters. Reward = 4,912.

SAC's off-policy learning + entropy regularization explores more effectively in continuous action spaces.

## Files

- `best/best_model.zip` — Best checkpoint (highest eval reward)
- `checkpoints/` — All 100K-step checkpoints
- `logs/evaluations.npz` — Evaluation metrics over training
- `go2_walking.mp4` — Demo video

## Environment

- **Simulator**: MuJoCo (via mujoco-python)
- **Robot**: Unitree Go2 (12 DOF) from MuJoCo Menagerie
- **Observation**: joint positions, velocities, torso orientation, height (37-dim)
- **Action**: joint torques (12-dim, continuous)

## License

Apache-2.0
