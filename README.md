# UnitPort

[![Website](https://img.shields.io/badge/Website-uniport.ai-blue)](https://uniport.ai)
[![Python](https://img.shields.io/badge/Python-3.11.9-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE.txt)

> [!WARNING]
> **Early alpha, still pretty rough.**
> You can already use it to mess with robot motion training, but it is not ready for serious research or production.

UnitPort is a free and open-source visual tool for robot behavior authoring and motion training. The main idea is simple: let people try robotics workflows without drowning in RL boilerplate on day one.

**Website:** [uniport.ai](https://uniport.ai)

---

## What It Does

Right now UnitPort has two main workspaces.

### Mission

This is the node-based canvas for wiring robot tasks visually.

It is still WIP. Controller integration and vendor SDK hookup are not fully done yet, might clean this up later. So yes, you can look around, but no, this part is not ready for full end-to-end use yet.

### Training

This is the part that actually works today.

It runs robot motion policy training on top of **MuJoCo**, with a GUI around the usual setup pain. Current support includes:

- **SAC**, **PPO**, and **TD3** via [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- **NPY** motion import for imitation-style training
- training parameter / observation / reward configuration from the UI
- built-in robot assets, plus custom **MJCF** assets if you want to bring your own stuff
- checkpoint export for later fine-tuning or deployment experiments

---

## Highlights

### Free and open source

No license fee, no paywall, no weird gated features. It is under Apache 2.0.

### Uses `mujoco_menagerie`

Several built-in robot assets come from [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie), including things like Go2, H1, G1, and Spot. Those models save a lot of setup time, so credit where credit is due.

### Bring your own assets

You are not locked into the bundled models. If you already have your own **MJCF** components or training assets, you can plug them in.

### GUI first

The whole point is to keep common training workflows visible and editable from the UI. Reward shaping, motion library management, and run configuration are handled in panels instead of buried in scripts.

---

## Quick Start

UnitPort uses a local virtual environment at `.venv311` with Python 3.11.9.

### Windows

Use `start.bat` to launch the app. That is the correct startup path on Windows.

```bat
git clone https://github.com/DrLavier/UnitPort.git
cd UnitPort
install.bat
start.bat
```

If `.venv311` is missing, `start.bat` will fail and tell you to run `install.bat` first.

### Linux

```bash
git clone https://github.com/DrLavier/UnitPort.git
cd UnitPort
chmod +x install.sh start.sh
./install.sh
./start.sh
```

---

## Project Structure

```text
UnitPort/
|-- main.py                     # app entry
|-- bin/                        # UI entry and app wiring
|-- compiler/                   # IR compiler pipeline: parser / lowering / codegen
|-- nodes/
|   |-- sys_nodes/              # built-in nodes
|   `-- custom_nodes/           # user/community nodes
|-- system/
|   |-- behavior/               # behavior logic and motor protocol handling
|   |-- training/               # trainer, env, motion library, training flow
|   |-- policy/                 # inference and policy bundle loading
|   |-- runtime/                # workflow execution and safety-related runtime logic
|   |-- service/
|   |   `-- adapters/           # vendor SDK adapter layer
|   `-- brand_packages/         # brand-specific robot packages
|-- models/                     # robot integration layer
|-- config/                     # INI config files
|-- assets/                     # icons and UI resources
|-- localisation/               # i18n files
|-- custom_checkpoints/         # user-saved checkpoints
|-- custom_motions/             # user-provided NPY motions
|-- training_assets/            # MuJoCo assets
|-- training_checkpoints/       # auto-saved checkpoints
|-- training_workspaces/        # saved training workspace data
`-- workflows/                  # saved .unitport workflow files
```

Some folders are more complete than others. A few areas still look like active construction because they are.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| GUI | PySide6 |
| Physics / Simulation | MuJoCo 3.0+ |
| RL Training | Stable-Baselines3 (SAC, PPO, TD3) |
| Robot SDKs | Unitree SDK2, Boston Dynamics Spot SDK, adapter layer |
| Language | Python 3.11.9 |

---

## Community

- Website: [uniport.ai](https://uniport.ai)
- Repository: [DrLavier/UnitPort](https://github.com/DrLavier/UnitPort)
- Issues: [GitHub Issues](https://github.com/DrLavier/UnitPort/issues)
- Discussions: [GitHub Discussions](https://github.com/DrLavier/UnitPort/discussions)

---

## License

Licensed under the [Apache License 2.0](LICENSE.txt).
