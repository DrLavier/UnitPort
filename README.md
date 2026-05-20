# UnitPort

[![Website](https://img.shields.io/badge/Website-uniport.ai-blue)](https://uniport.ai)
[![Python](https://img.shields.io/badge/Python-3.11.9-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE.txt)
[![Version](https://img.shields.io/badge/Version-1.0.0-brightgreen.svg)](src/config/system.ini)

**Website:** [uniport.ai](https://uniport.ai)

> [!NOTE]
> **First stable release (1.0.0).** The training pipeline (SB3 + MuJoCo) is stable and used internally for everyday work. Isaac Lab AMP / PPO‑WALK and the Mission Control real‑robot deploy paths are wired end‑to‑end but still marked experimental — please pin a version in any research you publish from this build.

UnitPort is a **FREE** and open‑source Studio for robot behaviour authoring, motion training and sim/real deployment. One canvas, many robots, two simulation engines, one click to a deployable policy bundle.

The guiding idea hasn't changed: **make robot training simpler — whether you are a student, hobbyist, or engineer.**

---

## What it does

UnitPort ships two cooperating workspaces around a shared canvas/IR pipeline.

### Training Ground — train policies on MuJoCo or Isaac Lab

This is the workspace most users start with. You wire a training graph on the canvas (robot · physics · observations · actions · rewards · terminations · algorithm), press Train, and watch loss / reward / FPS stream into the chart panel.

- **Two RL backends, picked per project:**
  - **Stable‑Baselines3** (stable): PPO, SAC, TD3 on MuJoCo.
  - **Isaac Lab** (experimental): AMP‑PPO, PPO‑WALK on PhysX. Requires a separately installed Isaac Lab.
- **Imitation learning** — Behavioral Cloning + IL‑PPO fine‑tuning, plus AMP discriminator nodes that consume `.npy` motion clips.
- **Mass‑matrix‑adaptive PD** — joints are tuned by `(ωn, ζ)` on the `ActuatorPDNode`; the engine gain solvers derive the engine‑specific `kp / kd` at compile time. No more hand‑tuning per simulator.
- **Bundled artifacts** — every export produces a portable `manifest.yaml` + ONNX policy + deploy contract. Bundles are self‑contained and round‑trip across machines.
- **Cross‑backend project compiler** — the canvas spec is lowered into a backend‑agnostic IR, then specialized for SB3 or Isaac Lab. The same project graph can train on either backend.

### Mission Control — author and deploy

Node‑based canvas for wiring real‑robot tasks: connect to the robot, stream telemetry, run a trained policy, drive joints from a gamepad / keyboard, replay a recorded clip.

- **Vendor adapters** for Unitree (Go2 family, WebRTC + DDS), Boston Dynamics Spot, and MangDang Mini Pupper (ROS 2).
- **Live policy runtime** loads any exported bundle and runs it against the connected robot or against a MuJoCo preview window.
- **Gamepad / keyboard / command‑bus input** so you can teleop or override the policy live.
- **SSH bring‑up** (paramiko) for robots that need an on‑board service started before the bridge can talk to them.

### Shared infrastructure

- **Visualized workflow** — like ComfyUI or LEGO Mindstorms: place nodes, set parameters, connect ports, run. Workflows can also be edited as Python directly from Mission Control.
- **Bilingual UI** — Simplified Chinese + English ship in the box. UI strings go through `tr()` / `i18n_bind`, so adding a locale is one folder under `localisation/`.
- **Cloud sync (Phase 1)** — opt‑in Supabase backend for login, profile, and selected artifact sync. Everything works fully offline if you skip auth.
- **In‑app updater** — checks GitHub Releases against `system.ini[System].version` (currently `1.0.0`).

---

## Supported robots (built‑in registry)

| Brand | Models |
|---|---|
| Unitree | Go2, Go2‑W, A1, B2, B2‑W, G1, H1, H1‑2 |
| Boston Dynamics | Spot |
| MangDang | Mini Pupper |
| Canonical templates | Generic Quadruped, Generic Humanoid |

Robots are registered through `registers/robots.py` + `registers/data/robots_canonical.json` and identified by an immutable SKU. Bringing a new robot online is an additive registry change — there is **no hardcoded brand string anywhere in the core pipeline.**

---

## Getting started

UnitPort uses a project‑local virtual environment at `.venv311` with **Python 3.11.9**. The launcher will create the venv on first run and re‑exec itself under it; you don't need to manage it manually.

### Windows

```bat
install.bat   :: first-time setup — creates .venv311, installs bootstrap deps
start.bat     :: launch (also runs install.bat automatically if .venv311 is missing)
```

`start.bat` paints the LoadingScreen first, then the in‑app **ProvisioningTask** installs the heavy dependencies (torch + CUDA wheel if an NVIDIA GPU is detected, loco‑mujoco, vendor SDKs) while streaming pip output into the log panel. First install typically takes 15–45 minutes depending on what optional components you select in the install wizard.

### Linux

```bash
chmod +x install.sh start.sh
./install.sh
./start.sh
```

### macOS

Not supported yet — sorry, this one is still on the roadmap.

### Reset

```bat
reset.bat   :: Windows — wipes .venv311 and runtime caches (keeps user data)
./reset.sh  :: Linux
```

User state (login tokens, exported bundles, project list) lives under `Paths.USER_CONFIG_DIR`, which is configured by the first‑launch wizard. `reset` does **not** touch it.

---

## System requirements

| Component | Minimum |
|---|---|
| OS | Windows 10 / 11, Ubuntu 22.04+ |
| Python | 3.11 (enforced — not 3.10, 3.12, or 3.13) |
| GPU | Any NVIDIA GPU for CUDA training; CPU‑only training works for SB3 on small policies |
| RAM | 8 GB for SB3 + MuJoCo, 16 GB+ recommended for Isaac Lab |
| Disk | ~10 GB after full provisioning (torch CUDA wheel dominates) |
| Isaac Lab | Optional, installed separately. Path is detected by the install wizard. |
| ROS 2 | Optional, only required for the MangDang adapter. The installer can fetch it. |

---

## Tech stack

| Component | Technology |
|---|---|
| GUI | PyQt6 |
| Simulation | MuJoCo ≥ 3.0, Isaac Lab (PhysX, optional) |
| RL | Stable‑Baselines3 (PPO / SAC / TD3), Isaac Lab AMP‑PPO / PPO‑WALK |
| Imitation Learning | In‑tree Behavioral Cloning + IL‑PPO + AMP discriminator |
| Policy export | ONNX (onnx, onnxruntime) |
| Robot SDKs | Unitree SDK2 (Go2 WebRTC + DDS), Boston Dynamics Spot SDK, ROS 2 (CycloneDDS) |
| Cloud | Supabase (auth + storage, optional) |
| Charts | pyqtgraph + TensorBoard |
| Language | Python 3.11.9 |

---

## Project structure

```text
UnitPort/
├── main.py                   # app entry — boots LoadingScreen, then MainWindow
├── start.bat | start.sh      # launcher (verifies .venv311, runs main.py)
├── install.bat | install.sh  # first-time setup (.venv311 + bootstrap deps)
├── reset.bat  | reset.sh     # wipe .venv311 and runtime caches
├── localisation.bat          # rebuild i18n .qm catalogs
├── requirements.txt
├── bootstrap/                # one-shot install / data-migration scripts
├── localisation/             # i18n source files (EN/, ZH/)
├── custom_mods/              # drop-in extension area
│   ├── canvas/               # canvas mods (e.g. isaac_lab)
│   ├── nodes/                # user-defined nodes (see example_node/)
│   ├── models/               # user-imported MJCF / URDF assets
│   ├── motions/              # user-imported .npy / clip files
│   └── runtime/              # custom runtime hooks
└── src/
    ├── config/system.ini     # SDK-shipped factory defaults (theme, fonts, system)
    ├── registers/            # central registries (nodes, robots, brands, IR, backends, ...)
    │   └── data/             # factory JSON catalogs
    ├── nodes/                # canvas node definitions (one folder per node type)
    ├── runtime/              # on-disk cache target (NOT a Python package)
    ├── unitport_sdk/         # in-tree PyQt6 infra SDK (frozen contract)
    └── application/
        ├── compiler/         # canvas spec → IR → lowering
        ├── engine/           # IR runtime contracts
        ├── physics/          # PD gain solvers (PhysX + MuJoCo)
        ├── training/         # SB3 + Isaac Lab + AMP + IL + motion + bundle export
        ├── service/          # auth, engines, input, models, projects, robot_assets, runtime
        │   ├── adapters/     # vendor SDK adapter layer (Unitree, Spot, MangDang)
        │   ├── brands/       # brand-package hooks (hot-pluggable)
        │   └── runtime/      # live policy + MuJoCo simulation runtime
        ├── ui/               # PyQt6 widgets (MainWindow, sidebar, canvas, dialogs, wizard)
        └── tools/            # background app tasks (provisioning, startup, post-setup)
```

Some folders are more complete than others. A few areas still look like active construction **because they are** — the 0.9 line stabilizes Training Ground; the 1.0 push closes out Mission Control's vendor matrix.

---

## Philosophy

- **Free and open source.** No license fee, no paywall, no gated features. Apache 2.0. The only money you ever spend is on optional upstream services (cloud LLM APIs, paid cloud compute) that you bring yourself.
- **Multi‑brand, no exceptions.** The core is brand‑agnostic. Anything that knows a specific robot lives in an adapter under `application/service/adapters/` or `application/service/brands/`. Adding a new vendor is an additive change to the registry, not a fork.
- **Fail loud, never silent.** When an input is missing or an operation fails, UnitPort raises with a clear message. We treat silent fallbacks (zero‑vector reads, "warn and continue", reverse‑guessing identifiers) as bugs, not features.
- **Portable artifacts.** Bundles, ONNX policies and deploy contracts are self‑contained. A bundle produced on your laptop must run on someone else's machine, or it is broken.
- **One source of truth for colors and fonts.** Everything the UI renders comes from `src/config/system.ini[Theme]` + `[Font]`. Reskinning is editing one INI.

---

## Community

- Website: [uniport.ai](https://uniport.ai)
- Repository: [DrLavier/UnitPort](https://github.com/DrLavier/UnitPort)
- Issues: [GitHub Issues](https://github.com/DrLavier/UnitPort/issues)
- Discussions: [GitHub Discussions](https://github.com/DrLavier/UnitPort/discussions)

If you find a robot whose URDF / MJCF you'd like upstreamed into the canonical registry, open an Issue with the asset and a body‑name → IR‑role mapping — that's the only thing the core can't auto‑derive (and shouldn't).

---

## Acknowledgements

- [google‑deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) — the bulk of our built‑in MJCF assets.
- [DLR‑RM/stable‑baselines3](https://github.com/DLR-RM/stable-baselines3) — the SB3 algorithm stack.
- [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab) — the PhysX side of the training pipeline.

---

## License

Licensed under the [Apache License 2.0](LICENSE.txt). See the LICENSE pre‑install panel (shown by `install.bat` / `install.sh` on first run) for the project‑specific notes around redistribution.
