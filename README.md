# UnitPort Studio

[![Python](https://img.shields.io/badge/Python-3.11.9-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE.txt)
[![Version](https://img.shields.io/badge/Version-1.0.0-brightgreen.svg)](src/config/system.ini)
[![Agent Used](https://img.shields.io/badge/Agent%20Used-Claude%20Code-orange.svg)](https://code.claude.com/docs/en/overview)

[![English](https://img.shields.io/badge/English-%E2%9C%94-success)](README.md)
[![Français](https://img.shields.io/badge/Fran%C3%A7ais-%E2%9C%94-success)](README.md)
[![中文](https://img.shields.io/badge/%E4%B8%AD%E6%96%87-%E2%9C%94-success)](README.md)
*Deutsch, Italiano, Русский, Español, and 日本語 will be added in the next version iteration.*

> [!NOTE]
> **First stable release (1.0.0).** The training pipeline (SB3 + IsaacLab) is stable and used internally for everyday work. IsaacLab support is present, including PPO and AMP-PPO workflows, but its advanced functions remain experimental. At present, general quadruped robots (e.g. Go2) have the most complete implementation and workflow support. General biped robots currently support a basic motion training pipeline. Robotic manipulators, wheeled/mobile-base platforms, and drones are not yet supported.
> 
> **首个稳定版本 (1.0.0)**。训练流程（SB3 + IsaacLab）稳定可靠，已在内部用于日常工作。该版本支持 IsaacLab，包括 PPO 和 AMP-PPO 工作流程，但其高级功能仍处于实验阶段。目前，通用四足机器人（例如 Go2）的实现和工作流程支持最为完善。通用双足机器人目前支持基本的运动训练流程。机械臂、轮式/移动式平台和无人机暂不支持。

UnitPort is a **FREE** and open‑source Studio for robot behaviour authoring, motion training and sim/real deployment. One canvas, many robots, two simulation engines, one click to a deployable policy bundle.
UnitPort 是一个**免费**的社区开源机器人行为设计、运动训练和仿真/实战部署工作室。它支持一个画布、多个机器人、两个仿真引擎，只需单击一下即可生成可部署的策略包。

The guiding idea hasn't changed: **make robot training simpler — whether you are a student, hobbyist, or engineer.**
指导理念始终未变：**让机器人训练更简单——无论你是学生、业余爱好者还是工程师。**

> [!NOTE]
> The project supports Email/GitHub/Google login, and each account has 100MB of cloud storage space for easy cross device deployment.
> We use the [Supabase](https://supabase.com/security) to ensure your login and data security. Of course, logging in is not mandatory; you can start the project offline and use all development features except for cloud synchronization.
> 
> 项目支持Email / Github / Google 登录，每个登录的账号拥有100MB的云端存储空间以方便你的跨设备部署。我们使用Supabase框架来保证你的登录和数据安全。当然，登录并不是必须的，你完全可以以离线模式启动项目，使用除了云同步之外的完整开发功能；

---

## Robot completeness | 功能完整性

[![Quadruped](https://img.shields.io/badge/Quadruped-%E2%9C%94-success)](#)
[![Bipedal](https://img.shields.io/badge/Bipedal-%E2%9C%94%20(fine--tuning%20required)-yellow)](#)
[![Arm](https://img.shields.io/badge/Arm-%E2%9A%A0%20WIP-orange)](#)
[![Drones](https://img.shields.io/badge/Arm-%E2%9A%A0%20WIP-orange)](#)

The current release is **most complete for quadrupeds** — out of the box, the SB3 and IsaacLab paths both train walking policies end‑to‑end with stock parameters. **Bipedal humanoids run correctly** through the same pipeline, but reaching a stable gait expects some prior knowledge: users should be comfortable hand‑tuning the finer details (reward weights, termination thresholds, `ωn / ζ` per joint group, AMP clip weighting). The training framework as a whole is **built around base‑locomotion learning** — waypoint following, complex scene interaction, VLA policies, and base‑type robots like **fixed‑base manipulator arms** and **UAVs** are *not* yet inside the training framework. They will land in later releases.
当前版本**对四足机器人而言最为完善**——SB3 和 IsaacLab 路径均开箱即用，可使用默认参数完成端到端的行走策略训练。**双足人形机器人也能**通过相同的流程正确运行，但要达到稳定的步态需要一些先验知识：用户需要能够手动调整一些细节参数（奖励权重、终止阈值、每个关节组的 `ωn / ζ` 值、AMP 裁剪权重）。整个训练框架**围绕基础运动学习构建**——路径点跟踪、复杂场景交互、VLA 策略以及诸如**固定式机械臂**和**无人机**等基础型机器人*尚未*集成到训练框架中。这些功能将在后续版本中加入。

---

>[!NOTE]
>The project includes an easy deployment solution for IsaacLab / IsaacSim.
>You can:
> - Connect to an existing local IsaacLab setup;
> - Attach your own cloud-training Docker server;
> - Or install **IsaacLab (0.54.3)** and **IsaacSim (5.1.0.0)** directly through the built-in guided installer;
>Everything is automatically integrated into the project's virtual environment. No more wasting hours dealing with dependency hell and environment conflicts.
>该项目包含一个 IsaacLab / IsaacSim 的一键部署解决方案。
>你可以：
> - 连接到现有的本地 IsaacLab 环境;
> - 连接您自己的云端培训 Docker 服务器;
> - 或者通过内置的引导式安装程序直接安装 **IsaacLab (0.54.3)** 和 **IsaacSim (5.1.0.0)**;
>所有内容都会自动集成到项目的虚拟环境中。
>无需再浪费时间处理依赖关系和环境冲突。

---

## Getting started | 上手入门

UnitPort uses a project‑local virtual environment at `.venv311` with **Python 3.11.9**. The launcher will create the venv on first run and re‑exec itself under it; you don't need to manage it manually.
UnitPort 使用项目本地虚拟环境 `.venv311`，并运行 Python 3.11.9。启动器会在首次运行时创建该虚拟环境，并在该环境中重新执行自身，无需手动管理。

### Windows

```bat
install.bat   :: first-time setup — creates .venv311, installs bootstrap deps
start.bat     :: launch (also runs install.bat automatically if .venv311 is missing)
```

`start.bat` paints the LoadingScreen first, then the in‑app **ProvisioningTask** installs the heavy dependencies (torch + CUDA wheel if an NVIDIA GPU is detected, loco‑mujoco, vendor SDKs) while streaming pip output into the log panel. First install typically takes 15–45 minutes depending on what optional components you select in the install wizard.
`start.bat` 首先会显示加载屏幕，然后应用内的 **ProvisioningTask** 会安装重要的依赖项（如果检测到 NVIDIA GPU，则安装 torch 和 CUDA wheel，以及 loco-mujoco 和厂商 SDK），同时将 pip 的输出流式传输到日志面板。首次安装通常需要 **15 到 45 分钟**，具体时间取决于你在安装向导中选择的可选组件。

### Linux

```bash
chmod +x install.sh start.sh
./install.sh
./start.sh
```

### macOS

Not supported yet, WIP.

### Reset

```bat
reset.bat   :: Windows — wipes .venv311 and runtime caches (keeps user data)
./reset.sh  :: Linux
```

User state (login tokens, exported bundles, project list) lives under `Paths.USER_CONFIG_DIR`, which is configured by the first‑launch wizard. `reset` does **not** touch it.
用户状态（登录令牌、导出的包、项目列表）存储在你选择的 `Paths.USER_CONFIG_DIR` 下，该目录由首次启动向导配置。`reset` 命令**不会**修改它。

---

## What UnitPort Studio does | 功能概要

Design, edit, and debug workflows from a unified workspace.
Build robot training pipelines visually with nodes, while keeping full control to source codes when needed. 
在统一的工作空间中设计、编辑和调试工作流程。
使用节点以可视化的方式构建机器人训练流程，并在需要时完全掌控源代码。

### Training Ground | 训练场 — train policies on MuJoCo or IsaacLab

This is the workspace most users start with. You wire a training graph on the canvas (robot + physics + observations + actions + rewards + terminations + algorithm), press Train, and watch loss / reward / episode length stream into the chart panel.
这是大多数用户开始使用的工作区。您可以在画布上绘制训练图（机器人资产+物理+观测+动作+奖励+终止+算法），点击【训练】，然后在图表面板中查看Loss/Rewards/片段长度等数据。

- **Two RL backends, picked per project | 每个项目选择两个强化学习后端:**
  - **Stable‑Baselines3** (stable): PPO, SAC, TD3 on MuJoCo.
  - **IsaacLab** (beta): AMP‑PPO, PPO on PhysX.
- **Imitation learning**: Behavioral Cloning + IL‑PPO fine‑tuning, plus AMP discriminator nodes that consume `.npy` motion clips. **模仿学习**: 行为克隆 + IL-PPO 微调，以及使用 `.npy` 运动片段的 AMP 鉴别器节点
- **Mass‑matrix‑adaptive PD** — joints are tuned by `(ωn, ζ)` on the `ActuatorPDNode`; the engine gain solvers derive the engine‑specific `kp / kd` at compile time. No more hand‑tuning per simulator. See [Sim2sim calibration](#sim2sim-calibration) below. **质量矩阵自适应PD** — 关节通过`ActuatorPDNode`上的`(ωn, ζ)`进行调整；引擎增益求解器在编译时导出引擎特定的`kp / kd`。无需再为每个仿真器手动调整。请参阅下方的[Sim2sim校准](#sim2sim-calibration)。
- **Bundled artifacts** — every export produces a portable `manifest.yaml` + ONNX policy + deploy contract. Bundles are self‑contained and round‑trip across machines. **训练产物** — 每次导出都会生成一个可移植的 `manifest.yaml` 文件、一个 ONNX 策略和一个部署合约。这些打包文件是自包含的，并且可以在不同机器之间往返传输。

### Mission Control: Simulation and deploy | 任务控制: 模拟和部署

Node‑based canvas for wiring real‑robot tasks: connect to the robot, stream telemetry, run a trained policy, drive joints from a gamepad / keyboard, replay a recorded clip.
基于节点的画布，用于连接真实机器人任务：连接到机器人、传输遥测数据、运行训练好的策略、通过游戏手柄/键盘驱动关节、回放录制的片段。

- **Vendor adapters** for Unitree (Go2 family, WebRTC + DDS), Boston Dynamics Spot, and MangDang Mini Pupper (ROS 2). **Unitree（Go2 系列，WebRTC + DDS）、Boston Dynamics Spot 和 MangDang Mini Pupper（ROS 2）的供应商适配器**。
- **Live policy runtime** loads any exported bundle and runs it against the connected robot or against a MuJoCo preview window. **实时策略运行时** 加载任何导出的包，并针对连接的机器人或 MuJoCo 预览窗口运行它。
- **Gamepad / keyboard / command‑bus input** so you can teleop or override the policy live. **支持游戏手柄/键盘/命令总线输入**，你可以进行远程操作或实时覆盖策略。
- **Ethernet/SSH/USB/Webrtc Connection** (paramiko) for robots that need an on‑board service started before the bridge can talk to them. **Ethernet/SSH/USB/Webrtc 连接**（paramiko），用于在桥接器能够与机器人通信之前启动板载服务的机器人。


### Shared infrastructure | 基础框架

- **Visualized workflow**: like ComfyUI or LEGO Mindstorms: place nodes, set parameters, connect ports, run. Workflows can also be edited as Python directly from Mission Control. **可视化工作流程**——类似于 ComfyUI 或 LEGO Mindstorms：放置节点、设置参数、连接端口、运行。工作流程也可以直接在 Mission Control 中使用 Python 进行编辑。
- **Multilingual User Interface:** Supports multilingual frameworks. User interface strings are processed using `tr()` / `i18n_bind`, therefore, adding a locale requires a folder within the `localisation/` directory. **多语种用户界面**：支持多语言框架。用户界面字符串通过 `tr()` / `i18n_bind` 进行处理，因此添加语言环境需要位于 `localisation/` 目录下的一个文件夹内。
- **Cloud sync**: opt‑in Supabase backend for login, profile, and selected artifact sync. Everything works fully offline if you skip auth. **云同步**：用户可选择加入 Supabase 后端，用于登录，实现个人资料和选定工件的同步。如果跳过身份验证，所有功能均可完全离线运行。
- **Auto updater**: checks GitHub Releases against `system.ini[System].version`. **应用内更新程序**：检查 GitHub Releases 是否与 `system.ini[System].version` 一致。

---

## Sim2sim calibration | Sim2sim 校准 (IsaacLab + Mujoco)

The default flow is **IsaacLab for training, MuJoCo for verification.** Two engines, two different rigid‑body solvers, two different conventions for what "PD gains" actually mean, yet the same trained policy has to behave the same in both.
默认流程是**IsaacLab 用于训练，MuJoCo 用于验证**。两个引擎，两种不同的刚体求解器，两种不同的“PD增益”定义, 然而同一个训练好的策略在两者中必须表现相同。

UnitPort handles this with a **mass‑matrix‑adaptive** approach: the canvas exposes a single, engine‑agnostic control semantic per joint group — natural frequency `ωn` and damping ratio `ζ` on the `ActuatorPDNode`. At compile time, each backend's gain solver reads the link / actuator inertia from its own dynamics representation (IsaacLab from the `articulated‑body mass matrix`, MuJoCo from `qM`) and **derives the engine‑specific `kp / kd`** from the shared `(ωn, ζ)` target.
UnitPort 采用一种基于**质量矩阵自适应（mass-matrix-adaptive）** 的统一控制抽象。在 `ActuatorPDNode` 中，系统仅暴露与具体物理引擎解耦的控制语义：每个关节组的自然频率 `ωn` 与阻尼比 `ζ`。编译阶段，各后端增益求解器会从对应动力学模型中提取系统惯量信息（IsaacLab 使用 `articulated-body mass matrix`，MuJoCo 使用 `qM`），并**基于共享目标 (ωn, ζ)** 自动求解引擎特定的 `kp / kd` 参数。

```
                  shared control semantics
                  ┌────────────────────────┐
canvas ──────►    │   (ωn, ζ) per joint    │   ────► same closed-loop response
                  └───────────┬────────────┘
                              │ compile-time gain solve
                ┌─────────────┴─────────────┐
                ▼                           ▼
       IsaacLab gain solver         MuJoCo gain solver
       (reads PhysX mass matrix)     (reads qM)
                │                           │
                ▼                           ▼
           kp_isaac, kd_isaac          kp_mj, kd_mj
```

The result is that an `(ωn, ζ)` value tuned in IsaacLab transfers to MuJoCo with the same closed‑loop bandwidth and damping — no per‑engine re‑tuning. This is the contract that lets a policy trained in IsaacLab be **sim2sim verified** in MuJoCo before going to the real robot.
结果是，在 IsaacLab 中调优的 `(ωn, ζ)` 值可以以相同的闭环带宽和阻尼传递到 MuJoCo，无需针对每个引擎进行重新调优。正是这种机制使得在 IsaacLab 中训练的策略能够在应用到真实机器人之前，在 MuJoCo 中进行 **sim2sim 验证** 。

For a more detailed infomation of this method, please see [our website](https://unitport.ai/).
有关此方法的更多详细信息，请参阅[我们的网站](https://unitport.ai/)。

---

## Supported robots (built‑in registry) | 支持的机器人（内置注册表）

| Brand | Models |
|---|---|
| Unitree | Go2, Go2‑W, A1, B2, B2‑W, G1, H1, H1‑2 |
| Boston Dynamics | Spot |
| MangDang | Mini Pupper |
| Canonical templates | Generic Quadruped, Generic Humanoid |

Robots are registered through `registers/robots.py` + `registers/data/robots_canonical.json` and identified by an immutable SKU. Bringing a new robot online is an additive registry change — there is **no hardcoded brand string anywhere in the core pipeline.**
机器人通过 `registers/robots.py` 和 `registers/data/robots_canonical.json` 进行注册，并使用不可变的 SKU 进行标识。新机器人上线只需对注册表进行一次更改——**核心流程中没有任何硬编码的品牌字符串。**

---

## System requirements | 系统要求

| Component | Minimum |
|---|---|
| OS | Windows 10 / 11, Ubuntu 22.04+ |
| Python | 3.11 (enforced — not 3.10, 3.12, or 3.13) |
| GPU | Any NVIDIA GPU for CUDA training; CPU‑only training works for SB3 on small policies |
| RAM | 8 GB for SB3 + MuJoCo, 16 GB+ recommended for IsaacLab |
| Disk | ~10 GB after full provisioning (torch CUDA wheel dominates) |
| IsaacLab | Optional, installed separately. Path is detected by the install wizard. |
| ROS 2 | Optional, only required for the MangDang adapter. The installer can fetch it. |

---

## Tech stack | 技术简要

| Component | Technology |
|---|---|
| GUI | PyQt6 |
| Simulation | MuJoCo ≥ 3.0, IsaacLab (PhysX, optional) |
| RL | Stable‑Baselines3 (PPO / SAC / TD3), IsaacLab AMP‑PPO / PPO‑WALK |
| Imitation Learning | In‑tree Behavioral Cloning + IL‑PPO + AMP discriminator |
| Policy export | ONNX (onnx, onnxruntime) |
| Robot SDKs | Unitree SDK2 (Go2 WebRTC + DDS), Boston Dynamics Spot SDK, ROS 2 (CycloneDDS) |
| Cloud | Supabase (auth + storage, optional) |
| Charts | pyqtgraph + TensorBoard |
| Language | Python 3.11.9 |

---

## Project structure | 项目结构

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
        ├── training/         # SB3 + IsaacLab + AMP + IL + motion + bundle export
        ├── service/          # auth, engines, input, models, projects, robot_assets, runtime
        │   ├── adapters/     # vendor SDK adapter layer (Unitree, Spot, MangDang)
        │   ├── brands/       # brand-package hooks (hot-pluggable)
        │   └── runtime/      # live policy + MuJoCo simulation runtime
        ├── ui/               # PyQt6 widgets (MainWindow, sidebar, canvas, dialogs, wizard)
        └── tools/            # background app tasks (provisioning, startup, post-setup)
```

Some folders are more complete than others. A few areas still look like active construction **because they are**... 
有些文件夹比其他文件夹更完整。部分区域看起来仍然像在积极建设中，**因为它们确实如此**...
---

## Community | 社区支持

- Website: [uniport.ai](https://uniport.ai)
- Repository: [DrLavier/UnitPort](https://github.com/DrLavier/UnitPort)
- Issues: [GitHub Issues](https://github.com/DrLavier/UnitPort/issues)
- Discussions: [GitHub Discussions](https://github.com/DrLavier/UnitPort/discussions)

If you encounter any issues during daily use that we may have overlooked, please feel free to submit a report via Issue.
如果您在日常使用过程中遇到任何我们可能忽略的问题，请随时通过 Issue 提交报告。

---

## Acknowledgements | 鸣谢

- [google‑deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) — the bulk of our built‑in MJCF assets.
- [robfiras/loco‑mujoco](https://github.com/robfiras/loco-mujoco) — reference motion library powering the locomotion / AMP demos.
- [isaac‑sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) — GPU‑accelerated RL training backend (paired with NVIDIA IsaacSim).
- [unitreerobotics/unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python) — Unitree Go2 / B2 / H1 native SDK.
- [unitreerobotics/unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) — Unitree's MuJoCo simulation bridge.
- [tfoldi/go2‑webrtc](https://github.com/tfoldi/go2-webrtc) — WebRTC transport for the Unitree Go2.
- [boston‑dynamics/spot‑sdk](https://github.com/boston-dynamics/spot-sdk) — Boston Dynamics Spot SDK.
- [mangdangroboticsclub/mini_pupper_ros](https://github.com/mangdangroboticsclub/mini_pupper_ros) — MangDang Mini Pupper ROS2 stack.
- [MiRoboticsLab/cyberdog_ros2](https://github.com/MiRoboticsLab/cyberdog_ros2) — XiaoMi CyberDog ROS2 stack.
- [escontra/AMP_for_hardware](https://github.com/escontra/AMP_for_hardware) — Adversarial Motion Priors reference implementation for legged hardware.
- [inspirai/MetalHead](https://github.com/inspirai/MetalHead) — Inspir.AI quadruped locomotion reference.
- [Tencent‑RoboticsX/lifelike‑agility‑and‑play](https://github.com/Tencent-RoboticsX/lifelike-agility-and-play) — Tencent Robotics X lifelike agility & play project.
- [fan‑ziqi/rl_amp](https://github.com/fan-ziqi/rl_amp) — RL + AMP training reference.
- [abizovnuralem/go2_ros2_sdk](https://github.com/abizovnuralem/go2_ros2_sdk) — community ROS2 SDK for the Unitree Go2.
- [isaac‑sim/IsaacGymEnvs](https://github.com/isaac-sim/IsaacGymEnvs) — legacy Isaac Gym training environments (kept for porting reference).
- [ak1raljl/amp_go2](https://github.com/ak1raljl/amp_go2) — excellent community AMP recipe tuned for Go2 (required by the shipped `Go2_AMP` canvas template).
- [The work of Gabriel B. Margolis & Pulkit Agrawal](https://gmargo11.github.io/walk-these-ways/) — Classic and inspiring work that provided endless inspiration for our project design.

---

## License

Licensed under the [Apache License 2.0](LICENSE.txt). See the LICENSE pre‑install panel (shown by `install.bat` / `install.sh` on first run) for the project‑specific notes around redistribution.
