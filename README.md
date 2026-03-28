# UnitPort

[![Website](https://img.shields.io/badge/Website-uniport.ai-blue)](https://uniport.ai)
[![Python](https://img.shields.io/badge/Python-3.11.9-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE.txt)
**Website:** [uniport.ai](https://uniport.ai)

> [!WARNING]
> **Early alpha, still pretty rough.**
> You can use it to mess with some simple motion training, but **I WILL NEVER SUGGEST YOU TO USE THIS ALPHA BUILD FOR SERIOUS RESEARCH & PRODUCTION**.

So anyway:
  UnitPort is a **FREE** and open-source tool for robot behavior authoring and motion training. 
  The main idea is simple: **Make your robot training simpler, whether you are a student, hobbyist, or engineer.**

---

**## ==================== WHAT IT DOES ==================== ##**

Right now we has two main workspaces:

- Mission Control -
   
    This is the node-based canvas for wiring robot tasks visually.
    
    It is still WIP. Controller integration and vendor SDK hookup are not fully done yet, might clean this up later. So yes, you can look around, but no, this part is not ready for full end-to-end use yet.


- Training Ground -

    This is the part that actually works today.

    It runs robot motion policy training on top of **MuJoCo**, with a GUI around the usual setup pain. Current support includes:

    - **SAC**, **PPO**, and **TD3** via [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
    - **NPY** motion import for imitation-style training
    - training parameter / observation / reward configuration from the UI
    - built-in robot assets, plus custom **MJCF** assets if you want to bring your own stuff
    - checkpoint export for later fine-tuning or deployment experiments

---

**## ==================== TO START ==================== ##**

UnitPort uses a local virtual environment at `.venv311` with Python 3.11.9.

**- Windows -**

Use `start.bat` to launch the app. That is the correct startup path on Windows.

```bat
install.bat
start.bat
```

**If `.venv311` is missing, `start.bat` will fail and tell you to run `install.bat` first.**


**- Linux -**

```bash
chmod +x install.sh start.sh
./install.sh
./start.sh
```

**- MacOS -**
Sry, will have to wait a little bit longer for this :(

---

**## ==================== WHAT U NEED TO KNOW ==================== ##**

**- Free & open source:**
    I personally firmly believe that resources should not be considered a barrier to the pursuit of knowledge.
    So we shall have no license fee, no paywall, no weird gated features. 
    Unless you're using agent API or other services that require payment from the upstream provider, ofc.
    For other rights: [![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE.txt)

**- Community Mujoco Support:**
    Major mujoco assets come from [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie). Check their pages for more info.

**- Build your own WorkSpace:**
    If you build your own mujoco assets, trained policy, models or anything, just plug them in.
    _But I might need to check if this functionality is complete enough..._

**- Visualized workflow:**
    Our interface operates similarly to ComfyUI & LEGO Mindstorms: 
    place nodes onto the canvas >> adjust the parameters >> connect them to the workflow.
    At the same time, I’ve retained the ability to edit workflows directly in **Python** within Mission Control. This stems from my personal habits developed while working with UE Blueprints, but I find it genuinely useful.

**- AI enhanced:**
    Yes we have it, but this feature **was not included in the Alpha release**. 
    It is currently in our internal version, in which we have built a hybrid RAG framework that integrates Ollama with APIs of more advanced agents. 
    It currently supports configuration and parameter tuning, reinforcement learning, workflow construction, and the generation of **.npy** reference action files.
    We are testing their stability and extending their application to more complex & composite tasks. 
  
---

**## ==================== Project Structure ==================== ##**

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

Some folders are more complete than others. A few areas still look like active construction **because they are**.

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
