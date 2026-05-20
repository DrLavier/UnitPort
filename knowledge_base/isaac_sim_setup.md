# NVIDIA Isaac Sim — Setup & Verification

> **Audience.** UnitPort users who want to train or review policies in
> the high-fidelity Isaac Sim / Isaac Lab path (USD scenes, RTX
> renderer) instead of the lightweight MuJoCo viewer.
>
> **Why this exists.** First-time onboarding to Isaac Sim is the single
> biggest source of "Launch Review does nothing" support tickets. The
> root cause is almost always one of three things: (a) Isaac Sim is not
> installed, (b) Isaac Lab is installed but UnitPort doesn't know its
> install root, or (c) the bundle was trained under SB3 but the user
> picked `isaac_sim` as the review backend. This page walks through all
> three.

---

## 1. What you actually need

Two things must be true for the Isaac Sim path to light up:

1. **Isaac Sim Kit is installed locally.** Either as part of Omniverse
   Launcher, a standalone Isaac Sim install, or via the Isaac Lab
   bundled installer — they all ship the same Kit binary.
2. **Isaac Lab is installed and UnitPort knows the path.** Isaac Lab is
   the layer UnitPort drives directly (the `isaaclab.bat` / `.sh`
   wrapper, the Python entrypoints, the actuator API). The Kit alone
   isn't enough.

There is no UnitPort-managed installer for either. They come from
NVIDIA. See [the official Isaac Lab installation guide][il-install]
for platform-specific steps.

[il-install]: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html

---

## 2. Register the Isaac Lab path with UnitPort

Once Isaac Lab is on disk, tell UnitPort where to find it:

1. Open the app. Sidebar → click the **User** rail button.
2. Scroll to the **Engines** section.
3. Find the **NVIDIA Isaac Lab** row.
   - If you see `Not registered — pick install root via the gear`, you
     need to do this step.
   - If you see a path + `(v0.54.3)` in green, you're done — skip to §3.
4. Click the gear button on that row.
5. Pick the **Isaac Lab root directory** — the folder that contains
   `isaaclab.bat` (Windows) or `isaaclab.sh` (Linux/Mac). On a typical
   Windows install this is `A:\Isaac\IsaacLab` or similar.

UnitPort will validate the path (probing for `isaaclab.bat/.sh` and
the underlying Kit Python) and either confirm the registration or
warn you that the path isn't a usable Isaac Lab root. If validation
fails, the most common cause is having selected the parent Omniverse
folder instead of the Isaac Lab subfolder.

### Behind the scenes

The registration writes to `<USER_CONFIG_DIR>/engines/isaac_lab.json`
(per-user state) and triggers
`registers.backends.refresh_engine_availability()`, which probes the
Isaac Lab install and writes the detection result to
`src/registers/data/backends_installed.json::engines.isaac_lab`. From
that point on, every part of UnitPort that needs Isaac Sim — Export
node `review_backend=isaac_sim`, Actor Setting `review_pose_engine=
isaac_sim`, IL training canvas — reads `is_available("isaac_lab")` and
decides whether to enable itself.

---

## 3. Verifying the path end-to-end

Before you try to launch a real review, you can sanity-check the wiring
without leaving the app:

### 3.1 Engines section shows green

Sidebar → User → Engines. Isaac Lab should read something like
`A:\Isaac\IsaacLab (v0.54.3)` in green. If it reads
`<path> (module not importable)` in red, the path was registered but
the Python under that root cannot import `isaaclab` — usually a
broken / partial install.

### 3.2 Export node picker shows Isaac Sim

Open any Export node on the canvas. The **Review backend** dropdown
should list `MuJoCo`, `Isaac Sim`, and `Newton`. Isaac Sim should
appear *enabled* (not greyed out). If it's greyed out with an
`(unavailable)` suffix, the registry hasn't picked up the install
yet — click the gear on the Isaac Lab row in the User panel once
more to force a refresh.

### 3.3 Actor Setting pose review uses Isaac Sim

Drag a Robot Node + Actor Setting onto a canvas. On Actor Setting,
change `review_pose_engine` from `mujoco` to `isaac_sim` and click
**▶ Review init pose in selected engine**. A second window will pop
up showing the robot in the Isaac Sim viewport at the configured
init pose. No bundle / training run is required for this — the pose
review path is bundle-free.

---

## 4. Common errors and what they mean

| Error message | Where surfaced | What it means |
|---|---|---|
| `bundle dir does not exist: ...` | CmdLog | `_resolve_bundle_dir` mapped `review_backend=isaac_sim` to `<project>/training/exported/isaac_lab/<name>/`, but you haven't trained anything under the IL canvas yet. Train first, then review. |
| `bundle ... was trained under [...], but review_backend='isaac_sim' needs a bundle trained under ['isaac_lab']` | CmdLog | You trained the bundle under SB3 (so it lives in `training/exported/sb3_mujoco/`) but selected Isaac Sim review. SB3-trained policies cannot run in the Isaac Sim path because the bundle's deploy_contract was emitted for MuJoCo. Switch `review_backend` to `mujoco`, or re-train on the IL canvas. |
| `Bundle '...' is SKU-locked to robot_sku='unitree.go2'; canvas Robot Config is currently bound to robot_sku='booster.t1'` | CmdLog | The bundle was trained for a different robot than the canvas is currently bound to. Switch the canvas Robot Node, or pick a bundle that matches. |
| `manifest.robot.sku is empty — re-export the bundle` | CmdLog | Legacy bundle pre-Phase-5 SKU contract. Re-export from the same project; the new bundle finalizer writes the SKU correctly. |
| `Isaac Lab 未安装或未在 Engine Settings 中注册路径` | QMessageBox (Actor Setting pose review) | The Actor Setting pose review is set to Isaac Sim but the registry says Isaac Lab is unavailable. Register the path per §2 or switch the engine back to `mujoco`. |
| `Robot {sku} has no joints_per_format['USD'] entries in the registry` | Subprocess stderr (pose review) | The robot's registry entry has MJCF joints declared but not USD. Run **Dump USD** on the Robot Node to populate the USD table, or use the MuJoCo pose review for this robot. |

---

## 5. The link chain, end-to-end

For when you need to debug deeper than the symptoms above:

```
Export node Launch Review (review_backend=isaac_sim)
   │
   ▼  scene.review_launch_requested signal
MainWindow._on_review_launch_requested (backend=="isaac_sim" branch)
   │ ├─ canvas SKU resolve (_resolve_canvas_robot_sku)
   │ ├─ bundle dir resolve (_resolve_bundle_dir, review_backend=isaac_sim
   │ │   → looks in training/exported/isaac_lab/<name>/; falls back to
   │ │   other backends with a diagnostic if missing)
   │ └─ submit IsaacSimReviewTask to TasksManager
   ▼
IsaacSimReviewTask.run() on SDK worker thread
   │ ├─ SKU hard-check: canvas SKU == manifest.robot.sku
   │ ├─ IsaacLabConfig.from_registers() → resolves Isaac Sim Python +
   │ │   isaaclab.bat/.sh wrapper from backends_installed.json
   │ └─ subprocess.Popen([isaac_python, il_review_launcher.py,
   │                     --bundle <dir>, --scene <id>, --max_play_steps N])
   ▼
il_review_launcher.py (Isaac Sim Kit subprocess)
   ├─ Load manifest.yaml + deploy_contract + policy.onnx
   ├─ Stage I: if pd_param present, re-derive PhysX gains via
   │   physx_gain_solver (kp = ωn² per joint group)
   ├─ Resolve SKU → spec → USD path via registers.robots
   ├─ Build minimal InteractiveScene (ground + USD articulation)
   └─ Replay loop:
       obs (base_ang_vel / projected_gravity / joint_pos / joint_vel /
            last_action / commands)
       → ONNX inference → target = action * scale + default_joint_pos
       → robot.set_joint_position_target → sim.step × decimation
```

The pose-review path (Actor Setting → ▶ Review init pose,
`review_pose_engine=isaac_sim`) follows the same shape but uses
`IsaacSimPoseReviewTask` + `il_pose_review_launcher.py` — no
bundle/manifest/policy, just `--sku`, `--base_pos`, `--joint_pos_by_ir`.

---

## 6. Files to read when something breaks

| Problem area | File |
|---|---|
| "Isaac Sim is greyed out in the picker" | `src/registers/review_backends.py` (availability mapping), `src/registers/backends.py::is_available` |
| "Bundle dir not found" | `src/application/ui/main_window.py::_resolve_bundle_dir` |
| "Subprocess fails to launch" | `src/application/service/runtime/simulation/isaac_sim/review_session.py` and `pose_review_session.py` |
| "Subprocess launches but Isaac Sim never shows up" | `src/application/training/isaac_lab/launcher/il_review_launcher.py` and `il_pose_review_launcher.py` — log lines `[isaac_sim_review] ...` / `[isaac_sim_pose_review] ...` in CmdLog |
| "Joint mapping is wrong" | `src/registers/data/robots_canonical.json` — check `joints_per_format['USD']` for the relevant SKU |
| "Engine path probe is wrong" | `src/registers/backends.py::_detect_isaac_lab` and the Engine row in `src/application/ui/sidebar_panels/user_panel.py` |
