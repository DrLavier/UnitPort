# Changelog

All notable changes to UnitPort are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-05-20

First stable release. Public API surface for the SDK (`unitport_sdk`), the
canvas spec → IR pipeline, the bundle/manifest format, and the registers/
catalogs (robots, brands, nodes, IR roles) is now considered stable and
follows semver from this point forward.

### Added
- **Mission Control** — vendor matrix wired end-to-end for Unitree (Go2 family,
  G1, H1, H1-2) over WebRTC + DDS, Boston Dynamics Spot, and MangDang Mini
  Pupper over ROS 2 (CycloneDDS).
- **Live policy runtime** — loads any exported bundle and runs it against the
  connected robot or a MuJoCo preview window.
- **Gamepad / keyboard / command-bus** input layer for teleop and live policy
  override.
- **SSH bring-up** (paramiko) for robots that need an on-board service started
  before the bridge can talk to them.
- **Cloud sync (Phase 1)** — opt-in Supabase backend for login, profile, and
  selected artifact sync; RLS-namespaced per user; works fully offline if you
  skip auth.
- **In-app updater** — checks GitHub Releases against
  `system.ini[System].version`.
- **Isaac Lab 5.x** integration with EULA gating in the install wizard.
- **AMP-PPO / PPO-WALK** training backends on PhysX (Isaac Lab side).
- **Behavioral Cloning + IL-PPO** fine-tuning, plus AMP discriminator nodes
  consuming `.npy` motion clips.
- **Mass-matrix-adaptive PD** — joints tuned by `(ωn, ζ)` on
  `ActuatorPDNode`; the engine gain solvers derive engine-specific `kp/kd`
  at compile time.
- **Bundle export** — portable `manifest.yaml` + ONNX policy + deploy contract;
  self-contained and round-trips across machines.
- **French (FR) localisation** added alongside EN and ZH.
- **Apache License 2.0 NOTICE** with third-party attribution.
- **CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md** community files.

### Changed
- Bumped `system.ini[System].version` to `1.0.0`.
- Default `cmd_log_debug` flipped to `false` for shipped builds.
- Canonical registry catalogs migrated to per-format variant metadata
  (`bootstrap/migrate_canonical_*` scripts).

### Migration notes
- Pre-1.0 canvases authored with vendor / physical joint names should be
  migrated with `bootstrap/migrate_canvas_joint_names_to_ir.py`.
- Pre-v2 reward graphs should be migrated with
  `bootstrap/migrate_canvas_rewards_v2.py`.
- IL observation `obs_terms` defaults changed; migrate with
  `bootstrap/migrate_il_observation_obs_terms.py`.

[1.0.0]: https://github.com/DrLavier/UnitPort/releases/tag/v1.0.0
