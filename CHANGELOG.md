<!--
SPDX-FileCopyrightText: 2026 SU CHANG
SPDX-License-Identifier: Apache-2.0
-->

<!-- version: 1.1.0 -->

> **Breaking change.** Canvases and bundles produced by earlier versions are not compatible with v1.1.0. We recommend re-importing your templates and re-training. The PD solver and torque model have changed at the foundation; legacy bundles will be rejected at load time by `DeployContract` with a directive to re-train. This is intentional, not a bug.

---

## Physics & simulation backends

### Fixed: PD-formula asymmetry between IsaacLab (PhysX) and MuJoCo that caused sim2sim gain mismatch

The PhysX side was over-estimating joint stiffness — in the worst case the knee ended up roughly **65× stiffer than intended** — so a policy that trained successfully in IsaacLab would collapse the moment it was handed to MuJoCo for verification.

After the fix, both engines derive `kp` / `kd` from a **shared `m_eff`** (sampled from `mj_fullM` at the home keyframe) using a single common formula; the two gain arrays are now byte-identical element-by-element. The cross-engine torque check has been upgraded to compare actual `τ`, and is enforced at two gates:

- **Export-time:** strict threshold (`1e-3`)
- **Load-time:** loose threshold (`1%`, tolerates YAML round-trip noise)

### Added: First-class support for torque-lookup (remotized) actuators

The previous backend could not faithfully model non-direct-drive machines such as Boston Dynamics Spot. Spot's knee is driven through a four-bar linkage; its torque ceiling varies with knee angle (roughly **30 – 113 N·m**), whereas we previously pinned a single scalar `effort_limit` of 45 N·m. The result on heavy quadrupeds was a knee that could neither push off nor absorb landings.

This release adds:

- A `torque_lookup_v1` schema
- A reusable `TorqueLookupTable` module
- A `RemotizedPDActuatorCfg` routing path on the IsaacLab side
- Angle-aware clamping inside `PDController` on the MuJoCo side

All four paths share the **same lookup-table data**, which is embedded into `deploy_contract` so that the bundle remains self-contained.

This is an extension at the **mechanism-type** layer, not a Spot-specific adapter. Any future machine with a torque table (Ghost Robotics and so on) only needs to contribute a YAML data file — no new code.

### Adjusted: Default `(ωn, ζ)` for quadruped families

The v1.0 family defaults `(ωn = 35, ζ = 0.8)` were chosen on intuition and had never been calibrated against measurement. We back-solved against independent empirical data from Boston Dynamics, Unitree, and Anybotics, and confirmed that the realistic damping ratio for a quadruped hip joint is **ζ ≈ 0.30** — far from critical. The new default makes quadruped hips feel responsive instead of mushy.

Knee defaults retain a relatively stiff `ζ = 0.78`, in line with the BD series. Lightweight Unitree quadrupeds are stiffer than ideal under this default but remain trainable — a real engineering difference between vendors, which we now document as a known limitation of the framework rather than papering it over.

---

## Termination system

### Added: Grace (time-gate) mechanism on `TerminationNode`

Each termination condition now exposes an independent `grace_period_s` field, controlling for how many seconds after the start of an episode that condition is suppressed. This addresses a common class of "robot looks fine but the episode cuts immediately" symptoms: PhysX always produces a brief contact-force spike and pose disturbance at spawn-touchdown, which a naive termination treats as failure.

- Grace is **orthogonal** to the curriculum system; the two do not interact.
- `time_out` does **not** accept a grace value (it is itself time-based).
- Suggested starting values: `base_height` / `bad_orientation` → 1 – 2 s; `illegal_contact` → 1 s or more.

Note: the v1.1 schema default remains `0` for backward compatibility. The next release will raise the defaults so that new users no longer fall into the "why does my training die instantly" trap.

### Changed: `termination_conditions` now uses a structured dict instead of a flat scalar

This schema evolution exists to carry `grace_period_s`. Each condition is now a `{threshold, grace_period_s}` object. The legacy scalar form still loads, but emits a migration warning.

---

## Reward system

### Reworked: `RewardNode` UI

The assignment flow for common reward functions (`base_height` and friends) has been simplified, and the slider component has been removed. The result looks plainer than before, but rendering cost drops noticeably and the canvas feels smoother.

### Added: Reward × Termination composition

You can now define rewards that fire **on a termination condition without actually terminating** — e.g. "apply a negative reward when condition X is violated, but do not end the episode." This is useful for shaping behaviours of the form "avoid this when you can, but a stray contact is acceptable."

---

## IsaacLab multi-version registration

### Added: Local management of multiple IsaacLab installations

Beyond the IsaacLab that ships with the project, you can now register **any number** of external IsaacLab installations at arbitrary versions. Use cases:

- Running community forks
- Comparing training behaviour across IsaacLab versions
- Keeping a stable build and an experimental build side by side

The management entry lives in the **User panel**. Each canvas can independently select which IsaacLab instance it runs against.

---

## Known limitations (stated up front)

### Cross-vendor knee damping differences are an engineering choice, not a physical universal

As noted above, the BD series uses a stiff knee damping (`ζ = 0.78`) while the Unitree series uses a low one (`ζ = 0.39`). Neither is wrong — these are different vendor trade-offs on knee response characteristics. The family default follows BD; Unitree-class machines are stiffer than ideal under this default but remain trainable. Use a per-brand override on the `RobotNode` if exact matching is required.

### Scope of v1.1 validation

All v1.1 changes have been fully validated on the **IsaacLab training side** and the **MuJoCo deployment-verification side**. The hardware-deployment portion of the sim2real chain is still under development.
