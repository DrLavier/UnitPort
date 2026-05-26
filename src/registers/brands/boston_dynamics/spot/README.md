<!--
SPDX-FileCopyrightText: 2026 SU CHANG
SPDX-License-Identifier: Apache-2.0
-->

# Brand data — `registers/brands/<brand>/<model>/`

This tree holds **per-robot DATA** that the registry/compiler layers query.
`boston_dynamics/spot/` is the first inhabitant; it sets the precedent for
every future brand.

## The one rule: DATA ONLY, no logic

Files here are **pure data** (YAML/JSON). There is **no Python**, no
`__init__.py`, no executable code, no `if brand == "spot"` anywhere — not in
this directory and not anywhere else in the core (CLAUDE.md §1).

The mechanism is generic: a *remotized actuator* is any joint whose maximum
torque depends on its angle, described by a **torque lookup table**
(`torque_lookup` v1 schema, see
`application/physics/actuators/torque_lookup_v1.yaml`). Brands contribute the
**table** (data); they never contribute **behavior** (logic). The logic lives
once, engine-side:

- IsaacLab (training): `RemotizedPDActuatorCfg` (engine-native).
- MuJoCo (deploy): `PDController` angle-aware clamp (Python-resident).

Both read the **same** table. Adding a new remotized robot is therefore a
**data-only contribution**: drop a `*_torque_table.yaml` + a `manifest.yaml`
into `registers/brands/<brand>/<model>/`. No code change should be required.

## Files in this directory

| File | What it is |
|---|---|
| `knee_torque_table.yaml` | The angle → max-torque curve for Spot's knees, conforming to `torque_lookup` v1. Imported from IsaacLab (provenance in the file header). |

> **Import precision.** Table values are written at 6-decimal fixed precision
> (`%.6f`), which reproduces the IsaacLab source's 6-decimal literals
> **exactly — zero precision loss** (verified: max abs error 0.0 on both the
> angle and torque columns). Noted here for reproducibility, so a future
> diff against a re-import is expected to be byte-identical, not "close".
| `manifest.yaml` | Declares which joints are remotized (`joint_pattern`) and which table backs each. Keys the robot by `(brand, model)`, resolved to the canonical SKU at consume-time. |
| `README.md` | This file. |

## What this data does NOT contain

- **No PD gains.** `(kp, kd)` come from the mass-weighted solver
  (`registers/data/pd_groups_defaults.json` → `mujoco_gain_solver` /
  `physx_gain_solver`). Remotization replaces only the **scalar effort
  ceiling** with the angle-dependent table.
- **No transmission ratio.** IsaacLab's source table carried a
  `transmission_ratio` column; it is **inert** (PV-2 (b): never referenced in
  `RemotizedPDActuator.compute()`) and is intentionally dropped on import.
- **No SKU hash, no display name.** Identity is `(brand, model)`; everything
  else is derived through `registers.robots` (CLAUDE.md §7).

## Adding another remotized robot

1. Create `registers/brands/<brand>/<model>/`.
2. Add `<joint>_torque_table.yaml` (v1 schema; the loader validates it).
3. Add `manifest.yaml` with `robot: {brand, model}` and one `remotized_joints`
   entry per joint group (`joint_pattern` + `lookup_table`).
4. Done — no Python. If you find yourself needing code, the mechanism is
   wrong; fix the generic path, not this directory.
