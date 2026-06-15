<!--
SPDX-FileCopyrightText: 2026 SU CHANG
SPDX-License-Identifier: Apache-2.0
-->

<!-- version: 1.1.5 -->

# v1.1.5

### Fixed: Joint-order & observation gaps in the automatic USD dump/alignment

The automatic asset dump could mis-order joints and drop observation entries while aligning a robot's articulation. A policy that trained cleanly in IsaacLab would then convulse or flip the moment it was deployed to MuJoCo, because the bundle's per-joint arrays followed the USD prim-authoring order instead of the **live PhysX articulation order the policy actually runs in**. The live order is now captured at train time and used as the single source of truth for the bundle, with a loud cross-check against the registry. New bundles deploy in the correct order; older bundles warn and ask for a re-export.

### Improved: Training Motion orchestration — explicit Weight Set

Per-command initial-sampling weights are now configured through a more explicit Weight Set flow. The values you enter are kept **verbatim** — the editor never auto-adjusts your numbers; it simply keeps the confirm button disabled until the weights add up to 100%.

### Changed: Canvas pan keys (community request)

You can now pan the canvas by holding **either the middle mouse button or the right mouse button**. The right button additionally hosts the context menu. The context menu is empty for now — convenience actions will be filled in over the coming releases.

### Added: Per-command curves in Mission Control

Mission Control's line charts can now break performance down **per command**, so you can inspect how each command (forward / turn / stand …) is doing individually instead of only in aggregate.
