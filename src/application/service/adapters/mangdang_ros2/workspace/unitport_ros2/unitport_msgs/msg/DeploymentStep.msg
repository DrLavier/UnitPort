# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

# Progress reports for an in-flight UnitPort deployment (bundle install,
# parameter push, runtime restart). Published on
# /unitport/deployment/progress by the robot side, consumed by PC's
# control_plane.DeploymentProgressChannel subscriber.

string step_id            # stable identifier per step (e.g. "p4b.colcon_build")
string status             # "running" | "ok" | "error" | "skipped"
float32 progress_pct      # 0.0..100.0; -1.0 if indeterminate
string detail             # human-readable extra context (errors, paths, …)
builtin_interfaces/Time timestamp
