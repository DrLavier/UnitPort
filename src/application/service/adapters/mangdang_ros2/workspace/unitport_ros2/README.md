<!--
SPDX-FileCopyrightText: 2026 SU CHANG
SPDX-License-Identifier: Apache-2.0
-->

# UnitPort ROS2 Workspace

Robot-side ROS2 packages deployed by the UnitPort host at
`/opt/unitport/ros2_ws/src/unitport_ros2/`. Everything here is meant to run
**alongside** a vendor image — UnitPort is an ADDITIVE overlay, never an
exclusive replacement. The systemd services live under `unitport.target`,
which declares no `Conflicts` on `default.target` — so sshd, the vendor
ROS2 bringup, and any factory panels (e.g. the Mangdang Flask panel on
`:8080`) keep running continuously while UnitPort adds its bridge /
e-stop / state publisher on top.

- `sudo systemctl start unitport.target` — activate the UnitPort overlay;
  vendor stack stays up.
- `sudo systemctl stop unitport.target` — deactivate the UnitPort overlay;
  vendor stack never noticed.

Do NOT use `systemctl isolate unitport.target`. Isolate would force
`default.target` to stop, which on Ubuntu brings down sshd and pulls the
vendor services down with it — the host can no longer reach the robot.
The history of why this matters is recorded in
`memory/architecture_additive_not_exclusive.md` on the host repo.

## Layout

```
unitport_msgs/           # RobotProfile, RobotStateGeneric, SensorEntry, PolicyAction
unitport_bridge/         # generic_bridge_node + controller adapters
unitport_state_pub/      # /unitport/robot_state publisher
unitport_param_srv/      # SetParam service backing the Host-side live-tuning UI
unitport_estop/          # emergency stop node (heartbeat + watchdog)
unitport_bringup/        # launch files + config
systemd/                 # service + target unit files
scripts/                 # install_robot.sh, unitport_mode.sh, etc.
```

## Install

Run the install script on the robot **once** after UnitPort pushes the
workspace:

```bash
sudo /opt/unitport/ros2_ws/src/unitport_ros2/scripts/install_robot.sh
```

This:
1. `colcon build --symlink-install` inside `/opt/unitport/ros2_ws`
2. Installs systemd units into `/etc/systemd/system/`
3. Enables (but does not start) `unitport.target`

Switching modes:

```bash
sudo /opt/unitport/ros2_ws/src/unitport_ros2/scripts/unitport_mode.sh on
sudo /opt/unitport/ros2_ws/src/unitport_ros2/scripts/unitport_mode.sh off
```

## Contract

- `/etc/unitport/identity.yaml` — authoritative robot identity record.
  Written by `unitport-identity --probe` (run automatically at the end of
  `install_robot.sh` and during host-side Start Deploy). Consumed by every
  host that connects to the robot; the host caches a copy under
  `~/.unitport/robots/<robot_id>/identity.yaml`.
- `/etc/unitport/robot_profile.json` — host-side inspector cache pushed by
  `cache_and_push` on the first Connect after Upgrade. Read by
  `generic_state_pub_node` to know which sensors to republish.
- `/etc/unitport/robot_mapping.json` — bundle-pushed adapter mapping. NOT
  yet authored anywhere (bundle deployment is deferred). When absent the
  bridge runs in passive mode (alive but no policy_action subscription).
- `/etc/unitport/deploy_manifest.json` — bundle hash record. Matched against
  `robot_mapping.json` only when both are present; otherwise the bridge
  skips the hash check entirely.
- `/etc/unitport/adapter_registry.yaml` — `controller_type` → adapter class
  lookup the bridge uses once a mapping arrives.
- `/etc/unitport/tunable_params.yaml` — whitelist the param service enforces.
