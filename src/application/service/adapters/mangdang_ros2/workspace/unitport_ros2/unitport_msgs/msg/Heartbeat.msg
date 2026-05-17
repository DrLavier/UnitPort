# Robot-side liveness + lightweight state report. Published on
# /unitport/heartbeat at ~2 Hz by HeartbeatPublisher (PC side) and
# consumed by unitport_estop watchdog + PC control_plane probes.
#
# Replaces the v0.1 std_msgs/Bool placeholder. Watchdog arming is now
# presence-based (any message in the timeout window keeps estop happy);
# the payload fields let probes read state via DDS instead of SSH.

builtin_interfaces/Time timestamp_monotonic   # publisher monotonic clock at send time
string robot_state                             # "idle" | "active" | "fault" | "" (unknown)
string active_bundle_name                      # currently-loaded policy/bundle name; "" if none
float32 inference_rate_measured_hz             # measured policy inference rate; 0.0 if not running
string systemd_unit_state                      # cached `systemctl is-active unitport-minipupper.service`
