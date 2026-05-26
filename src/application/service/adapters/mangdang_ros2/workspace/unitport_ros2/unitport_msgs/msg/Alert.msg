# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

# Published on /unitport/alerts by the bridge / estop / param_srv when
# something the host needs to know about happens. The host subscribes via
# BridgeClient and routes into RobotContext.emit_connection_failed.

std_msgs/Header header
string kind        # "mapping_hash_mismatch" | "estop_triggered" | "param_rejected" | ...
string severity    # "critical" | "warning" | "info"
string detail
