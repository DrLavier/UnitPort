# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

# Generic sensor passthrough used by /unitport/robot_state.
# The bridge serialises the raw ROS message to JSON and stores it in `data_json`.

string label
string ros_type
string data_json
