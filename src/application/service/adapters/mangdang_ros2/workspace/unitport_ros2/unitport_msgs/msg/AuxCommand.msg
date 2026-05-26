# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

# Non-joint actuator command — LED, LCD, speaker, button-rumble, etc.
#
# Published by the host on /unitport/aux_command. The generic bridge node
# dispatches by `surface` to a per-brand AuxHandler (see
# unitport_bridge/aux_handlers/). Payload is carried as JSON so new
# surfaces can be added without changing the wire schema.

# One of: led | lcd | speaker | button_led | ...
string surface

# Target within the surface — e.g. "left" / "right" / "both" for LEDs,
# "main" for a single screen, "default" for a single speaker. Empty string
# means "all targets of this surface".
string target

# Arbitrary JSON payload — schema is surface-defined:
#   led:     {"r": 0..1, "g": 0..1, "b": 0..1, "pattern": "solid|blink|..."}
#   lcd:     {"text": "...", "line": 0, "color": "white|red|..."}
#   speaker: {"sound": "estop.wav"} or {"frequency_hz": 440, "duration_ms": 200}
string payload_json
