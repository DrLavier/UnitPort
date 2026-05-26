# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Custom adapter — placeholder for robots whose controller isn't in the
built-in registry. Republish the PolicyAction payload unchanged on a
single configurable topic so the user can wire up downstream processing."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from unitport_bridge.adapters.base import UnitPortAdapterBase


class CustomAdapter(UnitPortAdapterBase):
    CONTROLLER_TYPE = "custom"

    DEFAULT_TOPIC = "/unitport/policy_action_out"

    def output_topics(self) -> List[Tuple[str, str]]:
        return [(self._topic(), "unitport_msgs/msg/PolicyAction")]

    def convert(self, policy_action, joint_state=None) -> Dict[str, Any]:
        return {self._topic(): policy_action}

    def _topic(self) -> str:
        for b in self.mapping.get("topic_bindings") or []:
            topic = str(b.get("robot_topic", ""))
            if topic:
                return topic
        return self.DEFAULT_TOPIC
