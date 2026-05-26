# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""UnitPortAdapterBase — abstract base for controller translators.

Each concrete adapter converts a generic :class:`PolicyAction` message
(+ optional JointState feedback) into one or more ROS2 publications
specific to the robot's controller stack. Adapters are resolved at
bridge startup from ``/etc/unitport/adapter_registry.yaml`` keyed by
``controller_type``.

Subclasses must implement:

* :meth:`output_topics`   — list of (topic, msg_type) pairs to publish on
* :meth:`convert`         — translate a PolicyAction into ``{topic: msg}``
* :meth:`validate_mapping` — verify the incoming robot_mapping.json shape
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


class UnitPortAdapterBase:
    """Abstract controller adapter."""

    CONTROLLER_TYPE: str = ""

    def __init__(self, mapping: Dict[str, Any], profile: Dict[str, Any]) -> None:
        self.mapping = dict(mapping or {})
        self.profile = dict(profile or {})
        self.pd_kp: Dict[str, float] = {}
        self.pd_kd: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Subclass API
    # ------------------------------------------------------------------

    def output_topics(self) -> List[Tuple[str, str]]:
        """Return [(topic, msg_type)] this adapter publishes on."""
        raise NotImplementedError

    def convert(self, policy_action, joint_state=None) -> Dict[str, Any]:
        """Translate one PolicyAction → ``{topic: ros_message}`` dict."""
        raise NotImplementedError

    def validate_mapping(self) -> Tuple[bool, str]:
        """Cheap sanity-check of ``self.mapping``. Returns (ok, reason)."""
        if not self.mapping.get("joint_assignments"):
            return False, "mapping.joint_assignments is empty"
        return True, ""

    # ------------------------------------------------------------------
    # Live-param reload (called by the bridge when /unitport/params/updated fires)
    # ------------------------------------------------------------------

    def on_params_updated(self, name: str, scope: str, value: Any) -> None:
        """Replace in-memory PD gains. No-op for controllers that don't use PD."""
        if name.startswith("pd.Kp"):
            self._apply_pd(self.pd_kp, scope, value)
        elif name.startswith("pd.Kd"):
            self._apply_pd(self.pd_kd, scope, value)

    @staticmethod
    def _apply_pd(target: Dict[str, float], scope: str, value: Any) -> None:
        try:
            v = float(value) if not isinstance(value, (list, tuple)) else float(value[0])
        except (TypeError, ValueError):
            return
        if scope == "global":
            for k in list(target):
                target[k] = v
            target["_global"] = v
        elif scope.startswith("joint:"):
            target[scope.split(":", 1)[1]] = v
        elif scope.startswith("leg:"):
            target[scope] = v

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def enabled_joints(self) -> List[Dict[str, Any]]:
        return [j for j in (self.mapping.get("joint_assignments") or [])
                if j.get("enabled", True)]

    def reorder_to_controller(self, action_vec) -> List[float]:
        """Permute policy action vector into controller slot order."""
        if not action_vec:
            return []
        joints = self.enabled_joints()
        if not joints:
            return [float(v) for v in action_vec]

        max_idx = 0
        for j in joints:
            try:
                max_idx = max(max_idx, int(j.get("controller_input_index", 0)))
            except (TypeError, ValueError):
                pass
        out = [0.0] * (max_idx + 1)
        for j in joints:
            try:
                c_idx = int(j.get("controller_input_index", -1))
                a_idx = int(j.get("unitport_action_index", -1))
            except (TypeError, ValueError):
                continue
            if c_idx < 0 or a_idx < 0 or a_idx >= len(action_vec):
                continue
            out[c_idx] = float(action_vec[a_idx])
        return out
