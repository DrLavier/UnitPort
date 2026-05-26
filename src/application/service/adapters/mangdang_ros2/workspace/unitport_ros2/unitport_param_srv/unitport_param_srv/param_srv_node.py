# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""param_srv_node — whitelist-gated live parameter update service.

Serves unitport_msgs/srv/SetParam. For each accepted request, publishes the
new value on /unitport/params/updated so the bridge adapter (and any other
consumer) can reload without a restart. Also writes to
/etc/unitport/live_params.json for persistence across reboots.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import rclpy
import yaml
from rclpy.node import Node

from std_msgs.msg import String
from unitport_msgs.srv import SetParam


WHITELIST_PATH = Path("/etc/unitport/tunable_params.yaml")
LIVE_PARAMS_PATH = Path("/etc/unitport/live_params.json")


class ParamSrvNode(Node):

    def __init__(self) -> None:
        super().__init__("unitport_param_srv")
        self._whitelist = self._load_whitelist()
        self._live: Dict[str, Dict[str, Any]] = self._load_live_params()

        self._updated_pub = self.create_publisher(String, "/unitport/params/updated", 10)
        self._srv = self.create_service(SetParam, "unitport_param_srv/SetParam",
                                        self._on_set_param)
        self.get_logger().info(
            f"param_srv_node up — whitelist has {len(self._whitelist)} entries"
        )

    def _on_set_param(self, req: SetParam.Request, resp: SetParam.Response) -> SetParam.Response:
        name, scope, raw_value = str(req.name), str(req.scope or "global"), str(req.value or "")

        ok, reason = self._check_whitelist(name, scope)
        if not ok:
            resp.ok = False
            resp.reason = reason
            return resp

        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            parsed = raw_value  # bare string payloads

        self._live.setdefault(name, {})[scope] = parsed
        self._persist_live_params()

        msg = String()
        msg.data = json.dumps({"name": name, "scope": scope, "value": parsed},
                              ensure_ascii=False)
        try:
            self._updated_pub.publish(msg)
        except Exception as exc:
            self.get_logger().warn(f"updated publish failed: {exc}")

        resp.ok = True
        resp.reason = "ok"
        return resp

    # ------------------------------------------------------------------
    # Whitelist enforcement
    # ------------------------------------------------------------------

    def _check_whitelist(self, name: str, scope: str) -> Tuple[bool, str]:
        if not self._whitelist:
            # Empty/missing whitelist → permissive mode, but warn so
            # operators notice before shipping.
            self.get_logger().warn(
                f"tunable_params.yaml missing/empty — accepting {name!r} in permissive mode"
            )
            return True, ""
        if name not in self._whitelist:
            return False, f"param {name!r} not in whitelist"
        allowed_scopes = self._whitelist[name].get("scopes") or ["global"]
        # Accept exact match, plus prefix match for scoped patterns ("joint:*").
        scope_root = scope.split(":", 1)[0]
        if scope not in allowed_scopes and scope_root not in allowed_scopes:
            return False, f"scope {scope!r} not allowed for {name!r}"
        return True, ""

    def _load_whitelist(self) -> Dict[str, Dict[str, Any]]:
        if not WHITELIST_PATH.exists():
            return {}
        try:
            data = yaml.safe_load(WHITELIST_PATH.read_text("utf-8")) or {}
        except yaml.YAMLError as exc:
            self.get_logger().warn(f"tunable_params.yaml parse failed: {exc}")
            return {}
        return dict(data.get("params") or {})

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_live_params(self) -> Dict[str, Dict[str, Any]]:
        if not LIVE_PARAMS_PATH.exists():
            return {}
        try:
            return json.loads(LIVE_PARAMS_PATH.read_text("utf-8")) or {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _persist_live_params(self) -> None:
        try:
            LIVE_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
            LIVE_PARAMS_PATH.write_text(
                json.dumps(self._live, sort_keys=True, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            self.get_logger().warn(f"live_params persist failed: {exc}")


def main(args=None):
    rclpy.init(args=args)
    node = ParamSrvNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
