"""generic_bridge_node — robot-side translator between the host's generic
/unitport/policy_action and the robot's native controller topics.

Startup:
  1. Load /etc/unitport/robot_profile.json (host-pushed inspector cache) and
     /etc/unitport/robot_mapping.json (bundle-pushed adapter mapping).
  2. If robot_mapping.json is absent, enter PASSIVE mode: the bridge stays
     alive and publishes a single info-level Alert ("no bundle mapping yet").
     This is the expected state after Upgrade and before any bundle deploy
     (bundle deploy is not yet implemented per native_dds_bridge_spec v1.0).
  3. With a mapping present, verify sha256(robot_mapping.json) matches
     /etc/unitport/deploy_manifest.json:mapping_hash. If not, publish a
     critical Alert and exit.
  4. Resolve (controller_type → adapter class) from
     /etc/unitport/adapter_registry.yaml + installed unitport_bridge.adapters.
  5. Subscribe /unitport/policy_action + /unitport/params/updated.
  6. Publish per the adapter's output_topics().
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Type

import rclpy
from rclpy.node import Node

from unitport_msgs.msg import Alert, AuxCommand, PolicyAction
from std_msgs.msg import String
import yaml

from unitport_bridge.aux_handlers import (
    BaseAuxHandler, NullAuxHandler, resolve_aux_handler,
)


PROFILE_PATH = Path("/etc/unitport/robot_profile.json")
MAPPING_PATH = Path("/etc/unitport/robot_mapping.json")
MANIFEST_PATH = Path("/etc/unitport/deploy_manifest.json")
REGISTRY_PATH = Path("/etc/unitport/adapter_registry.yaml")


class GenericBridgeNode(Node):

    def __init__(self) -> None:
        super().__init__("unitport_generic_bridge")
        self._alert_pub = self.create_publisher(Alert, "/unitport/alerts", 10)
        self._passive = False

        profile, mapping, manifest = self._load_config()
        if not mapping:
            # Passive mode: identity is set up but no bundle has been
            # pushed. Stay alive so the host's Connect doesn't time out
            # waiting for the bridge service, and so a future bundle
            # push can flip us into active mode via systemctl restart.
            self._passive = True
            self._publish_alert(
                "passive_mode", "info",
                f"{MAPPING_PATH} absent — bridge runs in passive mode "
                "until a bundle is pushed"
            )
            self.get_logger().info(
                "passive mode: no robot_mapping.json present; "
                "policy_action subscription disabled. Push a bundle and "
                "restart unitport-bridge.service to activate."
            )
            return

        self._check_mapping_hash_or_die(mapping, manifest)

        adapter_cls = self._resolve_adapter_class(mapping.get("controller_type", "custom"))
        self._adapter = adapter_cls(mapping=mapping, profile=profile)
        ok, reason = self._adapter.validate_mapping()
        if not ok:
            self._publish_alert(
                "mapping_invalid", "critical",
                f"adapter {adapter_cls.__name__} rejected mapping: {reason}"
            )
            self.get_logger().fatal(f"mapping invalid: {reason}")
            raise SystemExit(3)

        self._publishers: Dict[str, Any] = {}
        for topic, msg_type in self._adapter.output_topics():
            self._publishers[topic] = self.create_publisher(
                _resolve_msg_class(msg_type), topic, 10,
            )

        self.create_subscription(
            PolicyAction, "/unitport/policy_action", self._on_policy_action, 10,
        )
        self.create_subscription(
            String, "/unitport/params/updated", self._on_params_updated, 10,
        )

        # Aux handler resolution — brand-specific driver if the profile names
        # one, otherwise a no-op handler that logs and drops. The bridge
        # only requires the handler to exist; failures inside it are logged
        # but never bubble up.
        brand_id = str(profile.get("brand_id", "") or "")
        self._aux_handler = resolve_aux_handler(
            brand_id=brand_id,
            profile=profile,
            mapping=mapping,
            logger=self.get_logger(),
        )
        self.create_subscription(
            AuxCommand, "/unitport/aux_command", self._on_aux_command, 10,
        )

        self.get_logger().info(
            f"generic_bridge_node up — controller_type={mapping.get('controller_type')!r} "
            f"adapter={adapter_cls.__name__} "
            f"output_topics={[t for t, _ in self._adapter.output_topics()]} "
            f"aux_handler={type(self._aux_handler).__name__}"
        )

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    def _on_policy_action(self, msg: PolicyAction) -> None:
        try:
            outputs = self._adapter.convert(msg) or {}
        except Exception as exc:
            self.get_logger().error(f"adapter.convert raised: {exc}")
            return
        for topic, payload in outputs.items():
            pub = self._publishers.get(topic)
            if pub is None:
                self.get_logger().warn(f"no publisher for topic {topic!r}")
                continue
            try:
                pub.publish(payload)
            except Exception as exc:
                self.get_logger().error(f"publish {topic} failed: {exc}")

    def _on_aux_command(self, msg: AuxCommand) -> None:
        try:
            payload = json.loads(msg.payload_json or "{}")
        except json.JSONDecodeError as exc:
            self.get_logger().warn(
                f"aux_command payload is not valid JSON ({exc}); dropping"
            )
            return
        try:
            self._aux_handler.handle(
                surface=str(msg.surface),
                target=str(msg.target),
                payload=payload,
            )
        except Exception as exc:
            self.get_logger().warn(
                f"aux_handler raised for surface={msg.surface!r} target={msg.target!r}: {exc}"
            )

    def _on_params_updated(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data or "{}")
        except json.JSONDecodeError:
            return
        name = str(payload.get("name", ""))
        scope = str(payload.get("scope", "global"))
        value = payload.get("value")
        try:
            self._adapter.on_params_updated(name, scope, value)
        except Exception as exc:
            self.get_logger().warn(f"adapter.on_params_updated {name}: {exc}")

    # ------------------------------------------------------------------
    # Config loading + mapping_hash check
    # ------------------------------------------------------------------

    def _load_config(self):
        profile = _read_json(PROFILE_PATH) or {}
        mapping = _read_json(MAPPING_PATH) or {}
        manifest = _read_json(MANIFEST_PATH) or {}
        # Empty mapping is no longer fatal — caller flips to passive mode.
        return profile, mapping, manifest

    def _check_mapping_hash_or_die(self, mapping: Dict[str, Any], manifest: Dict[str, Any]) -> None:
        expected = str(manifest.get("mapping_hash", ""))
        if not expected:
            # No manifest — tolerate for first-run workflows, but log.
            self.get_logger().warn(
                f"{MANIFEST_PATH} missing mapping_hash; running without verification"
            )
            return
        raw = MAPPING_PATH.read_bytes()
        actual = hashlib.sha256(_canonicalise_json(raw)).hexdigest()
        if actual != expected:
            self._publish_alert(
                "mapping_hash_mismatch", "critical",
                f"expected={expected[:8]}… got={actual[:8]}…"
            )
            self.get_logger().fatal(
                f"mapping hash mismatch — refusing to start. "
                f"expected={expected[:8]}… got={actual[:8]}…"
            )
            # Give the alert a moment to make it out before we exit.
            time.sleep(0.2)
            raise SystemExit(4)

    # ------------------------------------------------------------------
    # Adapter resolution
    # ------------------------------------------------------------------

    def _resolve_adapter_class(self, controller_type: str) -> Type:
        registry = {}
        if REGISTRY_PATH.exists():
            try:
                registry = (yaml.safe_load(REGISTRY_PATH.read_text("utf-8")) or {}).get("built_in") or {}
            except yaml.YAMLError as exc:
                self.get_logger().warn(f"adapter_registry parse failed: {exc}")
        dotted = registry.get(controller_type)
        if not dotted:
            dotted = _BUILTIN_DEFAULTS.get(controller_type, _BUILTIN_DEFAULTS["custom"])
        module_path, _, class_name = dotted.rpartition(".")
        # Registry entries start "adapters.champ_adapter.ChampAdapter" — prefix
        # the package so the import is absolute.
        if module_path and not module_path.startswith("unitport_bridge"):
            module_path = f"unitport_bridge.{module_path}"
        try:
            module = __import__(module_path, fromlist=[class_name])
            return getattr(module, class_name)
        except Exception as exc:
            self._publish_alert(
                "adapter_resolve_failed", "critical",
                f"controller_type={controller_type} module={module_path}: {exc}"
            )
            self.get_logger().fatal(
                f"failed to resolve adapter for controller_type={controller_type}: {exc}"
            )
            raise SystemExit(5)

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------

    def _publish_alert(self, kind: str, severity: str, detail: str) -> None:
        msg = Alert()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.kind = str(kind)
        msg.severity = str(severity)
        msg.detail = str(detail)
        try:
            self._alert_pub.publish(msg)
        except Exception as exc:
            self.get_logger().error(f"alert publish failed: {exc}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_BUILTIN_DEFAULTS = {
    "champ":            "unitport_bridge.adapters.champ_adapter.ChampAdapter",
    "diff_drive":       "unitport_bridge.adapters.diff_drive_adapter.DiffDriveAdapter",
    "joint_trajectory": "unitport_bridge.adapters.joint_trajectory_adapter.JointTrajectoryAdapter",
    "custom":           "unitport_bridge.adapters.custom_adapter.CustomAdapter",
    # legged_gym mirrors joint_trajectory shape by default; dedicated adapter
    # can override later.
    "legged_gym":       "unitport_bridge.adapters.joint_trajectory_adapter.JointTrajectoryAdapter",
}


def _read_json(path: Path):
    try:
        return json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _canonicalise_json(raw: bytes) -> bytes:
    """Re-serialise the mapping bytes in the host's canonical form.

    The host computes mapping_hash over ``json.dumps(mapping, sort_keys=True,
    separators=(",",":"))``; the file on disk is pretty-printed. To compare
    hashes we parse + re-dump.
    """
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return raw
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _resolve_msg_class(msg_type: str):
    """Turn "geometry_msgs/msg/Twist" into the imported class."""
    parts = str(msg_type).split("/")
    if len(parts) == 3 and parts[1] == "msg":
        pkg, _, cls = parts
        module = __import__(f"{pkg}.msg", fromlist=[cls])
        return getattr(module, cls)
    if len(parts) == 2:
        pkg, cls = parts
        module = __import__(f"{pkg}.msg", fromlist=[cls])
        return getattr(module, cls)
    raise ValueError(f"bad msg_type spec: {msg_type!r}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(args=None):
    rclpy.init(args=args)
    try:
        node = GenericBridgeNode()
    except SystemExit as exc:
        rclpy.shutdown()
        sys.exit(exc.code)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
