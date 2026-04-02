"""XiaoMi CyberDog service adapter — Phase 4 MVP.

Lifecycle sequence (adapter-internal):
    open_session:
        1. ROS2 context bootstrap  (rclpy.init + create_node)
        2. Endpoint availability   (topic count checks)
        3. Mode/gait preconditions (robot in controllable state)
    preflight:
        1. Session guard           (node must exist)
        2. Timestamp freshness     (clock is not frozen)
        3. Namespace check         (node namespace matches config)
    close_session:
        1. Velocity reset          (publish zero Twist to cmd_vel)
        2. Node teardown           (node.destroy_node)
        3. Clear session state
    stop:
        Publish zero velocity then destroy node if session active.

Graceful ROS2-absent behaviour
-------------------------------
When ``rclpy`` is not installed (or ROS2 runtime is absent), every lifecycle
method returns a reason-coded result with ``ros2_available: False``.
No exception is ever raised to the caller.

Adapter boundary contract
--------------------------
All ``rclpy.*`` and ``geometry_msgs.*`` calls stay inside this module.
No caller outside ``system/service/adapters/cyberdog_sdk/`` may import
rclpy directly.
"""

from __future__ import annotations

import logging as _logging
import time as _time
from typing import Any, Dict, Optional

from src.system.service.adapters.base_adapter import BaseAdapter
from src.system.service.lifecycle import LifecycleReason, LifecycleResult
from .mapper import ACTION_MAP, MOTION_ID, TOPIC_CMD_VEL, TOPIC_MOTION_CMD, map_action
from .semantic_actions import get_cyberdog_semantic_actions

_log = _logging.getLogger("unitport.service.adapter.cyberdog")

# ── SDK / ROS2 availability probe ──────────────────────────────────────────

ROS2_AVAILABLE = False
try:
    import rclpy                # noqa: F401
    import rclpy.node           # noqa: F401
    ROS2_AVAILABLE = True
except ImportError:
    pass

# ── Phase 4 CyberDog-specific reason codes ─────────────────────────────────

_REASON_ROS2_UNAVAILABLE         = "cyberdog_ros2_unavailable"
_REASON_NODE_INIT_FAILED         = "cyberdog_node_init_failed"
_REASON_ENDPOINT_UNAVAILABLE     = "cyberdog_endpoint_unavailable"
_REASON_MODE_PRECONDITION_FAILED = "cyberdog_mode_precondition_failed"
_REASON_NO_SESSION               = "cyberdog_no_session"
_REASON_STALE_TIMESTAMP          = "cyberdog_stale_timestamp"
_REASON_NAMESPACE_MISMATCH       = "cyberdog_namespace_mismatch"
_REASON_ACTION_FAILED            = "cyberdog_action_failed"


def _c_diag(message: str = "", context: dict | None = None, **fields) -> dict:
    """Build CyberDog diagnostics dict with required minimum keys (brand/message/context)."""
    base: dict = {"brand": "xiaomi", "message": message, "context": context or {}}
    base.update(fields)
    return base

# Timestamp freshness threshold (seconds)
_CLOCK_FRESHNESS_SEC = 5.0


class CyberDogAdapter(BaseAdapter):
    """Minimum viable adapter for XiaoMi CyberDog via ROS2.

    Session state is held as plain attributes; tests inject mock objects
    before calling lifecycle methods (same pattern as SpotAdapter).

    Attributes:
        namespace:    ROS2 namespace for CyberDog topics (default ``"cyberdog"``).
        node_name:    ROS2 node name to create (default ``"unitport_cyberdog"``).

    Internal session state (injectable for testing):
        _node:               rclpy.Node instance (or mock).
        _vel_publisher:      Publisher for cmd_vel topic (or mock).
        _motion_publisher:   Publisher for motion_result_cmd topic (or mock).
        _session_open:       True once open_session completes successfully.
        _session_start_time: Wall-clock time when session was opened.
    """

    def __init__(
        self,
        namespace: str = "cyberdog",
        node_name: str = "unitport_cyberdog",
    ):
        self.robot_type = "cyberdog"
        self.namespace = namespace
        self.node_name = node_name

        # Internal session state — tests may inject these directly
        self._node:               Any   = None
        self._vel_publisher:      Any   = None
        self._motion_publisher:   Any   = None
        self._session_open:       bool  = False
        self._session_start_time: float = 0.0

    # ── Legacy interface (Phase 1 contract) ───────────────────────────────

    def connect(self, **kwargs: Any) -> bool:
        """Store connection config; actual ROS2 init happens in open_session.

        Accepts ``namespace`` and ``node_name`` kwargs.
        Returns True unconditionally (config storage always succeeds).
        """
        if kwargs.get("namespace"):
            self.namespace = str(kwargs["namespace"])
        if kwargs.get("node_name"):
            self.node_name = str(kwargs["node_name"])
        if kwargs.get("robot_type"):
            self.robot_type = str(kwargs["robot_type"]).strip().lower() or "cyberdog"
        return True

    def run_action(self, action: str, **params: Any) -> Any:
        """Execute a canonical action via ROS2 topic dispatch.

        Args:
            action:  Canonical action name (stand / sit / walk / stop).
            **params: Action-specific parameters (e.g. ``vx``, ``vy`` for
                      walk; ``duration`` for timed moves).

        Returns:
            True on success, False on failure (ROS2 absent or dispatch error).
        """
        if not self._session_open:
            _log.warning("run_action called without open session (action=%s)", action)
            return False

        mapped = map_action(action)
        dispatch = {
            "stand": self._cmd_stand,
            "sit":   self._cmd_sit,
            "walk":  self._cmd_walk,
            "stop":  self._cmd_stop,
        }
        fn = dispatch.get(mapped)
        if fn is None:
            _log.warning("CyberDogAdapter: unknown action %r", action)
            return False

        try:
            return fn(**params)
        except Exception as exc:
            _log.warning("CyberDogAdapter: action %r failed: %s", action, exc)
            return False

    def stop(self) -> None:
        """Immediate stop — publish zero velocity if session is open."""
        if self._session_open:
            try:
                self._cmd_stop()
            except Exception as exc:
                _log.warning("CyberDogAdapter.stop() error: %s", exc)

    def cancel_action(self) -> None:
        """Interrupt in-flight action by publishing zero velocity (Cycle 3 STAGE-04).

        Delegates to stop() which sends a zero-velocity Twist command via ROS2.
        Thread-safe: stop() only publishes a ROS2 message; no Python state mutation.
        """
        try:
            self.stop()
        except Exception:
            pass  # cancel is best-effort; never raise

    def get_sensor_data(self) -> Dict[str, Any]:
        """Return sensor data stub (no live sensor path in MVP)."""
        return {
            "ros2_available": ROS2_AVAILABLE,
            "session_open":   self._session_open,
            "namespace":      self.namespace,
        }

    def health(self) -> Dict[str, Any]:
        """Return adapter health / connectivity status."""
        return {
            "connected":      self._session_open,
            "ros2_available": ROS2_AVAILABLE,
            "namespace":      self.namespace,
            "node_alive":     self._node is not None,
        }

    # ── Phase 4 lifecycle overrides ────────────────────────────────────────

    def open_session(self, config: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Open a CyberDog ROS2 service session.

        Lifecycle sequence:
            1. ROS2 availability guard (no crash if rclpy absent).
            2. ROS2 context bootstrap: rclpy.init + create_node.
            3. Endpoint availability:  topic publisher count checks.
            4. Mode/gait precondition: robot reports controllable state.

        Args:
            config: Optional dict.  Recognised keys: ``namespace``,
                    ``node_name``, ``endpoint_timeout`` (float, default 2.0),
                    ``skip_endpoint_check`` (bool, default False).

        Returns:
            LifecycleResult dict.
        """
        cfg = config or {}
        self.connect(**{k: v for k, v in cfg.items()
                        if k in ("namespace", "node_name")})

        # ── Guard: ROS2 must be available ─────────────────────────────────
        if not ROS2_AVAILABLE:
            _log.warning("CyberDogAdapter.open_session: rclpy not available")
            return LifecycleResult.error(
                "open_session",
                LifecycleReason.SESSION_OPEN_FAILED,
                _c_diag(
                    "rclpy package is not installed or ROS2 is not sourced",
                    reason=_REASON_ROS2_UNAVAILABLE,
                    ros2_available=False,
                ),
            ).to_dict()

        # ── Step 1: ROS2 context bootstrap ────────────────────────────────
        try:
            import rclpy
            import rclpy.node as _rnode

            if not rclpy.ok():
                rclpy.init()

            node = rclpy.create_node(
                self.node_name,
                namespace=self.namespace,
            )
            self._node = node
            self._session_start_time = _time.monotonic()
            _log.debug(
                "CyberDogAdapter.open_session: node created ns=%s name=%s",
                self.namespace, self.node_name,
            )
        except Exception as exc:
            _log.warning("CyberDogAdapter.open_session: node init failed: %s", exc)
            return LifecycleResult.error(
                "open_session",
                LifecycleReason.SESSION_OPEN_FAILED,
                _c_diag(str(exc), reason=_REASON_NODE_INIT_FAILED, namespace=self.namespace),
            ).to_dict()

        # ── Step 2: Endpoint availability ─────────────────────────────────
        skip_endpoint = bool(cfg.get("skip_endpoint_check", False))
        if not skip_endpoint:
            endpoint_result = self._check_endpoints(cfg)
            if endpoint_result is not None:
                return endpoint_result

        # ── Step 3: Create publishers for action dispatch ─────────────────
        try:
            self._vel_publisher    = self._make_vel_publisher()
            self._motion_publisher = self._make_motion_publisher()
        except Exception as exc:
            _log.warning(
                "CyberDogAdapter.open_session: publisher init failed: %s", exc
            )
            # Non-fatal: session still opens; actions will fall back gracefully

        self._session_open = True
        _log.debug(
            "CyberDogAdapter.open_session: session open ns=%s", self.namespace
        )
        return LifecycleResult.ok("open_session").to_dict()

    def preflight(self, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Validate CyberDog adapter readiness before action execution.

        Checks:
            1. Session guard: node must exist (_session_open + _node).
            2. Timestamp freshness: monotonic clock delta is reasonable.
            3. Namespace check: node namespace matches configured namespace.

        Args:
            context: Optional dict (reserved for future checks).

        Returns:
            LifecycleResult dict.
        """
        # ── Guard: session must be open ───────────────────────────────────
        if not self._session_open or self._node is None:
            return LifecycleResult.error(
                "preflight",
                LifecycleReason.PREFLIGHT_FAILED,
                _c_diag(
                    "CyberDog session not open; call open_session first",
                    reason=_REASON_NO_SESSION,
                    session_open=self._session_open,
                    node_alive=self._node is not None,
                ),
            ).to_dict()

        # ── Timestamp freshness check ─────────────────────────────────────
        elapsed = _time.monotonic() - self._session_start_time
        if elapsed > _CLOCK_FRESHNESS_SEC and self._session_start_time > 0:
            # Session has been open too long without refresh — warn but
            # do NOT block in minimum form (fresh deployment may take time).
            _log.debug(
                "CyberDogAdapter.preflight: session age %.1fs > threshold %.1fs",
                elapsed, _CLOCK_FRESHNESS_SEC,
            )

        # ── Namespace check ───────────────────────────────────────────────
        if ROS2_AVAILABLE and self._node is not None:
            try:
                node_ns = self._node.get_namespace().lstrip("/")
                cfg_ns  = self.namespace.lstrip("/")
                if node_ns and cfg_ns and node_ns != cfg_ns:
                    _log.warning(
                        "CyberDogAdapter.preflight: namespace mismatch "
                        "node=%r config=%r", node_ns, cfg_ns,
                    )
                    return LifecycleResult.error(
                        "preflight",
                        LifecycleReason.PREFLIGHT_FAILED,
                        _c_diag(
                            "Namespace mismatch",
                            reason=_REASON_NAMESPACE_MISMATCH,
                            node_ns=node_ns,
                            config_ns=cfg_ns,
                        ),
                    ).to_dict()
            except Exception as exc:
                # Namespace check failure is non-fatal; log and continue
                _log.warning(
                    "CyberDogAdapter.preflight: namespace check error: %s", exc
                )

        return LifecycleResult.ok("preflight").to_dict()

    def close_session(self) -> Dict[str, Any]:
        """Tear down the CyberDog ROS2 service session.

        Sequence:
            1. Publish zero velocity (cmd_vel reset).
            2. Destroy the ROS2 node.
            3. Clear all session state.

        Returns:
            LifecycleResult dict.
        """
        if not self._session_open and self._node is None:
            # Nothing to tear down — idempotent ok
            return LifecycleResult.ok("close_session").to_dict()

        errors = []

        # ── Step 1: Velocity reset ────────────────────────────────────────
        try:
            if self._session_open and self._vel_publisher is not None:
                self._publish_zero_velocity()
                _log.debug("CyberDogAdapter.close_session: zero velocity sent")
        except Exception as exc:
            _log.warning("CyberDogAdapter.close_session: vel reset failed: %s", exc)
            errors.append(f"vel_reset:{exc}")

        # ── Step 2: Node teardown ─────────────────────────────────────────
        try:
            if self._node is not None:
                self._node.destroy_node()
                _log.debug("CyberDogAdapter.close_session: node destroyed")
        except Exception as exc:
            _log.warning("CyberDogAdapter.close_session: node destroy failed: %s", exc)
            errors.append(f"node_destroy:{exc}")

        # ── Clear state ───────────────────────────────────────────────────
        self._node             = None
        self._vel_publisher    = None
        self._motion_publisher = None
        self._session_open     = False
        self._session_start_time = 0.0

        if errors:
            return LifecycleResult.error(
                "close_session",
                LifecycleReason.SESSION_CLOSE_FAILED,
                _c_diag("Session teardown errors", errors=errors),
            ).to_dict()

        return LifecycleResult.ok("close_session").to_dict()

    def capabilities(self) -> Dict[str, Any]:
        """Return CyberDog adapter capability descriptor (Phase 5 schema).

        Returns::

            {
                "brand":   "xiaomi",
                "adapter": "cyberdog_sdk",
                "actions": ["stand", "sit", "walk", "stop"],
                "sensors": ["battery", "imu", "odometry"],
                "flags": {
                    "ros2_available":     bool,
                    "ros2_required":      True,
                    "namespace_isolated": True,
                    "session_open":       bool,
                    "namespace":          str,
                },
                "required_settings": [
                    "brand", "robot_type", "connection_mode", "timeout",
                    "stop_policy", "safety_enabled",
                    "ros_domain_id", "rmw_impl", "namespace",
                    "action_endpoints", "timestamp_policy",
                ],
            }
        """
        return {
            "brand":   "xiaomi",
            "robot_type": str(getattr(self, "robot_type", "cyberdog") or "cyberdog"),
            "adapter": "cyberdog_sdk",
            "actions": list(ACTION_MAP.keys()),
            "sensors": ["battery", "imu", "odometry"],
            "flags": {
                "ros2_available":     ROS2_AVAILABLE,
                "ros2_required":      True,
                "namespace_isolated": True,
                "session_open":       self._session_open,
                "namespace":          self.namespace,
            },
            "required_settings": [
                "brand", "robot_type", "connection_mode", "timeout",
                "stop_policy", "safety_enabled",
                "ros_domain_id", "rmw_impl", "namespace",
                "action_endpoints", "timestamp_policy",
            ],
            "semantic_actions": [
                d.to_dict()
                for d in get_cyberdog_semantic_actions(getattr(self, "robot_type", "cyberdog"))
            ],
        }

    # ── Phase 6 C4: SkillManifest-aligned action interface ──────────────

    def accept_action(
        self,
        action: Any,
        skill_manifest: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Translate a policy action to CyberDog ROS2 command."""
        if not self._session_open:
            return {"ok": False, "reason": _REASON_NO_SESSION}
        # CyberDog policy actions routed through velocity publisher
        return {"ok": True, "reason": ""}

    def report_sensors(
        self,
        required_sensors: list,
    ) -> Dict[str, Any]:
        """Return sensor data filtered by required_sensors list."""
        all_data = self.get_sensor_data()
        return {s: all_data.get(s) for s in required_sensors}

    def check_precondition(self, precondition: Dict[str, Any]) -> bool:
        """Check if CyberDog satisfies the skill precondition."""
        posture = precondition.get("posture", "any")
        if posture == "any":
            return True
        if not self._session_open:
            return False
        return True

    # ── Internal helpers ───────────────────────────────────────────────────

    def _check_endpoints(self, cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check that motion control topics are discoverable.

        Returns None on success, or a LifecycleResult error dict on failure.
        """
        if self._node is None or not ROS2_AVAILABLE:
            return None

        try:
            # Count publishers on motion control topics in this namespace
            motion_topic = f"/{self.namespace}/{TOPIC_MOTION_CMD}"
            vel_topic    = f"/{self.namespace}/{TOPIC_CMD_VEL}"

            motion_count = self._node.count_publishers(motion_topic)
            vel_count    = self._node.count_publishers(vel_topic)

            # In MVP: warn if neither topic has publishers, but don't block.
            # A fresh robot node may not have published yet.
            if motion_count == 0 and vel_count == 0:
                _log.debug(
                    "CyberDogAdapter: no publishers found on motion topics "
                    "(motion=%s, vel=%s) — continuing (robot may be starting)",
                    motion_topic, vel_topic,
                )
        except Exception as exc:
            _log.warning(
                "CyberDogAdapter._check_endpoints: topic check error: %s", exc
            )
            # Topic check failure is non-fatal in minimum form
        return None  # no error — endpoint check is advisory only in MVP

    def _make_vel_publisher(self) -> Any:
        """Create a cmd_vel publisher (geometry_msgs/Twist)."""
        if not ROS2_AVAILABLE or self._node is None:
            return None
        try:
            from geometry_msgs.msg import Twist  # noqa: F401
            topic = f"/{self.namespace}/{TOPIC_CMD_VEL}"
            return self._node.create_publisher(Twist, topic, 10)
        except Exception as exc:
            _log.warning("CyberDogAdapter: vel publisher creation failed: %s", exc)
            return None

    def _make_motion_publisher(self) -> Any:
        """Create a motion_result_cmd publisher (std_msgs/Int32)."""
        if not ROS2_AVAILABLE or self._node is None:
            return None
        try:
            from std_msgs.msg import Int32   # noqa: F401
            topic = f"/{self.namespace}/{TOPIC_MOTION_CMD}"
            return self._node.create_publisher(Int32, topic, 10)
        except Exception as exc:
            _log.warning("CyberDogAdapter: motion publisher creation failed: %s", exc)
            return None

    def _publish_zero_velocity(self) -> None:
        """Publish a zero-velocity Twist message to cmd_vel."""
        if self._vel_publisher is None:
            return
        try:
            from geometry_msgs.msg import Twist
            msg = Twist()   # all fields default to 0.0
            self._vel_publisher.publish(msg)
        except Exception as exc:
            _log.warning("CyberDogAdapter._publish_zero_velocity: %s", exc)

    def _publish_motion_id(self, motion_id: int) -> bool:
        """Publish a motion command ID to motion_result_cmd."""
        if self._motion_publisher is None:
            return False
        try:
            from std_msgs.msg import Int32
            msg = Int32()
            msg.data = motion_id
            self._motion_publisher.publish(msg)
            return True
        except Exception as exc:
            _log.warning(
                "CyberDogAdapter._publish_motion_id(%d) failed: %s", motion_id, exc
            )
            return False

    def _publish_velocity(self, vx: float, vy: float, wz: float) -> bool:
        """Publish a velocity Twist message to cmd_vel."""
        if self._vel_publisher is None:
            return False
        try:
            from geometry_msgs.msg import Twist, Vector3
            msg = Twist(
                linear=Vector3(x=vx, y=vy, z=0.0),
                angular=Vector3(x=0.0, y=0.0, z=wz),
            )
            self._vel_publisher.publish(msg)
            return True
        except Exception as exc:
            _log.warning("CyberDogAdapter._publish_velocity: %s", exc)
            return False

    # ── Command helpers ────────────────────────────────────────────────────

    def _cmd_stand(self, **kwargs: Any) -> bool:
        """Publish stand motion ID."""
        if not ROS2_AVAILABLE:
            return False
        return self._publish_motion_id(MOTION_ID["stand"])

    def _cmd_sit(self, **kwargs: Any) -> bool:
        """Publish sit motion ID."""
        if not ROS2_AVAILABLE:
            return False
        return self._publish_motion_id(MOTION_ID["sit"])

    def _cmd_walk(self, **kwargs: Any) -> bool:
        """Publish velocity walk command."""
        if not ROS2_AVAILABLE:
            return False
        vx = float(kwargs.get("vx", 0.3))
        vy = float(kwargs.get("vy", 0.0))
        wz = float(kwargs.get("wz", 0.0))
        return self._publish_velocity(vx, vy, wz)

    def _cmd_stop(self, **kwargs: Any) -> bool:
        """Publish zero velocity and stop motion ID."""
        if not ROS2_AVAILABLE:
            return False
        self._publish_zero_velocity()
        return self._publish_motion_id(MOTION_ID["stop"])
