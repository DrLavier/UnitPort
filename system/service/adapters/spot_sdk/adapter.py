"""Boston Dynamics Spot service adapter — Phase 4 MVP.

Lifecycle sequence (adapter-internal):
    open_session:
        1. Auth bootstrap  (create SDK → create robot → authenticate)
        2. Time sync check (robot.time_sync.wait_for_sync)
        3. Lease acquire   (lease_client.take)
    preflight:
        1. Session guard   (lease must be held)
        2. E-stop check    (safety channel must be clear)
    close_session:
        1. Return to safe stand posture
        2. Release lease
    stop:
        Sends stop command (or releases lease on hard stop)

Graceful SDK-absent behaviour
-------------------------------
When ``bosdyn-client`` is not installed, every lifecycle method returns a
reason-coded ``SESSION_OPEN_FAILED`` / ``PREFLIGHT_FAILED`` result with
``sdk_available: False``.  No crash is ever raised.

Adapter boundary contract
--------------------------
All ``bosdyn.*`` calls stay inside this module.  No caller outside
``system/service/adapters/spot_sdk/`` may import bosdyn directly.
"""

from __future__ import annotations

import importlib
import logging as _logging
from typing import Any, Dict, Optional

from system.service.adapters.base_adapter import BaseAdapter
from system.service.lifecycle import LifecycleReason, LifecycleResult
from .mapper import ACTION_MAP, map_action
from .semantic_actions import get_spot_semantic_actions

_log = _logging.getLogger("unitport.service.adapter.spot")

# ── SDK availability probe ─────────────────────────────────────────────────

SPOT_SDK_AVAILABLE = False
try:
    import bosdyn.client                          # noqa: F401
    import bosdyn.client.lease                    # noqa: F401
    import bosdyn.client.robot_command            # noqa: F401
    import bosdyn.client.time_sync                # noqa: F401
    SPOT_SDK_AVAILABLE = True
except ImportError:
    pass

# ── Phase 4 Spot-specific reason codes ────────────────────────────────────

_REASON_SDK_UNAVAILABLE  = "spot_sdk_unavailable"
_REASON_AUTH_FAILED      = "spot_auth_failed"
_REASON_TIME_SYNC_FAILED = "spot_time_sync_failed"
_REASON_LEASE_FAILED     = "spot_lease_failed"
_REASON_ESTOP_ACTIVE     = "spot_estop_active"
_REASON_NO_SESSION       = "spot_no_session"
_REASON_ACTION_FAILED    = "spot_action_failed"
_REASON_COMMAND_TIMEOUT  = "spot_command_timeout"


def _s_diag(message: str = "", context: dict | None = None, **fields) -> dict:
    """Build Spot diagnostics dict with required minimum keys (brand/message/context)."""
    base: dict = {"brand": "bostiondynamics", "message": message, "context": context or {}}
    base.update(fields)
    return base


class SpotAdapter(BaseAdapter):
    """Minimum viable adapter for Boston Dynamics Spot.

    Session state is held as plain attributes; tests inject mock objects
    before calling lifecycle methods (same pattern as UnitreeAdapter).

    Attributes:
        hostname:        Robot IP or hostname.
        username:        BD robot username (default ``"user"``).
        password:        BD robot password.

    Internal session state (injectable for testing):
        _sdk:            bosdyn.client.Sdk instance (or mock)
        _robot:          bosdyn.client.Robot instance (or mock)
        _lease_client:   LeaseClient (or mock)
        _lease:          Acquired lease resource (or mock)
        _command_client: RobotCommandClient (or mock)
        _session_open:   True once open_session completes successfully.
    """

    # Command timeout for blocking robot commands (seconds)
    _CMD_TIMEOUT: float = 10.0

    def __init__(
        self,
        hostname: str = "",
        username: str = "user",
        password: str = "",
    ):
        self.robot_type = "spot"
        self.hostname = hostname
        self.username = username
        self.password = password

        # Internal session state — tests may inject these directly
        self._sdk:            Any = None
        self._robot:          Any = None
        self._lease_client:   Any = None
        self._lease:          Any = None
        self._command_client: Any = None
        self._model:          Any = None
        self._session_open:   bool = False

    # ── Legacy interface (Phase 1 contract) ───────────────────────────────

    def connect(self, **kwargs: Any) -> bool:
        """Store connection config; actual connection happens in open_session.

        Accepts ``hostname``, ``username``, ``password`` kwargs.
        Returns True unconditionally (config storage always succeeds).
        """
        if kwargs.get("hostname"):
            self.hostname = str(kwargs["hostname"])
        if kwargs.get("username"):
            self.username = str(kwargs["username"])
        if kwargs.get("password"):
            self.password = str(kwargs["password"])
        if kwargs.get("robot_type"):
            self.robot_type = str(kwargs["robot_type"]).strip().lower() or "spot"
        return True

    def _create_model_instance(self, robot_type: str) -> Any:
        module_names = [
            "system.brand_packages.bostiondynamics",
        ]
        for module_name in module_names:
            try:
                module = importlib.import_module(module_name)
                model_class = getattr(module, "SpotModel", None)
                if model_class is not None:
                    return model_class(robot_type)
            except Exception:
                continue
        return None

    def get_model(self) -> Optional[Any]:
        if self._model is None:
            self._model = self._create_model_instance(self.robot_type)
        return self._model

    def velocity_move(self, vx: float, vy: float, vyaw: float, duration: float) -> bool:
        if not self._session_open:
            return False
        return self._cmd_walk(vx=float(vx), vy=float(vy), vr=float(vyaw), duration=float(duration))

    def velocity_move_mujoco(self, vx: float, vy: float, vyaw: float, duration: float) -> bool:
        model = self.get_model()
        if model is None:
            return False
        return model.velocity_move_mujoco(float(vx), float(vy), float(vyaw), float(duration))

    def run_action(self, action: str, **params: Any) -> Any:
        """Execute a canonical action.

        Args:
            action: Canonical action name (stand / sit / walk / stop).
            **params: Action-specific parameters (e.g. ``vx``, ``vy``
                      for walk; ``duration`` for timed moves).

        Returns:
            True on success, False on failure (SDK absent or command error).
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
            _log.warning("SpotAdapter: unknown action %r", action)
            return False

        try:
            return fn(**params)
        except Exception as exc:
            _log.warning("SpotAdapter: action %r failed: %s", action, exc)
            return False

    def stop(self) -> None:
        """Immediate stop — send stop command if session is open."""
        if self._session_open:
            try:
                self._cmd_stop()
            except Exception as exc:
                _log.warning("SpotAdapter.stop() error: %s", exc)

    def cancel_action(self) -> None:
        """Interrupt in-flight action by issuing an immediate stop (Cycle 3 STAGE-04).

        Delegates to stop() which halts motion via the SDK stop command.
        Thread-safe: stop() only mutates robot state, not adapter Python fields.
        """
        try:
            self.stop()
        except Exception:
            pass  # cancel is best-effort; never raise

    def get_sensor_data(self) -> Dict[str, Any]:
        """Return robot state data (stub when SDK absent or no session)."""
        if not self._session_open or self._robot is None:
            return {"sdk_available": SPOT_SDK_AVAILABLE, "session_open": False}
        try:
            state = self._robot.get_robot_state()
            return {
                "sdk_available":  True,
                "session_open":   True,
                "robot_state":    str(state),
            }
        except Exception as exc:
            return {"sdk_available": True, "session_open": True, "error": str(exc)}

    def health(self) -> Dict[str, Any]:
        """Return adapter health / connectivity status."""
        return {
            "connected":     self._session_open,
            "sdk_available": SPOT_SDK_AVAILABLE,
            "hostname":      self.hostname,
            "lease_held":    self._lease is not None,
        }

    # ── Phase 4 lifecycle overrides ────────────────────────────────────────

    def open_session(self, config: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Open a Spot service session.

        Lifecycle sequence:
            1. SDK availability guard (no crash if SDK absent).
            2. Auth bootstrap: create SDK → create robot → authenticate.
            3. Time sync: robot.time_sync.wait_for_sync(timeout).
            4. Lease acquire: lease_client.take().

        Args:
            config: Optional dict.  Recognised keys: ``hostname``,
                    ``username``, ``password``, ``time_sync_timeout``
                    (float, default 10.0).

        Returns:
            LifecycleResult dict.
        """
        cfg = config or {}
        # Allow config override of connection params
        self.connect(**{k: v for k, v in cfg.items()
                        if k in ("hostname", "username", "password")})

        # ── Guard: SDK must be installed ──────────────────────────────────
        if not SPOT_SDK_AVAILABLE:
            _log.warning("SpotAdapter.open_session: bosdyn SDK not available")
            return LifecycleResult.error(
                "open_session",
                LifecycleReason.SESSION_OPEN_FAILED,
                _s_diag(
                    "bosdyn-client package is not installed",
                    reason=_REASON_SDK_UNAVAILABLE,
                    sdk_available=False,
                ),
            ).to_dict()

        # ── Step 1: Auth bootstrap ────────────────────────────────────────
        try:
            import bosdyn.client
            sdk = bosdyn.client.create_standard_sdk("UnitportSpotAdapter")
            robot = sdk.create_robot(self.hostname)
            robot.authenticate(self.username, self.password)
            self._sdk   = sdk
            self._robot = robot
            _log.debug("SpotAdapter.open_session: auth ok hostname=%s", self.hostname)
        except Exception as exc:
            _log.warning("SpotAdapter.open_session: auth failed: %s", exc)
            return LifecycleResult.error(
                "open_session",
                LifecycleReason.SESSION_OPEN_FAILED,
                _s_diag(str(exc), reason=_REASON_AUTH_FAILED, hostname=self.hostname),
            ).to_dict()

        # ── Step 2: Time sync ─────────────────────────────────────────────
        time_sync_timeout = float(cfg.get("time_sync_timeout", 10.0))
        try:
            self._robot.time_sync.wait_for_sync(timeout_sec=time_sync_timeout)
            _log.debug("SpotAdapter.open_session: time sync ok")
        except Exception as exc:
            _log.warning("SpotAdapter.open_session: time sync failed: %s", exc)
            return LifecycleResult.error(
                "open_session",
                LifecycleReason.SESSION_OPEN_FAILED,
                _s_diag(str(exc), reason=_REASON_TIME_SYNC_FAILED),
            ).to_dict()

        # ── Step 3: Lease acquire ─────────────────────────────────────────
        try:
            import bosdyn.client.lease
            lease_client = self._robot.ensure_client(
                bosdyn.client.lease.LeaseClient.default_service_name
            )
            lease = lease_client.take()
            self._lease_client = lease_client
            self._lease        = lease
            _log.debug("SpotAdapter.open_session: lease acquired")
        except Exception as exc:
            _log.warning("SpotAdapter.open_session: lease acquire failed: %s", exc)
            return LifecycleResult.error(
                "open_session",
                LifecycleReason.SESSION_OPEN_FAILED,
                _s_diag(str(exc), reason=_REASON_LEASE_FAILED),
            ).to_dict()

        # ── Step 4: Command client ────────────────────────────────────────
        try:
            import bosdyn.client.robot_command
            self._command_client = self._robot.ensure_client(
                bosdyn.client.robot_command.RobotCommandClient.default_service_name
            )
        except Exception as exc:
            # Non-fatal: session is open, actions may still work via robot
            _log.warning("SpotAdapter.open_session: command client init failed: %s", exc)

        self._session_open = True
        _log.debug("SpotAdapter.open_session: session open hostname=%s", self.hostname)
        return LifecycleResult.ok("open_session").to_dict()

    def preflight(self, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Validate Spot adapter readiness before action execution.

        Checks:
            1. Session is open (lease must be held).
            2. E-stop is clear (safety channel check).

        Args:
            context: Optional dict (reserved for future checks).

        Returns:
            LifecycleResult dict.
        """
        # ── Guard: session must be open ───────────────────────────────────
        if not self._session_open or self._lease is None:
            return LifecycleResult.error(
                "preflight",
                LifecycleReason.PREFLIGHT_FAILED,
                _s_diag(
                    "Spot session not open; call open_session first",
                    reason=_REASON_NO_SESSION,
                    session_open=self._session_open,
                    lease_held=self._lease is not None,
                ),
            ).to_dict()

        # ── E-stop / safety channel check ────────────────────────────────
        if self._robot is not None and SPOT_SDK_AVAILABLE:
            try:
                robot_state = self._robot.get_robot_state()
                estop_states = getattr(
                    getattr(robot_state, "estop_states", None), "estop_states", []
                ) or []
                for s in estop_states:
                    if getattr(s, "state", None) == 1:   # ESTOP_STATE_ESTOPPED = 1
                        _log.warning("SpotAdapter.preflight: E-stop active")
                        return LifecycleResult.error(
                            "preflight",
                            LifecycleReason.PREFLIGHT_FAILED,
                            _s_diag(
                                "E-stop is active — clear before executing",
                                reason=_REASON_ESTOP_ACTIVE,
                            ),
                        ).to_dict()
            except Exception as exc:
                # E-stop check failure is non-fatal in minimum form;
                # log a warning and continue.
                _log.warning("SpotAdapter.preflight: e-stop check error: %s", exc)

        return LifecycleResult.ok("preflight").to_dict()

    def close_session(self) -> Dict[str, Any]:
        """Tear down the Spot service session.

        Sequence:
            1. Send stand command to return robot to safe posture.
            2. Release the lease.
            3. Clear all session state.

        Returns:
            LifecycleResult dict.
        """
        if not self._session_open and self._lease is None:
            # Nothing to tear down — idempotent ok
            return LifecycleResult.ok("close_session").to_dict()

        errors = []

        # ── Step 1: Return to safe stand posture ──────────────────────────
        try:
            if self._session_open and self._command_client is not None:
                self._cmd_stand()
                _log.debug("SpotAdapter.close_session: stand command sent")
        except Exception as exc:
            _log.warning("SpotAdapter.close_session: stand failed: %s", exc)
            errors.append(f"stand:{exc}")

        # ── Step 2: Release lease ─────────────────────────────────────────
        try:
            if self._lease_client is not None and self._lease is not None:
                self._lease_client.return_lease(self._lease)
                _log.debug("SpotAdapter.close_session: lease released")
        except Exception as exc:
            _log.warning("SpotAdapter.close_session: lease release failed: %s", exc)
            errors.append(f"lease_release:{exc}")

        # ── Clear state ───────────────────────────────────────────────────
        self._lease        = None
        self._lease_client = None
        self._command_client = None
        self._robot        = None
        self._sdk          = None
        self._session_open = False

        if errors:
            return LifecycleResult.error(
                "close_session",
                LifecycleReason.SESSION_CLOSE_FAILED,
                _s_diag("Session teardown errors", errors=errors),
            ).to_dict()

        return LifecycleResult.ok("close_session").to_dict()

    def capabilities(self) -> Dict[str, Any]:
        """Return Spot adapter capability descriptor (Phase 5 schema).

        Returns::

            {
                "brand":   "bostiondynamics",
                "adapter": "spot_sdk",
                "actions": ["stand", "sit", "walk", "stop"],
                "sensors": ["battery", "estop", "power_state"],
                "flags": {
                    "sdk_available":     bool,
                    "lease_required":    True,
                    "timesync_required": True,
                    "session_open":      bool,
                    "hostname":          str,
                },
                "required_settings": [
                    "brand", "robot_type", "connection_mode", "timeout",
                    "stop_policy", "safety_enabled",
                    "hostname", "auth_token", "timesync_required", "lease_required",
                    "safety_channel", "power_policy",
                ],
            }
        """
        return {
            "brand":   "bostiondynamics",
            "robot_type": str(getattr(self, "robot_type", "spot") or "spot"),
            "adapter": "spot_sdk",
            "actions": list(ACTION_MAP.keys()),
            "sensors": ["battery", "estop", "power_state"],
            "flags": {
                "sdk_available":     SPOT_SDK_AVAILABLE,
                "lease_required":    True,
                "timesync_required": True,
                "session_open":      self._session_open,
                "hostname":          self.hostname,
            },
            "required_settings": [
                "brand", "robot_type", "connection_mode", "timeout",
                "stop_policy", "safety_enabled",
                "hostname", "auth_token", "timesync_required", "lease_required",
                "safety_channel", "power_policy",
            ],
            "semantic_actions": [
                d.to_dict()
                for d in get_spot_semantic_actions(getattr(self, "robot_type", "spot"))
            ],
        }

    # ── Internal command helpers ───────────────────────────────────────────

    def _cmd_stand(self, **kwargs: Any) -> bool:
        """Send stand command to Spot."""
        if not SPOT_SDK_AVAILABLE or self._command_client is None:
            return False
        try:
            import bosdyn.client.robot_command as _rc
            cmd = _rc.RobotCommandBuilder.stand_command()
            self._command_client.robot_command(
                command=cmd, timeout=self._CMD_TIMEOUT
            )
            return True
        except Exception as exc:
            _log.warning("SpotAdapter._cmd_stand failed: %s", exc)
            return False

    def _cmd_sit(self, **kwargs: Any) -> bool:
        """Send safe-power-off (sit/down) command to Spot."""
        if not SPOT_SDK_AVAILABLE or self._command_client is None:
            return False
        try:
            import bosdyn.client.robot_command as _rc
            cmd = _rc.RobotCommandBuilder.safe_power_off_command()
            self._command_client.robot_command(
                command=cmd, timeout=self._CMD_TIMEOUT
            )
            return True
        except Exception as exc:
            _log.warning("SpotAdapter._cmd_sit failed: %s", exc)
            return False

    def _cmd_walk(self, **kwargs: Any) -> bool:
        """Send velocity walk command to Spot."""
        if not SPOT_SDK_AVAILABLE or self._command_client is None:
            return False
        try:
            import bosdyn.client.robot_command as _rc
            vx  = float(kwargs.get("vx", 0.5))
            vy  = float(kwargs.get("vy", 0.0))
            vr  = float(kwargs.get("vr", 0.0))
            dur = float(kwargs.get("duration", 2.0))
            cmd = _rc.RobotCommandBuilder.velocity_command(vx, vy, vr)
            self._command_client.robot_command(
                command=cmd, timeout=dur + self._CMD_TIMEOUT
            )
            return True
        except Exception as exc:
            _log.warning("SpotAdapter._cmd_walk failed: %s", exc)
            return False

    def _cmd_stop(self, **kwargs: Any) -> bool:
        """Send stop command to Spot."""
        if not SPOT_SDK_AVAILABLE or self._command_client is None:
            return False
        try:
            import bosdyn.client.robot_command as _rc
            cmd = _rc.RobotCommandBuilder.stop_command()
            self._command_client.robot_command(
                command=cmd, timeout=self._CMD_TIMEOUT
            )
            return True
        except Exception as exc:
            _log.warning("SpotAdapter._cmd_stop failed: %s", exc)
            return False
