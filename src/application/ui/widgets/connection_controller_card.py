# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""ConnectionControllerCard — sec3 capability-driven teleop card.

Shown inside RealRobotConnectionCard once the active connection reaches
``state == "connected"``. Hidden again on disconnect / error.

The card is **adapter-capability-driven**: it asks the live adapter
``capabilities()``, then renders the matching widget(s):

- ``CapabilitySet.teleop != None`` → :class:`QuadrupedTeleopWidget`
  (the only teleop layout shipped in Phase 3; arm / 6DOF arrive in Phase 6+
  with their own widgets dispatched here)
- ``CapabilitySet.estop != None`` → estop hook wired into the teleop widget
- Take Control / Release Control toggle drives ``adapter.enable_teleop`` /
  ``adapter.disable_teleop`` with a CommandBus-backed twist provider

Card never hardcodes ``"mangdang"`` or any brand string — every render path
goes through the capability descriptor.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from unitport_sdk import Config, I18nLabel, log_info, log_warning, setButton, tr

from application.service.adapters.base_adapter import BaseAdapter
from application.service.input.manager import get_global_input_manager
from application.ui.widgets.quadruped_teleop_widget import QuadrupedTeleopWidget


def _make_twist_provider(axes):
    """Build a CommandBus → Twist-dict provider for the given axis list.

    Maps axis labels onto the geometry_msgs/Twist field convention:

      vx  → linear.x          vy → linear.y         vz → linear.z
      vyaw → angular.z        vroll → angular.x     vpitch → angular.y

    Returns ``None`` when no axis has non-trivial input — that's the deadman
    tick the TeleopPump uses to stop sending commands when the user lets go.
    """
    mgr = get_global_input_manager()
    axis_set = tuple(axes)

    def _twist_field(axis: str, value: float, twist: dict) -> None:
        if axis == "vx":
            twist["linear"]["x"] = value
        elif axis == "vy":
            twist["linear"]["y"] = value
        elif axis == "vz":
            twist["linear"]["z"] = value
        elif axis == "vyaw":
            twist["angular"]["z"] = value
        elif axis == "vroll":
            twist["angular"]["x"] = value
        elif axis == "vpitch":
            twist["angular"]["y"] = value
        # Unknown axis → silently dropped; the adapter would not have
        # declared an axis it can't actuate.

    def _provide() -> Optional[dict]:
        try:
            live = mgr.get_live_values()
        except Exception:  # noqa: BLE001
            return None
        any_input = False
        twist = {
            "linear":  {"x": 0.0, "y": 0.0, "z": 0.0},
            "angular": {"x": 0.0, "y": 0.0, "z": 0.0},
        }
        for axis in axis_set:
            value = float(live.get(axis, 0.0))
            if abs(value) > 1e-6:
                any_input = True
            _twist_field(axis, value, twist)
        return twist if any_input else None

    return _provide


class ConnectionControllerCard(QFrame):
    """sec3 — capability-driven controller widget.

    Constructor takes the live adapter (already past ``open_session``) so
    the card can read ``capabilities()`` and wire the right schema.
    """

    take_control_changed = pyqtSignal(bool)

    def __init__(
        self,
        adapter: BaseAdapter,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("missionConnControllerCard")
        self._adapter = adapter
        self._teleop_widget: Optional[QuadrupedTeleopWidget] = None
        self._is_taking_control = False
        self._build_ui()
        self.apply_theme()
        # SYSTEM-channel estop (Slice 0): a bound estop key/button in the input
        # layer reaches the deploy adapter through the SAME stop path as this
        # card's Estop button. Subscribing here converges input-driven estop and
        # the UI button on ``adapter.disable_teleop()``. Disconnected in shutdown.
        from application.service.signals import get_app_signals
        get_app_signals().system_estop.connect(self._on_system_estop)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 8, 10, 8)
        outer.setSpacing(8)

        # Header row — title + Take/Release Control toggle.
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        self._title = I18nLabel(
            "mission.conn.section.controller", default="Controller", parent=self,
        )
        self._title.setObjectName("missionConnControllerTitle")
        header.addWidget(self._title, 0)
        header.addStretch(1)
        # [Take Control] doubles as the diagnose entry point. Pressing it
        # always kicks off run_diagnostics() in parallel with enable_teleop;
        # the existing diagnostics_ready -> ConnectionDiagnosticsDialog flow
        # auto-pops on >=WARNING, so a healthy connection stays silent and a
        # broken one surfaces the same dialog the user would otherwise have
        # to find a separate button for. Avoids a button-zoo in the header.
        self._btn_toggle = setButton(
            "mission.conn.controller.take_control", 130, 28,
            kind="normal", spec="save", default="Take Control", parent=self,
        )
        self._btn_toggle.clicked.connect(self._on_toggle_clicked)
        header.addWidget(self._btn_toggle, 0)
        outer.addLayout(header)

        # Capability-dispatch body. Phase 3 only ships QuadrupedTeleopWidget;
        # adapter capabilities not handled here render a "not yet supported"
        # placeholder so the card still draws something coherent.
        caps = self._adapter.capabilities()
        if caps.teleop is not None:
            self._teleop_widget = QuadrupedTeleopWidget(
                teleop=caps.teleop,
                on_estop=self._on_estop,
                parent=self,
            )
            outer.addWidget(self._teleop_widget, 0)
        else:
            placeholder = I18nLabel(
                "mission.conn.controller.no_teleop",
                default="This adapter does not declare teleop capability.",
                parent=self,
            )
            placeholder.setObjectName("missionConnControllerPlaceholder")
            outer.addWidget(placeholder, 0)

        if caps.arm is not None:
            # Phase 6+ — arm teleop UI dispatch lands here.
            arm_lbl = I18nLabel(
                "mission.conn.controller.arm_pending",
                default="Arm teleop arrives in Phase 6.",
                parent=self,
            )
            outer.addWidget(arm_lbl, 0)

        if caps.pid_tuning is not None:
            # Phase 6 — PID sub-card dispatched here.
            pid_lbl = I18nLabel(
                "mission.conn.controller.pid_pending",
                default="PID tuning UI arrives in Phase 6.",
                parent=self,
            )
            outer.addWidget(pid_lbl, 0)

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def apply_theme(self) -> None:
        bg = Config.get_color("bg_2")
        border = Config.get_color("border_1")
        title_color = Config.get_color("main_t1")
        font_normal = Config.get_font_size("size_normal")
        self.setStyleSheet(
            f"QFrame#missionConnControllerCard {{ background-color: {bg}; "
            f"border: 1px solid {border}; border-radius: 6px; }}"
            f"QLabel#missionConnControllerTitle {{ color: {title_color}; "
            f"font-size: {font_normal}px; font-weight: bold; "
            f"background: transparent; }}"
        )
        if self._teleop_widget is not None:
            self._teleop_widget.apply_theme()

    # ------------------------------------------------------------------
    # Take / Release Control
    # ------------------------------------------------------------------

    def _on_toggle_clicked(self) -> None:
        if self._is_taking_control:
            self._release_control()
        else:
            self._take_control()

    def _take_control(self) -> None:
        caps = self._adapter.capabilities()
        if caps.teleop is None:
            log_warning(
                "[controller_card] take_control: adapter has no teleop capability"
            )
            return
        provider = _make_twist_provider(caps.teleop.axes)
        ok = False
        try:
            ok = self._adapter.enable_teleop(provider, rate_hz=50.0)
        except Exception as exc:  # noqa: BLE001
            log_warning(f"[controller_card] enable_teleop raised: {exc}")
        if not ok:
            log_warning("[controller_card] adapter.enable_teleop returned False")
            return
        self._is_taking_control = True
        # Toggle the button label to "Release Control"; spec stays save.
        # I18nButton wrappers don't support live retext, so swap text directly.
        self._btn_toggle.setText(
            tr("mission.conn.controller.release_control", default="Release Control")
        )
        log_info("[controller_card] took control")
        self.take_control_changed.emit(True)
        # Phase 3.7: Take Control no longer triggers a fresh P9 — the
        # autonomous AutoRepairLoop already ran during connect. The
        # adapter's _validate_cmd_writer_match still polls 300ms after
        # pump start as a safety net; if the writer never matches a robot
        # reader it emits a connection_result so the ResultDialog opens.

    def _release_control(self) -> None:
        try:
            self._adapter.disable_teleop()
        except Exception as exc:  # noqa: BLE001
            log_warning(f"[controller_card] disable_teleop raised: {exc}")
        self._is_taking_control = False
        self._btn_toggle.setText(
            tr("mission.conn.controller.take_control", default="Take Control")
        )
        log_info("[controller_card] released control")
        self.take_control_changed.emit(False)

    def _on_estop(self) -> None:
        """Estop = release control + adapter publishes one zero twist."""
        if self._is_taking_control:
            self._release_control()
        else:
            # Even when not taking control, push one zero twist so any
            # ambient pump (future autonomy) clears its last command.
            try:
                self._adapter.disable_teleop()
            except Exception as exc:  # noqa: BLE001
                log_warning(f"[controller_card] estop disable_teleop: {exc}")
        log_info("[controller_card] estop fired")

    def _on_system_estop(self, source: str = "") -> None:
        """SYSTEM estop from the input layer → same adapter stop as the button."""
        self._on_estop()

    # ------------------------------------------------------------------
    # Lifecycle (called by parent card on disconnect)
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Release control + stop the teleop widget's poll timer."""
        try:
            from application.service.signals import get_app_signals
            get_app_signals().system_estop.disconnect(self._on_system_estop)
        except Exception:  # noqa: BLE001 — not connected / already torn down
            pass
        if self._is_taking_control:
            self._release_control()
        if self._teleop_widget is not None:
            self._teleop_widget.stop_polling()


__all__ = ["ConnectionControllerCard"]
