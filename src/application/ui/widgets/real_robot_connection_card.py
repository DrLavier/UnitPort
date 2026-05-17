"""RealRobotConnectionCard — mission_panel 右卡片：实机连接配置.

横向 2 个子区段，从左到右：
  1. _Ros2ConnectionSection      — 域 ID / 中间件 / 地址 / 端口 + 状态行 + [测试连接][连接]
  2. _RobotHardwareInfoSection   — 型号/序列号/固件 + 电量/温度 进度条 + 运行时长
                                   + IMU / 关节状态 / 激光雷达 / 相机 / 力矩传感器 健康状态

数据源：
  - 连接配置：``service.ros2_connection_config.Ros2ConnectionConfig``（user.ini[RobotConn]）
  - 连接状态：``signals.connection_state_changed(state, info)``
  - 选中机器人：``robot_assets.selected_robot()`` + ``signals.robot_selection_changed``
  - 硬件遥测：``service.telemetry`` 的 Heartbeat（battery/temperature/uptime/serial/firmware
    扩展字段 — 当前后端尚未推送，UI 显示 "—" 兜底）
  - 传感器健康：``service.sensor_health.SensorHealthModel`` + ``signals.sensor_status_changed``

颜色与字体严格走 ``Config.get_color`` / ``Config.get_font_size``，禁止 hex 字面量。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import (
    Config,
    I18nLabel,
    LaviComboBox,
    LaviLineEdit,
    log_debug,
    setButton,
    setComboBox,
    setLineEdit,
    setTitleGroupBox,
    tr,
)

from application.service.adapters import (
    AdapterFactory,
    AdapterUnavailable,
    available_strategies,
)
from application.ui.dialogs.connection_settings_dialog import (
    ConnectionSettingsDialog,
)
from application.service.connection.phases import (
    BROWNFIELD_SEQUENCE,
    ConnectionPhase,
)
from application.service.ros2_connection_config import (
    ADAPTER_CHOICES,
    DEFAULT_ADAPTER,
    DEFAULT_DOMAIN_ID,
    DEFAULT_MIDDLEWARE,
    DEFAULT_TRANSPORT,
    MIDDLEWARE_CHOICES,
    TRANSPORT_CHOICES,
    Ros2ConnectionConfig,
    Ros2ConnectionInfo,
    build_profile_from_info,
    get_ros2_connection_controller,
    transport_specific_visibility,
)
from application.service.diagnostics import DiagnosticReport, Severity
from application.ui.dialogs.connection_diagnostics_dialog import (
    ConnectionDiagnosticsDialog,
)
from application.ui.widgets.connection_controller_card import ConnectionControllerCard
from registers import robots as _robots_reg
from application.service.robot_assets import (
    RobotAsset,
    selected_robot,
    selected_robot_sku,
)
from application.service.sensor_health import (
    SENSOR_NAMES,
    get_sensor_health_model,
)
from application.service.signals import get_app_signals


# Fixed sensor display order + i18n keys for the hardware section's sensor list.
# Battery is intentionally excluded — overall battery state is shown via the
# dedicated battery progress bar above, not as a sensor health row.
_SENSOR_DISPLAY: Tuple[Tuple[str, str, str], ...] = (
    ("imu", "mission.conn.sensor.imu", "IMU"),
    ("joints", "mission.conn.sensor.joints", "关节状态"),
    ("lidar", "mission.conn.sensor.lidar", "激光雷达"),
    ("camera", "mission.conn.sensor.camera", "相机"),
    ("torque", "mission.conn.sensor.torque", "力矩传感器"),
)


def _level_color_slot(level: str) -> str:
    return {
        "ok": "safe_zone",
        "warn": "highlight",
        "error": "danger_zone",
    }.get(str(level or "").lower(), "sub_t2")


def _level_text(level: str) -> str:
    return {
        "ok": tr("mission.conn.level.ok", "正常"),
        "warn": tr("mission.conn.level.warn", "警告"),
        "error": tr("mission.conn.level.error", "异常"),
    }.get(str(level or "").lower(), tr("mission.conn.level.idle", "待命"))


def _conn_state_color_slot(state: str) -> str:
    return {
        "connected": "safe_zone",
        "connecting": "highlight",
        "testing": "highlight",
        "error": "danger_zone",
    }.get(str(state or "").lower(), "danger_zone")


def _conn_state_text(state: str) -> str:
    return {
        "disconnected": tr("mission.conn.state.disconnected", "未连接"),
        "testing": tr("mission.conn.state.testing", "测试中…"),
        "connecting": tr("mission.conn.state.connecting", "连接中…"),
        "connected": tr("mission.conn.state.connected", "已连接"),
        "error": tr("mission.conn.state.error", "错误"),
    }.get(str(state or "").lower(), tr("mission.conn.state.disconnected", "未连接"))


def _format_uptime(seconds: Optional[int]) -> str:
    if seconds is None or seconds < 0:
        return "—"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


# ---------------------------------------------------------------------------
# Shared section frame
# ---------------------------------------------------------------------------


class _SectionFrame(QFrame):
    def __init__(
        self,
        title_key: str,
        title_default: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("missionConnSection")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(10)

        self._title = I18nLabel(title_key, default=title_default, parent=self)
        self._title.setObjectName("missionConnSectionTitle")
        self._title.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        outer.addWidget(self._title, 0)

        self._body = QWidget(self)
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(0, 0, 0, 0)
        self._body_layout.setSpacing(8)
        outer.addWidget(self._body, 1)

    def body_layout(self) -> QVBoxLayout:
        return self._body_layout

    def apply_theme(self) -> None:
        bg = Config.get_color("bg_1")
        title_color = Config.get_color("sub_t1")
        font_small = Config.get_font_size("size_small")
        self.setStyleSheet(
            f"QFrame#missionConnSection {{ background-color: {bg}; "
            f"border-radius: 6px; }}"
        )
        self._title.setStyleSheet(
            f"QLabel#missionConnSectionTitle {{ color: {title_color}; "
            f"font-size: {font_small}px; background: transparent; }}"
        )


class _FormRow(QWidget):
    """label 上 / 控件下 的两行式表单 row（与左卡片同款）."""

    def __init__(
        self,
        label_key: str,
        label_default: str,
        control: QWidget,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.label = I18nLabel(label_key, default=label_default, parent=self)
        self.label.setObjectName("missionConnFormLabel")
        self.label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        layout.addWidget(self.label, 0)
        self.control = control
        layout.addWidget(self.control, 0)

    def apply_theme(self) -> None:
        sub = Config.get_color("sub_t2")
        font_small = Config.get_font_size("size_small")
        self.label.setStyleSheet(
            f"QLabel#missionConnFormLabel {{ color: {sub}; "
            f"font-size: {font_small}px; background: transparent; }}"
        )


# ---------------------------------------------------------------------------
# Section 1 — ROS2 connection
#
# Phase 3.8: SSH credentials and Domain ID moved into
# ConnectionSettingsDialog (opened via the card's [Settings] button).
# Reactive SSH credential collection (when AutoRepairLoop hits an
# ssh_required finding) is handled by SshCredentialPromptDialog (Phase 3.7).
# ---------------------------------------------------------------------------


class _Ros2ConnectionSection(_SectionFrame):
    """ROS2 connection fields + status + Test/Connect buttons.

    Phase 3.8 layout:
      Robot                    (single SKU enumerator combo)
      Adapter                  (auto-hidden when SKU offers ≤1 strategy)
      Middleware
      [Transport group]        (TitleGroupBox containing:
        Transport              (combo)
        [Address | Port]       (per-transport namespace)
      )
      status row + phase bar   (unchanged)
      [Settings] [Test] [Connect]

    Domain ID and SSH credentials moved into ConnectionSettingsDialog,
    accessed via the [Settings] button — they are "set once" knobs that
    cluttered the per-connect path.
    """

    test_clicked = pyqtSignal()
    connect_clicked = pyqtSignal()
    disconnect_clicked = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(
            "mission.conn.section.ros2", "ROS2 Connection", parent
        )

        info = Ros2ConnectionConfig.load()

        # --- Robot combo (Phase 3.8) ----------------------------------
        # Single SKU enumerator that replaces the previous Brand+Model
        # cascade. Items are (sku, display_name); persisted as
        # brand+robot_model (split via Ros2ConnectionConfig.set_robot_sku).
        current_sku = self._sku_for_info(info)
        self._cb_robot: LaviComboBox = setComboBox(
            self._robot_combo_items(), height=28, i18n=False, parent=self._body,
        )
        self._cb_robot.setCurrentKey(current_sku)

        # --- Middleware combo -----------------------------------------
        self._cb_mw: LaviComboBox = setComboBox(
            list(MIDDLEWARE_CHOICES), height=28, i18n=False, parent=self._body
        )
        self._set_combo_text(self._cb_mw, info.middleware, DEFAULT_MIDDLEWARE)

        # --- Adapter-strategy combo (Phase 3) -------------------------
        # Hidden when the active SKU exposes ≤1 strategy (the default is
        # auto-persisted in that case so downstream code never sees an
        # empty adapter_strategy field).
        self._cb_adapter: LaviComboBox = setComboBox(
            self._adapter_combo_items(current_sku),
            height=28, i18n=False, parent=self._body,
        )
        self._set_combo_text(
            self._cb_adapter, info.adapter_strategy, DEFAULT_ADAPTER,
        )

        # --- Transport combo + Address + Port (grouped) ---------------
        self._cb_transport: LaviComboBox = setComboBox(
            list(TRANSPORT_CHOICES), height=28, i18n=False, parent=self._body
        )
        self._set_combo_text(
            self._cb_transport, info.transport, DEFAULT_TRANSPORT,
        )
        self._le_addr: LaviLineEdit = setLineEdit(
            text=info.address, placeholder="192.168.55.1", parent=self._body,
        )
        self._le_addr.setFixedHeight(28)
        self._le_port: LaviLineEdit = setLineEdit(
            text=str(info.port), placeholder="9999", parent=self._body,
        )
        self._le_port.setFixedHeight(28)

        # Address + Port share a horizontal row inside the transport group.
        addr_row = _FormRow(
            "mission.conn.field.address", "Address",
            self._le_addr, self._body,
        )
        port_row = _FormRow(
            "mission.conn.field.port", "Port",
            self._le_port, self._body,
        )
        # Held as instance attrs so transport-specific visibility can
        # hide the port column for transports that do not use it (WebRTC
        # listens on hardcoded firmware ports 8081 / 9991).
        self._port_row = port_row
        self._addr_port_row = QWidget(self._body)
        ap_layout = QHBoxLayout(self._addr_port_row)
        ap_layout.setContentsMargins(0, 0, 0, 0)
        ap_layout.setSpacing(8)
        ap_layout.addWidget(addr_row, 2)
        ap_layout.addWidget(port_row, 1)

        transport_row = _FormRow(
            "mission.conn.field.transport", "Transport",
            self._cb_transport, self._body,
        )

        # Transport group — visually binds Transport+Address+Port.
        self._transport_group = setTitleGroupBox(
            "mission.conn.group.transport", default="Transport",
            parent=self._body,
        )
        tg_layout = QVBoxLayout(self._transport_group)
        tg_layout.setContentsMargins(8, 14, 8, 8)
        tg_layout.setSpacing(8)
        tg_layout.addWidget(transport_row, 0)
        tg_layout.addWidget(self._addr_port_row, 0)

        # Body row order — Robot at top (drives everything else).
        robot_row = _FormRow(
            "mission.conn.field.robot", "Robot",
            self._cb_robot, self._body,
        )
        self._adapter_row = _FormRow(
            "mission.conn.field.adapter", "Adapter",
            self._cb_adapter, self._body,
        )
        middleware_row = _FormRow(
            "mission.conn.field.middleware", "Middleware",
            self._cb_mw, self._body,
        )
        # Held as an instance attr so transport-specific visibility can
        # hide the middleware row on transports that do not use a ROS2
        # RMW (e.g. WebRTC, which speaks Unitree sport API directly).
        self._middleware_row = middleware_row

        # Phase 3.9 — sub-label below the middleware row that shows the
        # robot's actual RMW when in Auto mode. Hidden by default; updated
        # by ``_on_diag_for_middleware`` when MangdangRmwDetectionProbe
        # surfaces a result. Only visible while the combo is on "Auto".
        self._lbl_middleware_detail = QLabel("", self._body)
        self._lbl_middleware_detail.setObjectName("missionConnMiddlewareDetail")
        self._lbl_middleware_detail.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self._lbl_middleware_detail.setVisible(False)

        self.body_layout().addWidget(robot_row, 0)
        self.body_layout().addWidget(self._adapter_row, 0)
        self.body_layout().addWidget(middleware_row, 0)
        self.body_layout().addWidget(self._lbl_middleware_detail, 0)
        self.body_layout().addWidget(self._transport_group, 0)

        # All _FormRow widgets are kept in self._rows for apply_theme to
        # iterate (theme refresh updates the label colours).
        self._rows = [
            robot_row, self._adapter_row, middleware_row,
            transport_row, addr_row, port_row,
        ]

        # Persist on change. Robot → cascades to adapter strategy combo
        # refresh + Domain default; transport-change reloads Address+Port
        # from the new namespace.
        self._cb_robot.currentTextChanged.connect(self._on_robot_changed)
        self._cb_mw.currentTextChanged.connect(self._on_middleware_changed)
        self._cb_adapter.currentTextChanged.connect(
            lambda v: Ros2ConnectionConfig.set_field("adapter_strategy", str(v))
        )
        self._cb_transport.currentTextChanged.connect(self._on_transport_changed)
        self._le_addr.textChanged.connect(self._persist_address)
        self._le_port.textChanged.connect(self._persist_port)

        # Apply initial adapter row visibility for the loaded SKU.
        self._refresh_adapter_row_visibility(current_sku)

        # Apply initial transport-specific visibility -- if user.ini had
        # WebRTC as the persisted transport, middleware/port rows must be
        # hidden from the very first paint, not only after the user
        # touches the combo.
        self._apply_transport_visibility(info.transport)

        # Status row — dot + label.
        self._status_label_key = I18nLabel(
            "mission.conn.field.state", default="连接状态", parent=self._body
        )
        self._status_label_key.setObjectName("missionConnFormLabel")
        self.body_layout().addWidget(self._status_label_key, 0)

        self._status_row = QWidget(self._body)
        sr_layout = QHBoxLayout(self._status_row)
        sr_layout.setContentsMargins(0, 0, 0, 0)
        sr_layout.setSpacing(8)
        self._status_dot = QLabel("●", self._status_row)
        self._status_dot.setObjectName("missionConnStatusDot")
        self._status_dot.setAlignment(
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
        )
        self._status_text = QLabel("", self._status_row)
        self._status_text.setObjectName("missionConnStatusText")
        self._status_text.setAlignment(
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
        )
        sr_layout.addWidget(self._status_dot, 0)
        sr_layout.addWidget(self._status_text, 0)
        sr_layout.addStretch(1)
        self.body_layout().addWidget(self._status_row, 0)

        # Phase progress (Phase 1+) — 4 px slim bar + label showing the
        # current phase id. Hidden by default; shown once a connect starts
        # (driven by _on_phase) and auto-hidden after P8 completes.
        self._phase_bar = QProgressBar(self._body)
        self._phase_bar.setObjectName("missionConnPhaseBar")
        self._phase_bar.setRange(0, len(BROWNFIELD_SEQUENCE))
        self._phase_bar.setValue(0)
        self._phase_bar.setTextVisible(False)
        self._phase_bar.setFixedHeight(4)
        self._phase_bar.setVisible(False)
        self.body_layout().addWidget(self._phase_bar, 0)

        self._phase_label = QLabel("", self._body)
        self._phase_label.setObjectName("missionConnPhaseLabel")
        self._phase_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self._phase_label.setVisible(False)
        self.body_layout().addWidget(self._phase_label, 0)

        # Buttons row — Settings opens ConnectionSettingsDialog (Phase 3.8).
        self._btn_settings = setButton(
            "mission.conn.btn.settings", 100, 30,
            kind="normal", spec="none", default="⚙ Settings",
            parent=self._body,
        )
        self._btn_test = setButton(
            "mission.conn.btn.test", 92, 30,
            kind="normal", spec="none", default="测试连接", parent=self._body,
        )
        self._btn_connect = setButton(
            "mission.conn.btn.connect", 92, 30,
            kind="normal", spec="save",  # safe_zone palette
            default="连接", parent=self._body,
        )
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(8)
        btn_row.addWidget(self._btn_settings, 0)
        btn_row.addWidget(self._btn_test, 0)
        btn_row.addWidget(self._btn_connect, 0)
        btn_row.addStretch(1)
        self.body_layout().addLayout(btn_row)
        self.body_layout().addStretch(1)

        self._btn_settings.clicked.connect(self._on_open_settings)
        self._btn_test.clicked.connect(lambda: self.test_clicked.emit())
        # Phase 3.10 — single button toggles between Connect / Disconnect
        # based on the current state; ``_on_connect_button_clicked``
        # dispatches to the right signal.
        self._btn_connect.clicked.connect(self._on_connect_button_clicked)

        # Initial paint.
        self.set_state("disconnected")

        # Subscribe to controller state + phase changes.
        get_app_signals().connection_state_changed.connect(self._on_state)
        get_app_signals().connection_phase_changed.connect(self._on_phase)
        # Phase 3.9 — listen for the Auto-mode RMW detection finding so
        # the middleware sub-label can show "Auto: rmw_fastrtps_cpp" once
        # it's known. We do NOT pop the diagnostics dialog from here
        # (Phase 3.7 ResultDialog handles user-facing terminal events).
        get_app_signals().diagnostics_ready.connect(self._on_diag_for_middleware)

        # Initialise the middleware detail visibility for the loaded
        # middleware choice (no diagnostics report yet → label stays empty
        # but visible-state matches the combo).
        self._refresh_middleware_detail_visibility(self._cb_mw.currentText())

    # ------------------------------------------------------------------
    # Settings dialog (Phase 3.8) — Domain ID + SSH credentials
    # ------------------------------------------------------------------
    def _on_open_settings(self) -> None:
        ConnectionSettingsDialog(self).exec()

    # ------------------------------------------------------------------
    # Middleware (Phase 3.9) — Auto detection sub-label
    # ------------------------------------------------------------------
    def _on_middleware_changed(self, value: str) -> None:
        text = str(value or "")
        Ros2ConnectionConfig.set_field("middleware", text)
        self._refresh_middleware_detail_visibility(text)

    def _refresh_middleware_detail_visibility(self, middleware: str) -> None:
        """Show the detail label only while in Auto mode.

        When switching away from Auto we also drop any cached detection
        text — it would be stale next time the user re-enters Auto.
        """
        if str(middleware or "") == "Auto":
            self._lbl_middleware_detail.setVisible(
                bool(self._lbl_middleware_detail.text())
            )
        else:
            self._lbl_middleware_detail.setText("")
            self._lbl_middleware_detail.setVisible(False)

    def _on_diag_for_middleware(self, report) -> None:
        """Slot — pull the Auto-mode RMW from a DiagnosticReport.

        Looks for a finding produced by ``MangdangRmwDetectionProbe`` whose
        summary starts with ``Auto detected RMW:`` and updates the
        sub-label. Other diagnostics findings are ignored here (Phase 3.7
        ResultDialog handles them).
        """
        try:
            findings = list(getattr(report, "findings", []) or [])
        except Exception:
            return
        rmw = ""
        for f in findings:
            pid = str(getattr(f, "probe_id", "") or "")
            summary = str(getattr(f, "summary", "") or "")
            if pid == "mangdang_rmw_detection" and summary.startswith(
                "Auto detected RMW:"
            ):
                rmw = summary[len("Auto detected RMW:"):].strip()
                break
        if not rmw:
            return
        self._lbl_middleware_detail.setText(
            tr(
                "mission.conn.middleware.auto_detected",
                default="↳ Auto: {rmw}",
            ).format(rmw=rmw)
        )
        self._refresh_middleware_detail_visibility(self._cb_mw.currentText())

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def set_state(self, state: str) -> None:
        self._current_state = state
        self._status_text.setText(_conn_state_text(state))
        self._refresh_status_color(state)
        self._refresh_connect_button(state)

    def _refresh_connect_button(self, state: str) -> None:
        """Update the Connect/Disconnect button to match the current state.

        Phase 3.10 button states:
          disconnected / error  → "连接" enabled, click → connect_clicked
          testing / connecting  → "连接中…" disabled
          connected             → "断开" enabled, click → disconnect_clicked

        The single button is reused (rather than two siblings) so the
        button row layout doesn't shift on state changes.
        """
        s = str(state or "").lower()
        if s == "connected":
            self._btn_connect.setText(
                tr("mission.conn.btn.disconnect", default="断开")
            )
            self._btn_connect.setEnabled(True)
        elif s in {"connecting", "testing"}:
            self._btn_connect.setText(
                tr("mission.conn.btn.connecting", default="连接中…")
            )
            self._btn_connect.setEnabled(False)
        else:
            # disconnected / error / unknown — restore the connect entry point.
            self._btn_connect.setText(
                tr("mission.conn.btn.connect", default="连接")
            )
            self._btn_connect.setEnabled(True)

    def _on_connect_button_clicked(self) -> None:
        s = str(getattr(self, "_current_state", "disconnected") or "").lower()
        if s == "connected":
            self.disconnect_clicked.emit()
        elif s in {"connecting", "testing"}:
            # Disabled in this state, but be defensive — never fire either
            # signal so a stray click can't poke the controller mid-task.
            return
        else:
            self.connect_clicked.emit()

    def apply_theme(self) -> None:
        super().apply_theme()
        for r in self._rows:
            r.apply_theme()
        sub = Config.get_color("sub_t2")
        font_small = Config.get_font_size("size_small")
        font_normal = Config.get_font_size("size_normal")
        self._status_label_key.setStyleSheet(
            f"QLabel#missionConnFormLabel {{ color: {sub}; "
            f"font-size: {font_small}px; background: transparent; }}"
        )
        self._status_text.setStyleSheet(
            f"QLabel#missionConnStatusText {{ color: {Config.get_color('main_t1')}; "
            f"font-size: {font_normal}px; background: transparent; }}"
        )
        self._refresh_status_color(getattr(self, "_current_state", "disconnected"))

        # Phase progress (Phase 1+) — bar + label theme.
        bg_box = Config.get_color("bg_3")
        border = Config.get_color("border_1")
        accent = Config.get_color("safe_zone")
        self._phase_bar.setStyleSheet(
            f"QProgressBar#missionConnPhaseBar {{ background-color: {bg_box}; "
            f"border: 1px solid {border}; border-radius: 2px; }}"
            f"QProgressBar#missionConnPhaseBar::chunk {{ "
            f"background-color: {accent}; border-radius: 2px; }}"
        )
        self._phase_label.setStyleSheet(
            f"QLabel#missionConnPhaseLabel {{ color: {sub}; "
            f"font-size: {font_small}px; background: transparent; }}"
        )
        # Phase 3.9 — middleware Auto detection sub-label.
        self._lbl_middleware_detail.setStyleSheet(
            f"QLabel#missionConnMiddlewareDetail {{ color: {sub}; "
            f"font-size: {font_small}px; background: transparent; "
            f"padding-left: 4px; }}"
        )

        for cb in (
            self._cb_robot, self._cb_mw, self._cb_transport, self._cb_adapter,
        ):
            refresher = getattr(cb, "refresh_style", None)
            if callable(refresher):
                refresher()
        for le in (self._le_addr, self._le_port):
            refresher = getattr(le, "refresh_style", None)
            if callable(refresher):
                refresher()

    # ------------------------------------------------------------------
    # Slots / persistence
    # ------------------------------------------------------------------
    def _on_state(self, state: str, _info: Dict[str, Any]) -> None:
        self.set_state(state)
        s = str(state or "").lower()
        # Terminal / pre-connect states: clear the phase progress display.
        if s in {"disconnected", "error", "connected", "testing"}:
            self._phase_bar.setValue(0)
            self._phase_bar.setVisible(False)
            self._phase_label.setText("")
            self._phase_label.setVisible(False)

    def _on_phase(self, phase_id: str, status: str, msg: str) -> None:
        """connection_phase_changed slot — drives the phase bar+label."""
        st = str(status or "").lower()
        if st == "started":
            self._phase_bar.setVisible(True)
            self._phase_label.setVisible(True)
            self._phase_label.setText(f"{phase_id} …")
            self._refresh_phase_label_color("started")
            return
        if st == "success":
            try:
                idx = next(
                    i for i, p in enumerate(BROWNFIELD_SEQUENCE)
                    if p.value == phase_id
                )
            except StopIteration:
                return
            self._phase_bar.setValue(idx + 1)
            self._phase_label.setText(f"{phase_id} ok")
            self._refresh_phase_label_color("success")
            # After the terminal phase, hide the bar+label after a short
            # delay so the green "connected" status dot takes over cleanly.
            if phase_id == ConnectionPhase.P8_CONFIRMATION.value:
                QTimer.singleShot(500, self._hide_phase_widgets)
            return
        if st == "error":
            self._phase_label.setText(f"{phase_id}: {msg}")
            self._refresh_phase_label_color("error")
            return
        if st == "skipped":
            # Phase is unavailable on the active transport (e.g. SSH-required
            # phases on a USB-only path). Advance the bar so the user sees
            # forward progress, but switch the label to the idle grey colour
            # so it's visually distinct from a green real success.
            try:
                idx = next(
                    i for i, p in enumerate(BROWNFIELD_SEQUENCE)
                    if p.value == phase_id
                )
            except StopIteration:
                idx = self._phase_bar.value() - 1
            self._phase_bar.setVisible(True)
            self._phase_label.setVisible(True)
            self._phase_bar.setValue(idx + 1)
            self._phase_label.setText(f"{phase_id} (skipped: {msg})")
            self._refresh_phase_label_color("skipped")
            return
        # Unknown status: refresh the label only.
        self._phase_label.setText(f"{phase_id} ({st})")

    def _hide_phase_widgets(self) -> None:
        self._phase_bar.setVisible(False)
        self._phase_label.setVisible(False)
        self._phase_bar.setValue(0)
        self._phase_label.setText("")

    def _refresh_phase_label_color(self, kind: str) -> None:
        """Swap the phase label colour based on the latest status.

        ``kind`` ∈ {"started", "success", "skipped", "error"}. Falls back to
        the default sub_t2 slot for unknown values so the label is never
        themed inconsistently with the rest of the card.
        """
        if kind == "skipped":
            color = Config.get_color("mission_conn_phase_skipped")
        elif kind == "error":
            color = Config.get_color("danger_zone")
        elif kind == "success":
            color = Config.get_color("safe_zone")
        else:
            color = Config.get_color("sub_t2")
        font_small = Config.get_font_size("size_small")
        self._phase_label.setStyleSheet(
            f"QLabel#missionConnPhaseLabel {{ color: {color}; "
            f"font-size: {font_small}px; background: transparent; }}"
        )

    def _current_transport(self) -> str:
        return self._cb_transport.currentText() or DEFAULT_TRANSPORT

    def _persist_address(self, text: str) -> None:
        Ros2ConnectionConfig.set_address_for(
            self._current_transport(), str(text),
        )

    def _persist_port(self, text: str) -> None:
        Ros2ConnectionConfig.set_port_for(
            self._current_transport(), str(text).strip(),
        )

    def _on_transport_changed(self, new_transport: str) -> None:
        """Persist the new transport, then reload Address+Port from its
        per-transport namespace into the line edits.

        Block signals on the line edits while we set their text — otherwise
        ``setText`` would fire ``textChanged`` and immediately overwrite the
        new transport's stored value with whatever was on screen.
        """
        transport = str(new_transport).strip() or DEFAULT_TRANSPORT
        Ros2ConnectionConfig.set_field("transport", transport)
        addr, port = Ros2ConnectionConfig.load_address_port(transport)
        self._le_addr.blockSignals(True)
        self._le_port.blockSignals(True)
        try:
            self._le_addr.setText(addr)
            self._le_port.setText(str(port))
        finally:
            self._le_addr.blockSignals(False)
            self._le_port.blockSignals(False)
        self._apply_transport_visibility(transport)

    def _apply_transport_visibility(self, transport: str) -> None:
        """Hide / show fields that do not apply to the chosen transport.

        WebRTC for example does not use a ROS2 middleware (the sport API
        rides a single WebRTC data channel) and the port column is unused
        (the firmware listens on hardcoded ports). The visibility map is
        sourced from :func:`transport_specific_visibility` so the rule
        lives next to its TRANSPORT_DEFAULTS sibling, not in the widget.
        """
        vis = transport_specific_visibility(transport)
        if getattr(self, "_middleware_row", None) is not None:
            self._middleware_row.setVisible(vis.get("middleware", True))
        if getattr(self, "_port_row", None) is not None:
            self._port_row.setVisible(vis.get("port", True))
        # When middleware is hidden the Auto-detect sub-label is meaningless;
        # force-hide it so a stale detection from a prior connect does not
        # linger underneath an absent row.
        if not vis.get("middleware", True):
            self._lbl_middleware_detail.setVisible(False)

    # ------------------------------------------------------------------
    # Robot / Adapter strategy (Phase 3.8)
    # ------------------------------------------------------------------

    def _robot_combo_items(self) -> List[Tuple[str, str]]:
        """Return [(sku, display_name)] for every registered robot.

        First entry is the empty placeholder so an unconfigured profile
        renders as "—" rather than auto-selecting the first robot.
        """
        items: List[Tuple[str, str]] = [
            ("", tr("mission.conn.robot.none", default="—")),
        ]
        try:
            for sku in _robots_reg.list_skus():
                spec = _robots_reg.get_robot(sku)
                if not spec:
                    continue
                display = str(
                    spec.get("name")
                    or f"{spec.get('brand', '')} {spec.get('model', '')}".strip()
                    or sku
                )
                items.append((sku, display))
        except Exception as exc:  # noqa: BLE001
            log_debug(f"[card] list_skus failed: {exc}")
        return items

    @staticmethod
    def _sku_for_info(info: Ros2ConnectionInfo) -> str:
        """Resolve the SKU matching the brand+model in ``info``.

        Returns "" when either field is empty or the pair does not resolve
        to a registered SKU.
        """
        if not info.brand or not info.robot_model:
            return ""
        try:
            sku = _robots_reg.resolve_id(f"{info.brand}.{info.robot_model}")
        except Exception as exc:  # noqa: BLE001
            log_debug(f"[card] resolve_id failed: {exc}")
            return ""
        return sku or ""

    def _adapter_combo_items(self, sku: str) -> List[Tuple[str, str]]:
        """Return [(strategy_label, strategy_label)] for ``sku``.

        Falls back to the static ADAPTER_CHOICES when the SKU is empty or
        exposes no registered strategies, so the combo always has at least
        one option even before a robot is picked.
        """
        if sku:
            strategies = available_strategies(sku)
            if strategies:
                return [(s, s) for s in strategies]
        return [(s, s) for s in ADAPTER_CHOICES]

    def _on_robot_changed(self, _display_text: str) -> None:
        """Persist the new SKU (split into brand+model), then cascade.

        Cascades:
          1. Refresh Adapter strategy combo for the new SKU.
          2. Hide the Adapter row when only one strategy is registered.
          3. Auto-set the Domain ID to the adapter class's DEFAULT_DOMAIN_ID
             (written to user.ini so ConnectionSettingsDialog reflects it
             when next opened).
        """
        try:
            sku = str(self._cb_robot.currentKey()).strip()
        except AttributeError:
            sku = ""
        Ros2ConnectionConfig.set_robot_sku(sku)

        # Refresh strategy combo + visibility.
        self._cb_adapter.blockSignals(True)
        try:
            self._cb_adapter.setItems(self._adapter_combo_items(sku))
        finally:
            self._cb_adapter.blockSignals(False)
        self._refresh_adapter_row_visibility(sku)

        # Auto-set Domain default from the adapter class.
        if not sku:
            return
        spec = _robots_reg.get_robot(sku)
        brand = str((spec or {}).get("brand", "")).strip().lower()
        model = str((spec or {}).get("model", "")).strip()
        if not brand or not model:
            return
        try:
            cls = AdapterFactory.class_for_brand_model(
                brand, model,
                str(self._cb_adapter.currentText()) or DEFAULT_ADAPTER,
            )
            default_domain = int(
                getattr(cls, "DEFAULT_DOMAIN_ID", DEFAULT_DOMAIN_ID)
            )
            Ros2ConnectionConfig.set_field("domain_id", default_domain)
        except (AdapterUnavailable, Exception) as exc:  # noqa: BLE001
            log_debug(f"[card] adapter class lookup failed: {exc}")

    def _refresh_adapter_row_visibility(self, sku: str) -> None:
        """Hide the Adapter row when the SKU offers ≤1 strategy.

        When hidden, the sole available strategy (or DEFAULT_ADAPTER for an
        unselected SKU) is auto-persisted so downstream code never reads an
        empty value.
        """
        strategies: List[str] = []
        if sku:
            try:
                strategies = list(available_strategies(sku))
            except Exception as exc:  # noqa: BLE001
                log_debug(f"[card] available_strategies failed: {exc}")
        visible = len(strategies) >= 2
        self._adapter_row.setVisible(visible)
        if not visible:
            chosen = strategies[0] if strategies else DEFAULT_ADAPTER
            Ros2ConnectionConfig.set_field("adapter_strategy", chosen)

    def _refresh_status_color(self, state: str) -> None:
        slot = _conn_state_color_slot(state)
        color = Config.get_color(slot)
        font_normal = Config.get_font_size("size_normal")
        self._status_dot.setStyleSheet(
            f"QLabel#missionConnStatusDot {{ color: {color}; "
            f"font-size: {font_normal}px; background: transparent; }}"
        )

    @staticmethod
    def _set_combo_text(
        cb: LaviComboBox, text: str, fallback: str
    ) -> None:
        for i in range(cb.count()):
            if cb.itemText(i) == text:
                cb.setCurrentIndex(i)
                return
        for i in range(cb.count()):
            if cb.itemText(i) == fallback:
                cb.setCurrentIndex(i)
                return


# ---------------------------------------------------------------------------
# Section 2 — Robot hardware info
# ---------------------------------------------------------------------------


class _RobotHardwareInfoSection(_SectionFrame):
    """选中机器人的硬件信息卡 — 名称 + 型号/序列号/固件 + 电量/温度/运行时长 + 传感器健康."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(
            "mission.conn.section.hardware", "机器人硬件信息", parent
        )

        # Top: name label only (icon placeholder removed per UX request).
        self._name_label = QLabel("", self._body)
        self._name_label.setObjectName("missionConnRobotName")
        self._name_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self._name_label.setWordWrap(True)
        self.body_layout().addWidget(self._name_label, 0)

        # Field grid: 型号 / 序列号 / 固件版本 (label_left | value_right) × 3.
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)

        self._labels: List[I18nLabel] = []
        self._values: Dict[str, QLabel] = {}
        for row_idx, (key, lkey, ldef) in enumerate((
            ("model", "mission.conn.hw.model", "型号"),
            ("serial", "mission.conn.hw.serial", "序列号"),
            ("firmware", "mission.conn.hw.firmware", "固件版本"),
        )):
            label = I18nLabel(lkey, default=ldef, parent=self._body)
            label.setObjectName("missionConnHwLabel")
            label.setAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )
            value = QLabel("—", self._body)
            value.setObjectName("missionConnHwValue")
            value.setAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )
            grid.addWidget(label, row_idx, 0)
            grid.addWidget(value, row_idx, 1)
            self._labels.append(label)
            self._values[key] = value
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)
        self.body_layout().addLayout(grid)

        # Battery row.
        self._lbl_battery = I18nLabel(
            "mission.conn.hw.battery", default="电量", parent=self._body,
        )
        self._lbl_battery.setObjectName("missionConnHwLabel")
        self._labels.append(self._lbl_battery)
        self._battery_value = QLabel("—", self._body)
        self._battery_value.setObjectName("missionConnHwValue")
        self._battery_bar = QProgressBar(self._body)
        self._battery_bar.setObjectName("missionConnHwBatteryBar")
        self._battery_bar.setRange(0, 100)
        self._battery_bar.setTextVisible(False)
        self._battery_bar.setFixedHeight(6)
        self._battery_bar.setValue(0)

        bat_row = QHBoxLayout()
        bat_row.setContentsMargins(0, 0, 0, 0)
        bat_row.setSpacing(8)
        bat_row.addWidget(self._lbl_battery, 0)
        bat_row.addWidget(self._battery_value, 0)
        bat_row.addWidget(self._battery_bar, 1)
        self.body_layout().addLayout(bat_row)

        # Temperature row.
        self._lbl_temp = I18nLabel(
            "mission.conn.hw.temperature", default="温度", parent=self._body,
        )
        self._lbl_temp.setObjectName("missionConnHwLabel")
        self._labels.append(self._lbl_temp)
        self._temp_value = QLabel("—", self._body)
        self._temp_value.setObjectName("missionConnHwValue")
        self._temp_bar = QProgressBar(self._body)
        self._temp_bar.setObjectName("missionConnHwTempBar")
        self._temp_bar.setRange(0, 100)
        self._temp_bar.setTextVisible(False)
        self._temp_bar.setFixedHeight(6)
        self._temp_bar.setValue(0)

        temp_row = QHBoxLayout()
        temp_row.setContentsMargins(0, 0, 0, 0)
        temp_row.setSpacing(8)
        temp_row.addWidget(self._lbl_temp, 0)
        temp_row.addWidget(self._temp_value, 0)
        temp_row.addWidget(self._temp_bar, 1)
        self.body_layout().addLayout(temp_row)

        # Uptime row.
        self._lbl_uptime = I18nLabel(
            "mission.conn.hw.uptime", default="运行时长", parent=self._body,
        )
        self._lbl_uptime.setObjectName("missionConnHwLabel")
        self._labels.append(self._lbl_uptime)
        self._uptime_value = QLabel("—", self._body)
        self._uptime_value.setObjectName("missionConnHwValue")
        upt_row = QHBoxLayout()
        upt_row.setContentsMargins(0, 0, 0, 0)
        upt_row.setSpacing(8)
        upt_row.addWidget(self._lbl_uptime, 0)
        upt_row.addStretch(1)
        upt_row.addWidget(self._uptime_value, 0)
        self.body_layout().addLayout(upt_row)

        # Sensor health rows (moved here from the standalone sensor section).
        self._sensor_rows: Dict[str, _SensorRow] = {}
        for sensor_id, lkey, ldef in _SENSOR_DISPLAY:
            row = _SensorRow(sensor_id, lkey, ldef, self._body)
            self._sensor_rows[sensor_id] = row
            self.body_layout().addWidget(row, 0)

        self.body_layout().addStretch(1)

        # Initial render based on current selection.
        self._refresh_from_selection()

        # Initial sensor levels from the current model snapshot.
        sensor_model = get_sensor_health_model()
        for sid, row in self._sensor_rows.items():
            row.apply_level(sensor_model.get(sid))

        # Listen for selection changes; telemetry-driven dynamic fields are
        # left as TODOs until the heartbeat broadcaster is wired (see
        # service/telemetry.py — the EventBus emits Heartbeat dataclasses
        # but no Qt signal carries it yet).
        get_app_signals().robot_selection_changed.connect(self._on_selection)
        get_app_signals().sensor_status_changed.connect(self._on_sensor_changed)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def apply_theme(self) -> None:
        super().apply_theme()
        sub = Config.get_color("sub_t2")
        main = Config.get_color("main_t1")
        bg_box = Config.get_color("bg_3")
        border = Config.get_color("border_1")
        font_small = Config.get_font_size("size_small")
        font_normal = Config.get_font_size("size_normal")
        safe = Config.get_color("safe_zone")
        warn = Config.get_color("highlight")

        self._name_label.setStyleSheet(
            f"QLabel#missionConnRobotName {{ color: {main}; "
            f"font-size: {font_normal}px; font-weight: 600; "
            f"background: transparent; }}"
        )
        for label in self._labels:
            label.setStyleSheet(
                f"QLabel#missionConnHwLabel {{ color: {sub}; "
                f"font-size: {font_small}px; background: transparent; }}"
            )
        for value in (
            *self._values.values(),
            self._battery_value,
            self._temp_value,
            self._uptime_value,
        ):
            value.setStyleSheet(
                f"QLabel#missionConnHwValue {{ color: {main}; "
                f"font-size: {font_small}px; background: transparent; }}"
            )

        bar_qss = (
            f"QProgressBar {{ background-color: {bg_box}; "
            f"border: 1px solid {border}; border-radius: 3px; }}"
        )
        self._battery_bar.setStyleSheet(
            bar_qss
            + f"QProgressBar#missionConnHwBatteryBar::chunk {{ "
              f"background-color: {safe}; border-radius: 3px; }}"
        )
        self._temp_bar.setStyleSheet(
            bar_qss
            + f"QProgressBar#missionConnHwTempBar::chunk {{ "
              f"background-color: {warn}; border-radius: 3px; }}"
        )

        # Sensor rows: re-apply current levels so label/status colors track theme.
        sensor_model = get_sensor_health_model()
        for sid, row in self._sensor_rows.items():
            row.apply_level(sensor_model.get(sid))

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _on_selection(self, _sku: str) -> None:
        self._refresh_from_selection()

    def _on_sensor_changed(self, name: str, level: str) -> None:
        if name not in SENSOR_NAMES:
            return
        row = self._sensor_rows.get(name)
        if row is None:
            return
        row.apply_level(level)

    def _refresh_from_selection(self) -> None:
        sku = selected_robot_sku()
        if not sku:
            self._render_empty()
            return
        asset: Optional[RobotAsset] = selected_robot()
        if asset is None:
            self._render_empty()
            return
        # Static fields from the registry. Dynamic fields (serial / firmware
        # / battery / temperature / uptime) require a live Heartbeat — leave
        # them as "—" until that channel emits.
        self._name_label.setText(asset.name or asset.model or asset.sku)
        self._values["model"].setText(asset.model or "—")
        self._values["serial"].setText("—")
        self._values["firmware"].setText("—")
        self._battery_value.setText("—")
        self._battery_bar.setValue(0)
        self._temp_value.setText("—")
        self._temp_bar.setValue(0)
        self._uptime_value.setText(_format_uptime(None))

    def _render_empty(self) -> None:
        self._name_label.setText(
            tr("mission.conn.hw.no_selection", "未选择机器人")
        )
        for v in self._values.values():
            v.setText("—")
        self._battery_value.setText("—")
        self._battery_bar.setValue(0)
        self._temp_value.setText("—")
        self._temp_bar.setValue(0)
        self._uptime_value.setText("—")


# ---------------------------------------------------------------------------
# Sensor row (used inside the hardware section)
# ---------------------------------------------------------------------------


class _SensorRow(QWidget):
    """● sensor_name        状态  — 一行."""

    def __init__(
        self,
        sensor_id: str,
        label_key: str,
        label_default: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.sensor_id = sensor_id
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.dot = QLabel("●", self)
        self.dot.setObjectName("missionConnSensorDot")
        self.dot.setAlignment(
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
        )
        self.dot.setFixedWidth(14)

        self.label = I18nLabel(label_key, default=label_default, parent=self)
        self.label.setObjectName("missionConnSensorLabel")
        self.label.setAlignment(
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
        )

        self.status = QLabel("", self)
        self.status.setObjectName("missionConnSensorStatus")
        self.status.setAlignment(
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight
        )

        layout.addWidget(self.dot, 0)
        layout.addWidget(self.label, 1)
        layout.addWidget(self.status, 0)

    def apply_level(self, level: str) -> None:
        slot = _level_color_slot(level)
        color = Config.get_color(slot)
        main = Config.get_color("main_t1")
        font_small = Config.get_font_size("size_small")
        font_normal = Config.get_font_size("size_normal")
        self.dot.setStyleSheet(
            f"QLabel#missionConnSensorDot {{ color: {color}; "
            f"font-size: {font_normal}px; background: transparent; }}"
        )
        self.label.setStyleSheet(
            f"QLabel#missionConnSensorLabel {{ color: {main}; "
            f"font-size: {font_small}px; background: transparent; }}"
        )
        self.status.setStyleSheet(
            f"QLabel#missionConnSensorStatus {{ color: {color}; "
            f"font-size: {font_small}px; background: transparent; }}"
        )
        self.status.setText(_level_text(level))


# ---------------------------------------------------------------------------
# Card composer
# ---------------------------------------------------------------------------


class RealRobotConnectionCard(QFrame):
    """mission_panel 右卡片：实机连接配置."""

    connect_action_requested = pyqtSignal(str)        # "test" | "connect"

    _ACCENT_W = 4

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("missionConnCard")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        # Title bar with left accent stripe.
        title_host = QFrame(self)
        title_host.setObjectName("missionConnCardTitle")
        title_host.setFrameShape(QFrame.Shape.NoFrame)
        title_host.setFixedHeight(28)
        title_host.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        title_layout = QHBoxLayout(title_host)
        title_layout.setContentsMargins(8, 0, 8, 0)
        title_layout.setSpacing(8)
        self._title_label = I18nLabel(
            "mission.conn.title", default="实机连接配置", parent=title_host
        )
        self._title_label.setObjectName("missionConnCardTitleLabel")
        title_layout.addWidget(self._title_label, 0, Qt.AlignmentFlag.AlignVCenter)
        title_layout.addStretch(1)
        outer.addWidget(title_host, 0)

        body = QWidget(self)
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(8)

        self._sec_ros2 = _Ros2ConnectionSection(body)
        self._sec_hw = _RobotHardwareInfoSection(body)

        body_layout.addWidget(self._sec_ros2, 4)
        body_layout.addWidget(self._sec_hw, 6)
        outer.addWidget(body, 1)

        # sec3 — capability-driven Controller sub-card (Phase 3). Constructed
        # on demand when the connection reaches "connected" state; destroyed
        # on disconnect / error so we never reference a stale adapter handle.
        self._sec_controller: Optional[ConnectionControllerCard] = None
        self._sec_controller_host = QWidget(self)
        self._sec_controller_host.setObjectName("missionConnControllerHost")
        sec3_layout = QVBoxLayout(self._sec_controller_host)
        sec3_layout.setContentsMargins(0, 0, 0, 0)
        sec3_layout.setSpacing(0)
        self._sec_controller_host.setVisible(False)
        outer.addWidget(self._sec_controller_host, 0)

        # Wire ROS2 buttons → controller.
        self._sec_ros2.test_clicked.connect(self._on_test)
        self._sec_ros2.connect_clicked.connect(self._on_connect)
        self._sec_ros2.disconnect_clicked.connect(self._on_disconnect)

        # sec3 lifecycle — show after connected, hide on disconnect/error.
        get_app_signals().connection_state_changed.connect(
            self._on_connection_state_for_sec3
        )

        # Phase 3.7 — diagnostics report dialog is no longer auto-popped
        # on >=WARNING. The dialog is now a read-only details view opened
        # from ConnectionResultDialog's [View Details] button. We still
        # subscribe to diagnostics_ready to keep ``_latest_diag_report``
        # current so any user-triggered "show details" path has data.
        self._diag_dialog: Optional[ConnectionDiagnosticsDialog] = None
        self._latest_diag_report: Optional[DiagnosticReport] = None
        # Track the most recent ConnectionResult dialog so we don't stack
        # two on top of each other when the controller emits twice.
        self._result_dialog: Optional[Any] = None
        # Same idea for the SSH credential prompt — modal, max one alive.
        self._ssh_prompt_dialog: Optional[Any] = None
        get_app_signals().diagnostics_ready.connect(self._on_diag_report)
        get_app_signals().connection_needs_ssh.connect(self._on_needs_ssh)
        get_app_signals().connection_result.connect(self._on_connection_result)

        self.apply_theme()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def apply_theme(self) -> None:
        bg_card = Config.get_color("bg_2")
        border = Config.get_color("border_1")
        accent = Config.get_color("safe_zone")
        title_color = Config.get_color("main_t1")
        font_normal = Config.get_font_size("size_normal")

        self.setStyleSheet(
            f"QFrame#missionConnCard {{ background-color: {bg_card}; "
            f"border: 1px solid {border}; border-radius: 8px; }}"
            f"QFrame#missionConnCardTitle {{ background: transparent; "
            f"border-left: {self._ACCENT_W}px solid {accent}; }}"
        )
        self._title_label.setStyleSheet(
            f"QLabel#missionConnCardTitleLabel {{ color: {title_color}; "
            f"font-size: {font_normal}px; font-weight: 600; "
            f"background: transparent; }}"
        )
        self._sec_ros2.apply_theme()
        self._sec_hw.apply_theme()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _on_test(self) -> None:
        log_debug("[mission.conn] test connection requested")
        self.connect_action_requested.emit("test")
        get_ros2_connection_controller().test()

    def _on_connect(self) -> None:
        log_debug("[mission.conn] connect requested")
        self.connect_action_requested.emit("connect")
        get_ros2_connection_controller().connect()

    def _on_disconnect(self) -> None:
        log_info("[mission.conn] disconnect requested by user")
        self.connect_action_requested.emit("disconnect")
        get_ros2_connection_controller().disconnect()

    # ------------------------------------------------------------------
    # sec3 lifecycle
    # ------------------------------------------------------------------

    def _on_connection_state_for_sec3(
        self, state: str, _info: Dict[str, Any],
    ) -> None:
        """Mount / dismount the sec3 ConnectionControllerCard with state."""
        s = str(state or "").lower()
        if s == "connected":
            self._mount_controller_card()
        elif s in {"disconnected", "error"}:
            self._unmount_controller_card()

    def _mount_controller_card(self) -> None:
        if self._sec_controller is not None:
            return
        ctrl = get_ros2_connection_controller()
        adapter = ctrl.current_adapter()
        if adapter is None:
            log_debug(
                "[mission.conn] connected fired but no adapter — sec3 deferred"
            )
            return
        self._sec_controller = ConnectionControllerCard(
            adapter=adapter, parent=self._sec_controller_host,
        )
        self._sec_controller_host.layout().addWidget(self._sec_controller)
        self._sec_controller_host.setVisible(True)

    def _unmount_controller_card(self) -> None:
        if self._sec_controller is None:
            self._sec_controller_host.setVisible(False)
            return
        try:
            self._sec_controller.shutdown()
        except Exception as exc:  # noqa: BLE001
            log_debug(f"[mission.conn] sec3 shutdown raised: {exc}")
        self._sec_controller_host.layout().removeWidget(self._sec_controller)
        self._sec_controller.deleteLater()
        self._sec_controller = None
        self._sec_controller_host.setVisible(False)

    # ------------------------------------------------------------------
    # Phase 3.7 — diagnostics + result + SSH-prompt routing
    # ------------------------------------------------------------------

    def _on_diag_report(self, report: DiagnosticReport) -> None:
        """Cache the latest diagnostic report; never auto-pops a dialog.

        The dialog is opened explicitly from
        :class:`ConnectionResultDialog`'s [View Details] button.
        """
        if report is None:
            return
        self._latest_diag_report = report

    def _on_diag_dialog_closed(self, _result: int) -> None:
        self._diag_dialog = None

    def _on_needs_ssh(
        self, server_key: str, suggested_user: str, reason: str,
    ) -> None:
        """Pop SshCredentialPromptDialog when AutoRepairLoop needs creds.

        Phase 3.8 removed the in-card SSH section; the modal prompt stands
        alone now. Proactive credential management lives in
        ConnectionSettingsDialog (opened via the [Settings] button).
        """
        if self._ssh_prompt_dialog is not None:
            try:
                if self._ssh_prompt_dialog.isVisible():
                    return
            except Exception:
                pass
        # Lazy import — keep dialog dependencies out of the card module
        # boot path so a hypothetical import failure doesn't break the card.
        from application.ui.dialogs.ssh_credential_prompt_dialog import (
            SshCredentialPromptDialog,
        )
        dlg = SshCredentialPromptDialog(
            server_key=server_key,
            suggested_user=suggested_user,
            reason=reason,
            parent=self,
        )
        self._ssh_prompt_dialog = dlg
        dlg.finished.connect(lambda _r: self._clear_ssh_prompt())
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _clear_ssh_prompt(self) -> None:
        self._ssh_prompt_dialog = None

    def _on_connection_result(self, result: object) -> None:
        """Pop ConnectionResultDialog when AutoConnect emits its verdict."""
        if result is None:
            return
        # Re-pop if a dialog already shows the previous result (the user
        # clicked Retry → new result; old dialog has self.accept'd).
        from application.ui.dialogs.connection_result_dialog import (
            ConnectionResultDialog,
        )
        try:
            if self._result_dialog is not None:
                self._result_dialog.close()
        except Exception:
            pass
        dlg = ConnectionResultDialog(result, parent=self)  # type: ignore[arg-type]
        self._result_dialog = dlg
        dlg.finished.connect(lambda _r: self._clear_result_dialog())
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _clear_result_dialog(self) -> None:
        self._result_dialog = None


__all__ = ["RealRobotConnectionCard"]
