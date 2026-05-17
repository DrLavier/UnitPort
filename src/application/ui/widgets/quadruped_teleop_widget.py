"""QuadrupedTeleopWidget — capability-driven teleop sub-widget.

Renders the 2D quadruped teleop layout (vx / vy / vyaw + Estop) when the
active adapter declares ``CapabilitySet.teleop`` with ≤3 axes from the
linear-x / linear-y / angular-z subset (i.e. a wheeled / quadruped base).

Design:

- **Driven by capabilities**: axes / max_rates come from
  ``TeleopCapability``; widget reads them per-instance and renders the
  correct number of meters. No hardcoded "vx vy vyaw" assumption in
  paint code — replacing the capability schema with a 6DOF arm spec
  swaps :class:`QuadrupedTeleopWidget` for an arm widget without
  touching this module.
- **Live values from GlobalInputManager**: 100 ms refresh polls
  ``get_live_values()`` and renders the magnitude as a coloured bar.
- **Estop button**: publishes one zero twist via ``adapter.disable_teleop``
  (which already does the safe-stop publish) — no extra topic-specific
  code in the UI.
- **Take Control / Release Control** lives in the parent
  :class:`ConnectionControllerCard`; this widget only displays values.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QFrame, QGridLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from unitport_sdk import Config, log_warning, setButton, tr

from application.service.adapters.capabilities import TeleopCapability
from application.service.input.manager import get_global_input_manager


_REFRESH_MS = 100


class QuadrupedTeleopWidget(QWidget):
    """Live axis readouts + Estop, capability-driven.

    Construction args:
        teleop: the active adapter's TeleopCapability (drives axis labels +
                clamps used for the colour blend).
        on_estop: callback fired when the Estop button is pressed (typically
                  ``adapter.disable_teleop``).
    """

    def __init__(
        self,
        teleop: TeleopCapability,
        on_estop,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._teleop = teleop
        self._on_estop = on_estop
        self._mgr = get_global_input_manager()
        self._value_labels: dict[str, QLabel] = {}
        self._build_ui()
        self._timer = QTimer(self)
        self._timer.setInterval(_REFRESH_MS)
        self._timer.timeout.connect(self._refresh_live_values)
        self._timer.start()
        # First paint — don't wait for the first 100 ms tick.
        self._refresh_live_values()
        self.apply_theme()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        # Live values grid — N columns wide where N = len(axes).
        grid_host = QFrame(self)
        grid_host.setObjectName("missionConnTeleopGrid")
        grid = QGridLayout(grid_host)
        grid.setContentsMargins(8, 6, 8, 6)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(2)

        for col, axis in enumerate(self._teleop.axes):
            name_lbl = QLabel(axis, grid_host)
            name_lbl.setObjectName("missionConnTeleopAxisName")
            name_lbl.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom)
            grid.addWidget(name_lbl, 0, col)

            val_lbl = QLabel("0.000", grid_host)
            val_lbl.setObjectName("missionConnTeleopAxisValue")
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
            val_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            grid.addWidget(val_lbl, 1, col)
            self._value_labels[axis] = val_lbl

        outer.addWidget(grid_host, 0)

        # Estop button — full width, distinct red palette.
        self._btn_estop = setButton(
            "mission.conn.controller.estop", 0, 32,
            kind="normal", spec="danger", default="Estop", parent=self,
        )
        self._btn_estop.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._btn_estop.clicked.connect(self._on_estop_clicked)
        outer.addWidget(self._btn_estop, 0)

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def apply_theme(self) -> None:
        bg = Config.get_color("bg_3")
        border = Config.get_color("border_1")
        sub = Config.get_color("sub_t2")
        font_small = Config.get_font_size("size_small")
        font_normal = Config.get_font_size("size_normal")
        # The grid host frame is themed as a dark inset panel; the axis
        # value labels paint with a magnitude-driven colour in
        # _refresh_live_values so we don't set their colour here.
        self.setStyleSheet(
            f"QFrame#missionConnTeleopGrid {{ background-color: {bg}; "
            f"border: 1px solid {border}; border-radius: 4px; }}"
            f"QLabel#missionConnTeleopAxisName {{ color: {sub}; "
            f"font-size: {font_small}px; background: transparent; }}"
            f"QLabel#missionConnTeleopAxisValue {{ "
            f"font-size: {font_normal}px; background: transparent; "
            f"font-weight: bold; }}"
        )
        self._refresh_live_values()  # repaint colour with live magnitude

    # ------------------------------------------------------------------
    # Live update
    # ------------------------------------------------------------------

    def _refresh_live_values(self) -> None:
        try:
            live = self._mgr.get_live_values()
        except Exception as exc:  # noqa: BLE001
            log_warning(f"[teleop_widget] get_live_values: {exc}")
            return
        idle_color = Config.get_color("mission_conn_teleop_axis_idle")
        active_color = Config.get_color("safe_zone")
        for axis, lbl in self._value_labels.items():
            value = float(live.get(axis, 0.0))
            lbl.setText(f"{value:+.3f}")
            # Magnitude blend: 0% → idle, 100% → active. Per-axis max rate
            # is the brand's hard envelope (TeleopCapability.max_rates).
            try:
                idx = self._teleop.axes.index(axis)
                cap = self._teleop.max_rates[idx] if idx < len(self._teleop.max_rates) else 1.0
            except ValueError:
                cap = 1.0
            magnitude = min(1.0, abs(value) / max(cap, 1e-6))
            color = active_color if magnitude > 0.05 else idle_color
            lbl.setStyleSheet(
                f"QLabel#missionConnTeleopAxisValue {{ "
                f"color: {color}; "
                f"font-size: {Config.get_font_size('size_normal')}px; "
                f"background: transparent; font-weight: bold; }}"
            )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_estop_clicked(self) -> None:
        try:
            self._on_estop()
        except Exception as exc:  # noqa: BLE001
            log_warning(f"[teleop_widget] estop callback: {exc}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def stop_polling(self) -> None:
        """Stop the live-value refresh timer.

        Called by the parent card when the connection drops so we don't
        keep polling GlobalInputManager after the adapter is gone.
        """
        if self._timer.isActive():
            self._timer.stop()


__all__ = ["QuadrupedTeleopWidget"]
