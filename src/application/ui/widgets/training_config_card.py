"""TrainingConfigPerspectiveCard — mission_panel 左卡片：训练配置透视.

横向 4 个子区段：
  1. _TrainingStatusSection   — 状态徽标 + 7 项实时指标 + [Start][Pause] 按钮
                                 （合并 Start/Stop，状态联动顶部 main_row 按钮）
  2. _TaskConfigSection        — 任务环境 (readonly) / 任务名 (readonly) / 链路 / 训练设备
  3. _HyperparamSection        — 超参字段，与画布 ``is_trainer`` 节点双向同步
  4. _RobotConfigSection       — Training Asset / Asset Files / Joint Mapping / Target Height
                                  → 与画布 ``robot`` 节点 4 个 param 双向同步,
                                    数据直接落盘到 ``<canvas>.canvas.json``。

关键设计：
  - 卡片 ``set_canvas(page, file_id)`` 切换画布上下文；超参 / robot 字段从节点
    读，订阅 ``signals.canvas_param_changed`` + ``canvas_topology_changed``
    双向同步。
  - 卡片 ``bind_run_buttons(start_btn, stop_btn)`` 把状态区按钮改为转发顶部按钮
    click，并通过 EventFilter 镜像 enabled 状态。
  - 卡片 ``bind_link_combo(top_combo)`` 把卡片"链路"combo 与顶部 [Local/Cloud]
    选单做双向同步（``_syncing`` 标志位防回环）。
  - body_mapping 编辑同时写 ``RobotAssetService.set_body_ir_overrides``（单一
    真相源 state.json）+ 镜像 JSON 到 ``body_mapping`` param（与画布
    ``BodyMappingTableRow._persist`` 完全一致）。
  - 颜色 / 字体 / 文本 一律走 SDK：Config.get_color / get_font_size / tr / I18nLabel。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import QEvent, QObject, Qt, pyqtSignal
from PyQt6.QtGui import QCursor, QDoubleValidator, QIntValidator
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from unitport_sdk import (
    Config,
    I18n,
    I18nLabel,
    LaviComboBox,
    LaviLineEdit,
    log_debug,
    log_warning,
    setButton,
    setComboBox,
    setLineEdit,
    tr,
)

from application.service.signals import get_app_signals
from application.service.training_status_model import (
    get_training_status_model,
    initial_status,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_USER_INI_SECTION = "Training"               # 仅训练设备落 user.ini（用户偏好）


def _param_kind_from_type(param_type: str) -> str:
    """ParamSpec.type → mission_panel 内部 kind / validator 类别.

    int → 'int'；float → 'float'；其余（string / enum / bool / json）统一用
    'string'（_parse_value 会按需解析；ent_coef='auto' 这种走此分支）。
    """
    t = (param_type or "").strip().lower()
    if t == "int":
        return "int"
    if t == "float":
        return "float"
    return "string"


def _collect_trainer_params(page: Any) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """遍历画布找 ``manifest.is_trainer == True`` 的节点 + 收集暴露字段。

    收集判据（来自 ParamSpec.meta）：
      * ``mission_expose=True`` 必填
      * ``mission_label_i18n`` 必填（缺则字段不暴露 — 强制 i18n 化）

    多 trainer 节点时取第一个并 log_warning。无 trainer 或无 expose 字段返回 (None, [])。

    Returns ``(trainer_node_id, [{key, kind, default, label_i18n, label_default}, ...])``，
    顺序按 manifest.parameters 列表顺序。
    """
    if page is None:
        return None, []
    instances = getattr(page, "_instances", None) or {}
    trainer_node_id: Optional[str] = None
    trainer_manifest = None
    for inst_id, item in instances.items():
        manifest = getattr(item, "manifest", None)
        if manifest is None:
            continue
        if not bool(getattr(manifest, "is_trainer", False)):
            continue
        if trainer_node_id is None:
            trainer_node_id = inst_id
            trainer_manifest = manifest
        else:
            log_warning(
                f"[mission.train] 多个 trainer 节点 — 已取 '{trainer_node_id}'，"
                f"忽略后续 '{inst_id}' (schema={getattr(manifest, 'id', '?')})"
            )

    if trainer_node_id is None or trainer_manifest is None:
        return None, []

    exposed: List[Dict[str, Any]] = []
    for spec in getattr(trainer_manifest, "parameters", []) or []:
        meta = getattr(spec, "meta", None) or {}
        if not bool(meta.get("mission_expose")):
            continue
        i18n_key = str(meta.get("mission_label_i18n") or "").strip()
        if not i18n_key:
            continue
        exposed.append({
            "key": str(spec.key),
            "kind": _param_kind_from_type(getattr(spec, "type", "")),
            "default": getattr(spec, "default", None),
            "label_i18n": i18n_key,
            # description 是 ParamSpec 上的中文裸串（"学习率" 等），用作 i18n 兜底。
            "label_default": str(getattr(spec, "description", "") or spec.key),
        })
    return trainer_node_id, exposed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state_label(state: str) -> str:
    return {
        "idle": tr("mission.train.state.idle", "未运行"),
        "running": tr("mission.train.state.running", "运行中"),
        "paused": tr("mission.train.state.paused", "已暂停"),
        "finished": tr("mission.train.state.finished", "已结束"),
        "failed": tr("mission.train.state.failed", "失败"),
    }.get(state, tr("mission.train.state.idle", "未运行"))


def _state_color_slot(state: str) -> str:
    return {
        "idle": "safe_zone",
        "running": "highlight",
        "paused": "sub_t1",
        "finished": "safe_zone",
        "failed": "danger_zone",
    }.get(state, "safe_zone")


def _format_elapsed(seconds: int) -> str:
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _empty_text() -> str:
    return tr("mission.train.placeholder.empty", "—")


def _parse_value(text: str, kind: str) -> Any:
    """Parse user-entered text per spec kind.

    Returns the parsed value, or raises ValueError on failure. ``ent_coef``-style
    string fields accept either ``"auto"`` or any float literal (kept as float).
    """
    s = (text or "").strip()
    if not s:
        raise ValueError("empty")
    if kind == "int":
        return int(s)
    if kind == "float":
        return float(s)
    if kind == "string":
        if s.lower() == "auto":
            return "auto"
        # strict: anything else must be a parseable float
        return float(s)
    raise ValueError(f"unknown kind: {kind!r}")


def _format_value(value: Any, kind: str) -> str:
    if value is None:
        return ""
    if kind == "int":
        try:
            return str(int(value))
        except Exception:
            return str(value)
    if kind == "float":
        try:
            f = float(value)
        except Exception:
            return str(value)
        # 用足够精度，尽量避免 1.0e-4 → 0.0001 的语义改变
        return f"{f:g}"
    return str(value)


# Slider step = 0.01 m (matches Robot manifest meta.step). Slider ticks are
# integers; ``int(value × _HEIGHT_TICKS_PER_M)`` converts m ↔ ticks.
_HEIGHT_TICKS_PER_M = 100
# Fallback height-slider bounds (m) when the asset's MJCF / URDF can't be
# parsed. Mirrors the Robot node manifest's ``target_height.meta`` defaults
# so the widget still works on assets without packaged geometry.
_HEIGHT_RANGE_FALLBACK = (0.0, 5.0)


def _parse_asset_nominal_height(asset: Any) -> Optional[float]:
    """Best-effort nominal standing Z (m) extracted from MJCF, else URDF.

    MJCF: first ``<body>`` under ``<worldbody>`` with a positive Z component
    in its ``pos`` attribute — that's the floating-base body Z (e.g. Go2's
    trunk at 0.34 m).
    URDF: largest positive Z found among any ``<joint><origin xyz="…">`` —
    a coarse proxy for the leg/base stack height.

    Returns ``None`` when nothing parseable found; callers fall back to the
    manifest meta range.
    """
    import xml.etree.ElementTree as ET

    mjcf_path = getattr(asset, "mjcf_path", None)
    if mjcf_path is not None:
        p = Path(str(mjcf_path))
        if p.exists():
            try:
                root = ET.parse(str(p)).getroot()
                wb = root.find(".//worldbody")
                if wb is not None:
                    for body in wb.iter("body"):
                        pos = body.get("pos", "")
                        parts = pos.split() if pos else []
                        if len(parts) < 3:
                            continue
                        try:
                            z = float(parts[2])
                        except ValueError:
                            continue
                        if z > 0.0:
                            return z
            except Exception as exc:
                log_warning(f"[mission.robot] MJCF parse failed: {exc!r}")

    urdf_path = getattr(asset, "urdf_path", None)
    if urdf_path is not None:
        p = Path(str(urdf_path))
        if p.exists():
            try:
                root = ET.parse(str(p)).getroot()
                max_z = 0.0
                for joint in root.iter("joint"):
                    origin = joint.find("origin")
                    if origin is None:
                        continue
                    xyz = origin.get("xyz", "")
                    parts = xyz.split() if xyz else []
                    if len(parts) < 3:
                        continue
                    try:
                        z = float(parts[2])
                    except ValueError:
                        continue
                    if z > max_z:
                        max_z = z
                if max_z > 0.0:
                    return max_z
            except Exception as exc:
                log_warning(f"[mission.robot] URDF parse failed: {exc!r}")

    return None


def _resolve_height_range(asset: Any) -> Tuple[float, float]:
    """Return ``(min_m, max_m)`` for the target-height slider, tied to the asset.

    Range = ``[0.0, max(2 × nominal, 1.0)]`` when MJCF/URDF parsing succeeds;
    0 m always remains the "use asset nominal" sentinel. Falls back to
    :data:`_HEIGHT_RANGE_FALLBACK` (matches the Robot manifest meta) when
    the asset has no packaged geometry to read from.
    """
    if asset is None:
        return _HEIGHT_RANGE_FALLBACK
    nominal = _parse_asset_nominal_height(asset)
    if nominal is None or nominal <= 0.0:
        return _HEIGHT_RANGE_FALLBACK
    upper = max(round(float(nominal) * 2.0, 2), 1.0)
    return (0.0, upper)


# ---------------------------------------------------------------------------
# Section base
# ---------------------------------------------------------------------------


class _SectionFrame(QFrame):
    """子区段公共底盘：左对齐 sub_t1 标题 + body 容器."""

    def __init__(
        self,
        title_key: str,
        title_default: str,
        parent: Optional[QWidget] = None,
        *,
        body_stretch: bool = True,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("missionTrainSection")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(10)

        self._title = I18nLabel(title_key, default=title_default, parent=self)
        self._title.setObjectName("missionTrainSectionTitle")
        self._title.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        outer.addWidget(self._title, 0)

        self._body = QWidget(self)
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(0, 0, 0, 0)
        self._body_layout.setSpacing(8)
        outer.addWidget(self._body, 1 if body_stretch else 0)

    def body_layout(self) -> QVBoxLayout:
        return self._body_layout

    def apply_theme(self) -> None:
        bg = Config.get_color("bg_1")
        title_color = Config.get_color("sub_t1")
        font_small = Config.get_font_size("size_small")
        self.setStyleSheet(
            f"QFrame#missionTrainSection {{ background-color: {bg}; "
            f"border-radius: 6px; }}"
        )
        self._title.setStyleSheet(
            f"QLabel#missionTrainSectionTitle {{ color: {title_color}; "
            f"font-size: {font_small}px; background: transparent; }}"
        )


class _ElidingValueLabel(QLabel):
    """右对齐只读值标签：宽度不够时按 ElideRight 截断，并把完整文本放进 tooltip。"""

    def __init__(self, text: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._full_text = str(text or "")
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.setToolTip(self._full_text)
        self._apply_elided()

    def setText(self, text: str) -> None:  # type: ignore[override]
        self._full_text = str(text or "")
        self.setToolTip(self._full_text)
        self._apply_elided()

    def text(self) -> str:  # type: ignore[override]
        return self._full_text

    def resizeEvent(self, event) -> None:  # noqa: D401
        super().resizeEvent(event)
        self._apply_elided()

    def changeEvent(self, event) -> None:  # noqa: D401
        super().changeEvent(event)
        if event.type() == QEvent.Type.FontChange:
            self._apply_elided()

    def _apply_elided(self) -> None:
        avail = max(0, int(self.width()) - 2)
        if avail <= 0 or not self._full_text:
            super().setText(self._full_text)
            return
        elided = self.fontMetrics().elidedText(
            self._full_text, Qt.TextElideMode.ElideRight, avail
        )
        super().setText(elided)


class _StatusRow(QWidget):
    def __init__(
        self,
        label_key: str,
        label_default: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.label = I18nLabel(label_key, default=label_default, parent=self)
        self.label.setObjectName("missionTrainStatRowLabel")
        self.label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self.value = QLabel("", self)
        self.value.setObjectName("missionTrainStatRowValue")
        self.value.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        layout.addWidget(self.label, 1)
        layout.addWidget(self.value, 0)

    def apply_theme(self) -> None:
        sub = Config.get_color("sub_t2")
        main = Config.get_color("main_t1")
        font_small = Config.get_font_size("size_small")
        font_normal = Config.get_font_size("size_normal")
        self.label.setStyleSheet(
            f"QLabel#missionTrainStatRowLabel {{ color: {sub}; "
            f"font-size: {font_small}px; background: transparent; }}"
        )
        self.value.setStyleSheet(
            f"QLabel#missionTrainStatRowValue {{ color: {main}; "
            f"font-size: {font_normal}px; background: transparent; }}"
        )


# ---------------------------------------------------------------------------
# Section 1 — Training status
# ---------------------------------------------------------------------------


class _TrainingStatusSection(_SectionFrame):
    """运行状态徽标 + 7 行实时指标 + 底部 [Start] 按钮.

    底部按钮承担原 ``_RunControlSection`` 的全部行为:
      - Start (spec=save) ↔ Stop (spec=danger) 由顶部 main_row 的 start/stop
        enabled 状态联动切换。
      - ``bind_run_buttons(top_start, top_stop)`` 装配 ``_MergedRunFilter`` 监
        听顶部 EnabledChange，把 click 转发到正确的顶部按钮。

    Pause 按钮被移除：SB3 的 ``model.learn`` 与 Isaac Lab / RSL_RL 的 ``runner.learn``
    都是无中断点的紧循环，后端没有 pause 原语，留 disabled 占位徒增噪声。
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(
            "mission.train.section.status", "训练状态", parent
        )

        self._badge = QLabel("", self._body)
        self._badge.setObjectName("missionTrainStateBadge")
        self._badge.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self.body_layout().addWidget(self._badge, 0)

        self._row_elapsed = _StatusRow(
            "mission.train.field.elapsed", "已训练时间", self._body)
        self._row_iter = _StatusRow(
            "mission.train.field.iterations", "迭代次数", self._body)
        self._row_cur_reward = _StatusRow(
            "mission.train.field.current_reward", "当前奖励", self._body)
        self._row_mean_reward = _StatusRow(
            "mission.train.field.mean_reward", "平均奖励", self._body)
        self._row_best_reward = _StatusRow(
            "mission.train.field.best_reward", "最佳奖励", self._body)
        self._row_episode_length = _StatusRow(
            "mission.train.field.episode_length", "回合时长", self._body)
        self._row_fps = _StatusRow(
            "mission.train.field.fps", "帧率(FPS)", self._body)
        for r in (
            self._row_elapsed, self._row_iter, self._row_cur_reward,
            self._row_mean_reward, self._row_best_reward,
            self._row_episode_length, self._row_fps,
        ):
            self.body_layout().addWidget(r, 0)

        # ---- bottom Start button row ------------------------------------
        # 按钮 row 紧贴在实时数据行下方(无上方 stretch),按钮右对齐;
        # 多余的纵向空间放到按钮 row 之后。
        self._btn_run: QPushButton = setButton(
            "mission.train.btn.start", 100, 32,
            kind="normal", spec="save", default="Start", parent=self._body,
        )
        # 默认禁用直到 bind_run_buttons 接通顶部按钮。
        self._btn_run.setEnabled(False)

        btn_row = QWidget(self._body)
        btn_row_layout = QHBoxLayout(btn_row)
        btn_row_layout.setContentsMargins(0, 0, 0, 0)
        btn_row_layout.setSpacing(8)
        btn_row_layout.addStretch(1)
        btn_row_layout.addWidget(self._btn_run, 0)
        self.body_layout().addWidget(btn_row, 0)
        self.body_layout().addStretch(1)

        # Run-mode state machine — moved verbatim from the old _RunControlSection.
        self._run_mode: str = "start"
        self._top_start: Optional[QPushButton] = None
        self._top_stop: Optional[QPushButton] = None
        self._run_filter: Optional[_MergedRunFilter] = None
        self._btn_run.clicked.connect(self._fire_run)
        # 语种切换 → 按当前 mode 重新翻译合并按钮文本（覆盖 setButton 内部
        # 绑定的原始 i18n key 的回放结果）。
        I18n.instance().language_changed.connect(self._retranslate_run_button)

        self._render_status(initial_status())
        # Singleton constructed lazily — guaranteed safe under QApplication.
        get_training_status_model()
        get_app_signals().training_status_changed.connect(self._on_status)

    # ── run-button state machine ─────────────────────────────────────────
    def bind_run_buttons(
        self, top_start: QPushButton, top_stop: QPushButton,
    ) -> None:
        self._top_start = top_start
        self._top_stop = top_stop
        self._run_filter = _MergedRunFilter(
            top_start, top_stop, self._update_run_mode
        )
        self._update_run_mode(
            bool(top_start.isEnabled()), bool(top_stop.isEnabled())
        )

    def _update_run_mode(self, start_enabled: bool, stop_enabled: bool) -> None:
        # stop 可点 → 已在训练中，显示 Stop；否则按 start 状态显示 Start。
        if stop_enabled:
            self._set_mode("stop")
            self._btn_run.setEnabled(True)
        else:
            self._set_mode("start")
            self._btn_run.setEnabled(bool(start_enabled))

    def _set_mode(self, mode: str) -> None:
        if mode not in ("start", "stop") or mode == self._run_mode:
            return
        self._run_mode = mode
        btn = self._btn_run
        if mode == "stop":
            btn._i18n_key = "mission.train.btn.stop"
            btn._i18n_default = "Stop"
            btn._spec = "danger"
        else:
            btn._i18n_key = "mission.train.btn.start"
            btn._i18n_default = "Start"
            btn._spec = "save"
        self._retranslate_run_button()
        refresher = getattr(btn, "refresh_style", None)
        if callable(refresher):
            refresher()

    def _retranslate_run_button(self) -> None:
        btn = self._btn_run
        key = getattr(btn, "_i18n_key", "")
        default = getattr(btn, "_i18n_default", "")
        QPushButton.setText(btn, I18n.tr(key, default))

    def _fire_run(self) -> None:
        if self._run_mode == "stop":
            if self._top_stop is not None and self._top_stop.isEnabled():
                log_debug("[mission.train] forward stop to top main_row")
                self._top_stop.click()
        else:
            if self._top_start is not None and self._top_start.isEnabled():
                log_debug("[mission.train] forward start to top main_row")
                self._top_start.click()

    def apply_theme(self) -> None:
        super().apply_theme()
        from application.service.training_status_model import (
            get_training_status_model as _get,
        )
        status = _get().status()
        self._refresh_badge_color(str(status.get("state", "idle")))
        for r in (
            self._row_elapsed, self._row_iter, self._row_cur_reward,
            self._row_mean_reward, self._row_best_reward,
            self._row_episode_length, self._row_fps,
        ):
            r.apply_theme()
        refresher = getattr(self._btn_run, "refresh_style", None)
        if callable(refresher):
            refresher()

    def _on_status(self, status: Dict[str, Any]) -> None:
        self._render_status(status)
        self._refresh_badge_color(str(status.get("state", "idle")))

    def _render_status(self, status: Dict[str, Any]) -> None:
        state = str(status.get("state", "idle"))
        self._badge.setText(_state_label(state))
        self._row_elapsed.value.setText(
            _format_elapsed(int(status.get("elapsed_s", 0) or 0))
        )
        self._row_iter.value.setText(str(int(status.get("iterations", 0) or 0)))
        self._row_cur_reward.value.setText(
            f"{float(status.get('current_reward', 0.0) or 0.0):.2f}"
        )
        self._row_mean_reward.value.setText(
            f"{float(status.get('mean_reward', 0.0) or 0.0):.2f}"
        )
        self._row_best_reward.value.setText(
            f"{float(status.get('best_reward', 0.0) or 0.0):.2f}"
        )
        self._row_episode_length.value.setText(
            f"{float(status.get('episode_length', 0.0) or 0.0):.0f}"
        )
        self._row_fps.value.setText(
            f"{float(status.get('fps', 0.0) or 0.0):.0f}"
        )

    def _refresh_badge_color(self, state: str) -> None:
        font_huge = Config.get_font_size("size_huge")
        slot = _state_color_slot(state)
        color = Config.get_color(slot)
        self._badge.setStyleSheet(
            f"QLabel#missionTrainStateBadge {{ color: {color}; "
            f"font-size: {font_huge}px; font-weight: 600; "
            f"background: transparent; }}"
        )


# ---------------------------------------------------------------------------
# Form rows
# ---------------------------------------------------------------------------


class _FormRow(QWidget):
    """label 上 / 控件下 两行式表单 row."""

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
        self.label.setObjectName("missionTrainFormLabel")
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
            f"QLabel#missionTrainFormLabel {{ color: {sub}; "
            f"font-size: {font_small}px; background: transparent; }}"
        )


def _list_devices() -> List[str]:
    """训练设备下拉：列出可用 CUDA 设备 + CPU."""
    devices: List[str] = []
    try:
        import torch
        if torch.cuda.is_available():
            n = int(torch.cuda.device_count() or 0)
            for i in range(n):
                try:
                    name = str(torch.cuda.get_device_name(i))
                except Exception:
                    name = f"CUDA {i}"
                devices.append(f"GPU ({i}) {name}")
    except Exception:
        pass
    if not devices:
        devices.append("GPU (0)")
    devices.append("CPU")
    return devices


# ---------------------------------------------------------------------------
# Section 2 — Task config
# ---------------------------------------------------------------------------


class _TaskConfigSection(_SectionFrame):
    """任务环境（readonly）/ 任务名（readonly）/ 链路（与顶部 combo 同步）/ 训练设备."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(
            "mission.train.section.task", "任务配置", parent
        )

        # readonly 值标签（任务环境 / 任务名）— 用 _ElidingValueLabel：宽度不够
        # 时自动 ElideRight 截断（任务名可能是较长的画布文件名）。
        self._lbl_env_value = _ElidingValueLabel(_empty_text(), self._body)
        self._lbl_env_value.setObjectName("missionTrainReadonlyValue")
        self._lbl_name_value = _ElidingValueLabel(_empty_text(), self._body)
        self._lbl_name_value.setObjectName("missionTrainReadonlyValue")

        # 链路 combo：local / cloud（与顶部 [Local|Cloud] 双向同步）
        self._cb_link: LaviComboBox = setComboBox(
            [
                ("mission.link.local", "本地"),
                ("mission.link.cloud", "云端"),
            ],
            height=28,
            i18n=True,
            parent=self._body,
        )
        # 把 itemData 设成 "local" / "cloud" 字面值，便于和顶部 combo 对应
        self._cb_link.setItemData(0, "local")
        self._cb_link.setItemData(1, "cloud")

        # 训练设备 combo（保持 user.ini 持久化）
        self._cb_device: LaviComboBox = setComboBox(
            _list_devices(), height=28, i18n=False, parent=self._body
        )
        last_device = str(Config.get_value(_USER_INI_SECTION, "device", "")).strip()
        if last_device:
            for i in range(self._cb_device.count()):
                if self._cb_device.itemText(i) == last_device:
                    self._cb_device.setCurrentIndex(i)
                    break
        self._cb_device.currentTextChanged.connect(
            lambda v: Config.set_value(_USER_INI_SECTION, "device", str(v))
        )

        # 任务环境 / 任务名：横向 label-left / value-right（与 _StatusRow 风格一致，
        # 值列宽度不足时 ElideRight 截断）。链路 / 训练设备：保留竖排 _FormRow，
        # combo 在窄列里更适合 label 上 / 控件下。
        self._row_env = self._make_inline_row(
            "mission.train.field.task_env", "任务环境", self._lbl_env_value
        )
        self._row_name = self._make_inline_row(
            "mission.train.field.task_name", "任务名称", self._lbl_name_value
        )
        self._row_link = _FormRow(
            "mission.train.field.link", "链路", self._cb_link, self._body
        )
        self._row_device = _FormRow(
            "mission.train.field.device", "训练设备", self._cb_device, self._body
        )
        for r in (self._row_env, self._row_name, self._row_link, self._row_device):
            self.body_layout().addWidget(r, 0)
        self._rows: List[_FormRow] = [self._row_link, self._row_device]
        self._inline_rows: List[QWidget] = [self._row_env, self._row_name]
        self.body_layout().addStretch(1)

        # 顶部 combo 句柄（bind_link_combo 注入）+ 防回环标志
        self._top_link_combo: Optional[QComboBox] = None
        self._link_syncing: bool = False
        self._cb_link.currentIndexChanged.connect(self._on_local_link_changed)

    def _make_inline_row(
        self,
        label_key: str,
        label_default: str,
        value_widget: QWidget,
    ) -> QWidget:
        """Build a label-left / value-right inline row (consistent with `_StatusRow`)."""
        row = QWidget(self._body)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        label = I18nLabel(label_key, default=label_default, parent=row)
        label.setObjectName("missionTrainStatRowLabel")
        label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        layout.addWidget(label, 0)
        layout.addWidget(value_widget, 1)
        row.label = label  # type: ignore[attr-defined]
        row.value = value_widget  # type: ignore[attr-defined]
        return row

    def apply_theme(self) -> None:
        super().apply_theme()
        for r in self._rows:
            r.apply_theme()
        sub = Config.get_color("sub_t2")
        main = Config.get_color("main_t1")
        font_small = Config.get_font_size("size_small")
        font_normal = Config.get_font_size("size_normal")
        label_style = (
            f"QLabel#missionTrainStatRowLabel {{ color: {sub}; "
            f"font-size: {font_small}px; background: transparent; }}"
        )
        for row in self._inline_rows:
            row.label.setStyleSheet(label_style)  # type: ignore[attr-defined]
        ro_style = (
            f"QLabel#missionTrainReadonlyValue {{ color: {main}; "
            f"font-size: {font_normal}px; background: transparent; "
            f"padding: 4px 6px; }}"
        )
        self._lbl_env_value.setStyleSheet(ro_style)
        self._lbl_name_value.setStyleSheet(ro_style)
        for cb in (self._cb_link, self._cb_device):
            refresher = getattr(cb, "refresh_style", None)
            if callable(refresher):
                refresher()

    # ------ canvas binding ------------------------------------------------
    def set_canvas_context(self, backend_id: str, file_id: str) -> None:
        """画布切换时刷新任务环境 / 任务名 readonly 值。

        任务环境字段走 ``registers.backends.get_display_name(eid)`` 拿面向用户
        的显示名（"isaac_lab" → "IsaacLab"），core 不让 raw engine_id 出 UI。
        """
        be = (backend_id or "").strip()
        if be:
            try:
                from registers import backends as _backends
                display = str(_backends.get_display_name(be) or be).strip()
            except Exception:
                display = be
            self._lbl_env_value.setText(display or _empty_text())
        else:
            self._lbl_env_value.setText(_empty_text())
        # 任务名 = file_id 文件名 stem（去掉 .canvas.json）
        if file_id:
            stem = Path(str(file_id).replace("\\", "/").rsplit("/", 1)[-1])
            name = stem.name
            if name.endswith(".canvas.json"):
                name = name[: -len(".canvas.json")]
            self._lbl_name_value.setText(name or _empty_text())
        else:
            self._lbl_name_value.setText(_empty_text())

    # ------ link combo sync ----------------------------------------------
    def bind_link_combo(self, top_combo: QComboBox) -> None:
        """与顶部 [Local|Cloud] combo 双向同步。"""
        self._top_link_combo = top_combo
        # 卡片初始值 ← 顶部
        self._sync_card_from_top()
        # 顶部变化 → 卡片
        top_combo.currentIndexChanged.connect(self._sync_card_from_top)

    def _sync_card_from_top(self, *_: Any) -> None:
        if self._top_link_combo is None:
            return
        if self._link_syncing:
            return
        data = str(self._top_link_combo.currentData() or "").strip().lower()
        if not data:
            return
        target = 0 if data == "local" else 1
        if self._cb_link.currentIndex() == target:
            return
        self._link_syncing = True
        try:
            self._cb_link.setCurrentIndex(target)
        finally:
            self._link_syncing = False

    def _on_local_link_changed(self, _idx: int) -> None:
        if self._top_link_combo is None:
            return
        if self._link_syncing:
            return
        data = str(self._cb_link.currentData() or "").strip().lower()
        if not data:
            return
        # 找顶部 combo 中匹配 itemData 的 index
        target_idx = -1
        for i in range(self._top_link_combo.count()):
            if str(self._top_link_combo.itemData(i) or "").strip().lower() == data:
                target_idx = i
                break
        if target_idx < 0:
            return
        if self._top_link_combo.currentIndex() == target_idx:
            return
        self._link_syncing = True
        try:
            self._top_link_combo.setCurrentIndex(target_idx)
        finally:
            self._link_syncing = False


# ---------------------------------------------------------------------------
# Section 3 — Hyperparams (canvas-bound, bidirectional)
# ---------------------------------------------------------------------------


class _HyperparamSection(_SectionFrame):
    """超参字段 — 与画布上 ``manifest.is_trainer`` 节点的暴露字段双向同步.

    UX 行为：
      - ``set_canvas(page, file_id)`` 切换画布：调 ``_collect_trainer_params(page)``
        遍历画布找第一个 ``manifest.is_trainer == True`` 的节点 → 按其
        ``parameters`` 顺序收集所有标记 ``meta["mission_expose"]=True`` +
        ``meta["mission_label_i18n"]`` 非空的 ParamSpec → 重建行 + 读初值。
        无画布或无 trainer 节点 → 全字段 disable，显示空占位。
      - 订阅 ``signals.canvas_topology_changed``：节点增 / 减 / 切画布时重新
        发现 trainer，重建行。
      - 用户 ``editingFinished``：解析文本，验证通过则调
        ``page.set_node_param(node_id, key, value, save=False)`` —— 只写
        画布 in-memory params + push 到 undo stack + emit
        ``canvas_param_changed`` + 触发 DirtyTracker，**不落盘**。落盘走
        项目级 Save 流程。
      - 订阅 ``signals.canvas_param_changed``：画布上节点参数变了时刷新
        对应字段（``_syncing`` 防回环）。
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(
            "mission.train.section.hyper", "超参数配置", parent
        )

        # body 容器只放一个动态 grid host，重建 profile 时整段替换内容。
        self._grid_host = QWidget(self._body)
        self._grid_layout = QGridLayout(self._grid_host)
        self._grid_layout.setContentsMargins(0, 0, 0, 0)
        self._grid_layout.setHorizontalSpacing(10)
        self._grid_layout.setVerticalSpacing(8)
        self._grid_layout.setColumnStretch(0, 0)
        self._grid_layout.setColumnStretch(1, 1)
        self.body_layout().addWidget(self._grid_host, 0)
        self.body_layout().addStretch(1)

        # 当前显示的字段集与 widget 索引
        self._fields: Dict[str, LaviLineEdit] = {}
        self._labels: List[I18nLabel] = []
        self._field_specs: Dict[str, Dict[str, Any]] = {}

        # canvas 上下文
        self._page: Any = None
        self._file_id: str = ""
        self._algo_node_id: Optional[str] = None
        self._syncing: bool = False           # canvas → card 同步时置 True

        # 用空 profile 渲染初始空态
        self._rebuild_rows([])

        sigs = get_app_signals()
        sigs.canvas_param_changed.connect(self._on_canvas_param)
        sigs.canvas_topology_changed.connect(self._on_topology_changed)

    # ------ topology change → re-discover trainer ---------------------------
    def _on_topology_changed(self, file_id: str) -> None:
        """画布拓扑变化（节点增 / 减 / 切画布）时重新发现 trainer 节点。"""
        if str(file_id or "") != self._file_id:
            return
        if self._page is None:
            return
        # 重新跑 set_canvas 流程，按当前画布最新 _instances 重建超参面板。
        self.set_canvas(self._page, self._file_id)

    # ------ profile rebuild ----------------------------------------------
    def _rebuild_rows(self, profile: List[Dict[str, Any]]) -> None:
        """按 profile 重建 grid 行；profile 为空时显示一行 disable 占位."""
        # 清空旧 widget。
        while self._grid_layout.count():
            item = self._grid_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._fields.clear()
        self._labels.clear()
        self._field_specs.clear()

        if not profile:
            # 空态：一行灰显占位标签
            placeholder = I18nLabel(
                "mission.train.placeholder.no_canvas",
                default="未加载画布",
                parent=self._grid_host,
            )
            placeholder.setObjectName("missionTrainHyperLabel")
            placeholder.setAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )
            self._grid_layout.addWidget(placeholder, 0, 0, 1, 2)
            self._labels.append(placeholder)
            return

        for row_idx, spec in enumerate(profile):
            label = I18nLabel(
                str(spec["label_i18n"]),
                default=str(spec["label_default"]),
                parent=self._grid_host,
            )
            label.setObjectName("missionTrainHyperLabel")
            label.setAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )
            le = setLineEdit(text="", parent=self._grid_host)
            le.setFixedHeight(26)
            le.setEnabled(False)
            self._install_validator(le, str(spec["kind"]))
            self._grid_layout.addWidget(label, row_idx, 0)
            self._grid_layout.addWidget(le, row_idx, 1)
            self._fields[str(spec["key"])] = le
            self._labels.append(label)
            self._field_specs[str(spec["key"])] = spec
            le.editingFinished.connect(self._make_commit_handler(str(spec["key"])))

    # ------ public --------------------------------------------------------
    def apply_theme(self) -> None:
        super().apply_theme()
        self._apply_theme_to_dynamic_rows()

    def _apply_theme_to_dynamic_rows(self) -> None:
        sub = Config.get_color("sub_t2")
        font_small = Config.get_font_size("size_small")
        for label in self._labels:
            label.setStyleSheet(
                f"QLabel#missionTrainHyperLabel {{ color: {sub}; "
                f"font-size: {font_small}px; background: transparent; }}"
            )
        for le in self._fields.values():
            refresher = getattr(le, "refresh_style", None)
            if callable(refresher):
                refresher()

    def set_canvas(self, page: Any, file_id: str) -> None:
        """切换画布上下文；遍历 manifest.is_trainer 节点收集暴露字段，按
        ParamSpec.meta 驱动行数 / 顺序 / validator。
        """
        self._page = page
        self._file_id = str(file_id or "")
        node_id, exposed = _collect_trainer_params(page)
        self._algo_node_id = node_id
        if node_id is None or not exposed:
            self._rebuild_rows([])
            return
        self._rebuild_rows(exposed)
        self._apply_theme_to_dynamic_rows()
        self._refresh_all_from_canvas()

    def _refresh_all_from_canvas(self) -> None:
        if self._page is None or not self._algo_node_id:
            return
        self._syncing = True
        try:
            for key, le in self._fields.items():
                spec = self._field_specs[key]
                value = self._page.get_node_param(
                    self._algo_node_id, key, spec.get("default")
                )
                le.setEnabled(True)
                le.setPlaceholderText("")
                le.setText(_format_value(value, str(spec["kind"])))
        finally:
            self._syncing = False

    def _on_canvas_param(
        self, file_id: str, node_id: str, key: str, value: Any
    ) -> None:
        if self._syncing:
            return
        if not self._file_id or file_id != self._file_id:
            return
        if not self._algo_node_id or node_id != self._algo_node_id:
            return
        if key not in self._fields:
            return
        spec = self._field_specs[key]
        le = self._fields[key]
        new_text = _format_value(value, str(spec["kind"]))
        if le.text() == new_text:
            return
        self._syncing = True
        try:
            le.setText(new_text)
        finally:
            self._syncing = False

    def _make_commit_handler(self, key: str):
        def _commit() -> None:
            if self._syncing:
                return
            if self._page is None or not self._algo_node_id:
                return
            spec = self._field_specs[key]
            le = self._fields[key]
            text = le.text()
            try:
                value = _parse_value(text, str(spec["kind"]))
            except Exception:
                # 解析失败：回滚到画布当前值
                cur = self._page.get_node_param(
                    self._algo_node_id, key, spec.get("default")
                )
                self._syncing = True
                try:
                    le.setText(_format_value(cur, str(spec["kind"])))
                finally:
                    self._syncing = False
                log_warning(
                    f"[mission.train] hyper {key!r} parse failed, reverted: "
                    f"text={text!r}"
                )
                return
            try:
                # save=False — mission_panel edits flow into the canvas
                # in-memory state and flip the project dirty flag via
                # ParamChangeCmd / DirtyTracker; the user persists via the
                # explicit project Save action, never on each keystroke.
                self._page.set_node_param(
                    self._algo_node_id, key, value, save=False,
                )
            except Exception as exc:
                log_warning(f"[mission.train] set_node_param failed: {exc!r}")
        return _commit

    @staticmethod
    def _install_validator(le: QLineEdit, kind: str) -> None:
        if kind == "int":
            v = QIntValidator(0, 2_000_000_000, le)
            le.setValidator(v)
        elif kind == "float":
            v = QDoubleValidator(0.0, 1.0e9, 12, le)
            v.setNotation(QDoubleValidator.Notation.ScientificNotation)
            le.setValidator(v)
        # kind=="string" → no validator (accepts "auto" or any text;
        # _parse_value enforces actual contract on commit)


# ---------------------------------------------------------------------------
# Run-button filter (used by _TrainingStatusSection's bottom button row)
# ---------------------------------------------------------------------------


class _MergedRunFilter(QObject):
    """监听顶部 start / stop 两个按钮的 EnabledChange，回调当前 enabled 组合。"""

    def __init__(
        self,
        top_start: QWidget,
        top_stop: QWidget,
        callback,
    ) -> None:
        super().__init__(top_start)
        self._top_start = top_start
        self._top_stop = top_stop
        self._callback = callback
        top_start.installEventFilter(self)
        top_stop.installEventFilter(self)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.EnabledChange and obj in (
            self._top_start, self._top_stop
        ):
            try:
                self._callback(
                    bool(self._top_start.isEnabled()),
                    bool(self._top_stop.isEnabled()),
                )
            except Exception:
                pass
        return False


# ---------------------------------------------------------------------------
# Section 4 — Robot config (bidirectional with canvas `robot` node)
# ---------------------------------------------------------------------------


class _HeightInlineEdit(QLineEdit):
    """One-shot numeric inline editor for the Target Height value chip.

    Behaves like a plain QLineEdit but emits :attr:`cancelled` on Escape
    so the section can swap the chip back to its display button without
    committing. ``editingFinished`` (Enter / focus-out) is left to commit
    through the section's handler.
    """

    cancelled = pyqtSignal()

    def keyPressEvent(self, ev) -> None:  # noqa: D401
        if ev.key() == Qt.Key.Key_Escape:
            self.cancelled.emit()
            ev.accept()
            return
        super().keyPressEvent(ev)


def _active_file_kind_available(asset: Any, kind: str) -> bool:
    """True iff ``asset`` has a usable resource for ``kind`` (mjcf/usd/urdf).

    Mirror of canvas ``ActiveFileTableRow._kind_available`` — USD also accepts
    a cloud Nucleus URL (``asset.usd_url``) so IsaacLab-served assets count
    as available even with no local file.
    """
    if asset is None:
        return False
    if getattr(asset, f"{kind}_path", None) is not None:
        return True
    if kind == "usd" and getattr(asset, "usd_url", None):
        return True
    return False


class _BodyMappingTable(QTableWidget):
    """body→IR-role 2 列编辑表 (Body / Role).

    - Role 单元左键点击 → 弹出 canonical role 菜单（含 Out of scope / Unassign）。
    - 行右键 → 上下文菜单：mark/clear out-of-scope, unassign。
    - 不持有 mapper，所有修改通过 signal 上抛父 section；父 section 负责
      调用 ``set_data(asset, mapper)`` 重新渲染。
    """

    role_reassigned = pyqtSignal(str, str)        # (body_link, new_role_id or "")
    out_of_scope_changed = pyqtSignal(str, bool)  # (body_link, is_oos)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(0, 2, parent)
        self.setObjectName("missionRobotJointTable")
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setShowGrid(True)
        self.setHorizontalHeaderLabels([
            tr("mission.train.body.col_robot", "Robot"),
            tr("mission.train.body.col_role", "Role"),
        ])
        self.verticalHeader().setVisible(False)
        hdr = self.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        hdr.setHighlightSections(False)
        self.cellClicked.connect(self._on_cell_clicked)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context_menu)

        self._asset: Any = None   # RobotAsset | None (joint name source)
        self._mapper: Any = None  # BodyIRMapper | None (override state + family)
        # Per-row metadata: [{"kind": "role"/"unmapped"/"oos"/"required_empty",
        #                     "body": body_link, "role_id": role_id}, ...]
        # body_link here is the *body-link* form (FL_hip_joint → FL_hip) used
        # by BodyIRMapper.reassign_role; the displayed Robot column is the
        # raw joint name from the asset.
        self._row_info: List[Dict[str, str]] = []

    def set_data(self, asset: Any, mapper: Any) -> None:
        self._asset = asset
        self._mapper = mapper
        self._rebuild_rows()

    def _rebuild_rows(self) -> None:
        """Build the table as user spec:

          1. Load every actuated joint from the asset (``asset.joints`` —
             ``{joint_name: ir_role_canonical}`` from ``robots_canonical.json``).
          2. Load the set of REQUIRED IR joint roles for the family
             (``body_ir.get_joint_ir_roles(family)`` — already filtered to
             actuated-joint categories, so ``base`` / ``feet`` / ``head`` /
             ``hands`` are NOT in this set by construction).
          3. Reconcile via the registry: each asset joint is matched to its
             current IR role through the mapper (mapper.roles holds the
             post-override state, so user reassignments show through).
          4. Emit one row per asset joint (matched) + one row per required
             IR joint role that no asset joint covers (to-be-matched).

        No special-case skipping — non-joint roles never enter the required
        set in the first place.
        """
        self.setRowCount(0)
        self._row_info.clear()
        if self._mapper is None or self._asset is None:
            return

        try:
            from application.training.body_ir import (
                get_canonical_roles,
                get_joint_ir_roles,
                joint_name_to_link_name,
            )
        except Exception as exc:
            log_warning(f"[mission.robot] body_ir import failed: {exc!r}")
            return

        family = str(getattr(self._mapper, "family", "") or "generic")

        # Step 1: asset joints (preserve declaration order from the registry).
        try:
            joints_map = dict(getattr(self._asset, "joints", {}) or {})
        except Exception:
            joints_map = {}

        # Step 2: required IR joint roles for this morphology.
        try:
            required_role_ids = list(get_joint_ir_roles(family))
        except Exception as exc:
            log_warning(f"[mission.robot] get_joint_ir_roles failed: {exc!r}")
            required_role_ids = []

        # Canonical labels for the family (so the Role column shows
        # human-readable "Hip AA FL" rather than the raw id "hip_FL").
        try:
            role_labels: Dict[str, str] = {
                str(r.role_id): str(r.label)
                for r in get_canonical_roles(family)
            }
        except Exception:
            role_labels = {}

        # Step 3a: current-state index of body_link → (role_id, label) coming
        # off the mapper (= canonical + user overrides). When the user has
        # reassigned a joint, this is what shows in the Role column.
        body_to_role: Dict[str, Tuple[str, str]] = {}
        for role in self._mapper.roles:
            if role.body:
                body_to_role[str(role.body)] = (
                    str(role.role_id), str(role.label),
                )
        try:
            oos_set = set(self._mapper.out_of_scope_bodies())
        except Exception:
            oos_set = set()

        unassigned_label = tr(
            "mission.train.body.unassigned", "(Unassigned)",
        )
        oos_label = tr(
            "mission.train.body.out_of_scope", "(Out of Scope)",
        )

        # Step 3b: one row per asset joint.
        matched_required: set = set()
        for joint_name in joints_map.keys():
            jn = str(joint_name)
            body_link = joint_name_to_link_name(jn)
            if body_link in oos_set:
                self._add_row(
                    kind="oos", body_text=jn, role_text=oos_label,
                    body_link=body_link, role_id="",
                )
                continue
            if body_link in body_to_role:
                role_id, role_label = body_to_role[body_link]
                self._add_row(
                    kind="role", body_text=jn, role_text=role_label,
                    body_link=body_link, role_id=role_id,
                )
                if role_id in required_role_ids:
                    matched_required.add(role_id)
            else:
                self._add_row(
                    kind="unmapped", body_text=jn,
                    role_text=unassigned_label,
                    body_link=body_link, role_id="",
                )

        # Step 4: required IR joint roles with no asset joint covering them.
        for role_id in required_role_ids:
            if role_id in matched_required:
                continue
            label = role_labels.get(role_id, role_id)
            self._add_row(
                kind="required_empty",
                body_text="—",
                role_text=label,
                body_link="",
                role_id=role_id,
            )

    def _add_row(
        self, *, kind: str, body_text: str, role_text: str,
        body_link: str, role_id: str,
    ) -> None:
        r = self.rowCount()
        self.insertRow(r)
        body_item = QTableWidgetItem(body_text)
        body_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        body_item.setToolTip(body_link or body_text)
        role_item = QTableWidgetItem(role_text)
        role_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        role_item.setToolTip(role_text)
        self.setItem(r, 0, body_item)
        self.setItem(r, 1, role_item)
        self._row_info.append({
            "kind": kind, "body": body_link, "role_id": role_id,
        })

    def _on_cell_clicked(self, row: int, col: int) -> None:
        if col != 1 or self._mapper is None:
            return
        if row < 0 or row >= len(self._row_info):
            return
        info = self._row_info[row]
        body = info.get("body") or ""
        if not body:
            # 空 role slot — 没有 body 可移动，跳过
            return
        self._open_role_picker(body)

    def _open_role_picker(self, body: str) -> None:
        try:
            from application.training.body_ir import get_canonical_roles
        except Exception as exc:
            log_warning(f"[mission.robot] get_canonical_roles import failed: {exc!r}")
            return
        try:
            family = str(getattr(self._mapper, "family", "") or "generic")
            roles = list(get_canonical_roles(family))
        except Exception as exc:
            log_warning(f"[mission.robot] get_canonical_roles failed: {exc!r}")
            roles = []
        menu = QMenu(self)
        for canon in roles:
            act = menu.addAction(str(canon.label))
            act.setData(str(canon.role_id))
        if roles:
            menu.addSeparator()
        unmap_act = menu.addAction(
            tr("mission.train.body.menu_unassign", "Unassign")
        )
        unmap_act.setData("__none__")
        oos_act = menu.addAction(
            tr("mission.train.body.menu_oos", "Mark out of scope")
        )
        oos_act.setData("__oos__")
        chosen = menu.exec(QCursor.pos())
        if chosen is None:
            return
        choice = str(chosen.data() or "")
        if choice == "__oos__":
            self.out_of_scope_changed.emit(body, True)
        elif choice == "__none__":
            self.role_reassigned.emit(body, "")
        elif choice:
            self.role_reassigned.emit(body, choice)

    def _on_context_menu(self, pos) -> None:
        idx = self.indexAt(pos)
        if not idx.isValid():
            return
        row = idx.row()
        if row < 0 or row >= len(self._row_info):
            return
        info = self._row_info[row]
        body = info.get("body") or ""
        if not body:
            return
        menu = QMenu(self)
        if info.get("kind") == "oos":
            act = menu.addAction(
                tr("mission.train.body.menu_restore", "Restore (Unassigned)")
            )
            act.triggered.connect(
                lambda _=None, b=body: self.out_of_scope_changed.emit(b, False)
            )
        else:
            act_oos = menu.addAction(
                tr("mission.train.body.menu_oos", "Mark out of scope")
            )
            act_oos.triggered.connect(
                lambda _=None, b=body: self.out_of_scope_changed.emit(b, True)
            )
            if info.get("role_id"):
                act_un = menu.addAction(
                    tr("mission.train.body.menu_unassign", "Unassign role")
                )
                act_un.triggered.connect(
                    lambda _=None, b=body: self.role_reassigned.emit(b, "")
                )
        menu.exec(self.viewport().mapToGlobal(pos))

    def apply_theme(self) -> None:
        bg = Config.get_color("bg_1")
        border = Config.get_color("border_1")
        text = Config.get_color("main_t1")
        sub = Config.get_color("sub_t2")
        hdr_bg = Config.get_color("bg_1")
        font_small = Config.get_font_size("size_small")
        self.setStyleSheet(
            f"QTableWidget#missionRobotJointTable {{ "
            f"background-color: {bg}; color: {text}; "
            f"gridline-color: {border}; border: 1px solid {border}; "
            f"font-size: {font_small}px; }}"
            f"QTableWidget#missionRobotJointTable::item {{ padding: 2px 4px; }}"
            f"QHeaderView::section {{ background-color: {hdr_bg}; color: {sub}; "
            f"font-size: {font_small}px; border: none; "
            f"padding: 3px 6px; }}"
        )


class _RobotConfigSection(_SectionFrame):
    """Robot Config — 与画布 ``robot`` 节点 4 个 param 双向同步.

    顺序按 manifest:
      ``asset_id``        (string) — Training Asset 下拉
      ``active_override`` (enum)   — Asset Files 3 行 mini-table
      ``body_mapping``    (json)   — Joint Mapping 编辑表
      ``target_height``   (float)  — Target Height 输入框

    任一控件提交 → ``page.set_node_param(key, value, save=False)`` 走
    ParamChangeCmd —— 只更新画布 in-memory params + push 到 undo stack +
    emit ``canvas_param_changed`` + 通过 indexChanged → DirtyTracker 把项目
    标记为 dirty。**不在每次编辑时落盘**;由项目级 Save 操作 (Ctrl+S /
    toolbar Save) 决定何时把 ``<canvas>.canvas.json`` 写盘。
    反向：订阅 ``canvas_param_changed`` / ``canvas_topology_changed`` /
    ``RobotAssetService.changed``, ``_syncing`` 防回环。

    ``body_mapping`` 写入 = 写 RobotAssetService.set_body_ir_overrides (state.json
    单一真相源) + 镜像 JSON 到 ``body_mapping`` param —— 与画布
    ``BodyMappingTableRow._persist`` (``param_rows.py:2530``) 完全一致。
    """

    _ROBOT_KEYS = frozenset({
        "asset_id", "active_override", "body_mapping", "target_height",
    })

    # Emitted whenever the resolved Robot SKU changes (canvas-driven).
    # PolicySimulationCard consumes this so Run / Review Robot always
    # target whichever robot the user has selected in Robot Config,
    # instead of the SKU baked into the policy bundle at training time.
    robot_sku_changed = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(
            "mission.train.section.robot", "Robot Config", parent
        )

        self._page: Any = None
        self._file_id: str = ""
        self._robot_node_id: Optional[str] = None
        self._syncing: bool = False
        self._current_sku: str = ""
        self._asset: Any = None      # RobotAsset | None
        self._mapper: Any = None     # BodyIRMapper | None

        body = self.body_layout()

        # ---- Training Asset (asset_id) ----
        self._cb_asset: LaviComboBox = setComboBox(
            [], height=28, i18n=False, parent=self._body,
        )
        self._populate_asset_choices()
        self._row_asset = _FormRow(
            "mission.train.field.training_asset", "Training Asset",
            self._cb_asset, self._body,
        )
        body.addWidget(self._row_asset, 0)
        self._cb_asset.currentIndexChanged.connect(self._on_asset_picked)

        # ---- Asset Files (active_override) ----
        # 单选下拉,只罗列 asset 实际存在的文件类型 (mjcf/usd/urdf)。
        # 当前选项 = active_override 的值,选 "Auto" 等价于写回 "auto"。
        self._cb_files: LaviComboBox = setComboBox(
            [], height=28, i18n=False, parent=self._body,
        )
        self._cb_files.currentIndexChanged.connect(
            self._on_active_files_combo_changed
        )
        self._row_files = _FormRow(
            "mission.train.field.active_files", "Asset Files",
            self._cb_files, self._body,
        )
        body.addWidget(self._row_files, 0)

        # ---- Joint Mapping (body_mapping) ----
        self._lbl_joints = I18nLabel(
            "mission.train.field.body_mapping", default="body_mapping",
            parent=self._body,
        )
        self._lbl_joints.setObjectName("missionTrainFormLabel")
        self._lbl_joints.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self._joints_tbl = _BodyMappingTable(self._body)
        self._joints_tbl.role_reassigned.connect(self._on_role_reassigned)
        self._joints_tbl.out_of_scope_changed.connect(self._on_oos_changed)
        body.addWidget(self._lbl_joints, 0)
        body.addWidget(self._joints_tbl, 1)  # expanding row

        # ---- Target Height (target_height) ----
        # Slider 取值离散到 0.01 m 一格 (Robot manifest meta.step 一致);
        # range 来自 _resolve_height_range(asset) — 解析 MJCF/URDF 拿到的
        # nominal 标准高度,严格随机器人资产联动。
        self._height_min: float = _HEIGHT_RANGE_FALLBACK[0]
        self._height_max: float = _HEIGHT_RANGE_FALLBACK[1]

        self._height_host = QWidget(self._body)
        height_layout = QHBoxLayout(self._height_host)
        height_layout.setContentsMargins(0, 0, 0, 0)
        height_layout.setSpacing(8)

        self._slider_height = QSlider(
            Qt.Orientation.Horizontal, self._height_host,
        )
        # tracking=False → valueChanged 只在 release / 键盘步进时触发,中间拖
        # 拽不会产生 ParamChangeCmd 风暴;sliderMoved 单独用来实时预览数值。
        self._slider_height.setTracking(False)
        self._slider_height.setSingleStep(1)
        self._slider_height.setPageStep(10)
        self._slider_height.setRange(
            int(self._height_min * _HEIGHT_TICKS_PER_M),
            int(self._height_max * _HEIGHT_TICKS_PER_M),
        )
        self._slider_height.setValue(0)
        self._slider_height.setToolTip(
            tr("mission.train.field.target_height_hint",
               "0 = use asset nominal")
        )
        self._slider_height.sliderMoved.connect(self._on_slider_moved)
        self._slider_height.valueChanged.connect(
            self._on_slider_value_changed
        )

        # Value chip: a QPushButton showing the current numeric value
        # (canvas RangeRow's end-button equivalent). Click → flip to an
        # inline QLineEdit; Enter commits / Esc cancels; focus-out also
        # commits. The widgets share the same fixed slot via swap-visibility.
        _HEIGHT_VAL_W = 56
        self._btn_height_val = QPushButton("0.00", self._height_host)
        self._btn_height_val.setObjectName("missionRobotHeightVal")
        self._btn_height_val.setFixedWidth(_HEIGHT_VAL_W)
        self._btn_height_val.setFixedHeight(24)
        self._btn_height_val.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_height_val.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._btn_height_val.setToolTip(
            tr("mission.train.field.target_height_hint",
               "0 = use asset nominal")
        )
        self._btn_height_val.clicked.connect(self._open_height_editor)

        self._le_height_edit = _HeightInlineEdit("", self._height_host)
        self._le_height_edit.setObjectName("missionRobotHeightEdit")
        self._le_height_edit.setFixedWidth(_HEIGHT_VAL_W)
        self._le_height_edit.setFixedHeight(24)
        self._le_height_edit.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._le_height_edit.setVisible(False)
        # Validator range is updated alongside the slider's range when the
        # asset changes (see _apply_height_range_for_asset).
        self._le_height_validator = QDoubleValidator(
            self._height_min, self._height_max, 3, self._le_height_edit,
        )
        self._le_height_validator.setNotation(
            QDoubleValidator.Notation.StandardNotation
        )
        self._le_height_edit.setValidator(self._le_height_validator)
        self._le_height_edit.editingFinished.connect(
            self._commit_height_editor
        )
        self._le_height_edit.cancelled.connect(self._cancel_height_editor)

        height_layout.addWidget(self._slider_height, 1)
        height_layout.addWidget(self._btn_height_val, 0)
        height_layout.addWidget(self._le_height_edit, 0)

        self._row_height = _FormRow(
            "mission.train.field.target_height", "Target Height (m)",
            self._height_host, self._body,
        )
        body.addWidget(self._row_height, 0)

        # ---- placeholder (visible only when no robot node) ----
        self._lbl_placeholder = I18nLabel(
            "mission.train.placeholder.no_robot",
            default="No robot node",
            parent=self._body,
        )
        self._lbl_placeholder.setObjectName("missionTrainHyperLabel")
        self._lbl_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        body.addWidget(self._lbl_placeholder, 0)

        # 信号订阅
        sigs = get_app_signals()
        sigs.canvas_param_changed.connect(self._on_canvas_param)
        sigs.canvas_topology_changed.connect(self._on_topology_changed)
        try:
            from application.service.robot_assets.service import (
                get_robot_asset_service,
            )
            get_robot_asset_service().changed.connect(self._on_overrides_changed)
        except Exception as exc:
            log_warning(
                f"[mission.robot] subscribe RobotAssetService.changed failed: "
                f"{exc!r}"
            )

        # 初始空态：等 set_canvas() 接入
        self._set_disabled_state(True)

    # ── theme ─────────────────────────────────────────────────────────
    def apply_theme(self) -> None:
        super().apply_theme()
        sub = Config.get_color("sub_t2")
        font_small = Config.get_font_size("size_small")
        lbl_style = (
            f"QLabel#missionTrainFormLabel {{ color: {sub}; "
            f"font-size: {font_small}px; background: transparent; }}"
        )
        self._lbl_joints.setStyleSheet(lbl_style)
        self._row_asset.apply_theme()
        self._row_files.apply_theme()
        self._row_height.apply_theme()
        self._joints_tbl.apply_theme()
        self._lbl_placeholder.setStyleSheet(
            f"QLabel#missionTrainHyperLabel {{ color: {sub}; "
            f"font-size: {font_small}px; background: transparent; }}"
        )
        # Height value chip + inline editor: matches the canvas RangeRow's
        # end-button look (rounded border + hover lift). Button background
        # uses the input-bg slot to read as an interactive chip rather than
        # a flat label.
        main = Config.get_color("main_t1")
        chip_bg = Config.get_color("bg_1")
        border = Config.get_color("border_1")
        hover = Config.get_color("hover_1", "#525252")
        font_normal = Config.get_font_size("size_normal")
        self._btn_height_val.setStyleSheet(
            f"QPushButton#missionRobotHeightVal {{ color: {main}; "
            f"background-color: {chip_bg}; "
            f"border: 1px solid {border}; border-radius: 4px; "
            f"font-size: {font_normal}px; padding: 0 6px; "
            f"text-align: right; }}"
            f"QPushButton#missionRobotHeightVal:hover {{ "
            f"background-color: {hover}; }}"
        )
        self._le_height_edit.setStyleSheet(
            f"QLineEdit#missionRobotHeightEdit {{ color: {main}; "
            f"background-color: {chip_bg}; "
            f"border: 1px solid {border}; border-radius: 4px; "
            f"font-size: {font_normal}px; padding: 0 6px; }}"
        )
        for cb in (self._cb_asset, self._cb_files):
            refresher = getattr(cb, "refresh_style", None)
            if callable(refresher):
                refresher()

    # ── populate registry ────────────────────────────────────────────
    def _populate_asset_choices(self) -> None:
        skus: List[str] = []
        registry = None
        try:
            from registers import robots as _robots
            registry = _robots
            skus = list(_robots.list_skus())
        except Exception as exc:
            log_warning(f"[mission.robot] list_skus failed: {exc!r}")
        self._cb_asset.blockSignals(True)
        try:
            self._cb_asset.clear()
            for sku in skus:
                entry = None
                if registry is not None:
                    try:
                        entry = registry.get_robot(sku)
                    except Exception:
                        entry = None
                display = (entry or {}).get("name") if entry else sku
                self._cb_asset.addItem(str(display or sku), sku)
        finally:
            self._cb_asset.blockSignals(False)

    # ── public ────────────────────────────────────────────────────────
    def set_canvas(self, page: Any, file_id: Optional[str]) -> None:
        """切换画布上下文; 找 ``robot`` 节点 → 渲染；找不到 → 空态。"""
        self._page = page
        self._file_id = str(file_id or "")
        node_id: Optional[str] = None
        if page is not None:
            try:
                node_id = page.find_first_node_by_schema("robot")
            except Exception as exc:
                log_warning(
                    f"[mission.robot] find_first_node_by_schema failed: {exc!r}"
                )
                node_id = None
        self._robot_node_id = node_id
        if node_id is None:
            self._set_disabled_state(True)
            return
        self._set_disabled_state(False)
        self._refresh_all_from_canvas()

    # ── visibility / placeholder ─────────────────────────────────────
    def _set_disabled_state(self, disabled: bool) -> None:
        self._lbl_placeholder.setVisible(disabled)
        for w in (
            self._row_asset, self._row_files,
            self._lbl_joints, self._joints_tbl, self._row_height,
        ):
            w.setVisible(not disabled)

    # ── inbound signals ──────────────────────────────────────────────
    def _on_topology_changed(self, file_id: str) -> None:
        if str(file_id or "") != self._file_id:
            return
        if self._page is None:
            return
        # robot 节点可能被增/删 — 重跑发现流程
        self.set_canvas(self._page, self._file_id)

    def _on_overrides_changed(self) -> None:
        if (
            self._page is None or self._robot_node_id is None
            or not self._current_sku
        ):
            return
        if self._syncing:
            return
        # 仅刷新 body mapping (overrides 不在 canvas.json 里, 不会触发
        # canvas_param_changed)
        self._syncing = True
        try:
            self._refresh_body_mapping_from_overrides()
        finally:
            self._syncing = False

    def _on_canvas_param(
        self, file_id: str, node_id: str, key: str, value: Any,
    ) -> None:
        if self._syncing:
            return
        if not self._file_id or file_id != self._file_id:
            return
        if not self._robot_node_id or node_id != self._robot_node_id:
            return
        if key not in self._ROBOT_KEYS:
            return
        self._syncing = True
        try:
            if key == "asset_id":
                self._refresh_asset_picker(value)
                self._refresh_active_files_from_canvas()
                self._refresh_body_mapping_from_overrides()
                # Slider range just changed (new asset) — re-pull current
                # target_height so the displayed value reflects the new bounds.
                cur_h = self._page.get_node_param(
                    self._robot_node_id, "target_height", 0.0,
                )
                self._refresh_target_height_from_canvas(cur_h)
            elif key == "active_override":
                self._refresh_active_files_from_canvas()
            elif key == "body_mapping":
                self._refresh_body_mapping_from_param(value)
            elif key == "target_height":
                self._refresh_target_height_from_canvas(value)
        finally:
            self._syncing = False

    # ── refresh helpers ──────────────────────────────────────────────
    def _refresh_all_from_canvas(self) -> None:
        if self._page is None or self._robot_node_id is None:
            return
        self._syncing = True
        try:
            asset_id = self._page.get_node_param(
                self._robot_node_id, "asset_id", "",
            )
            self._refresh_asset_picker(asset_id)
            self._refresh_active_files_from_canvas()
            self._refresh_body_mapping_from_overrides()
            target_h = self._page.get_node_param(
                self._robot_node_id, "target_height", 0.0,
            )
            self._refresh_target_height_from_canvas(target_h)
        finally:
            self._syncing = False

    def _refresh_asset_picker(self, asset_id_value: Any) -> None:
        raw = str(asset_id_value or "").strip()
        sku = raw
        try:
            from registers import robots as _robots
            if raw and _robots.get_robot(raw) is None:
                resolved = _robots.resolve_id(raw)
                if resolved:
                    sku = resolved
        except Exception:
            pass
        new_sku = str(sku or "")
        sku_flipped = new_sku != self._current_sku
        self._current_sku = new_sku
        # Resolve RobotAsset (drives file table + body mapper + height range)
        self._asset = None
        if self._current_sku:
            try:
                from application.service.robot_assets.service import (
                    get_robot_asset_service,
                )
                self._asset = get_robot_asset_service().resolve(self._current_sku)
            except Exception as exc:
                log_warning(
                    f"[mission.robot] resolve {self._current_sku!r}: {exc!r}"
                )
        # Slider range is asset-derived (MJCF/URDF nominal Z × 2);
        # recompute on every asset switch.
        self._apply_height_range_for_asset()
        # Combo: select matching SKU (no signal echo)
        target_idx = -1
        for i in range(self._cb_asset.count()):
            if str(self._cb_asset.itemData(i) or "") == self._current_sku:
                target_idx = i
                break
        if target_idx >= 0 and self._cb_asset.currentIndex() != target_idx:
            self._cb_asset.blockSignals(True)
            try:
                self._cb_asset.setCurrentIndex(target_idx)
            finally:
                self._cb_asset.blockSignals(False)
        # Notify external subscribers (PolicySimulationCard) that the
        # Robot Config's effective SKU changed. Emit on every refresh
        # where the resolved SKU actually flipped — including the empty
        # → SKU and SKU → empty transitions on canvas (un)load.
        if sku_flipped:
            self.robot_sku_changed.emit(self._current_sku)

    def _refresh_active_files_from_canvas(self) -> None:
        """Rebuild the Asset Files combo from the resolved asset + current
        ``active_override``. Items: ``Auto`` plus only the file kinds the
        asset actually has (mjcf/usd/urdf). Missing kinds are not listed.
        """
        if self._page is None or self._robot_node_id is None:
            return
        active = str(self._page.get_node_param(
            self._robot_node_id, "active_override", "auto",
        ) or "auto").strip().lower() or "auto"
        items: List[Tuple[str, str]] = [("Auto", "auto")]
        for label, kind in (("MJCF", "mjcf"), ("USD", "usd"), ("URDF", "urdf")):
            if _active_file_kind_available(self._asset, kind):
                items.append((label, kind))
        self._cb_files.blockSignals(True)
        try:
            self._cb_files.clear()
            for label, data in items:
                self._cb_files.addItem(label, data)
            # If active points to an unavailable kind, fall back to Auto entry.
            target_idx = 0
            for i in range(self._cb_files.count()):
                if str(self._cb_files.itemData(i) or "") == active:
                    target_idx = i
                    break
            self._cb_files.setCurrentIndex(target_idx)
        finally:
            self._cb_files.blockSignals(False)

    def _refresh_body_mapping_from_overrides(self) -> None:
        """Rebuild ``BodyIRMapper`` from (asset, overrides) and feed the table."""
        if not self._current_sku or self._asset is None:
            self._mapper = None
            self._joints_tbl.set_data(self._asset, None)
            return
        try:
            from application.service.robot_assets.service import (
                get_robot_asset_service,
            )
            from application.training.body_ir import (
                BodyIRMapper, apply_user_overrides,
            )
        except Exception as exc:
            log_warning(f"[mission.robot] body_ir import failed: {exc!r}")
            self._mapper = None
            self._joints_tbl.set_data(self._asset, None)
            return
        try:
            mapper = BodyIRMapper.from_robot_asset(self._asset)
            overrides = get_robot_asset_service().get_body_ir_overrides(
                self._current_sku
            )
            if overrides:
                apply_user_overrides(mapper, overrides)
            self._mapper = mapper
        except Exception as exc:
            log_warning(f"[mission.robot] build mapper failed: {exc!r}")
            self._mapper = None
        self._joints_tbl.set_data(self._asset, self._mapper)

    def _refresh_body_mapping_from_param(self, raw_value: Any) -> None:
        """画布写了新的 body_mapping JSON → 据此重建 mapper + 表格."""
        try:
            from application.training.body_ir import BodyIRMapper
        except Exception as exc:
            log_warning(f"[mission.robot] BodyIRMapper import failed: {exc!r}")
            return
        try:
            if isinstance(raw_value, str) and raw_value.strip():
                self._mapper = BodyIRMapper.from_dict(json.loads(raw_value))
            elif isinstance(raw_value, dict) and raw_value:
                self._mapper = BodyIRMapper.from_dict(raw_value)
            else:
                # 空字符串 / None → 回到 overrides 重建
                self._refresh_body_mapping_from_overrides()
                return
        except Exception as exc:
            log_warning(f"[mission.robot] parse body_mapping failed: {exc!r}")
            self._refresh_body_mapping_from_overrides()
            return
        self._joints_tbl.set_data(self._asset, self._mapper)

    def _apply_height_range_for_asset(self) -> None:
        """根据当前 asset 重新计算 slider range + inline-edit validator
        (MJCF/URDF 解析)."""
        lo, hi = _resolve_height_range(self._asset)
        # 防御:确保 hi > lo 且 step 至少 1 tick。
        if hi <= lo:
            hi = lo + max(_HEIGHT_RANGE_FALLBACK[1] - _HEIGHT_RANGE_FALLBACK[0], 1.0)
        self._height_min = float(lo)
        self._height_max = float(hi)
        # 重设 slider tick range (blockSignals → 不触发 valueChanged 回调)
        self._slider_height.blockSignals(True)
        try:
            self._slider_height.setRange(
                int(self._height_min * _HEIGHT_TICKS_PER_M),
                int(self._height_max * _HEIGHT_TICKS_PER_M),
            )
        finally:
            self._slider_height.blockSignals(False)
        # Keep the inline-edit validator's range in lockstep — without
        # this, switching to a robot with a tighter range and then typing
        # the old value would silently reject the input.
        self._le_height_validator.setRange(
            self._height_min, self._height_max, 3,
        )

    def _format_height_label(self, value_m: float) -> str:
        """Chip text is always a fixed-width numeric (``"0.00"`` etc.). The
        ``0 = use asset nominal`` hint lives in the chip's tooltip — keeps
        the button width stable and makes Enter-to-commit unambiguous.
        """
        return f"{float(value_m):.2f}"

    def _refresh_target_height_from_canvas(self, raw_value: Any) -> None:
        try:
            f = float(raw_value)
        except Exception:
            f = 0.0
        # 夹到当前 slider range,避免值超出 (asset 切换时旧值可能落到新范围外)
        f = max(self._height_min, min(self._height_max, f))
        ticks = int(round(f * _HEIGHT_TICKS_PER_M))
        if self._slider_height.value() != ticks:
            self._slider_height.blockSignals(True)
            try:
                self._slider_height.setValue(ticks)
            finally:
                self._slider_height.blockSignals(False)
        self._btn_height_val.setText(self._format_height_label(f))

    # ── outbound (user edits) ────────────────────────────────────────
    def _commit_param(self, key: str, value: Any) -> None:
        """Write to the in-memory canvas params and let the global
        DirtyTracker flip the project to dirty — **never** force a disk
        write here. ``save=False`` keeps the change in
        ``page._instances[id].params``,still pushes ``ParamChangeCmd``
        onto the undo stack (so Ctrl+Z works), still emits
        ``canvas_param_changed`` (so the canvas Robot Node + any other
        mirror refresh), and the indexChanged → ``_refresh_dirty_state``
        chain flips ``dirty_state_changed(True)``. The user persists
        through the normal project Save flow (Ctrl+S / toolbar).
        """
        if self._page is None or self._robot_node_id is None:
            return
        # 已在 _syncing=True 中调用时,直接写;否则裸写也走 _syncing 守卫,
        # 防止 set_node_param 的 emit 把信号反弹回 _on_canvas_param 自己。
        outer = self._syncing
        self._syncing = True
        try:
            self._page.set_node_param(
                self._robot_node_id, key, value, save=False,
            )
        except Exception as exc:
            log_warning(
                f"[mission.robot] set_node_param {key!r} failed: {exc!r}"
            )
        finally:
            self._syncing = outer

    def _on_asset_picked(self, _idx: int) -> None:
        if self._syncing:
            return
        sku = str(self._cb_asset.currentData() or "").strip()
        if not sku:
            return
        # 写完后 canvas_param_changed 会反弹回来,但被 _syncing 拦截;
        # 同步执行下面 refresh 链确保 file table / mapper / 滑块立刻更新。
        self._commit_param("asset_id", sku)
        self._syncing = True
        try:
            self._refresh_asset_picker(sku)
            self._refresh_active_files_from_canvas()
            self._refresh_body_mapping_from_overrides()
            # Refresh slider against the new asset's height range.
            if self._page is not None and self._robot_node_id is not None:
                cur_h = self._page.get_node_param(
                    self._robot_node_id, "target_height", 0.0,
                )
                self._refresh_target_height_from_canvas(cur_h)
        finally:
            self._syncing = False

    def _on_active_files_combo_changed(self, _idx: int) -> None:
        if self._syncing:
            return
        if self._page is None or self._robot_node_id is None:
            return
        new_val = str(self._cb_files.currentData() or "auto").strip().lower()
        if new_val not in ("auto", "mjcf", "usd", "urdf"):
            new_val = "auto"
        current = str(self._page.get_node_param(
            self._robot_node_id, "active_override", "auto",
        ) or "auto").strip().lower()
        if current == new_val:
            return
        self._commit_param("active_override", new_val)

    def _on_slider_moved(self, ticks: int) -> None:
        """实时预览:拖动过程中只更新右侧数值按钮文字,不写画布。"""
        value_m = float(ticks) / _HEIGHT_TICKS_PER_M
        self._btn_height_val.setText(self._format_height_label(value_m))

    def _on_slider_value_changed(self, ticks: int) -> None:
        """提交:slider release / 键盘步进 / 点击导轨触发 (tracking=False)."""
        if self._syncing:
            return
        if self._page is None or self._robot_node_id is None:
            return
        value_m = round(float(ticks) / _HEIGHT_TICKS_PER_M, 2)
        self._btn_height_val.setText(self._format_height_label(value_m))
        self._commit_param("target_height", value_m)

    # ---- inline numeric editor for the value chip --------------------
    def _open_height_editor(self) -> None:
        """User clicked the value chip → swap to QLineEdit for direct entry.

        Editor seed = current slider value (the truth source). Validator
        range matches the slider's asset-derived bounds. Selection is
        primed so a single keystroke replaces the value, matching the
        canvas RangeRow's number-popup feel.
        """
        cur_value = float(self._slider_height.value()) / _HEIGHT_TICKS_PER_M
        # Sync validator to the current asset's range (slider was already
        # reset in _apply_height_range_for_asset; mirror that here so a
        # stale validator doesn't reject otherwise-valid input).
        self._le_height_validator.setRange(
            self._height_min, self._height_max, 3,
        )
        self._le_height_edit.setText(f"{cur_value:.2f}")
        self._btn_height_val.setVisible(False)
        self._le_height_edit.setVisible(True)
        self._le_height_edit.selectAll()
        self._le_height_edit.setFocus(Qt.FocusReason.MouseFocusReason)

    def _commit_height_editor(self) -> None:
        """Enter / focus-out: parse → clamp → drive slider → swap back to chip."""
        if not self._le_height_edit.isVisible():
            return
        text = self._le_height_edit.text().strip()
        try:
            value = 0.0 if not text else float(text)
        except Exception:
            value = float(self._slider_height.value()) / _HEIGHT_TICKS_PER_M
        # Clamp to slider bounds; the validator should reject most invalid
        # input but a stray ``-`` or stale range can still leak through.
        value = max(self._height_min, min(self._height_max, value))
        # Drive the slider — its valueChanged handler does the canvas
        # commit + DirtyTracker flip + chip text refresh. blockSignals is
        # NOT used here on purpose; we WANT the slider's change pipeline
        # to fire so undo/dirty work end-to-end.
        ticks = int(round(value * _HEIGHT_TICKS_PER_M))
        if self._slider_height.value() != ticks:
            self._slider_height.setValue(ticks)
        else:
            # Value didn't actually change — slider won't emit, refresh
            # the chip text ourselves and we're done.
            self._btn_height_val.setText(self._format_height_label(value))
        self._le_height_edit.setVisible(False)
        self._btn_height_val.setVisible(True)

    def _cancel_height_editor(self) -> None:
        """Escape: discard edits, snap chip text back to the current slider."""
        if not self._le_height_edit.isVisible():
            return
        cur_value = float(self._slider_height.value()) / _HEIGHT_TICKS_PER_M
        self._btn_height_val.setText(self._format_height_label(cur_value))
        self._le_height_edit.setVisible(False)
        self._btn_height_val.setVisible(True)

    def _on_role_reassigned(self, body: str, new_role_id: str) -> None:
        if self._mapper is None or not self._current_sku or not body:
            return
        try:
            self._mapper.reassign_role(body, new_role_id or None)
        except Exception as exc:
            log_warning(f"[mission.robot] reassign_role failed: {exc!r}")
            return
        self._persist_body_mapping()

    def _on_oos_changed(self, body: str, is_oos: bool) -> None:
        if self._mapper is None or not self._current_sku or not body:
            return
        try:
            if is_oos:
                self._mapper.mark_out_of_scope(body)
            else:
                self._mapper.clear_out_of_scope(body)
        except Exception as exc:
            log_warning(f"[mission.robot] oos change failed: {exc!r}")
            return
        self._persist_body_mapping()

    def _persist_body_mapping(self) -> None:
        if self._mapper is None or not self._current_sku:
            return
        # 1. 单一真相源: state.json overrides (RobotAssetService.changed will
        #    fire — our own _on_overrides_changed handler is suppressed by
        #    the surrounding _syncing flag below).
        self._syncing = True
        try:
            try:
                from application.service.robot_assets.service import (
                    get_robot_asset_service,
                )
                from application.training.body_ir import extract_user_overrides
                overrides = extract_user_overrides(self._mapper)
                get_robot_asset_service().set_body_ir_overrides(
                    self._current_sku, overrides if overrides else None,
                )
            except Exception as exc:
                log_warning(
                    f"[mission.robot] persist overrides failed: {exc!r}"
                )
            # 2. 镜像到画布 param (与 BodyMappingTableRow._persist 一致 — 让
            #    其它 consumer 能读出当前 mapper 快照)
            try:
                snapshot = json.dumps(
                    self._mapper.to_dict(), ensure_ascii=False,
                )
                self._commit_param("body_mapping", snapshot)
            except Exception as exc:
                log_warning(
                    f"[mission.robot] serialise mapper failed: {exc!r}"
                )
            # 3. 重新渲染表格 (mapper 内部已变)
            self._joints_tbl.set_data(self._asset, self._mapper)
        finally:
            self._syncing = False


# ---------------------------------------------------------------------------
# Card — composes the four sections
# ---------------------------------------------------------------------------


class TrainingConfigPerspectiveCard(QFrame):
    """mission_panel 左卡片：训练配置透视."""

    # 对外的语义信号。run 控制按钮已直接转发顶部，不再需要 train_action_requested
    # 信号；保留 task_configure_requested / hyper_edit_requested 兼容现有钩子。
    task_configure_requested = pyqtSignal()
    hyper_edit_requested = pyqtSignal()
    # Forwarded from _RobotConfigSection.robot_sku_changed.
    # MissionControlPanel wires this into PolicySimulationCard so the
    # Run / Review Robot buttons always target the Robot Config's
    # current SKU, not the policy bundle's baked SKU.
    robot_sku_changed = pyqtSignal(str)

    _ACCENT_W = 4

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("missionTrainCard")
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
        title_host.setObjectName("missionTrainCardTitle")
        title_host.setFrameShape(QFrame.Shape.NoFrame)
        title_host.setFixedHeight(28)
        title_host.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        title_layout = QHBoxLayout(title_host)
        title_layout.setContentsMargins(8, 0, 8, 0)
        title_layout.setSpacing(8)
        self._title_label = I18nLabel(
            "mission.train.title", default="训练配置透视", parent=title_host
        )
        self._title_label.setObjectName("missionTrainCardTitleLabel")
        title_layout.addWidget(self._title_label, 0, Qt.AlignmentFlag.AlignVCenter)
        title_layout.addStretch(1)
        outer.addWidget(title_host, 0)

        # Sections row.
        body = QWidget(self)
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(8)

        self._sec_status = _TrainingStatusSection(body)
        self._sec_task = _TaskConfigSection(body)
        self._sec_hyper = _HyperparamSection(body)
        self._sec_robot = _RobotConfigSection(body)

        body_layout.addWidget(self._sec_status, 3)
        body_layout.addWidget(self._sec_task, 3)
        body_layout.addWidget(self._sec_hyper, 3)
        body_layout.addWidget(self._sec_robot, 4)
        outer.addWidget(body, 1)

        # Forward Robot Config's SKU changes so external listeners
        # (PolicySimulationCard) don't have to reach into _sec_robot.
        self._sec_robot.robot_sku_changed.connect(self.robot_sku_changed)

        self.apply_theme()

    # ------------------------------------------------------------------
    # Public — robot SKU access
    # ------------------------------------------------------------------
    def current_robot_sku(self) -> str:
        """The Robot Config's currently-resolved SKU ('' when empty)."""
        return str(getattr(self._sec_robot, "_current_sku", "") or "")

    # ------------------------------------------------------------------
    # Public — wired by MissionControlPanel / MainWindow
    # ------------------------------------------------------------------
    def set_canvas(self, page: Any, file_id: Optional[str]) -> None:
        """切换当前画布上下文。``page=None`` 进入空态。"""
        fid = str(file_id or "")
        backend_id = ""
        if page is not None:
            try:
                backend_id = str(page.backend_id or "")
            except Exception:
                backend_id = ""
        self._sec_task.set_canvas_context(backend_id, fid)
        self._sec_hyper.set_canvas(page, fid)
        self._sec_robot.set_canvas(page, fid)

    def bind_run_buttons(self, top_start: QPushButton, top_stop: QPushButton) -> None:
        # Buttons live in the status section now; forward through.
        self._sec_status.bind_run_buttons(top_start, top_stop)

    def bind_link_combo(self, top_combo: QComboBox) -> None:
        self._sec_task.bind_link_combo(top_combo)

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------
    def apply_theme(self) -> None:
        bg_card = Config.get_color("bg_2")
        border = Config.get_color("border_1")
        accent = Config.get_color("safe_zone")
        title_color = Config.get_color("main_t1")
        font_normal = Config.get_font_size("size_normal")

        self.setStyleSheet(
            f"QFrame#missionTrainCard {{ background-color: {bg_card}; "
            f"border: 1px solid {border}; border-radius: 8px; }}"
            f"QFrame#missionTrainCardTitle {{ background: transparent; "
            f"border-left: {self._ACCENT_W}px solid {accent}; }}"
        )
        self._title_label.setStyleSheet(
            f"QLabel#missionTrainCardTitleLabel {{ color: {title_color}; "
            f"font-size: {font_normal}px; font-weight: 600; "
            f"background: transparent; }}"
        )
        self._sec_status.apply_theme()
        self._sec_task.apply_theme()
        self._sec_hyper.apply_theme()
        self._sec_robot.apply_theme()


__all__ = ["TrainingConfigPerspectiveCard"]
