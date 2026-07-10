# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""application.service.signals — 全局应用级信号总线 / App-level signal bus.

把"registers 内容变了""当前后端引擎切换了"这类跨模块事件挂到一个 ``QObject``
单例上，让任意 widget / service 模块解耦订阅。

契约信号 / Signals:
- ``nodes_changed()``                 ``registers.nodes.reload()`` 后由调用方手动 emit
- ``current_backend_changed(str)``    后端切换；payload 为 engine_id（空串表示未选）

辅助 helpers 维护一个模块级的 ``_current_backend`` 字符串，让新订阅者在订阅前
也能拿到最新状态（``current_backend()``），无需 replay 信号。

Usage:
    >>> from application.service.signals import get_app_signals, set_current_backend
    >>> sig = get_app_signals()
    >>> sig.nodes_changed.connect(panel.refresh)
    >>> sig.current_backend_changed.connect(lambda eid: panel.refresh())
    >>> set_current_backend("isaac_lab")  # 自动 emit current_backend_changed
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal


class AppSignals(QObject):
    """跨模块事件广播 / Cross-module event hub.

    单例由 ``get_app_signals()`` 持有；不要在业务代码里直接实例化。
    """

    # registers.nodes 内容变了（手动 reload 后由调用方 emit）
    nodes_changed = pyqtSignal()

    # 当前训练/仿真后端切换；payload 为 engine_id（空串 = 未选）
    current_backend_changed = pyqtSignal(str)

    # 训练 run 生命周期 / metrics 广播 — 由 SB3TrainingTask 在 SDK Task 线程
    # emit；接收方（TrainingChartPanel / RunSourceSelector）在主线程，Qt 自动
    # QueuedConnection 跨线程派发，不需要手动 invokeMethod。
    training_run_started = pyqtSignal(str, str)        # (run_id, label)
    training_metrics = pyqtSignal(str, dict)            # (run_id, sample dict)
    training_run_finished = pyqtSignal(str, bool)       # (run_id, success)

    # 聚合后的训练状态 — TrainingStatusModel 把 metrics 流聚合成一个 status dict
    # （state/elapsed/iterations/current_reward/mean_reward/episode_length/fps），
    # 供 mission_panel 训练状态卡消费，避免每个 widget 自己重复聚合。
    training_status_changed = pyqtSignal(dict)         # status dict

    # 实机连接状态 — Ros2ConnectionController 在 test/connect/disconnect 时 emit。
    # state ∈ {"disconnected","connecting","connected","error"}；info 是 free-form
    # 附加信息（rtt/error msg 等），UI 可选消费。
    connection_state_changed = pyqtSignal(str, dict)   # (state, info)

    # SYSTEM channel — unconditional estop (skill_command_path_design.md §4.1).
    # A bound estop key/button fires this; it NEVER enters the CommandBus / the
    # policy command vector (that would be a soft velocity-zero, not preemption).
    # Emitted by GlobalInputManager (gamepad button / keyboard Esc). Consumers:
    # SessionController (cancels the live-sim/review task) and ConnectionController
    # Card (adapter.disable_teleop for deploy). ``source`` = "gamepad"/"keyboard"/…
    # for logging. Emitted cross-thread from the gamepad poll thread → Qt marshals
    # to the receivers' (GUI) thread via QueuedConnection; no manual invokeMethod.
    system_estop = pyqtSignal(str)                     # (source)

    # 传感器健康 — sensor_health model 对每项传感器 emit 一次。
    # name ∈ {"imu","joints","lidar","camera","torque","battery"}；
    # level ∈ {"ok","warn","error","idle"}。
    sensor_status_changed = pyqtSignal(str, str)       # (name, level)

    # 当前选中的机器人 SKU — robot_assets.service.set_selected_robot 触发；
    # 空串表示未选中。机器人硬件信息卡订阅这个信号决定显示哪台机器人。
    robot_selection_changed = pyqtSignal(str)          # sku ('' = none)

    # Real-robot connection phase progress — emitted by the Task running
    # inside Ros2ConnectionController during brownfield phase transitions.
    # Phase 1 source = the fake state machine; Phase 3 source = the real
    # connector. The signal shape is intentionally identical so the UI is
    # unaffected by the swap. ``phase_id`` is one of ``ConnectionPhase``
    # values; ``status`` is one of {"started","success","error","skipped"};
    # ``msg`` is human-readable detail.
    connection_phase_changed = pyqtSignal(str, str, str)   # (phase_id, status, msg)

    # Real-robot telemetry heartbeat — to be emitted by the Heartbeat
    # broadcaster in Phase 6. Payload is a free-form dict (e.g.
    # battery_pct / temperature_c / uptime_s / serial). Declared in Phase 1
    # as a forward-compat placeholder with no emitter yet.
    connection_telemetry = pyqtSignal(dict)

    # Phase 3.5 — diagnostics framework signals.
    # ``diagnostics_ready`` carries a ``DiagnosticReport``; emitted by
    # DiagnosticsTask at the end of P9 and by manual [Diagnose] runs.
    # ``diagnostics_repair_started`` fires when RepairTask begins (UI may
    # disable the dialog buttons). ``diagnostics_repair_finished`` carries
    # a ``RepairReport``; the dialog reads it to auto re-diagnose + reconnect.
    diagnostics_ready = pyqtSignal(object)            # DiagnosticReport
    diagnostics_repair_started = pyqtSignal()
    diagnostics_repair_finished = pyqtSignal(object)  # RepairReport
    # Per-action progress emitted by RepairTask before each action runs so
    # the dialog can render "[i/N] Running <action.name>" instead of going
    # silent for the duration of the sequence. payload: (action_name, idx, total).
    diagnostics_repair_progress = pyqtSignal(str, int, int)
    # Manual diagnose-run lifecycle so the dialog can render
    # "Re-running diagnostics..." between repair-finished and the next
    # diagnostics_ready arrival. payload: empty.
    diagnostics_started = pyqtSignal()

    # Phase 3.7 — autonomous connect signals.
    # ``connection_needs_ssh`` fires when AutoRepairLoop hits a finding
    # that requires SSH but has no credentials in SecureCredentialStore.
    # The connect task blocks on a threading.Event until the UI shows
    # SshCredentialPromptDialog and the user either saves credentials or
    # cancels. payload: (server_key, suggested_user, reason).
    connection_needs_ssh = pyqtSignal(str, str, str)
    # UI -> connect task wake-up: fired by SshCredentialPromptDialog when
    # the user clicks Save & Continue (ok=True) or Cancel (ok=False).
    # payload: (server_key, ok).
    connection_ssh_response = pyqtSignal(str, bool)
    # Final outcome of one AutoConnect run; payload is a ConnectionResult
    # (application.service.connection.result.ConnectionResult). UI listens
    # to pop ConnectionResultDialog exactly once per connect attempt.
    connection_result = pyqtSignal(object)

    # 画布节点参数变化 — CanvasPage.set_node_param 在写入参数 + 落盘后 emit。
    # mission_panel 训练配置卡片订阅此信号实现「画布 → 卡片」回流；
    # 卡片自身写入时调用 set_node_param(source="mission_panel") 并在订阅端
    # 用 _syncing 标志位过滤回环。
    canvas_param_changed = pyqtSignal(str, str, str, object)   # (canvas_file_id, node_id, key, value)

    # 画布拓扑变化（节点增 / 减 / 切换文件）— CanvasPage 在 spawn / despawn / set_file_id
    # 之后 emit。mission_panel 卡片用此信号触发动态 trainer 发现重建超参面板。
    canvas_topology_changed = pyqtSignal(str)         # canvas_file_id（'' = 未关联）

    # 画布连线变化（边的建立 / 断开）— ConnectionItem attach/disconnect 经
    # CanvasScene.notify_edge_changed → CanvasPage._emit_edge_changed 转发。
    # canvas_topology_changed 只覆盖节点增删/加载，不覆盖边；订阅方：
    # CommandContractModel（Command Pipe 契约预览随 actor-pipe / command-pipe
    # 接线实时重算）。批量加载期间静默，from_workflow_dict 末尾的
    # topology_changed 作为收尾信号。
    canvas_edge_changed = pyqtSignal(str)             # canvas_file_id（'' = 未关联）

    # 策略回放启动（MuJoCo live sim）— policy_simulation_card 在提交
    # MujocoReviewTask 时 emit。payload = (robot_sku, deploy_contract.commands
    # dict | None)。订阅方：ControllerPanel（按契约通道渲染遥控绑定区并
    # 安装 per-SKU 轴路由）。
    policy_review_started = pyqtSignal(str, object)   # (sku, commands block)

    # 当前选中策略的命令契约变更 — PolicySimulationCard._on_policy_changed 在下拉
    # 选中/初选/取消时触发,广播 (sku, deploy_contract.commands | None)，与 live-review
    # 是否在跑无关。订阅方：ControllerPanel（常驻渲染契约通道绑定区 + 装 per-SKU 路由）。
    # policy_review_started 保留不动（review 启动时另发一次，两者数据一致）。
    policy_contract_changed = pyqtSignal(str, object)  # (sku, commands block | None)

    # 当前项目切换 — MainWindow._bind_project 在解析到新的 ProjectInfo 后通过
    # ``application.service.projects.store.set_current_project_info`` 触发。
    # payload 为 ProjectInfo（已切换为有效项目）或 None（项目被卸载）。
    # 订阅方典型：MainWindow 顶部 row 的 History/Policies 下拉、未来的 deploy 面板。
    project_changed = pyqtSignal(object)              # ProjectInfo or None

    # User-script variant mutation — emitted by
    # ``application.service.scripts.resolver.save_variant`` /
    # ``delete_variant`` after a successful disk write. Payload:
    # (kind, key) where ``kind`` is one of
    # {"reward","termination","observation","discriminator"} and
    # ``key`` is the preset / function key the variant belongs to.
    # Sidebar Script panels subscribe to refresh their per-key
    # fold-out without polling.
    user_scripts_changed = pyqtSignal(str, str)        # (kind, key)

    # In-app update checker (application.service.updater).
    # ``update_check_started`` — fires when ``CheckUpdateTask`` is submitted at
    #     Stage 2. UI uses it to show a "checking..." hint if relevant.
    # ``update_check_complete`` — payload is a ``ReleaseInfo`` (newer available)
    #     or ``None`` (no update / network failure / cache-throttled hit). The
    #     sidebar Update button listens to this to flip its icon + short label.
    # ``update_apply_progress`` — emitted by ``ApplyUpdateTask`` during download
    #     / extract / merge. (fraction in [0,1], status label).
    # ``update_apply_complete`` — terminal signal for ``ApplyUpdateTask``;
    #     (ok, message). The progress dialog flips to its restart screen on ok.
    update_check_started = pyqtSignal()
    update_check_complete = pyqtSignal(object)        # ReleaseInfo or None
    update_apply_progress = pyqtSignal(float, str)    # (fraction 0..1, label)
    update_apply_complete = pyqtSignal(bool, str)     # (ok, message)

    # Resources panel (application.service.resources).
    # Lifecycle signals for third-party assets downloaded into
    # ``custom_mods/`` from GitHub clone / GitHub Release / HuggingFace.
    # The ResourcesPanel subscribes to all four so cards reflect download
    # progress without polling the registry.
    #
    # ``resource_added`` — emitted by ResourceManager.queue_download right
    #     after the new entry is persisted to ``.resources.json`` with
    #     state="downloading". UI inserts a card immediately.
    # ``resource_progress`` — emitted by DownloadResourceTask once per
    #     progress tick from the underlying transport (git/httpx/HF stream).
    #     payload: (entry_id, fraction in [0,1], one-line status). Per the
    #     project's progress contract, the task does NOT interleave
    #     log_info/log_warning between these emissions.
    # ``resource_finished`` — terminal signal. payload: (entry_id, ok, msg).
    #     On ok=True the entry's state is "local"; on ok=False it is "error"
    #     and ``msg`` is the user-facing error.
    # ``resource_removed`` — fires when the entry is deleted (user clicked
    #     Remove, or the partial-download cleanup ran after a cancel).
    resource_added = pyqtSignal(str)                   # entry_id
    resource_progress = pyqtSignal(str, float, str)    # (entry_id, fraction 0..1, line)
    resource_finished = pyqtSignal(str, bool, str)     # (entry_id, ok, message)
    resource_removed = pyqtSignal(str)                 # entry_id

    # Isaac Lab in-app installer (application.service.installers).
    # Emitted by ``IsaacLabInstallTask`` running in the global thread pool
    # after the wizard's EULA modal is accepted; consumed by
    # ``IsaacInstallProgressDialog`` and by ``main.py`` (the latter calls
    # ``registers.backends.refresh_engine_availability`` on success).
    #
    # ``isaac_install_phase`` — fires once per stage transition. ``phase``
    #     ∈ {"preflight","download","extract","clone","install","register",
    #     "fallback_external","done","error"}. ``label`` is human-readable
    #     status for the dialog header (already localised by the emitter).
    # ``isaac_install_progress`` — 0..1 progress fraction across the whole
    #     install (not per-stage). ``label`` is a single-line status used
    #     for the progress-bar caption; per the project's progress contract,
    #     emit progress in tight loops WITHOUT interleaved log_info/warning.
    # ``isaac_install_complete`` — terminal signal. ``ok=True`` means
    #     EngineService has been updated and Isaac Lab is registered;
    #     ``ok=False`` means a hard failure (message is the user-facing
    #     summary; full stderr lives in Paths.LOGS_DIR).
    isaac_install_phase = pyqtSignal(str, str)        # (phase, label)
    isaac_install_progress = pyqtSignal(float, str)   # (fraction 0..1, label)
    isaac_install_complete = pyqtSignal(bool, str)    # (ok, message)


_singleton: Optional[AppSignals] = None
_current_backend: str = ""


def get_app_signals() -> AppSignals:
    """返回 ``AppSignals`` 单例（lazy 构造，主线程调用安全）.

    第一次调用必须在 ``QApplication`` 存在之后；订阅信号本身不需要 QApplication。
    """
    global _singleton
    if _singleton is None:
        _singleton = AppSignals()
    return _singleton


def current_backend() -> str:
    """当前选中的后端 engine_id；空串表示未选."""
    return _current_backend


def set_current_backend(engine_id: str) -> None:
    """切换当前后端并 emit ``current_backend_changed``.

    - 若与当前值相同，不重复 emit（避免 UI 抖动）
    - ``engine_id=""`` 表示清除选择（恢复"全部节点"视图）
    """
    global _current_backend
    eid = str(engine_id or "")
    if eid == _current_backend:
        return
    _current_backend = eid
    get_app_signals().current_backend_changed.emit(eid)


__all__ = [
    "AppSignals",
    "get_app_signals",
    "current_backend",
    "set_current_backend",
]
