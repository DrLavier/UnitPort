"""logger.py — 纯日志 API（无 Qt widget 依赖）。

整合在一个文件内的三类东西：

- 全局可见性开关（DEBUG_FLAG / INFO_FLAG）+ set/get
- LogSignal：跨线程信号中枢
- log / log_info / log_debug / log_warning / log_error / log_success：顶层便捷函数
- LoguruHandler / install_loguru_bridge：可选 loguru 桥接（loguru 未安装时降级为 no-op）

设计取舍：本文件只负责「往 LogSignal 上发信号」+「往终端/文件 sink 写日志」两件事，
渲染层在 ui.CmdLogWidget 内完成。这样 logger.py 在 headless 测试 / CLI 下也能直接用。

终端格式（简洁模式，不再依赖 loguru）：
- 普通级别：``[HH:MM:SS.mmm] text``
- DEBUG 级：``[HH:MM:SS.mmm] | DEBUG | text``
染色按级别走 ANSI（debug=blue / info=bold / success=green / warning=yellow / error=red）。
"""

from __future__ import annotations

import datetime as _dt
import os
import sys
import threading
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal


# =============================================================================
# Windows 控制台 VT 处理（让 ANSI 色码在 cmd.exe / PowerShell 生效）
# =============================================================================
def _enable_windows_vt() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        ENABLE_VT = 0x0004
        for handle_id in (-11, -12):  # STDOUT, STDERR
            h = kernel32.GetStdHandle(handle_id)
            if not h or h == ctypes.c_void_p(-1).value:
                continue
            mode = ctypes.c_uint32()
            if not kernel32.GetConsoleMode(h, ctypes.byref(mode)):
                continue
            kernel32.SetConsoleMode(h, mode.value | ENABLE_VT)
    except Exception:
        pass


_enable_windows_vt()


# =============================================================================
# ANSI 染色 —— 颜色统一从 system.ini[Theme].log_* 槽位读取并转 24-bit truecolor。
# =============================================================================
_RESET = "\033[0m"


def _hex_to_ansi(hex_str: str) -> str:
    """``#RRGGBB`` → ``\\033[38;2;R;G;Bm``。无效输入返回空串（=不染色）。"""
    if not hex_str:
        return ""
    s = hex_str.strip().lstrip("#")
    if len(s) != 6:
        return ""
    try:
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
    except ValueError:
        return ""
    return f"\033[38;2;{r};{g};{b}m"


def _level_ansi(level: str) -> str:
    # 延迟 import：logger 可能在 Config 完成 user.ini overlay 之前被调用；
    # 每次读取走 Config.get_color，确保 user.ini 主题切换即时生效。
    try:
        from .sys import Config
    except Exception:
        return ""
    return _hex_to_ansi(Config.get_color(f"log_{level}"))


def _ts_ansi() -> str:
    try:
        from .sys import Config
    except Exception:
        return ""
    return _hex_to_ansi(Config.get_color("log_time"))


def _now_ts() -> str:
    # 与旧格式对齐到毫秒
    now = _dt.datetime.now()
    return now.strftime("%H:%M:%S.") + f"{now.microsecond // 1000:03d}"


def _colorize(level: str, text: str, *, colorize: bool) -> str:
    if not colorize:
        return text
    code = _level_ansi(level)
    if not code:
        return text
    return f"{code}{text}{_RESET}"


# =============================================================================
# 终端 + 文件 sink
# =============================================================================
_progress_active = False
_sink_lock = threading.Lock()
_file_handle: Optional[object] = None
_file_date: Optional[str] = None


def _resolve_logs_dir() -> Path:
    from .sys import Paths
    return Paths.get_logs_dir()


def _ensure_file_handle() -> Optional[object]:
    """按日切换 log 文件；同一天内复用句柄。失败时返回 None（仅影响文件落盘）。"""
    global _file_handle, _file_date
    today = _dt.date.today().isoformat()
    if _file_handle is not None and _file_date == today:
        return _file_handle
    try:
        logs_dir = _resolve_logs_dir()
        logs_dir.mkdir(parents=True, exist_ok=True)
        if _file_handle is not None:
            try:
                _file_handle.close()  # type: ignore[union-attr]
            except Exception:
                pass
        _file_handle = open(logs_dir / f"sdk_{today}.log", "a", encoding="utf-8")
        _file_date = today
    except Exception:
        _file_handle = None
    return _file_handle


def _format_line(level: str, text: str, *, colorize: bool) -> str:
    """统一格式化：DEBUG 多一段 ``| DEBUG |``，其余仅 ``[ts] text``。"""
    ts = _now_ts()
    if colorize:
        ts_code = _ts_ansi()
        ts_str = f"{ts_code}[{ts}]{_RESET}" if ts_code else f"[{ts}]"
    else:
        ts_str = f"[{ts}]"
    body = _colorize(level, str(text), colorize=colorize)
    if level == "debug":
        debug_tag = _colorize("debug", "DEBUG", colorize=colorize)
        return f"{ts_str} | {debug_tag} | {body}"
    return f"{ts_str} {body}"


def _emit_terminal(level: str, text: str, *, wrap: bool) -> None:
    """根据 wrap 决定换行 vs 回车覆写；wrap=False 用于进度行。"""
    global _progress_active
    is_progress = not wrap
    line = _format_line(level, text, colorize=True)
    with _sink_lock:
        if is_progress:
            sys.stderr.write("\033[2K\r" + line)
            sys.stderr.flush()
            _progress_active = True
        else:
            if _progress_active:
                sys.stderr.write("\n")
                _progress_active = False
            sys.stderr.write(line + "\n")
            sys.stderr.flush()


def _emit_file(level: str, text: str) -> None:
    """文件落盘：无染色、固定 ``{ts} | LEVEL | text`` 全量格式。"""
    fh = _ensure_file_handle()
    if fh is None:
        return
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.") + f"{_dt.datetime.now().microsecond // 1000:03d}"
    line = f"{now} | {level.upper():<8} | {text}\n"
    try:
        with _sink_lock:
            fh.write(line)  # type: ignore[union-attr]
            fh.flush()      # type: ignore[union-attr]
    except Exception:
        pass


def _broadcast(level: str, text: str, *, wrap: bool) -> None:
    """终端 + 文件双写。LogSignal 由顶层 log_* 直接 emit。"""
    _emit_terminal(level, text, wrap=wrap)
    _emit_file(level, text)


# =============================================================================
# 全局可见性开关（GUI 控制）
# =============================================================================
DEBUG_FLAG: bool = False  # 默认关闭：debug 行平时不展示
INFO_FLAG: bool = True    # 默认打开：info 行平时展示


def set_debug_flag(value: bool) -> None:
    global DEBUG_FLAG
    DEBUG_FLAG = bool(value)


def get_debug_flag() -> bool:
    return DEBUG_FLAG


def set_info_flag(value: bool) -> None:
    global INFO_FLAG
    INFO_FLAG = bool(value)


def get_info_flag() -> bool:
    return INFO_FLAG


# =============================================================================
# LogSignal — 跨线程信号中枢（懒加载单例）
# =============================================================================
class LogSignal(QObject):
    """所有日志通过本类的 ``log_message`` 信号在线程间安全传递。"""

    log_message = pyqtSignal(str, str, bool, bool)  # (text, log_type, wrap, typer)

    _instance: Optional["LogSignal"] = None
    _lock = threading.Lock()

    @classmethod
    def instance(cls) -> "LogSignal":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def emit_log(self, text: str, log_type: str = "info", wrap: bool = True, typer: bool = False):
        self.log_message.emit(text, log_type, wrap, typer)

    def debug(self, text, wrap=True, typer=False):
        self.emit_log(text, "debug", wrap, typer)

    def info(self, text, wrap=True, typer=False):
        self.emit_log(text, "info", wrap, typer)

    def warning(self, text, wrap=True, typer=False):
        self.emit_log(text, "warning", wrap, typer)

    def error(self, text, wrap=True, typer=False):
        self.emit_log(text, "error", wrap, typer)

    def success(self, text, wrap=True, typer=False):
        self.emit_log(text, "success", wrap, typer)


def get_log_signal() -> LogSignal:
    return LogSignal.instance()


# =============================================================================
# 顶层便捷函数（同时往终端/文件 sink + 内置屏幕）
# =============================================================================
def log(text, log_type: str = "info", wrap: bool = True, typer: bool = False):
    get_log_signal().emit_log(str(text), log_type, wrap, typer)


def log_info(text, wrap=True, typer=False):
    text = str(text)
    get_log_signal().info(text, wrap, typer)
    _broadcast("info", text, wrap=wrap)


def log_debug(text, wrap=True, typer=False):
    text = str(text)
    # 文件日志不受 DEBUG_FLAG 影响；始终向 widget 发射，由 widget 按 DEBUG_FLAG 过滤渲染。
    _broadcast("debug", text, wrap=wrap)
    get_log_signal().debug(text, wrap, typer)


def log_warning(text, wrap=True, typer=False):
    text = str(text)
    get_log_signal().warning(text, wrap, typer)
    _broadcast("warning", text, wrap=wrap)


def log_error(text, wrap=True, typer=False):
    text = str(text)
    get_log_signal().error(text, wrap, typer)
    _broadcast("error", text, wrap=wrap)


def log_success(text, wrap=True, typer=False):
    text = str(text)
    get_log_signal().success(text, wrap, typer)
    _broadcast("success", text, wrap=wrap)


# =============================================================================
# LoguruHandler — 可选桥接（loguru 未安装时降级为 no-op）
# =============================================================================
class LoguruHandler:
    """把任何 ``from loguru import logger`` 的输出镜像到内置 cmd 屏。

    仅当宿主环境装了 loguru 且显式调用 ``install_loguru_bridge()`` 时生效。
    SDK 自身不再依赖 loguru。
    """

    def write(self, message):
        try:
            record = message.record
            level = record["level"].name.lower()
            text = record["message"]
        except Exception:
            level = "info"
            text = str(message)

        level_map = {
            "trace": "debug",
            "debug": "debug",
            "info": "info",
            "success": "success",
            "warning": "warning",
            "error": "error",
            "critical": "error",
        }
        log_type = level_map.get(level, "info")
        get_log_signal().emit_log(text, log_type, wrap=True, typer=False)


_loguru_installed = False


def install_loguru_bridge() -> None:
    """安装 LoguruHandler 把 loguru 输出镜像到内置屏。

    幂等。loguru 未安装时静默跳过（SDK 不再强依赖 loguru）。
    """
    global _loguru_installed
    if _loguru_installed:
        return
    try:
        from loguru import logger as _loguru_logger  # type: ignore[import-not-found]
    except Exception:
        return
    try:
        _loguru_logger.add(LoguruHandler(), level="DEBUG", format="{message}")
        _loguru_installed = True
    except Exception:
        pass
