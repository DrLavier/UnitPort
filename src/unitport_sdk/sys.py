"""sys.py — SDK 系统/运行时基础设施。

整合四块原本分散的能力，按 class 划分：

- Paths       — SDK 内部路径常量（替代旧 _path.py）
- Config      — ini 配置存储（替代旧 _config.py）
- CrashHook   — sys.excepthook + threading.excepthook 路由（替代旧 crash.py）
- Task / TaskWorker / TaskSlot / TasksManager / TaskSignal — 任务系统（替代旧 tasks.py）

模块底部还导出便捷别名（get_color、install_crash_hook、get_tasks_manager 等），
保证既有调用点不需要改写。

注：本模块名为 sys，但 Python 3 默认绝对导入，模块内 ``import sys`` 仍解析为 stdlib。
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys as _stdlib_sys  # 显式重命名以与本模块区分，纯防御
import threading
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from configparser import ConfigParser
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from PyQt6.QtCore import QCoreApplication, QObject, QMutex, QThread, QWaitCondition, pyqtSignal


# =============================================================================
# Paths — 项目路径常量
# =============================================================================
class Paths:
    """Project path anchors — Tier-A bootstrap stays derived from ``__file__``;
    Tier-B resource directories resolve from ``system.ini[Resources]`` with
    ``user.ini[Resources]`` overlay, falling back to project-rooted defaults.

    Layout::

        RELEASE/                       ← PROJECT_ROOT (Tier-A, install root)
        ├── log/                       ← LOGS_DIR        (Tier-B, configurable)
        └── src/
            ├── unitport_sdk/          ← SDK_ROOT        (Tier-A, install anchor)
            ├── config/                ← CONFIG_DIR      (Tier-A, bootstrap; system.ini only)
            ├── registers/             ← REGISTERS_DIR   (Tier-B, configurable)
            ├── runtime/               ← RUNTIME_DIR     (Tier-B, configurable)
            └── application/
                └── ui/assets/         ← ASSETS_DIR      (Tier-B, configurable)

        ~/UnitPort/                    ← USER_CONFIG_DIR (Tier-B, per-user state root)
            └── user.ini               ← USER_INI        (Tier-B, derived from USER_CONFIG_DIR)

    Tier-A is hardcoded because it describes where the SDK *lives* on disk
    (you can't read where ``system.ini`` is from inside ``system.ini``).
    Tier-B is overlay-resolvable so deployments can relocate logs/assets/
    runtime/per-user-state without code edits. User-customised settings
    (user.ini and any future user-side state files) always live under
    USER_CONFIG_DIR — outside the project tree — so a clean ``git pull`` /
    redeploy never clobbers user data.

    Bootstrap exception: ``USER_INI`` is Tier-B but its location depends on
    ``USER_CONFIG_DIR``. Therefore ``[Resources].user_config_dir`` is the *one*
    key that does NOT support user.ini overlay (chicken-egg) — it is read
    from ``system.ini`` only.
    """

    # ---- Tier-A: bootstrap anchors (derived from __file__, NOT configurable) ----
    SDK_ROOT: Path = Path(__file__).resolve().parent
    SRC_ROOT: Path = SDK_ROOT.parent
    PROJECT_ROOT: Path = SRC_ROOT.parent
    APP_ROOT: Path = SRC_ROOT / "application"
    CONFIG_DIR: Path = SRC_ROOT / "config"
    SYSTEM_INI: Path = CONFIG_DIR / "system.ini"
    SDK_INI: Path = SYSTEM_INI  # legacy alias (older code may reference SDK_INI)

    # ---- Tier-B: configurable directories ----
    # Class-body values are *defaults*; _resolve_dynamic() (called once at
    # module load, see bottom of this class) overwrites them with the
    # overlay-resolved values. Keeping them as class attributes (vs methods)
    # preserves the ``Paths.LOGS_DIR`` static-attribute API used by main.py
    # and other call sites.
    LOGS_DIR: Path = PROJECT_ROOT / "log"
    ASSETS_DIR: Path = APP_ROOT / "ui" / "assets"
    REGISTERS_DIR: Path = SRC_ROOT / "registers"
    RUNTIME_DIR: Path = SRC_ROOT / "runtime"
    USER_CONFIG_DIR: Path = Path.home() / "UnitPort"
    USER_INI: Path = USER_CONFIG_DIR / "user.ini"
    PROJECTS_DIR: Path = USER_CONFIG_DIR / "projects"

    # SDK install-location anchor for assets layering — Assets.find() searches
    # both ASSETS_DIR (overlay-resolved) and this canonical install dir.
    _ASSETS_INTERNAL_DIR: Path = APP_ROOT / "ui" / "assets"

    @classmethod
    def resolve_under_project_root(cls, raw_path: str, fallback: Path) -> Path:
        """Resolve [Resources] relative paths against PROJECT_ROOT; absolute paths pass through; empty → fallback."""
        raw = str(raw_path).strip()
        if not raw:
            return fallback
        path = Path(raw).expanduser()
        return path if path.is_absolute() else (cls.PROJECT_ROOT / path)

    # Note: _read_resource and _read_resource_from read INI directly via raw
    # ConfigParser, bypassing Config. This is the bootstrap-bypass that breaks
    # the logger import cycle: logger.py calls Paths.get_logs_dir() at module
    # load; routing through Config → DataManager.write failure path would
    # ``from .logger import log_error`` against a half-initialized module.
    # Both system.ini and user.ini reads go through this same path.
    @classmethod
    def _read_resource_from(cls, ini_path: Path, key: str) -> str:
        try:
            cp = ConfigParser(strict=False)
            if ini_path.exists():
                cp.read(ini_path, encoding="utf-8")
            return cp.get("Resources", key, fallback="").strip()
        except Exception:
            return ""

    @classmethod
    def _read_resource(cls, key: str) -> str:
        """[Resources].<key> with user.ini overlay (user wins over system)."""
        user = cls._read_resource_from(cls.USER_INI, key)
        if user:
            return user
        return cls._read_resource_from(cls.SYSTEM_INI, key)

    # Bootstrap shim for the one Tier-B path that user.ini can't store:
    # ``user_config_dir`` (chicken-egg — user.ini lives under USER_CONFIG_DIR).
    # The install wizard writes this file when the user picks a non-default
    # data directory. Located at ``Path.home() / ".unitport_paths.ini"``
    # because ``Path.home()`` is the only location guaranteed to be readable
    # without any prior path resolution.
    _BOOTSTRAP_SHIM: Path = Path.home() / ".unitport_paths.ini"

    @classmethod
    def _read_bootstrap_user_config_dir(cls) -> str:
        """Return ``[Resources] user_config_dir`` from ``~/.unitport_paths.ini``
        if the shim exists, else "" (empty = use system.ini fallback)."""
        try:
            if not cls._BOOTSTRAP_SHIM.exists():
                return ""
            cp = ConfigParser(strict=False)
            cp.read(cls._BOOTSTRAP_SHIM, encoding="utf-8")
            return cp.get("Resources", "user_config_dir", fallback="").strip()
        except Exception:
            return ""

    @classmethod
    def _resolve_dynamic(cls) -> None:
        """Resolve Tier-B paths once at module load. Idempotent — safe to call again.

        Order matters:
          1. USER_CONFIG_DIR — resolved from system.ini ONLY (no user.ini overlay).
             user.ini lives under USER_CONFIG_DIR, so it cannot itself determine
             where USER_CONFIG_DIR points (chicken-egg).
          2. USER_INI — derived from the freshly-resolved USER_CONFIG_DIR.
          3. Remaining Tier-B paths — resolved with full system.ini ⊕ user.ini
             overlay (now that USER_INI is correctly located).
        """
        # 1) USER_CONFIG_DIR: bootstrap shim first, then system.ini fallback.
        # The install wizard writes ``~/.unitport_paths.ini`` when the user
        # picks a non-default data directory; we read it here so subsequent
        # launches honor the choice without touching system.ini (which is a
        # repo-tracked factory default).
        shim_value = cls._read_bootstrap_user_config_dir()
        if shim_value:
            shim_path = Path(shim_value).expanduser()
            cls.USER_CONFIG_DIR = (
                shim_path
                if shim_path.is_absolute()
                else (cls.PROJECT_ROOT / shim_path)
            )
        else:
            cls.USER_CONFIG_DIR = cls.resolve_under_project_root(
                cls._read_resource_from(cls.SYSTEM_INI, "user_config_dir"),
                Path.home() / "UnitPort",
            )
        # 2) USER_INI lives under USER_CONFIG_DIR.
        cls.USER_INI = cls.USER_CONFIG_DIR / "user.ini"

        # 3) Other Tier-B paths use full overlay (user.ini wins, falls back to system.ini).
        cls.LOGS_DIR = cls.resolve_under_project_root(
            cls._read_resource("logs_dir"), cls.PROJECT_ROOT / "log"
        )
        cls.ASSETS_DIR = cls.resolve_under_project_root(
            cls._read_resource("assets_dir"), cls._ASSETS_INTERNAL_DIR
        )
        cls.REGISTERS_DIR = cls.resolve_under_project_root(
            cls._read_resource("registers_dir"), cls.SRC_ROOT / "registers"
        )
        cls.RUNTIME_DIR = cls.resolve_under_project_root(
            cls._read_resource("runtime_dir"), cls.SRC_ROOT / "runtime"
        )
        cls.PROJECTS_DIR = cls.resolve_under_project_root(
            cls._read_resource("projects_dir"), cls.USER_CONFIG_DIR / "projects"
        )

    # ---- Backward-compat shims (kept for any external callers) ----
    @classmethod
    def get_logs_dir(cls) -> Path:
        return cls.LOGS_DIR

    @classmethod
    def get_assets_dirs(cls) -> List[Path]:
        """Layered asset search: overlay-resolved ASSETS_DIR first, then canonical
        install dir. Used by ``Assets.find()`` so SDK-shipped icons remain
        reachable even when an external override is configured."""
        dirs: List[Path] = []
        if cls.ASSETS_DIR.exists() and cls.ASSETS_DIR.is_dir():
            dirs.append(cls.ASSETS_DIR)
        if (
            cls._ASSETS_INTERNAL_DIR.exists()
            and cls._ASSETS_INTERNAL_DIR.is_dir()
            and cls._ASSETS_INTERNAL_DIR not in dirs
        ):
            dirs.append(cls._ASSETS_INTERNAL_DIR)
        return dirs


# Resolve Tier-B paths from system.ini ⊕ user.ini overlay.
# Runs once at module load so ``Paths.LOGS_DIR`` etc. are static-attribute-correct
# by the time logger.py imports this module and calls Paths.get_logs_dir().
Paths._resolve_dynamic()


# =============================================================================
# Config — 极简 ini 配置助手
# =============================================================================
class Config:
    """读写 ULTIMA_SDK/config/sdk.ini 的轻量封装。

    设计原则：
    - 单文件 ini，stdlib configparser
    - RLock 线程安全
    - 同节内重复键容忍（strict=False）
    - 空字符串视为「未填写」，回退到 fallback
    """

    _DEFAULTS: Dict[str, Dict[str, str]] = {
        "Theme": {
            # ----- Text -----
            "main_t1": "#D6D3C7",
            "main_t2": "#B8B4A6",
            "main_c1": "#D6D3C7",
            "main_c2": "#999999",
            "sub_t1": "#A0A0A0",
            "sub_t2": "#777777",
            "highlight": "#F6D393",
            "safe_zone": "#00D26A",
            "danger_zone": "#FF6B6B",
            # ----- Card -----
            "bg_1": "#101010",
            "bg_2": "#1E1E1E",
            "row_1": "#2A2A2A",
            "row_2": "#3D3D3D",
            "btn_1": "#2A2C33",
            "btn_2": "#3D3D3D",
            "hover_1": "#525252",
            "hover_2": "#212121",
            "checked_1": "#525252",
            "checked_2": "#212121",
            "border_1": "#444444",
            "border_2": "#3D3D3D",
            "spec_cmd": "#000000",
            # ----- Spec hover (用于 setButton 的 spec hover bg) -----
            "highlight": "#FFE0A8",
            "safe_zone_hover": "#4DE08C",
            "danger_zone_hover": "#FF8888",
            # ----- Spec checked (用于 setButton 的 spec checked bg) -----
            "highlight_checked": "#E0BC78",
            "safe_zone_checked": "#00A555",
            "danger_zone_checked": "#D14545",
            # ----- Misc -----
            "misc_1": "#B5CEA8",
            "misc_2": "#569CD6",
            "misc_3": "#4EC9B0",
            "misc_4": "#DCDCAA",
            "theme_1": "#9989E0",
            "theme_2": "#F6D393",
        },
        "Font": {
            "family": "Microsoft YaHei",
            "size_mini": "14",
            "size_small": "18",
            "size_normal": "20",
            "size_large": "22",
        },
        "SwitchButton": {
            "width": "44",
            "height": "22",
            "slider_margin": "2",
            "animation_duration": "200",
        },
        "System": {
            "cmd_log_debug": "false",
            "cmd_log_info": "true",
        },
        "Localisation": {
            # 翻译文件根目录，相对 SDK_ROOT；也可填绝对路径；
            # 留空则回退到 SDK_ROOT/localisation。
            # 实际文件按 {dir}/{LANG}/{category}.txt 分块存放
            # （e.g. localisation/EN/ui.txt 内含 [button]/[menu] 等子节）。
            "dir": "localisation",
            # 当前语种代码；I18n.set_language() 会自动写回
            "lang": "EN",
        },
        "Resources": {
            # 外部 asset 目录；与 SDK 内置 assets/ 取并集（外部优先）。
            # 相对路径基于 SDK_ROOT；留空则只用内置 assets/。
            "assets_dir": "",
            # 日志目录；留空则使用 SDK 内置 logs/。
            "logs_dir": "",
        },
    }

    _lock = threading.RLock()
    _system_parser: Optional[ConfigParser] = None
    _user_parser: Optional[ConfigParser] = None

    # ---- Internal -----------------------------------------------------------
    @classmethod
    def _ensure_loaded(cls) -> ConfigParser:
        """Load both layers if not yet cached. Returns the SYSTEM parser
        (factory defaults). Use _ensure_user_parser() for the overlay layer."""
        with cls._lock:
            if cls._system_parser is not None:
                # User parser still loads lazily on first access — it may have
                # been written to disk by another process between our two reads.
                if cls._user_parser is None and Paths.USER_INI.exists():
                    cls._user_parser = DataManager.load(Paths.USER_INI)
                return cls._system_parser

            Paths.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            file_existed = Paths.SYSTEM_INI.exists()
            cp = DataManager.load(Paths.SYSTEM_INI)

            # Self-healing schema migration: fill missing keys from _DEFAULTS.
            # This is SDK-driven (not user-driven) — runs once on first import
            # when system.ini is shipped without a newly-added field. Safe to
            # mutate system.ini here because it's a deterministic, idempotent
            # filling of *gaps* — never overwrites existing values.
            dirty = False
            for section, kv in cls._DEFAULTS.items():
                if not cp.has_section(section):
                    cp.add_section(section)
                    dirty = True
                for k, v in kv.items():
                    if not cp.has_option(section, k):
                        cp.set(section, k, v)
                        dirty = True

            if dirty or not file_existed:
                cls._flush(cp, Paths.SYSTEM_INI)

            cls._system_parser = cp

            if Paths.USER_INI.exists():
                cls._user_parser = DataManager.load(Paths.USER_INI)

            return cp

    @classmethod
    def _ensure_user_parser(cls) -> ConfigParser:
        """Get-or-create the user-overlay parser. Used by set_value when
        user.ini may not yet exist on disk."""
        with cls._lock:
            cls._ensure_loaded()  # ensures system layer + lazy-loads user if file exists
            if cls._user_parser is None:
                cls._user_parser = ConfigParser(strict=False)
            return cls._user_parser

    @classmethod
    def _flush(cls, cp: ConfigParser, path: Path) -> None:
        """Write parser to disk via DataManager (atomic temp + os.replace).
        Also invalidates DataManager's read-cache for that path so subsequent
        reads through ``get_data_value`` see the updated content."""
        DataManager.write(path, cp)
        DataManager.clear(path)

    @staticmethod
    def _coerce_ini(raw: Any, value_type: type, fallback: Any) -> Any:
        """Type-coerce a raw INI string. Empty/None → fallback."""
        if raw is None:
            return fallback
        if isinstance(raw, str) and raw.strip() == "":
            return fallback
        try:
            if value_type is bool:
                return str(raw).strip().lower() in ("1", "true", "yes", "on")
            if value_type is int:
                return int(raw)
            if value_type is float:
                return float(raw)
            return raw
        except (TypeError, ValueError):
            return fallback

    # ---- Public API ----------------------------------------------------
    @classmethod
    def get_value(
        cls, section: str, key: str, fallback: Any = None, value_type: type = str
    ) -> Any:
        """Layered read: user.ini wins over system.ini, both fall back to the
        ``fallback`` argument. Empty string in either layer counts as 'not set'
        and falls through to the next layer."""
        system_cp = cls._ensure_loaded()
        with cls._lock:
            user_cp = cls._user_parser
            if user_cp is not None and user_cp.has_option(section, key):
                raw = user_cp.get(section, key)
                if not (isinstance(raw, str) and raw.strip() == ""):
                    return cls._coerce_ini(raw, value_type, fallback)
            if system_cp.has_option(section, key):
                raw = system_cp.get(section, key)
                return cls._coerce_ini(raw, value_type, fallback)
            return fallback

    @classmethod
    def set_value(cls, section: str, key: str, value: Any) -> None:
        """Write to user.ini overlay. system.ini is treated as immutable factory
        defaults at runtime — never written by ``set_value``. Creates
        USER_CONFIG_DIR (and any parents) if it doesn't yet exist."""
        with cls._lock:
            user_cp = cls._ensure_user_parser()
            if not user_cp.has_section(section):
                user_cp.add_section(section)
            user_cp.set(section, key, str(value))
            Paths.USER_INI.parent.mkdir(parents=True, exist_ok=True)
            cls._flush(user_cp, Paths.USER_INI)

    @classmethod
    def reload(cls) -> None:
        """Drop both parser caches; next read re-loads from disk. Use this when
        user.ini was mutated through DataManager (outside Config) and you need
        Config's overlay view to refresh."""
        with cls._lock:
            cls._system_parser = None
            cls._user_parser = None

    @classmethod
    def get_color(cls, slot: str, fallback: str = "#000000") -> str:
        return str(cls.get_value("Theme", slot, fallback))

    @classmethod
    def get_font_size(cls, slot: str, fallback_size: int = 12) -> int:
        return int(cls.get_value("Font", slot, fallback_size, int))


# 模块级便捷别名（兼容旧调用点）
get_value = Config.get_value
set_value = Config.set_value
get_color = Config.get_color
get_font_size = Config.get_font_size


# =============================================================================
# DataManager — 通用线程安全文件 I/O，所有写入都走原子替换
# =============================================================================
#
# 设计要点：
# - 类方法接口（与 Config 同款风格），无需实例化。
# - 全格式统一缓存：INI / JSON / YAML / TOML / pickle / 文本 / DataFrame / bytes。
# - 所有写入走 临时文件 + os.replace 原子替换，进程中断不会损坏目标文件。
# - 每个绝对路径独立 RLock。
# - 通过 register_handler 可挂载自定义格式 Handler。
# - 重依赖 (pandas / pyyaml / tomli / tomli_w) 都是惰性 import，缺失时给友好报错。

def _coerce(raw: Any, value_type: Optional[type], fallback: Any) -> Any:
    """把字符串/原值转成 value_type；失败回退 fallback。INI/JSON 的 get 共用。"""
    if raw is None:
        return fallback
    if value_type is None or value_type is str:
        if isinstance(raw, str) and raw.strip() == "":
            return fallback
        return raw
    try:
        if value_type is bool:
            if isinstance(raw, bool):
                return raw
            return str(raw).strip().lower() in ("1", "true", "yes", "on")
        if value_type is int:
            return int(raw)
        if value_type is float:
            return float(raw)
        return value_type(raw)
    except (TypeError, ValueError):
        return fallback


def _atomic_write(path: Path, write_fn: Callable[[Path], None]) -> None:
    """把内容写到同目录临时文件，成功后 os.replace 原子替换目标文件。

    write_fn 收到一个临时路径，自行把数据写进去；中途异常会清理临时文件后再抛。
    Windows 与 POSIX 上 os.replace 都是原子的（同卷内）。
    """
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        write_fn(tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        raise


class _BaseHandler:
    """格式 Handler 基类。read 返回解析后对象；write 必须通过 _atomic_write 落盘。"""

    def read(self, path: Path, **kw) -> Any:
        raise NotImplementedError

    def write(self, path: Path, value: Any, **kw) -> None:
        raise NotImplementedError


class _IniHandler(_BaseHandler):
    def read(self, path: Path, **kw) -> ConfigParser:
        cp = ConfigParser(strict=kw.pop("strict", False), **kw)
        if path.exists():
            cp.read(path, encoding="utf-8")
        return cp

    def write(self, path: Path, value: ConfigParser, **kw) -> None:
        if not isinstance(value, ConfigParser):
            raise TypeError(f"IniHandler requires ConfigParser, got {type(value).__name__}")

        def _w(tmp: Path) -> None:
            with tmp.open("w", encoding="utf-8") as f:
                value.write(f)

        _atomic_write(path, _w)


class _JsonHandler(_BaseHandler):
    def read(self, path: Path, **kw) -> Any:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            return json.load(f, **kw)

    def write(
        self,
        path: Path,
        value: Any,
        *,
        indent: int = 2,
        ensure_ascii: bool = False,
        **kw,
    ) -> None:
        def _w(tmp: Path) -> None:
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(value, f, indent=indent, ensure_ascii=ensure_ascii, **kw)

        _atomic_write(path, _w)


class _YamlHandler(_BaseHandler):
    @staticmethod
    def _yaml():
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise ImportError("YAML support requires pyyaml: pip install pyyaml") from e
        return yaml

    def read(self, path: Path, **kw) -> Any:
        yaml = self._yaml()
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        return {} if loaded is None else loaded

    def write(self, path: Path, value: Any, **kw) -> None:
        yaml = self._yaml()

        def _w(tmp: Path) -> None:
            with tmp.open("w", encoding="utf-8") as f:
                yaml.safe_dump(value, f, allow_unicode=True, **kw)

        _atomic_write(path, _w)


class _TomlHandler(_BaseHandler):
    def read(self, path: Path, **kw) -> dict:
        if not path.exists():
            return {}
        try:
            import tomllib  # py3.11+
        except ImportError:
            try:
                import tomli as tomllib  # type: ignore
            except ImportError as e:
                raise ImportError(
                    "TOML reading requires Python 3.11+ with tomllib, or pip install tomli"
                ) from e
        with path.open("rb") as f:
            return tomllib.load(f)

    def write(self, path: Path, value: Any, **kw) -> None:
        try:
            import tomli_w  # type: ignore
        except ImportError as e:
            raise ImportError("TOML writing requires pip install tomli_w") from e

        def _w(tmp: Path) -> None:
            with tmp.open("wb") as f:
                tomli_w.dump(value, f)

        _atomic_write(path, _w)


class _PickleHandler(_BaseHandler):
    def read(self, path: Path, **kw) -> Any:
        if not path.exists():
            return None
        with path.open("rb") as f:
            return pickle.load(f, **kw)

    def write(
        self,
        path: Path,
        value: Any,
        *,
        protocol: int = pickle.HIGHEST_PROTOCOL,
        **kw,
    ) -> None:
        def _w(tmp: Path) -> None:
            with tmp.open("wb") as f:
                pickle.dump(value, f, protocol=protocol, **kw)

        _atomic_write(path, _w)


class _TextHandler(_BaseHandler):
    def read(self, path: Path, *, encoding: str = "utf-8", **kw) -> str:
        if not path.exists():
            return ""
        return path.read_text(encoding=encoding, **kw)

    def write(self, path: Path, value: str, *, encoding: str = "utf-8", **kw) -> None:
        if not isinstance(value, str):
            raise TypeError(f"TextHandler requires str, got {type(value).__name__}")

        def _w(tmp: Path) -> None:
            tmp.write_text(value, encoding=encoding, **kw)

        _atomic_write(path, _w)


class _BytesHandler(_BaseHandler):
    """兜底 Handler：未知扩展名按原始字节读写。"""

    def read(self, path: Path, **kw) -> bytes:
        if not path.exists():
            return b""
        return path.read_bytes()

    def write(self, path: Path, value, **kw) -> None:
        if not isinstance(value, (bytes, bytearray, memoryview)):
            raise TypeError(
                f"BytesHandler requires bytes/bytearray/memoryview, got {type(value).__name__}"
            )

        def _w(tmp: Path) -> None:
            tmp.write_bytes(bytes(value))

        _atomic_write(path, _w)


class _DataFrameHandler(_BaseHandler):
    """parquet / xlsx / csv 共用，依赖 pandas 惰性导入。"""

    @staticmethod
    def _pd():
        try:
            import pandas as pd  # type: ignore
        except ImportError as e:
            raise ImportError("DataFrame support requires pip install pandas") from e
        return pd

    def read(self, path: Path, **kw):
        pd = self._pd()
        ext = path.suffix.lower()
        if not path.exists():
            return pd.DataFrame()
        if ext == ".parquet":
            return pd.read_parquet(path, **kw)
        if ext == ".xlsx":
            return pd.read_excel(path, **kw)
        if ext == ".csv":
            return pd.read_csv(path, **kw)
        raise ValueError(f"DataFrameHandler unsupported extension: {ext}")

    def write(self, path: Path, value, *, index: bool = False, **kw) -> None:
        pd = self._pd()
        if not isinstance(value, pd.DataFrame):
            raise TypeError(
                f"DataFrameHandler requires DataFrame, got {type(value).__name__}"
            )
        ext = path.suffix.lower()
        if ext == ".parquet":
            def _w(tmp: Path) -> None:
                value.to_parquet(tmp, index=index, **kw)
        elif ext == ".xlsx":
            def _w(tmp: Path) -> None:
                value.to_excel(tmp, index=index, **kw)
        elif ext == ".csv":
            def _w(tmp: Path) -> None:
                value.to_csv(tmp, index=index, **kw)
        else:
            raise ValueError(f"DataFrameHandler unsupported extension: {ext}")
        _atomic_write(path, _w)


class DataManager:
    """通用线程安全文件 I/O 管理器（类方法接口，免实例化）。

    支持的扩展名：
        .ini / .cfg                       → ConfigParser
        .json                             → dict / list / 任何 JSON 兼容对象
        .yaml / .yml                      → dict (需要 pyyaml)
        .toml                             → dict (读 stdlib/tomli; 写 tomli_w)
        .pkl / .pickle                    → 任意 Python 对象
        .txt / .md / .log                 → str
        .parquet / .xlsx / .csv           → pandas.DataFrame
        其它扩展名                         → bytes (兜底)

    所有写入路径都走 _atomic_write：先写到同目录的隐藏临时文件，再 os.replace
    替换目标。即使写到一半进程崩溃，原文件也不会被破坏。

    并发模型：每个绝对路径独立 RLock；缓存与锁表本身的更改在类级 _lock 下进行。

    用法：
        DataManager.write("conf/app.json", {"a": 1})
        cfg = DataManager.read("conf/app.json")
        DataManager.update("conf/app.json", key="a", value=99)
        v   = DataManager.get("conf/app.json", "a", fallback=0)
        DataManager.register_handler(".myfmt", MyHandler())
    """

    _lock: threading.RLock = threading.RLock()
    _cache: Dict[str, Any] = {}
    _file_locks: Dict[str, threading.RLock] = {}
    _handlers: Dict[str, _BaseHandler] = {}
    _fallback_handler: _BaseHandler = _BytesHandler()

    # ---- 内部 ---------------------------------------------------------------
    @classmethod
    def _abs(cls, path) -> Path:
        return Path(path).expanduser().resolve()

    @classmethod
    def _key(cls, path) -> str:
        return str(cls._abs(path))

    @classmethod
    def _file_lock(cls, abs_key: str) -> threading.RLock:
        with cls._lock:
            lk = cls._file_locks.get(abs_key)
            if lk is None:
                lk = threading.RLock()
                cls._file_locks[abs_key] = lk
            return lk

    @classmethod
    def _resolve_handler(cls, path: Path, fmt: Optional[str]) -> _BaseHandler:
        if fmt:
            k = fmt.lower()
            if not k.startswith("."):
                k = "." + k
            return cls._handlers.get(k, cls._fallback_handler)
        return cls._handlers.get(path.suffix.lower(), cls._fallback_handler)

    @staticmethod
    def _return_value(value: Any) -> Any:
        # DataFrame 返回副本，防止外部就地改动污染缓存
        try:
            import pandas as pd  # type: ignore

            if isinstance(value, pd.DataFrame):
                return value.copy()
        except ImportError:
            pass
        return value

    # ---- 扩展点 ------------------------------------------------------------
    @classmethod
    def register_handler(cls, ext, handler: _BaseHandler) -> None:
        """注册扩展名 → Handler。ext 可以是 '.foo' / 'foo' / 多个组成的可迭代。"""
        with cls._lock:
            exts = [ext] if isinstance(ext, str) else list(ext)
            for e in exts:
                k = e.lower()
                if not k.startswith("."):
                    k = "." + k
                cls._handlers[k] = handler

    # ---- 公共 API ----------------------------------------------------------
    @classmethod
    def load(
        cls,
        path,
        *,
        force_reload: bool = False,
        format: Optional[str] = None,
        **handler_kw,
    ) -> Any:
        """从磁盘读到缓存（命中缓存且未 force_reload 时直接复用）。"""
        abs_path = cls._abs(path)
        key = str(abs_path)
        with cls._file_lock(key):
            if not force_reload and key in cls._cache:
                return cls._return_value(cls._cache[key])
            value = cls._resolve_handler(abs_path, format).read(abs_path, **handler_kw)
            cls._cache[key] = value
            return cls._return_value(value)

    @classmethod
    def read(cls, path, *, format: Optional[str] = None) -> Any:
        """读缓存；缓存缺失时自动 load。"""
        key = cls._key(path)
        with cls._file_lock(key):
            if key in cls._cache:
                return cls._return_value(cls._cache[key])
        return cls.load(path, format=format)

    @classmethod
    def write(
        cls,
        path,
        value: Any,
        *,
        format: Optional[str] = None,
        **handler_kw,
    ) -> bool:
        """整体写入并刷新缓存；失败返回 False 并尝试写日志。"""
        abs_path = cls._abs(path)
        key = str(abs_path)
        with cls._file_lock(key):
            try:
                cls._resolve_handler(abs_path, format).write(abs_path, value, **handler_kw)
            except Exception as e:
                try:
                    from .logger import log_error

                    log_error(f"[DataManager] write failed: {abs_path}: {e}")
                except Exception:
                    pass
                return False
            cls._cache[key] = value
            return True

    @classmethod
    def get(
        cls,
        path,
        *keys,
        fallback: Any = None,
        value_type: Optional[type] = None,
        format: Optional[str] = None,
    ) -> Any:
        """统一寻址：

        - INI ：get(path, section, key, fallback=, value_type=)
        - dict-like：get(path, "a", "b", "c") 或 get(path, "a.b.c")
        - 其它：未提供 keys 时返回整体；提供了 keys 但目标无法寻址时返回 fallback。
        """
        data = cls.read(path, format=format)

        if isinstance(data, ConfigParser):
            if len(keys) != 2:
                raise TypeError("INI get requires two keys: (section, key)")
            section, key = keys
            if not data.has_option(section, key):
                return fallback
            raw = data.get(section, key)
            if isinstance(raw, str) and raw.strip() == "":
                return fallback
            return _coerce(raw, value_type, fallback)

        path_keys: List[Any] = []
        for k in keys:
            if isinstance(k, str) and "." in k:
                path_keys.extend(k.split("."))
            else:
                path_keys.append(k)

        if not path_keys:
            return data if data is not None else fallback

        cur: Any = data
        for k in path_keys:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return fallback
        return _coerce(cur, value_type, fallback) if value_type is not None else cur

    @classmethod
    def update(
        cls,
        path,
        *,
        section: Optional[str] = None,
        key: Any = None,
        value: Any = None,
        data: Optional[dict] = None,
        merge: bool = True,
        format: Optional[str] = None,
        **handler_kw,
    ) -> bool:
        """局部更新（兼容老 up_data 语义）：

        - INI ：update(path, section=, key=, value=)
                或 update(path, data={section: {k: v, ...}})
        - dict-like (.json/.yaml/.toml/.pkl)：
                update(path, key=, value=)
                update(path, data={...}, merge=True|False)

        其它格式（DataFrame / 文本 / bytes）请直接用 write()。
        """
        abs_path = cls._abs(path)
        with cls._file_lock(str(abs_path)):
            current = cls.read(abs_path, format=format)

            if isinstance(current, ConfigParser):
                if section is not None and key is not None:
                    if not current.has_section(section):
                        current.add_section(section)
                    current.set(section, str(key), str(value))
                if data:
                    for sec, items in data.items():
                        if not current.has_section(sec):
                            current.add_section(sec)
                        for k, v in items.items():
                            current.set(sec, str(k), str(v))
                return cls.write(abs_path, current, format=format, **handler_kw)

            if not isinstance(current, dict):
                if current in (None, b"", ""):
                    current = {}
                else:
                    raise TypeError(
                        f"update() does not support {type(current).__name__}; use write() instead"
                    )

            if not merge:
                current = {}

            if key is not None:
                current[key] = value
            if data:
                current.update(data)

            return cls.write(abs_path, current, format=format, **handler_kw)

    # ---- 缓存管理 -----------------------------------------------------------
    @classmethod
    def clear(cls, path=None) -> None:
        """清单文件缓存或全量清空。"""
        with cls._lock:
            if path is None:
                cls._cache.clear()
            else:
                cls._cache.pop(cls._key(path), None)

    @classmethod
    def reload_all(cls) -> None:
        """重新加载所有已缓存的文件。"""
        with cls._lock:
            keys = list(cls._cache.keys())
        for k in keys:
            try:
                cls.load(k, force_reload=True)
            except Exception:
                pass


# 注册内置 Handler（顺序：后注册的覆盖先注册的——.csv 由 DataFrameHandler 接管）
DataManager.register_handler([".ini", ".cfg"], _IniHandler())
DataManager.register_handler(".json", _JsonHandler())
DataManager.register_handler([".yaml", ".yml"], _YamlHandler())
DataManager.register_handler(".toml", _TomlHandler())
DataManager.register_handler([".pkl", ".pickle"], _PickleHandler())
DataManager.register_handler(
    # 普通文本 + XML 家族（XML/URDF/XACRO 用文本读写最稳——保留注释/空白/xacro 宏片段）
    [".txt", ".md", ".log", ".xml", ".urdf", ".xacro", ".svg", ".html", ".htm"],
    _TextHandler(),
)
DataManager.register_handler([".parquet", ".xlsx", ".csv"], _DataFrameHandler())


# 模块级便捷别名（与 core/bin.py 同名，便于后续把业务调用点切到 SDK）
load_data = DataManager.load
read_data = DataManager.read
up_data = DataManager.update
save_data = DataManager.write
# 注意：get_value 已被 Config.get_value 占用（L205），DataManager.get 用 get_data_value 暴露
get_data_value = DataManager.get


# =============================================================================
# Storage — Generic data-push dispatch (local FS + cloud Supabase stub)
# =============================================================================
#
# Whereas DataManager operates on caller-owned absolute Paths, Storage takes
# a *relative* path/key and routes to a channel-specific destination:
#
#   - 'local' channel  → writes under Paths.USER_CONFIG_DIR via DataManager.
#                        Per-user state lives outside the project tree
#                        (~/UnitPort by default; configurable via
#                        [Resources].user_config_dir overlay).
#   - 'cloud' channel  → reserved for Supabase push tied to the logged-in
#                        account. Currently a WIP stub that logs an error
#                        and returns False. Wiring callsites today is fine.
#
# This split is intentional: DataManager stays "I/O for a given Path";
# Storage owns destination routing. Same data formats flow through both
# (Storage.push delegates to DataManager for the actual local write).
class Storage:
    """Generic data-push dispatch with two channels: 'local' (per-user FS) and
    'cloud' (Supabase, WIP). See module docstring above for design rationale."""

    @classmethod
    def push(
        cls,
        rel_path,
        value: Any,
        *,
        channel: str = "local",
        format: Optional[str] = None,
        **handler_kw,
    ) -> bool:
        """Push ``value`` to ``rel_path`` over the named channel.

        - ``rel_path`` is interpreted relative to the channel's destination
          (USER_CONFIG_DIR for local; account-bound space for cloud).
        - Format dispatch is delegated to DataManager (extension-based, or
          override with ``format=``).
        - Returns True on success, False on failure or unknown channel.
        """
        if channel == "local":
            target = Paths.USER_CONFIG_DIR / Path(rel_path)
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                try:
                    from .logger import log_error

                    log_error(f"[Storage.push] mkdir failed for {target.parent}: {e}")
                except Exception:
                    pass
                return False
            return DataManager.write(target, value, format=format, **handler_kw)

        if channel == "cloud":
            # WIP: user-explicitly-requested stub. Cloud transport (Supabase
            # bound to the logged-in account) lands in a follow-up. Do not
            # remove this branch — call sites are wired against it today.
            try:
                from .logger import log_error

                log_error(
                    f"[Storage.push] WIP: cloud channel (Supabase) not yet "
                    f"implemented; target={rel_path!r} dropped"
                )
            except Exception:
                pass
            return False

        try:
            from .logger import log_error

            log_error(f"[Storage.push] unknown channel: {channel!r}")
        except Exception:
            pass
        return False


def push_data(rel_path, value, *, channel: str = "local", format: Optional[str] = None, **handler_kw) -> bool:
    """Module-level alias mirroring load_data / read_data / save_data style."""
    return Storage.push(rel_path, value, channel=channel, format=format, **handler_kw)


# =============================================================================
# DirtyTracker — 全局"当前编辑文件"脏状态广播（SHA-1 hash 对比，回基线即清洁）
# =============================================================================
class DirtyTracker(QObject):
    """Single-file dirty-state tracker, SDK-wide singleton broadcaster.

    维护一份 "当前编辑文件 (currentPath, baselineHash, baselineLen)" 状态；调用方在文本
    变化时调用 ``reportText(text)``，本类按 ``len → SHA-1`` 顺序对比基线，仅在 dirty
    翻转时 emit ``dirtyChanged(path, isDirty)``。

    "打一个空格又删掉"路径上：长度复原 → hash 一致 → 仍判为 clean，**不会**误报脏。

    与 ``unitport_sdk.compiler.editor.CodeEditorWidget`` 内部 per-widget 的
    ``dirtyChanged`` / ``is_modified`` 是**互补关系**：本类是全局单例广播，editor 内部
    机制保留不动。

    Naming
    ------
    本类公开 API 采用驼峰命名，**不使用下划线前缀或分割**——这是项目对该子系统的命名约定，
    覆盖 PEP8 默认（其余 SDK 旧 API 保留各自的历史风格）。
    """

    dirtyChanged = pyqtSignal(str, bool)   # (path, isDirty)
    pathChanged = pyqtSignal(str)          # active path 切换（emit 时不含 dirty 翻转）

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.currentPath: Optional[str] = None
        self.baselineHash: Optional[str] = None
        self.baselineLen: int = 0
        self.dirty: bool = False
        # 持有最近一次 reportText / setBaseline 的明文，用于 markClean() 固化新基线
        # 而无需调用方再次传文本。单文件版可接受此常驻拷贝。
        self.lastSeenText: str = ""

    # —— Hash helper ——
    @staticmethod
    def hashText(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()

    # —— Mutators ——
    def setBaseline(self, path: str, text: str) -> None:
        """Set active file and snapshot ``text`` as the clean baseline.

        若 path 与现 currentPath 不同则 emit ``pathChanged(path)``；
        若先前 dirty=True，emit ``dirtyChanged(prevPath, False)`` 再切换。
        """
        prevPath = self.currentPath
        prevDirty = self.dirty

        # 先把旧 path 的 dirty 拉回 False，避免遗留状态被误解为新文件的脏
        if prevDirty and prevPath is not None:
            self.dirty = False
            self.dirtyChanged.emit(prevPath, False)

        self.currentPath = path
        self.baselineHash = self.hashText(text)
        self.baselineLen = len(text)
        self.lastSeenText = text
        self.dirty = False

        if prevPath != path:
            self.pathChanged.emit(path)

    def reportText(self, text: str) -> None:
        """Report the editor's current text; recompute dirty; emit on flip only."""
        if self.currentPath is None or self.baselineHash is None:
            return
        self.lastSeenText = text
        if len(text) != self.baselineLen:
            newDirty = True
        else:
            newDirty = self.hashText(text) != self.baselineHash
        if newDirty != self.dirty:
            self.dirty = newDirty
            self.dirtyChanged.emit(self.currentPath, newDirty)

    def markClean(self) -> None:
        """Promote the most recently reported text to the new baseline.

        典型调用时机：业务侧保存成功后。若当前已 clean，emit 抑制；否则 emit
        ``dirtyChanged(path, False)``。
        """
        if self.currentPath is None:
            return
        self.baselineHash = self.hashText(self.lastSeenText)
        self.baselineLen = len(self.lastSeenText)
        if self.dirty:
            self.dirty = False
            self.dirtyChanged.emit(self.currentPath, False)

    def clear(self) -> None:
        """Drop active tracking; emit dirty=False then pathChanged("")."""
        prevPath = self.currentPath
        prevDirty = self.dirty
        self.currentPath = None
        self.baselineHash = None
        self.baselineLen = 0
        self.lastSeenText = ""
        self.dirty = False
        if prevDirty and prevPath is not None:
            self.dirtyChanged.emit(prevPath, False)
        if prevPath is not None:
            self.pathChanged.emit("")

    # —— Read-only queries ——
    def getCurrentPath(self) -> Optional[str]:
        return self.currentPath

    def isDirty(self) -> bool:
        return self.dirty

    def getBaselineHash(self) -> Optional[str]:
        return self.baselineHash


gDirtyTracker: Optional[DirtyTracker] = None


def getDirtyTracker() -> DirtyTracker:
    """Lazy singleton accessor. Must be called after a ``QApplication`` /
    ``QCoreApplication`` exists (QObject construction requirement)."""
    global gDirtyTracker
    if gDirtyTracker is None:
        gDirtyTracker = DirtyTracker()
    return gDirtyTracker


def setDirtyBaseline(path: str, text: str) -> None:
    getDirtyTracker().setBaseline(path, text)


def reportDirtyText(text: str) -> None:
    getDirtyTracker().reportText(text)


def markDirtyClean() -> None:
    getDirtyTracker().markClean()


def clearDirtyTracker() -> None:
    getDirtyTracker().clear()


def isDirty() -> bool:
    return getDirtyTracker().isDirty()


# =============================================================================
# Assets — 资源（图标 / 图形 / 音频）查找：内置 + 外部，外部优先
# =============================================================================
class Assets:
    """资源查找器：在 SDK 内置 ``assets/`` 与 sdk.ini ``[Resources].assets_dir``
    指定的外部目录之间求并集，外部优先。

    典型用法：

        Assets.find("icon/icon_close.svg")     # 任意相对路径
        Assets.find_icon("icon_close")          # icon/ 子目录的快捷查找
        Assets.list_icons()                     # 列出所有可用图标 stem
    """

    _ICON_EXTS = (".svg", ".png", ".ico")

    @classmethod
    def search_dirs(cls) -> List[Path]:
        """返回所有有效搜索目录，外部排在前（优先级高），内置在后。"""
        return Paths.get_assets_dirs()

    @classmethod
    def find(cls, rel: str) -> Optional[Path]:
        """按相对路径在搜索目录里查找；返回首个命中的绝对路径，找不到返回 None。

        - 绝对路径直通（存在则返回）。
        - 相对路径在每个搜索目录下尝试拼接。
        """
        if not rel:
            return None
        p = Path(rel)
        if p.is_absolute():
            return p if p.exists() else None
        for d in cls.search_dirs():
            candidate = d / rel
            if candidate.exists():
                return candidate
        return None

    @classmethod
    def find_icon(cls, name: str) -> Optional[Path]:
        """图标专用查找。接受三种输入：

        - 绝对路径：``C:/some/x.svg`` → 直通
        - 相对路径含分隔符 / 含扩展名：``icon/foo.png`` → 走 ``find()``
        - 仅 stem：``icon_close`` → 在每个搜索目录的 ``icon/`` 子目录里
          按 svg / png / ico 顺序探测扩展名
        """
        if not name:
            return None
        p = Path(name)
        # 绝对路径或显式相对路径
        if p.is_absolute() or p.suffix.lower() in cls._ICON_EXTS or "/" in name or "\\" in name:
            return cls.find(name)
        # 仅 stem：扫 icon/ 子目录的多种扩展名
        for d in cls.search_dirs():
            for ext in cls._ICON_EXTS:
                candidate = d / "icon" / f"{name}{ext}"
                if candidate.exists():
                    return candidate
        return None

    @classmethod
    def list_icons(cls) -> List[str]:
        """列出所有可用图标 stem（外部覆盖内置同名）。"""
        seen: Dict[str, Path] = {}
        for d in cls.search_dirs():
            icon_dir = d / "icon"
            if not icon_dir.exists():
                continue
            for f in icon_dir.iterdir():
                if f.suffix.lower() in cls._ICON_EXTS:
                    seen.setdefault(f.stem, f)  # 外部优先（先遍历）
        return sorted(seen.keys())


# =============================================================================
# CrashHook — 未捕获异常路由
# =============================================================================
class CrashHook:
    """注册 sys.excepthook + threading.excepthook，把未捕获异常路由到 log_error。

    用法：
        from ULTIMA_SDK import install_crash_hook
        install_crash_hook()                   # 同时写文件
        install_crash_hook(write_file=False)   # 只输出到 logger
    """

    _installed: bool = False
    _orig_excepthook = None
    _orig_threading_excepthook = None

    @classmethod
    def _crash_log_path(cls) -> Path:
        Paths.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        return Paths.LOGS_DIR / f"crash_{datetime.now():%Y%m%d}.log"

    @classmethod
    def _append_crash_log(cls, text: str) -> None:
        try:
            with cls._crash_log_path().open("a", encoding="utf-8") as f:
                f.write(f"\n========== {datetime.now():%Y-%m-%d %H:%M:%S} ==========\n")
                f.write(text)
                f.write("\n")
        except Exception:
            # 崩溃捕获器自己绝不能再抛——静默失败优先于级联崩溃
            pass

    @classmethod
    def install(cls, write_file: bool = True) -> None:
        if cls._installed:
            return

        cls._orig_excepthook = _stdlib_sys.excepthook
        cls._orig_threading_excepthook = getattr(threading, "excepthook", None)

        def _handle(exc_type, exc, tb):
            # KeyboardInterrupt 走原 hook，方便 Ctrl+C 退出
            if issubclass(exc_type, KeyboardInterrupt):
                if cls._orig_excepthook:
                    cls._orig_excepthook(exc_type, exc, tb)
                return
            text = "".join(traceback.format_exception(exc_type, exc, tb))
            try:
                from .logger import log_error  # 延迟导入避免初始化顺序耦合
                log_error(f"Unhandled exception:\n{text}")
            except Exception:
                _stdlib_sys.__stderr__.write(text)
            if write_file:
                cls._append_crash_log(text)

        def _thread_hook(args):
            _handle(args.exc_type, args.exc_value, args.exc_traceback)

        _stdlib_sys.excepthook = _handle
        if hasattr(threading, "excepthook"):
            threading.excepthook = _thread_hook

        cls._installed = True

    @classmethod
    def uninstall(cls) -> None:
        if not cls._installed:
            return
        if cls._orig_excepthook is not None:
            _stdlib_sys.excepthook = cls._orig_excepthook
        if cls._orig_threading_excepthook is not None and hasattr(threading, "excepthook"):
            threading.excepthook = cls._orig_threading_excepthook
        cls._installed = False


install_crash_hook = CrashHook.install
uninstall_crash_hook = CrashHook.uninstall


# =============================================================================
# I18n — 多语种支持
# =============================================================================
class I18n(QObject):
    """i18n 单例。

    职责：
    - 维护当前语种 + 已加载的翻译文件缓存
    - tr(key, default) 三层回退查找：当前语种 → EN → default
    - 自动收集所有 ``tr()`` 调用过的 (key, default)，便于 export_template() 同步
    - set_language() 切换语种、持久化到 sdk.ini、发射 language_changed 信号

    磁盘布局（分块导出）：
        ``{localisation_dir}/{LANG}/{category}.txt`` —— 每个 LANG 一个子目录，
        目录内按 key 首段（category）分文件；文件内按 key 第二段作 INI section、
        剩余部分作 option。例：``ui.button.ok`` → ``EN/ui.txt`` 内
        ``[button] ok = OK``。

    继承 QObject 仅为发射信号；状态全部走 classmethod，方便 `I18n.tr(...)` 直接调。
    """

    language_changed = pyqtSignal(str)  # 新语种代码

    BASE_LANG = "EN"  # 三层回退中的 base 语种

    _instance: Optional["I18n"] = None
    _lock = threading.Lock()

    _current_lang: str = "EN"
    # lang -> category -> 已解析 ini（每个 category 对应磁盘上的 {lang}/{category}.txt）
    _translations: Dict[str, Dict[str, ConfigParser]] = {}
    _registered_keys: Dict[str, str] = {}             # key -> default（保留首次出现的 default）

    def __init__(self):
        super().__init__()

    @classmethod
    def instance(cls) -> "I18n":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            # Parent to QCoreApplication once it exists so the singleton outlives
            # every widget that connected to ``language_changed``. Without this
            # the Python GC can collect the I18n QObject before the widgets'
            # destructors run, producing the noisy
            # ``QObject::disconnect: No such signal QObject::language_changed(QString)``
            # warning on every connected receiver during shutdown.
            if cls._instance.parent() is None:
                app = QCoreApplication.instance()
                if app is not None:
                    cls._instance.setParent(app)
            return cls._instance

    # ---- 生命周期 -------------------------------------------------------
    @classmethod
    def init(cls) -> None:
        """读 sdk.ini 的 [Localisation].lang 并加载对应文件。SDK 导入时调用。"""
        lang = Config.get_value("Localisation", "lang", cls.BASE_LANG) or cls.BASE_LANG
        cls._current_lang = lang
        cls._ensure_loaded(lang)
        if lang != cls.BASE_LANG:
            cls._ensure_loaded(cls.BASE_LANG)

    # ---- 公开 API -------------------------------------------------------
    @classmethod
    def tr(cls, key: str, default: str = "") -> str:
        """查翻译。三层回退：当前语种 → EN.txt → default 参数。"""
        # 自动收集（保留首次出现的 default 作为「源语种」基准）
        if key not in cls._registered_keys:
            cls._registered_keys[key] = default

        v = cls._lookup(cls._current_lang, key)
        if v:
            return v
        if cls._current_lang != cls.BASE_LANG:
            v = cls._lookup(cls.BASE_LANG, key)
            if v:
                return v
        return default if default else key

    @classmethod
    def set_language(cls, lang: str) -> None:
        cls._ensure_loaded(lang)
        cls._current_lang = lang
        Config.set_value("Localisation", "lang", lang)
        cls.instance().language_changed.emit(lang)

    @classmethod
    def get_language(cls) -> str:
        return cls._current_lang

    @classmethod
    def available_languages(cls) -> List[str]:
        """扫描翻译根目录，返回所有语种子目录名（按字母序）。

        新版本分块布局：每个语种对应一个子目录 ``localisation/{LANG}/``，
        其中放置若干 ``{category}.txt`` 文件。
        """
        d = cls._localisation_dir()
        if not d.exists():
            return []
        return sorted(p.name for p in d.iterdir() if p.is_dir())

    @classmethod
    def export_template(cls, lang: Optional[str] = None, merge: bool = True) -> Path:
        """导出已收集 key 到 ``{lang}/{category}.txt`` 分块文件。

        按 key 的首段（category）拆分写入对应文件；每个文件内按 key 第二段
        作为 section、剩余部分作为 option 组织：
          ``ui.button.ok`` → ``localisation/EN/ui.txt`` 内 ``[button] ok = ...``

        Args:
            lang: 目标语种。省略则用当前语种。
            merge: True 时保留已有翻译，仅追加缺失项（值=default）；False 时全量重写。
        Returns:
            写入的语种目录路径 ``localisation/{lang}/``。
        """
        lang = lang or cls._current_lang
        lang_dir = cls._lang_dir(lang)
        lang_dir.mkdir(parents=True, exist_ok=True)

        # 按 category 分组：category -> [(section, option, default), ...]
        grouped: Dict[str, List[tuple[str, str, str]]] = {}
        for key, default in cls._registered_keys.items():
            cat, sec, opt = cls._split_key(key)
            grouped.setdefault(cat, []).append((sec, opt, default))

        for category, entries in grouped.items():
            target = lang_dir / f"{category}.txt"

            # 翻译文件扩展名是 .txt，用 format='ini' 显式让 DataManager 走 INI Handler
            if merge and target.exists():
                cp = DataManager.load(target, format="ini", force_reload=True)
            else:
                cp = ConfigParser(strict=False)

            # 排序后写入，保证文件输出稳定
            for section, opt, default in sorted(entries):
                if not cp.has_section(section):
                    cp.add_section(section)
                if not cp.has_option(section, opt):
                    cp.set(section, opt, default or "")

            # 原子写入（临时文件 + os.replace）
            DataManager.write(target, cp, format="ini")

        # 让导出的文件下次 tr() 立即生效：失效 I18n 自身缓存
        cls._translations.pop(lang, None)
        return lang_dir

    @classmethod
    def registered_keys(cls) -> Dict[str, str]:
        """返回所有已观察到的 key -> default 映射的副本。"""
        return dict(cls._registered_keys)

    # ---- 内部 -----------------------------------------------------------
    @staticmethod
    def _split_key(key: str) -> tuple[str, str, str]:
        """三级拆分：返回 ``(category, section, option)``。

        约定（key 以 ``.`` 分段）：
        - ≥3 段：前两段分别为 category / section，剩余部分整体为 option。
          e.g. ``ui.button.actions.save`` → ``("ui", "button", "actions.save")``
        - 2 段：首段为 category，section 落入 ``"default"``。
          e.g. ``app.title`` → ``("app", "default", "title")``
        - 1 段：category 与 section 均为 ``"default"``。
          e.g. ``loose_key`` → ``("default", "default", "loose_key")``
        """
        parts = key.split(".", 2)
        if len(parts) >= 3:
            return parts[0], parts[1], parts[2]
        if len(parts) == 2:
            return parts[0], "default", parts[1]
        return "default", "default", parts[0]

    @classmethod
    def _localisation_dir(cls) -> Path:
        # sdk.ini 中 [Localisation].dir 为空时（Config.get_value 把空串当作
        # 未填写并返回 fallback），回退到 SDK_ROOT/localisation。
        # When sdk.ini's [Localisation].dir is blank, fall back to
        # SDK_ROOT/localisation.
        raw = Config.get_value("Localisation", "dir", "localisation")
        p = Path(raw)
        if not p.is_absolute():
            p = (Paths.SDK_ROOT / p).resolve()
        return p

    @classmethod
    def _lang_dir(cls, lang: str) -> Path:
        """返回 ``localisation/{lang}/`` 目录路径。"""
        return cls._localisation_dir() / lang

    @classmethod
    def _ensure_loaded(cls, lang: str) -> Dict[str, ConfigParser]:
        """加载并缓存某语种下所有 ``{category}.txt``，返回 ``{category: ConfigParser}``。"""
        if lang in cls._translations:
            return cls._translations[lang]

        bucket: Dict[str, ConfigParser] = {}
        d = cls._lang_dir(lang)
        if d.exists() and d.is_dir():
            for p in sorted(d.glob("*.txt")):
                if not p.is_file():
                    continue
                # 翻译文件 .txt 但内容是 INI；让 DataManager 走 IniHandler
                try:
                    bucket[p.stem] = DataManager.load(p, format="ini")
                except Exception:
                    # 单文件解析失败时静默跳过：后续查询走 EN / default 回退
                    bucket[p.stem] = ConfigParser(strict=False)
        cls._translations[lang] = bucket
        return bucket

    @classmethod
    def _lookup(cls, lang: str, key: str) -> Optional[str]:
        bucket = cls._ensure_loaded(lang)
        cat, section, opt = cls._split_key(key)
        cp = bucket.get(cat)
        if cp is None:
            return None
        if cp.has_option(section, opt):
            v = cp.get(section, opt).strip()
            return v or None
        return None


def tr(key: str, default: str = "") -> str:
    """顶层便捷函数。等价于 ``I18n.tr(key, default)``。"""
    return I18n.tr(key, default)


# =============================================================================
# Task 系统
# =============================================================================
class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskSignal(QObject):
    """任务相关全局信号中枢（线程安全单例）。"""

    progress_updated = pyqtSignal(int, float, str)
    status_changed = pyqtSignal(int, str, str)
    notify_requested = pyqtSignal(str, str, str)
    input_requested = pyqtSignal(str, str, str, str, list)
    input_response = pyqtSignal(str, object)
    task_finished = pyqtSignal(str, bool, object)
    slot_released = pyqtSignal(int)
    file_select_requested = pyqtSignal(str, str, str)
    file_select_response = pyqtSignal(str, list)

    _instance: Optional["TaskSignal"] = None
    _lock = threading.Lock()

    def __init__(self):
        super().__init__()

    @classmethod
    def instance(cls) -> "TaskSignal":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance


def get_task_signal() -> TaskSignal:
    return TaskSignal.instance()


# ---- CMD 屏进度条单一所有权 ------------------------------------------------
# cmd 屏每次只允许一个 task 输出 wrap=False 进度条，否则多任务会互相覆写。
# 其他 task 在并发期间只更新图形进度条（progress_updated 信号），不写 cmd。
_cmd_progress_owner: Optional[str] = None
_cmd_progress_lock = threading.Lock()


def _claim_cmd_progress(task_id: str) -> bool:
    """尝试把 cmd 屏进度行授予 task_id。返回 True 表示当前持有者就是 task_id。"""
    global _cmd_progress_owner
    with _cmd_progress_lock:
        if _cmd_progress_owner is None:
            _cmd_progress_owner = task_id
        return _cmd_progress_owner == task_id


def _release_cmd_progress(task_id: str) -> bool:
    """task_id 释放 cmd 屏进度行所有权（仅当它是当前持有者）。"""
    global _cmd_progress_owner
    with _cmd_progress_lock:
        if _cmd_progress_owner == task_id:
            _cmd_progress_owner = None
            return True
        return False


class TaskCancelledException(Exception):
    pass


class Task(ABC):
    """任务基类。子类实现 ``run()``，返回值即任务结果。"""

    def __init__(self, name: str = "Task"):
        self.id = str(uuid.uuid4())
        self.name = name
        self.status = TaskStatus.PENDING
        self.progress = 0.0
        self.result: Any = None
        self.error: Optional[Exception] = None

        self._slot_index: int = -1
        self._cancelled = False
        self._signal: Optional[TaskSignal] = None

        self._input_mutex = QMutex()
        self._input_condition = QWaitCondition()
        self._input_response: Any = None
        self._input_received = False

    def _bind(self, slot_index: int, signal: TaskSignal):
        self._slot_index = slot_index
        self._signal = signal

    @abstractmethod
    def run(self) -> Any:
        """子类实现。返回值即任务结果。"""

    # ---- 进度 / 日志 -----------------------------------------------------
    def set_progress(self, progress: float, text: str = ""):
        self.progress = max(0.0, min(1.0, progress))
        if self._signal and self._slot_index >= 0:
            self._signal.progress_updated.emit(self._slot_index, self.progress, text)

    def progress_line(self, ratio: float, text: str = "", bar_width: int = 30):
        """同时刷新 UI 槽位进度条 + cmd 屏单行进度条。

        cmd 屏单行进度条只允许一个活跃 task 同时占用：
        - 其他并发 task 自动跳过 cmd 输出（图形进度条仍正确）
        - ratio>=1.0 时所有者用 wrap=True 写最终一行（保留在日志），并释放所有权
        """
        from .logger import log_info  # 延迟导入避免循环

        ratio = max(0.0, min(1.0, ratio))
        self.set_progress(ratio, text)

        if not _claim_cmd_progress(self.id):
            return  # 别人在用 cmd 进度条；本任务只更新图形进度条

        filled = int(ratio * bar_width)
        bar = "#" * filled + "-" * (bar_width - filled)
        suffix = f" {text}" if text else ""
        line = f"[{self.name}] [{bar}] {ratio * 100:5.1f}%{suffix}"

        if ratio >= 1.0:
            log_info(line, wrap=True)
            _release_cmd_progress(self.id)
        else:
            log_info(line, wrap=False)

    def log(self, text: str, level: str = "info"):
        """统一桥接到 SDK 顶层 logger。"""
        from . import logger as _logger  # 延迟导入

        fn = {
            "info": _logger.log_info,
            "success": _logger.log_success,
            "warning": _logger.log_warning,
            "error": _logger.log_error,
            "debug": _logger.log_debug,
        }.get(level, _logger.log_info)
        fn(f"[{self.name}] {text}")

    def log_info(self, text: str):
        self.log(text, "info")

    def log_success(self, text: str):
        self.log(text, "success")

    def log_warning(self, text: str):
        self.log(text, "warning")

    def log_error(self, text: str):
        self.log(text, "error")

    def log_debug(self, text: str):
        self.log(text, "debug")

    # ---- 用户交互 --------------------------------------------------------
    def notify(self, title: str, message: str, notify_type: str = "info"):
        if self._signal:
            self._signal.notify_requested.emit(title, message, notify_type)

    def confirm(self, message: str, title: str = "Confirm") -> bool:
        return bool(self._request_input(title, message, "confirm", []))

    def input_text(self, prompt: str, title: str = "Input") -> Optional[str]:
        return self._request_input(title, prompt, "input", [])

    def input_choice(
        self, prompt: str, options: List[str], title: str = "Select"
    ) -> Optional[int]:
        return self._request_input(title, prompt, "choice", options)

    def _request_input(
        self, title: str, message: str, input_type: str, options: list
    ) -> Any:
        if not self._signal:
            return None

        request_id = str(uuid.uuid4())

        def on_response(rid, response):
            if rid == request_id:
                self._input_mutex.lock()
                self._input_response = response
                self._input_received = True
                self._input_condition.wakeAll()
                self._input_mutex.unlock()

        self._signal.input_response.connect(on_response)

        self._input_received = False
        self._signal.input_requested.emit(
            request_id, title, message, input_type, options
        )

        self._input_mutex.lock()
        while not self._input_received and not self._cancelled:
            self._input_condition.wait(self._input_mutex, 100)
        self._input_mutex.unlock()

        try:
            self._signal.input_response.disconnect(on_response)
        except Exception:
            pass

        return self._input_response if self._input_received else None

    # ---- 控制 ------------------------------------------------------------
    def cancel(self):
        self._cancelled = True
        self._input_mutex.lock()
        self._input_condition.wakeAll()
        self._input_mutex.unlock()

    def check_cancelled(self):
        if self._cancelled:
            raise TaskCancelledException(f"Task {self.name} was cancelled")

    def sleep(self, seconds: float):
        """可中断的睡眠。"""
        interval = 0.1
        elapsed = 0.0
        while elapsed < seconds:
            if self._cancelled:
                raise TaskCancelledException(f"Task {self.name} was cancelled")
            time.sleep(min(interval, seconds - elapsed))
            elapsed += interval


class TaskWorker(QThread):
    """单任务工作线程。"""

    finished_signal = pyqtSignal(str, bool, object)

    def __init__(self, task: Task, parent=None):
        super().__init__(parent)
        self._task = task

    def run(self):
        task = self._task
        success = False
        result = None
        try:
            task.status = TaskStatus.RUNNING
            result = task.run()
            task.result = result
            task.status = TaskStatus.COMPLETED
            success = True
        except TaskCancelledException:
            task.status = TaskStatus.CANCELLED
        except Exception as e:
            task.error = e
            task.status = TaskStatus.FAILED
            from .logger import log_error
            log_error(f"Task {task.name} failed: {e}")
        finally:
            # 任务无论成败/取消都释放 cmd 屏进度所有权（若它持有），
            # 让队列中下一个任务能接手 cmd 进度行。
            _release_cmd_progress(task.id)
            self.finished_signal.emit(task.id, success, result)


class TaskSlot:
    """单任务执行槽位。"""

    def __init__(self, index: int, signal: TaskSignal):
        self.index = index
        self._signal = signal
        self._task: Optional[Task] = None
        self._worker: Optional[TaskWorker] = None
        self._on_finished: Optional[Callable] = None

    @property
    def is_busy(self) -> bool:
        return self._task is not None

    @property
    def current_task(self) -> Optional[Task]:
        return self._task

    def execute(self, task: Task, on_finished: Callable[[Task], None]):
        if self.is_busy:
            raise RuntimeError(f"Slot {self.index} is busy")

        self._task = task
        self._on_finished = on_finished

        task._bind(self.index, self._signal)
        self._signal.status_changed.emit(self.index, task.id, "running")

        self._worker = TaskWorker(task)
        self._worker.finished_signal.connect(self._on_worker_finished)
        self._worker.start()

    def _on_worker_finished(self, task_id: str, success: bool, result: Any):
        task = self._task
        if task is None:
            return

        if success:
            status = "completed"
        elif task.status == TaskStatus.CANCELLED:
            status = "cancelled"
        else:
            status = "failed"
        self._signal.status_changed.emit(self.index, task_id, status)
        self._signal.task_finished.emit(task_id, success, result)

        if self._worker:
            self._worker.quit()
            self._worker.wait(1000)
            self._worker.deleteLater()
            self._worker = None

        callback = self._on_finished
        finished_task = self._task

        self._task = None
        self._on_finished = None

        self._signal.slot_released.emit(self.index)

        if callback and finished_task:
            callback(finished_task)

    def cancel(self):
        if self._task:
            self._task.cancel()


class TasksManager:
    """N 槽位并发任务管理器（懒加载单例）。

    支持运行时通过 ``configure(max_slots=...)`` 调整槽位数（仅在所有槽位空闲时允许）。
    """

    DEFAULT_SLOTS = 4
    HARD_MAX = 16

    _instance: Optional["TasksManager"] = None
    _lock = threading.Lock()

    def __init__(self, max_slots: int = DEFAULT_SLOTS):
        self._signal = get_task_signal()
        self._max_slots = self._clamp(max_slots)
        self._slots: List[TaskSlot] = [
            TaskSlot(i, self._signal) for i in range(self._max_slots)
        ]
        self._queue: List[Task] = []
        self._tasks: Dict[str, Task] = {}
        self._queue_lock = threading.Lock()

    @classmethod
    def _clamp(cls, n: int) -> int:
        return min(max(1, int(n)), cls.HARD_MAX)

    @classmethod
    def instance(cls) -> "TasksManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @property
    def max_slots(self) -> int:
        return self._max_slots

    def configure(self, max_slots: int) -> bool:
        """调整槽位数（仅在所有槽位空闲且队列为空时生效）。"""
        n = self._clamp(max_slots)
        with self._queue_lock:
            if any(s.is_busy for s in self._slots) or self._queue:
                return False
            self._max_slots = n
            self._slots = [TaskSlot(i, self._signal) for i in range(n)]
            return True

    def submit(self, task: Task) -> str:
        self._tasks[task.id] = task
        slot = self._get_free_slot()
        if slot:
            slot.execute(task, self._on_task_finished)
        else:
            with self._queue_lock:
                self._queue.append(task)
        return task.id

    def _get_free_slot(self) -> Optional[TaskSlot]:
        for slot in self._slots:
            if not slot.is_busy:
                return slot
        return None

    def _on_task_finished(self, task: Task):
        with self._queue_lock:
            if self._queue:
                next_task = self._queue.pop(0)
                slot = self._get_free_slot()
                if slot:
                    slot.execute(next_task, self._on_task_finished)

    def cancel(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if not task:
            return False

        with self._queue_lock:
            if task in self._queue:
                self._queue.remove(task)
                task.status = TaskStatus.CANCELLED
                return True

        for slot in self._slots:
            if slot.current_task and slot.current_task.id == task_id:
                slot.cancel()
                return True

        return False

    def shutdown(self, timeout_ms: int = 3000) -> None:
        """取消所有排队/运行中的任务，等待 worker 线程退出。

        在 ``QApplication.aboutToQuit`` 时调用，避免 Qt 在 worker QThread
        仍在运行时销毁它（"QThread: Destroyed while thread is still
        running" 会让进程以非零码退出）。子进程层面的强制中止由各 Task
        在 ``check_cancelled()`` 抛 ``TaskCancelledException`` 时自然解栈
        负责；本方法只保证 QThread 干净退出。

        - 清空队列（队列项标记为 cancelled，不会再被调度）。
        - 对每个 busy 槽位调用 ``cancel()``（设置 task 的取消标志）。
        - 等 worker 在 ``timeout_ms`` 内退出；超时则放弃等待，由 Qt 警告。
        """
        with self._queue_lock:
            for t in self._queue:
                t.status = TaskStatus.CANCELLED
            self._queue.clear()

        for slot in self._slots:
            try:
                slot.cancel()
            except Exception:
                pass

        deadline_per_slot = max(50, int(timeout_ms) // max(1, len(self._slots)))
        for slot in self._slots:
            worker = getattr(slot, "_worker", None)
            if worker is None:
                continue
            try:
                if worker.isRunning():
                    worker.wait(deadline_per_slot)
            except Exception:
                pass

    def get_task(self, task_id: str) -> Optional[Task]:
        return self._tasks.get(task_id)

    def get_slot_status(self) -> List[Dict]:
        result = []
        for slot in self._slots:
            task = slot.current_task
            result.append(
                {
                    "index": slot.index,
                    "busy": slot.is_busy,
                    "task_id": task.id if task else None,
                    "task_name": task.name if task else None,
                    "progress": task.progress if task else 0.0,
                }
            )
        return result


def get_tasks_manager() -> TasksManager:
    return TasksManager.instance()


def set_max_slots(n: int) -> bool:
    """调整全局 TasksManager 的槽位数。当前有任务运行/排队时返回 False。"""
    return get_tasks_manager().configure(n)
