"""registers.nodes — 节点 discovery facade / Hot-pluggable node discovery.

升级自原 catalog stub。两条扫描路径，共享同一文件夹契约：

- 第一方（出厂内置）: ``Paths.SRC_ROOT / "nodes" / <id>/``
- 三方（社区/用户）  : ``Paths.PROJECT_ROOT / "custom_mods" / "nodes" / <id>/``

每个节点文件夹必须包含三件套：
- ``__init__.py``    暴露 ``NODE_CLASSES = [SomeNodeClass, ...]``
- ``manifest.toml``  静态元数据（schema/id/kind/category/version/inputs/outputs/parameters）
- ``node.py``        ``BaseNode`` 子类实现

启动期由 ``RegistryHub.load_all()`` 调用 ``load()`` 一次性扫描；冲突规则：
- 内置 vs 内置 id 撞车 → 后扫到的拒绝并 ``log_error``
- 三方 vs 内置 / 三方 vs 三方 → 后扫到的拒绝并 ``log_warning``
- manifest 不合法 / 类型非 BaseNode 子类 → 单个节点拒绝，不影响其他节点

第一方节点 manifest 写出到 ``data/nodes_system.json`` 作快照（人类可读、可 git diff review）；
三方节点不写入快照，每次启动重新扫盘以避免 stale。

API:
    load() -> int                                 总注册数
    get_node_class(node_id) -> type[BaseNode] | None
    get_manifest(node_id) -> NodeManifest | None
    list_nodes(category=None) -> list[NodeManifest]
    iter_node_classes() -> Iterable[(node_id, cls)]
    # 兼容旧 API
    list_system_nodes() -> list[dict]              builtin manifest dict 列表
    list_custom_scan_dirs() -> list[str]
    list_hb_nodes() -> list[dict]                  Behavior tree 节点（阶段 C 填）
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

from unitport_sdk import (
    Paths,
    log_debug,
    log_error,
    log_success,
    log_warning,
    read_data,
    save_data,
)

from application.compiler.nodes import BaseNode, NodeManifest


_DATA_DIR = Paths.REGISTERS_DIR / "data"
_SYSTEM_PATH = _DATA_DIR / "nodes_system.json"

_BUILTIN_ROOT = Paths.SRC_ROOT / "nodes"
_CUSTOM_ROOT = Paths.CUSTOM_MODS_DIR / "nodes"


# Registry entry shape:
#   _state["registry"][node_id] = {
#       "class": Type[BaseNode],
#       "manifest": NodeManifest,
#       "source": "builtin" | "custom",
#       "path": str,
#   }
_state: Dict[str, Any] = {
    "loaded": False,
    "registry": {},
    "builtin_count": 0,
    "custom_count": 0,
    "conflicts": [],
    "hb_nodes": {},  # 阶段 C 填
}


# =============================================================================
# 主入口：load / reload
# =============================================================================

def load() -> int:
    """启动期一次性扫描两条路径并注册所有节点 / Scan-and-register on app boot.

    Returns: 注册成功的节点总数（不含被拒的冲突 / 不合法）。
    """
    if _state["loaded"]:
        return len(_state["registry"])

    _state["registry"] = {}
    _state["conflicts"] = []

    _state["builtin_count"] = _scan_dir(_BUILTIN_ROOT, source="builtin")
    _state["custom_count"] = _scan_dir(_CUSTOM_ROOT, source="custom")

    _refresh_snapshot()
    _state["loaded"] = True

    total = len(_state["registry"])
    log_success(
        f"[registers.nodes] discovery: built_in={_state['builtin_count']} "
        f"custom={_state['custom_count']} conflicts={len(_state['conflicts'])} "
        f"total={total}"
    )
    return total


def reload() -> int:
    """清空状态后重新扫盘 / Clear state and rescan."""
    _state["loaded"] = False
    _state["registry"] = {}
    _state["builtin_count"] = 0
    _state["custom_count"] = 0
    _state["conflicts"] = []
    return load()


# =============================================================================
# 内部：扫描 + importlib 加载
# =============================================================================

def _scan_dir(root: Path, source: str) -> int:
    """扫一条路径下所有节点子目录 / Scan one root for node packages.

    Args:
        root: 路径根（``src/nodes/`` 或 ``custom_mods/nodes/``）
        source: ``"builtin"`` 或 ``"custom"``

    Returns: 成功注册数
    """
    if not root.exists() or not root.is_dir():
        log_debug(f"[registers.nodes] {source} root 不存在，跳过: {root}")
        return 0

    count = 0
    root_resolved = root.resolve()
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        if sub.name.startswith("_") or sub.name.startswith("."):
            continue  # 隐藏 / 私有目录跳过
        # Symlink-traversal boundary (Plan finding P2-3): refuse to load a
        # node package whose resolved path escapes ``root``. ``iterdir``
        # by itself doesn't follow symlinks during enumeration, but a
        # symlinked child still resolves outside — defence in depth so a
        # planted symlink under ``custom_mods/nodes/`` cannot pull code
        # from arbitrary disk locations.
        try:
            sub_resolved = sub.resolve()
        except OSError as exc:
            log_warning(f"[registers.nodes] {sub.name}: resolve 失败 — {exc}")
            continue
        if not sub_resolved.is_relative_to(root_resolved):
            log_warning(
                f"[registers.nodes] {sub.name}: 解析后位于 {root} 之外，已拒绝"
            )
            continue
        manifest_path = sub / "manifest.toml"
        init_path = sub / "__init__.py"
        if not manifest_path.exists() or not init_path.exists():
            # 不是节点包（缺三件套）— 静默跳过
            continue
        if _load_one(sub, manifest_path, init_path, source):
            count += 1
    return count


def _validate_params(manifest: NodeManifest) -> List[str]:
    """对每个 ParamSpec 跑 ``param_rows.validate_param_spec``；返回错误列表.

    校验是 UI 自动转换契约的一部分（widget / type / choices / meta 是否合
    法），对应 ``create_param_row`` 工厂分发能否找到合适的 ParamRow 子类。
    """
    try:
        from application.ui.canvas.param_rows import validate_param_spec
    except Exception:
        # UI 还没初始化 / 测试环境无 PyQt → 跳过校验，不阻塞 headless 流程
        return []
    errors: List[str] = []
    for spec in manifest.parameters:
        msg = validate_param_spec(spec)
        if msg:
            errors.append(msg)
    return errors


def _load_one(
    node_dir: Path,
    manifest_path: Path,
    init_path: Path,
    source: str,
) -> bool:
    """加载并注册单个节点包 / Load and register one node package.

    Returns: True 注册成功；False 拒绝（冲突 / manifest 不合法 / 类型不符）
    """
    # 1. 读 manifest.toml
    try:
        manifest_data = read_data(manifest_path)
    except Exception as e:
        log_warning(f"[registers.nodes] {node_dir.name}: 读 manifest 失败 — {e}")
        return False

    if not isinstance(manifest_data, dict):
        log_warning(f"[registers.nodes] {node_dir.name}: manifest 不是 dict")
        return False

    # 2. 校验 schema / id / kind
    try:
        manifest = NodeManifest.from_dict(manifest_data)
    except ValueError as e:
        log_warning(f"[registers.nodes] {node_dir.name}: manifest 不合法 — {e}")
        return False

    # 2b. 校验 ParamSpec 与 UI 自动转换 contract 对齐（widget/type/choices/meta）
    #     失败抛 warn 但不阻断 discovery，便于 dev 期发现而不是直接整体失败。
    param_errors = _validate_params(manifest)
    if param_errors:
        for err in param_errors:
            log_warning(f"[registers.nodes] {node_dir.name}: {err}")

    # 3. id 唯一性（先发现的赢）
    if manifest.id in _state["registry"]:
        existing = _state["registry"][manifest.id]
        if source == "builtin" and existing["source"] == "builtin":
            log_error(
                f"[registers.nodes] 内置节点 id 撞车: '{manifest.id}' "
                f"已由 {existing['path']} 注册，{node_dir} 被拒绝"
            )
        else:
            log_warning(
                f"[registers.nodes] 节点 id 撞车: '{manifest.id}' "
                f"已由 {existing['source']}={existing['path']} 注册，"
                f"{source}={node_dir} 被拒绝"
            )
        _state["conflicts"].append({
            "id": manifest.id,
            "loser_source": source,
            "loser_path": str(node_dir),
            "winner_path": existing["path"],
        })
        return False

    # 4. importlib 加载节点包（不污染 sys.path；用合成 module 名避免撞车）
    sanitized = manifest.id.replace(".", "_").replace("-", "_").replace("/", "_")
    module_name = f"_unitport_nodes_{source}_{sanitized}"
    try:
        spec = importlib.util.spec_from_file_location(
            module_name,
            init_path,
            submodule_search_locations=[str(node_dir)],
        )
        if spec is None or spec.loader is None:
            log_warning(f"[registers.nodes] {node_dir.name}: importlib spec 创建失败")
            return False
        module = importlib.util.module_from_spec(spec)
        # 必须先注册到 sys.modules，相对导入（``from .node import X``）才能解析。
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            # 失败时清理 sys.modules，避免污染
            sys.modules.pop(module_name, None)
            raise
    except Exception as e:
        log_warning(f"[registers.nodes] {node_dir.name}: 导入失败 — {e!r}")
        return False

    # 5. 读 NODE_CLASSES
    node_classes = getattr(module, "NODE_CLASSES", None)
    if not node_classes:
        log_warning(f"[registers.nodes] {node_dir.name}: 未暴露 NODE_CLASSES")
        return False
    if not isinstance(node_classes, (list, tuple)):
        log_warning(
            f"[registers.nodes] {node_dir.name}: NODE_CLASSES 必须是 list/tuple"
        )
        return False

    # 6. 选择主类（其 MANIFEST.id 必须与 manifest.toml 的 id 对齐）
    chosen_cls: Optional[Type[BaseNode]] = None
    for cls in node_classes:
        if not isinstance(cls, type) or not issubclass(cls, BaseNode):
            log_warning(
                f"[registers.nodes] {node_dir.name}: NODE_CLASSES 含非 BaseNode 子类 {cls!r}"
            )
            continue
        cls_manifest = getattr(cls, "MANIFEST", None)
        if not isinstance(cls_manifest, NodeManifest):
            log_warning(
                f"[registers.nodes] {node_dir.name}: {cls.__name__} 缺 MANIFEST 类属性"
            )
            continue
        if cls_manifest.id == manifest.id:
            chosen_cls = cls
            break

    if chosen_cls is None:
        log_warning(
            f"[registers.nodes] {node_dir.name}: NODE_CLASSES 中无 id={manifest.id} 的节点类"
        )
        return False

    # 7. 注册
    _state["registry"][manifest.id] = {
        "class": chosen_cls,
        "manifest": manifest,
        "source": source,
        "path": str(node_dir),
    }
    log_debug(
        f"[registers.nodes] 注册 {source} 节点: {manifest.id} ({chosen_cls.__name__})"
    )
    return True


def _refresh_snapshot() -> None:
    """把 built-in 节点 manifest 写到 ``data/nodes_system.json``.

    三方节点不写入此文件——每次启动重扫盘以避免 stale 与隐私泄露。
    """
    builtin_manifests = {
        nid: entry["manifest"].to_dict()
        for nid, entry in _state["registry"].items()
        if entry["source"] == "builtin"
    }
    payload = {
        "schema": "unitport.nodes_system/v1",
        "version": "1.0.0",
        "_doc": "由 registers.nodes.load() 自动维护的内置节点快照；可 git diff review",
        "system": builtin_manifests,
        "custom_scan_dirs": [str(_CUSTOM_ROOT)],
    }
    try:
        save_data(_SYSTEM_PATH, payload)
    except Exception as e:
        log_warning(f"[registers.nodes] 写 nodes_system.json 失败: {e!r}")


# =============================================================================
# 公开查询 API
# =============================================================================

def get_node_class(node_id: str) -> Optional[Type[BaseNode]]:
    """通过 id 取节点类 / Get a node class by id."""
    if not _state["loaded"]:
        load()
    entry = _state["registry"].get(node_id)
    return entry["class"] if entry else None


def get_manifest(node_id: str) -> Optional[NodeManifest]:
    """通过 id 取节点 manifest / Get a node manifest by id."""
    if not _state["loaded"]:
        load()
    entry = _state["registry"].get(node_id)
    return entry["manifest"] if entry else None


def list_nodes(category: Optional[str] = None) -> List[NodeManifest]:
    """列出所有 manifest，可按 category 过滤 / List all manifests, optionally filter."""
    if not _state["loaded"]:
        load()
    result: List[NodeManifest] = []
    for entry in _state["registry"].values():
        m: NodeManifest = entry["manifest"]
        if category is None or m.category == category:
            result.append(m)
    return result


def iter_node_classes() -> Iterable[Tuple[str, Type[BaseNode]]]:
    """迭代 (node_id, class) 对 / Iterate (node_id, class) pairs."""
    if not _state["loaded"]:
        load()
    for nid, entry in _state["registry"].items():
        yield nid, entry["class"]


def get_conflicts() -> List[Dict[str, Any]]:
    """返回本次启动期遇到的所有冲突（用于 UI 节点管理面板）/ Return all conflicts seen at boot."""
    if not _state["loaded"]:
        load()
    return list(_state["conflicts"])


# =============================================================================
# 兼容旧 API（plan.md §11.7 #6 列出的契约 + 原 stub）
# =============================================================================

def list_system_nodes() -> List[Dict[str, Any]]:
    """builtin 节点 manifest dict 列表 / Builtin node manifest dicts."""
    if not _state["loaded"]:
        load()
    return [
        {"id": nid, **entry["manifest"].to_dict()}
        for nid, entry in _state["registry"].items()
        if entry["source"] == "builtin"
    ]


def list_custom_scan_dirs() -> List[str]:
    """扫描路径列表 / Scan path list."""
    return [str(_CUSTOM_ROOT)]


def list_hb_nodes() -> List[Dict[str, Any]]:
    """Behavior tree 节点目录（阶段 C 填）/ HB node catalog (phase C)."""
    if not _state["loaded"]:
        load()
    return list(_state["hb_nodes"].values())


__all__ = [
    "load",
    "reload",
    "get_node_class",
    "get_manifest",
    "list_nodes",
    "iter_node_classes",
    "get_conflicts",
    # 兼容旧 API
    "list_system_nodes",
    "list_custom_scan_dirs",
    "list_hb_nodes",
]
