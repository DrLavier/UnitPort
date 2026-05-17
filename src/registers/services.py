"""registers.services — 运行时服务/绑定/检查点/路由/存储通道 一站式注册.

合并自 5 个原子目录 / Merged from 5 former subdirs:
- services/{adapter_registry,bindings,checkpoints}.py    → 本文件 "live services" 区
- runtime_routes/{ros2_idl,ros2_topics,motors,controllers}.py → 本文件 "routes" 区
- storage/channels.py                                    → 本文件 "storage channels" 区

实例数据生命周期：
- 实例（adapter 单例 / binding 表 / checkpoint 条目）由 application/service/ 在
  阶段 C 启动时主动 ``register_*()``。
- 路由（ROS2 IDL / topic / motor）由 application/engine/ ROS2 桥在阶段 C 注册。
- 存储通道为**文档化契约**（无运行时状态），表征 SDK ``Storage.push_data`` 的三种通道。

API（按区分组）:
    # live services -----------
    register_adapter / get_adapter / list_adapters
    register_binding / get_binding / list_bindings
    register_checkpoint / get_checkpoint / list_checkpoints
    # routes ------------------
    register_ros2_idl / get_ros2_idl
    register_ros2_topic / get_ros2_topic
    register_motor / get_motor / list_motors
    # storage channels --------
    list_channels / get_channel
    # framework ---------------
    load() -> int
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


_state: Dict[str, Any] = {
    "loaded": False,
    "adapters": {},     # adapter_id → info
    "bindings": {},     # (intent_id, brand, robot_type, backend) → binding
    "checkpoints": {},  # policy_id → CheckpointEntry
    "ros2_idl": {},     # msg_type → Type
    "ros2_topics": {},  # (brand_id, kind, name) → TopicBinding
    "motors": {},       # motor_id → MotorSpec
}


def load() -> int:
    """阶段 B 占位 / Stage B placeholder. 实例由阶段 C 启动时通过 register_*() 主动注册。"""
    _state["loaded"] = True
    return 0


# ---------------------------------------------------------------------------
# Live services（来自 services/{adapter_registry,bindings,checkpoints}.py）
# ---------------------------------------------------------------------------

def register_adapter(adapter_id: str, info: Dict[str, Any]) -> None:
    if not _state["loaded"]:
        load()
    _state["adapters"][adapter_id] = info


def get_adapter(adapter_id: str) -> Optional[Dict[str, Any]]:
    if not _state["loaded"]:
        load()
    return _state["adapters"].get(adapter_id)


def list_adapters() -> List[Dict[str, Any]]:
    if not _state["loaded"]:
        load()
    return list(_state["adapters"].values())


def register_binding(key: tuple, binding: Dict[str, Any]) -> None:
    if not _state["loaded"]:
        load()
    _state["bindings"][key] = binding


def get_binding(key: tuple) -> Optional[Dict[str, Any]]:
    if not _state["loaded"]:
        load()
    return _state["bindings"].get(key)


def list_bindings() -> List[Dict[str, Any]]:
    if not _state["loaded"]:
        load()
    return list(_state["bindings"].values())


def register_checkpoint(policy_id: str, entry: Dict[str, Any]) -> None:
    if not _state["loaded"]:
        load()
    _state["checkpoints"][policy_id] = entry


def get_checkpoint(policy_id: str) -> Optional[Dict[str, Any]]:
    if not _state["loaded"]:
        load()
    return _state["checkpoints"].get(policy_id)


def list_checkpoints() -> List[Dict[str, Any]]:
    if not _state["loaded"]:
        load()
    return list(_state["checkpoints"].values())


# ---------------------------------------------------------------------------
# Runtime routes（来自原 runtime_routes/{ros2_idl,ros2_topics,motors}.py）
# controller 硬件 profile 属 per-machine 用户级数据，**实例数据走 USER_CONFIG_DIR**
# （~/UnitPort/robots/profile/controllers.json），此处仅保留 schema/loader 模板。
# ---------------------------------------------------------------------------

def register_ros2_idl(msg_type: str, type_obj: Any) -> None:
    if not _state["loaded"]:
        load()
    _state["ros2_idl"][msg_type] = type_obj


def get_ros2_idl(msg_type: str) -> Optional[Any]:
    if not _state["loaded"]:
        load()
    return _state["ros2_idl"].get(msg_type)


def register_ros2_topic(key: tuple, binding: Dict[str, Any]) -> None:
    if not _state["loaded"]:
        load()
    _state["ros2_topics"][key] = binding


def get_ros2_topic(key: tuple) -> Optional[Dict[str, Any]]:
    if not _state["loaded"]:
        load()
    return _state["ros2_topics"].get(key)


def register_motor(motor_id: str, spec: Dict[str, Any]) -> None:
    if not _state["loaded"]:
        load()
    _state["motors"][motor_id] = spec


def get_motor(motor_id: str) -> Optional[Dict[str, Any]]:
    if not _state["loaded"]:
        load()
    return _state["motors"].get(motor_id)


def list_motors() -> List[Dict[str, Any]]:
    if not _state["loaded"]:
        load()
    return list(_state["motors"].values())


# ---------------------------------------------------------------------------
# Storage channels（来自原 storage/channels.py）
# 业务代码必须经 SDK ``Storage.push_data(rel, value, channel=...)`` 调用，
# **禁止**绕过通道直接读写底层介质。
# ---------------------------------------------------------------------------

CHANNELS: Dict[str, Dict[str, Any]] = {
    "local": {
        "id": "local",
        "label": "Local prefs",
        "medium": "filesystem",
        "root_path_constant": "Paths.USER_CONFIG_DIR",
        "default_root": "~/UnitPort/",
        "secret_safe": False,
        "_doc": (
            "用户档案级文件存储。非敏感设置（主题、最近项目、controller 绑定、"
            "连接元数据等）。SDK Storage.push 'channel=local' 走此通道。"
        ),
        "examples": [
            "user.ini",
            "registers/ir_custom.json",
            "registers/robots_custom.json",
            "engines/<engine>.json",
            "auth/session.json",
            "recents.json",
        ],
    },
    "secure": {
        "id": "secure",
        "label": "OS credential manager",
        "medium": "keyring",
        "root_path_constant": None,
        "default_root": "OS keyring (Windows Credential Manager / macOS Keychain / Secret Service)",
        "secret_safe": True,
        "_doc": (
            "敏感凭据存 OS 钥匙串。auth refresh token、SSH 密码、机器人凭据、"
            "LLM API key 走此通道。SDK 当前未在顶层暴露 secure-only API；"
            "通过 application/service/auth.py 与 SDK Storage 内部桥接。"
        ),
        "service_namespaces": [
            "AUTH_SERVICE",
            "SSH_SERVICE",
            "ROBOT_SERVICE",
            "LLM_SERVICE",
        ],
    },
    "cloud": {
        "id": "cloud",
        "label": "Cloud sync (WIP)",
        "medium": "supabase|sync",
        "root_path_constant": None,
        "default_root": "Supabase（待实装）",
        "secret_safe": False,
        "_doc": (
            "用户授权的云端同步通道。当前 SDK 实现是 stub —— "
            "Storage.push_data(channel='cloud') 记录 log_error 后返回 False，"
            "不应作为关键路径。**Secrets never sync via cloud.**"
        ),
        "categories": [
            "preferences",
            "connection_profiles",
            "training_servers",
            "robot_settings",
        ],
        "wip": True,
    },
}


def get_channel(channel_id: str) -> Dict[str, Any]:
    return dict(CHANNELS.get(channel_id, {}))


def list_channels() -> List[Dict[str, Any]]:
    return [dict(v) for v in CHANNELS.values()]


__all__ = [
    "load",
    # live services
    "register_adapter", "get_adapter", "list_adapters",
    "register_binding", "get_binding", "list_bindings",
    "register_checkpoint", "get_checkpoint", "list_checkpoints",
    # routes
    "register_ros2_idl", "get_ros2_idl",
    "register_ros2_topic", "get_ros2_topic",
    "register_motor", "get_motor", "list_motors",
    # storage channels
    "CHANNELS", "get_channel", "list_channels",
]
