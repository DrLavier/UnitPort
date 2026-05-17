# UnitPort 三方节点插件目录 / Custom Nodes Directory

把社区节点放到本目录的子文件夹里，**重启 UnitPort** 后即被发现并加入调色板。

## 文件夹契约 / Folder contract

每个节点 = 一个子文件夹 `<node_id>/`，必须包含：

```
<node_id>/
├── __init__.py        # 必需：暴露 NODE_CLASSES = [YourNodeClass]
├── manifest.toml      # 必需：静态元数据
└── node.py            # 必需：BaseNode 子类
```

可选文件：
- `icon.svg`           调色板图标
- `README.md`          文档（在 UI tooltip 中显示）
- `assets/`            节点自带资源
- `requirements.txt`   第三方 Python 依赖（**仅声明**，UnitPort 不会自动安装）

## manifest.toml 模板

```toml
schema = "unitport.node/v1"
id = "my_company.my_node"      # 全局唯一；建议加命名空间前缀避免撞车
kind = "CUSTOM"                 # 当前 NodeKind 只保留一个变体：CUSTOM
category = "custom"             # 调色板分组
layer = "A"                     # 可选：A | B | C | D | IL；决定标题栏配色
version = "0.1.0"
display_name_key = "node.my_node.display"      # 走 tr() 多语言；可缺省由 id 兜底
description_key = "node.my_node.desc"
icon = "icon.svg"

[[inputs]]
name = "in_data"
type = "any"                    # 端口 type 决定连线颜色（参见 src/application/ui/canvas/port_palette.py）

[[outputs]]
name = "out_data"
type = "any"

[[parameters]]
key = "message"
type = "string"
default = "hello"
description = "要打印的消息"
```

`NodeKind` 当前只剩 `CUSTOM`（mission canvas FSM 控制流变体已在 RELEASE 废弃）。

## __init__.py 模板

```python
from .node import MyNode

NODE_CLASSES = [MyNode]   # 一个文件夹可注册多个节点
```

## node.py 模板

```python
from __future__ import annotations
from typing import Any, Dict

from application.compiler.nodes import (
    BaseNode, NodeKind, NodeManifest, PortSpec, ParamSpec,
    NODE_MANIFEST_SCHEMA,
)


class MyNode(BaseNode):
    MANIFEST = NodeManifest(
        schema=NODE_MANIFEST_SCHEMA,
        id="my_company.my_node",
        kind=NodeKind.CUSTOM,
        version="0.1.0",
        category="custom",
        layer="A",
        inputs=[PortSpec(name="in_data", type="any")],
        outputs=[PortSpec(name="out_data", type="any")],
        parameters=[
            ParamSpec(key="message", type="string", default="hello"),
        ],
    )

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # 后端搭线时把 raise 换成真实装配逻辑；
        # 前端阶段保留 NotImplementedError 让误调直接报错而非静默 no-op。
        raise NotImplementedError(
            f"{self.MANIFEST.id}.execute pending backend wiring"
        )
```

## 冲突与校验

- **id 撞车（与内置）**：三方节点拒绝注册，UnitPort 启动日志会出现 `log_warning`
- **id 撞车（三方之间）**：先扫到的赢，后者拒绝 + warning
- **manifest 不识别 / kind 非法 / 必填字段缺失**：单个节点拒绝，不影响其他节点加载
- **NODE_CLASSES 中类不是 BaseNode 子类**：拒绝 + warning

## 不做什么 / Anti-patterns

- ❌ 在 `manifest.toml` 中塞可执行 Python 代码（manifest 只能是声明式数据）
- ❌ 跨节点目录 `import`（每个节点应自包含；公共逻辑应放进 `requirements.txt` 引用的 PyPI 包）
- ❌ 在节点 `node.py` 顶层执行重计算或 I/O（importlib 加载所有节点；顶层副作用会拖慢启动）
- ❌ 期望运行时热加载（设计上**不**支持——增删节点都需要重启）

## 安装

直接把 `<node_id>/` 文件夹复制（或 git clone）到本目录下，重启 UnitPort 即可。
