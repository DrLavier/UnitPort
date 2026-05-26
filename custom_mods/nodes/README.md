<!--
SPDX-FileCopyrightText: 2026 SU CHANG
SPDX-License-Identifier: Apache-2.0
-->

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
layer = "A"                     # 可选：A | B | C | D | IL | TOOLS；决定标题栏配色
version = "0.1.0"
display_name_key = "node.my_node.display"      # 走 tr() 多语言；可缺省由 id 兜底
description_key = "node.my_node.desc"
icon = "icon.svg"

# 可选：承载训练超参的节点必须设为 true，否则训练配置卡发现不到。
# is_trainer = true

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

`layer` 影响画布上节点标题栏的配色 tier，目前合法值为 `A | B | C | D | IL | TOOLS`
（内置 `note` 节点用的就是 `TOOLS`）。

## __init__.py 模板

```python
from .node import MyNode

NODE_CLASSES = [MyNode]   # 一个文件夹可注册多个节点
```

## node.py 模板

```python
from __future__ import annotations

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

    # 通常无需重写 execute。``BaseNode.execute`` 默认 passthrough：
    # 为每个声明的 output port 发出 ``{node_id, schema_id, **params, _inputs}``
    # 载荷，适用于 25 个 Layer A/B/C/D 配置型节点。
    # 仅 trainer（is_trainer=True）或资产解析节点（如 robot / base_asset）
    # 才需要 override。
```

## 参数规约 / ParamSpec 进阶

`ParamSpec.type` 取值：`"string" | "int" | "float" | "bool" | "enum" | "json"`。
`type="enum"` 时必须给 `choices=[...]`。

`ParamSpec.widget`（可选）—— UI 行类型 hint，缺省由 `type` 推断；启动期由
`registers.nodes._validate_params` 调 `param_rows.validate_param_spec` 校验，
不识别的 widget 会 `log_warning`（非致命，单条目跳过）。可选值：

- `"path"`               文件/目录选择
- `"code"`               多行代码框（配合 `meta={"language": "python"|"text"|...}`）
- `"range"`              滑条（配合 `meta={"min": ..., "max": ..., "step": ...}`）
- `"index"`              索引选择
- `"badge"`              徽章/标签型
- `"table_readonly"`     只读表格

`ParamSpec.meta` —— 行特定附加参数（具体 key 由各 ParamRow 子类定义）。

## 端口规约 / PortSpec 进阶

`PortSpec.type` 取值：`"flow" / "bool" / "int" / "float" / "string" / "dict" /
"list" / "any" / 自定义协议名（如 "protocol.cmd_vel"）`。

`PortSpec.multi=True` 允许多连接；`PortSpec.optional=True` 允许悬空。

`PortSpec.meta.conditional_on` —— UI 据 host params 决定端口显隐：

```toml
[[inputs]]
name = "advanced_in"
type = "any"
optional = true
meta = { conditional_on = { key = "mode", op = "==", value = "advanced" } }
```

`op` 取值：`"==" | "!=" | "in" | "not in"`。

## i18n / 多语言

- `display_name_key` / `description_key` 走 `tr()`，缺省由 id 兜底；
- 每个参数的 label key 约定为 **`node.<id>.param.<key>`**（去掉 `node.` 前缀
  写到 `localisation/{EN,ZH}/node.txt`）。例如 `id="example.print_hello"` 的
  `message` 参数：

  ```ini
  [example]
  print_hello.display = Print Hello
  print_hello.desc    = Print a message to the log
  print_hello.param.message = Message
  ```

## 冲突与校验

- **id 撞车（与内置）**：三方节点拒绝注册，UnitPort 启动日志会出现 `log_warning`
- **id 撞车（三方之间）**：先扫到的赢，后者拒绝 + warning
- **manifest 不识别 / kind 非法 / 必填字段缺失 / layer 不在合法表内**：单个节点拒绝，不影响其他节点加载
- **NODE_CLASSES 中类不是 BaseNode 子类**：拒绝 + warning
- **ParamSpec 的 widget/type/choices/meta 组合不合法**：`log_warning`（非致命）

## 不做什么 / Anti-patterns

- ❌ 在 `manifest.toml` 中塞可执行 Python 代码（manifest 只能是声明式数据）
- ❌ 跨节点目录 `import`（每个节点应自包含；公共逻辑应放进 `requirements.txt` 引用的 PyPI 包）
- ❌ 在节点 `node.py` 顶层执行重计算或 I/O（importlib 加载所有节点；顶层副作用会拖慢启动）
- ❌ 期望运行时热加载（设计上**不**支持——增删节点都需要重启）
- ❌ 不必要地 override `execute`（默认 passthrough 已覆盖配置型节点；只有 trainer / 资产解析才需要重写）

## 安装

直接把 `<node_id>/` 文件夹复制（或 git clone）到本目录下，重启 UnitPort 即可。
