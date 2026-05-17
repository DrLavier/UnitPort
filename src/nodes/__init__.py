"""src.nodes — 出厂第一方节点插件树 / First-party node plugin tree.

每个子目录 ``<node_id>/`` 是一个独立节点包，契约：
- ``__init__.py``       必需 — 暴露 ``NODE_CLASSES = [SomeNodeClass, ...]``
- ``manifest.toml``     必需 — 静态元数据（schema/id/kind/category/version/inputs/outputs/parameters）
- ``node.py``           必需 — ``BaseNode`` 子类
- ``icon.svg``          可选 — 调色板图标
- ``README.md``         可选 — 文档/tooltip
- ``assets/``           可选 — 节点自带资源

Discovery 在 ``registers.nodes.load()`` 完成。冲突规则：
- 内置节点（本目录）id 撞车 → 后扫到的内置拒绝注册并 ``log_error``
- 三方节点（``custom_mods/nodes/``）与内置撞车 → 三方拒绝并 ``log_warning``

新增节点：直接在本目录新建 ``<id>/`` 子文件夹 + 三件套即可，重启后自动可用。
"""
