# Celebrimbor 项目结构说明

## 重构概览

本项目已从单文件 `celebrimbor.py` (2138行) 重构为模块化结构，便于维护和扩展。

## 核心改动

### 1. 目录结构调整

- **原路径**: `./unitree_mujoco` 和 `./unitree_sdk2_python`
- **新路径**: `./models/unitree/` 下
- **原因**: 支持多品牌机器人的热插拔扩展

### 2. 配置文件系统

- `config/system.ini`: 系统级配置（路径、仿真参数等）
- `config/user.ini`: 用户偏好配置

### 3. 模块拆分

| 原文件 | 新位置 | 说明 |
|--------|--------|------|
| MujocoSimulationThread | bin/core/simulation_thread.py | 仿真线程基类 |
| UnitreeModel相关 | models/unitree/unitree_model.py | Unitree控制逻辑 |
| GraphScene, GraphView | bin/components/graph_view.py | 图形编辑组件（待提取）|
| ModuleCard, ModulePalette | bin/components/module_cards.py | 模块卡片（待提取）|
| CodeEditor | bin/components/code_editor.py | 代码编辑器 |
| MainWindow | bin/ui.py | 主窗口 |

## 文件映射表

### 核心功能 (原 celebrimbor.py → 新结构)

```
celebrimbor.py (1-73行: 导入部分)
  → models/unitree/unitree_model.py (导入和初始化)
  
celebrimbor.py (96-441行: MujocoSimulationThread)
  → bin/core/simulation_thread.py (基类)
  → models/unitree/unitree_model.py (Unitree专用实现)
  
celebrimbor.py (443-1700行: GraphScene, GraphView等)
  → bin/components/graph_view.py (待创建，需要手动提取)
  
celebrimbor.py (1701-2020行: ModuleCard, ModulePalette)
  → bin/components/module_cards.py (待创建，需要手动提取)
  
celebrimbor.py (2043-2130行: MainWindow)
  → bin/ui.py (已简化实现)
```

## 扩展指南

### 添加新机器人品牌

1. 在 `models/` 下创建新目录，如 `models/boston_dynamics/`
2. 创建模型类继承 `BaseRobotModel`
3. 在 `models/__init__.py` 中注册
4. 在 `config/system.ini` 中添加路径配置

示例：
```python
# models/boston_dynamics/spot_model.py
from models.base import BaseRobotModel

class SpotModel(BaseRobotModel):
    def __init__(self, robot_type):
        super().__init__(robot_type)
        # 实现Spot专用逻辑
```

```python
# models/__init__.py
from .boston_dynamics import SpotModel
register_model("spot", SpotModel)
```

### 添加新功能节点

1. 在 `nodes/` 下创建节点类
2. 继承 `BaseNode`
3. 在 `nodes/node_registry.py` 中注册

## 待完成工作

### 高优先级
1. **提取图形编辑器组件**
   - 从原 `celebrimbor.py` 提取 `GraphScene` (443-1700行)
   - 提取 `GraphView`
   - 创建 `bin/components/graph_view.py`

2. **提取模块面板组件**
   - 提取 `ModuleCard` (1701-1841行)
   - 提取 `ModulePalette` (1842-2020行)
   - 创建 `bin/components/module_cards.py`

3. **完善动作库**
   - 实现 `sit` 动作
   - 实现 `walk` 动作
   - 添加更多复杂动作

### 中优先级
4. **节点执行引擎**
   - 完善 `NodeExecutor` 的代码生成功能
   - 实现节点图的序列化/反序列化

5. **UI增强**
   - 添加项目文件管理
   - 实现代码导出功能
   - 添加设置界面

### 低优先级
6. **文档和测试**
   - 编写完整的API文档
   - 添加更多单元测试
   - 添加示例项目

## 迁移步骤

如果你有原 `celebrimbor.py` 的完整代码：

1. **备份原文件**
2. **提取UI组件**:
   ```python
   # 从原文件复制 GraphScene, GraphView, ModuleCard, ModulePalette
   # 调整导入路径
   # 保存到 bin/components/
   ```
3. **更新主窗口**:
   - 用提取的组件替换 `bin/ui.py` 中的占位符
4. **测试功能**:
   ```bash
   python main.py
   python tests/test_unitree.py
   ```

## 配置路径检查

在首次运行前，请检查 `config/system.ini`:

```ini
[PATH]
unitree_sdk = ./models/unitree/unitree_sdk2_python
unitree_mujoco = ./models/unitree/unitree_mujoco
unitree_robots = ./models/unitree/unitree_mujoco/unitree_robots
```

确保这些路径下有对应的文件：
- `unitree_sdk2_python/` 目录存在
- `unitree_mujoco/` 目录存在
- `unitree_mujoco/unitree_robots/go2/scene.xml` 等模型文件存在

## 运行测试

```bash
# 测试Unitree模型
python tests/test_unitree.py

# 运行主程序
python main.py
```

## 常见问题

**Q: 为什么图形编辑器显示占位符？**
A: 当前版本是重构框架，完整的图形编辑器代码需要从原 `celebrimbor.py` 手动提取。

**Q: 如何快速迁移原代码？**
A: 参考上面的"文件映射表"，逐个组件复制并调整导入路径。

**Q: 模型加载失败怎么办？**
A: 检查 `config/system.ini` 中的路径配置，确保Unitree SDK和MuJoCo文件在正确位置。

## 技术栈

- **GUI**: PySide6
- **仿真**: MuJoCo 3.0+
- **机器人**: Unitree SDK 2
- **配置**: ConfigParser (Python标准库)
- **日志**: logging (Python标准库)
