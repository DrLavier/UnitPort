# Celebrimbor - 机器人可视化编程平台

基于PySide6的可视化机器人控制系统，支持图形化编程和MuJoCo仿真。

## 项目结构

```
celebrimbor/
├── main.py                    # 主入口文件
├── config/                    # 配置文件
│   ├── system.ini            # 系统配置
│   └── user.ini              # 用户配置
├── bin/                       # UI和核心组件
│   ├── ui.py                 # 主窗口
│   ├── components/           # UI组件
│   │   ├── code_editor.py   # 代码编辑器
│   │   ├── graph_view.py    # 图形编辑器（待实现）
│   │   └── module_cards.py  # 模块卡片（待实现）
│   ├── core/                 # 核心功能
│   │   ├── config_manager.py # 配置管理
│   │   ├── data_manager.py   # 数据读写（线程安全）
│   │   ├── theme_manager.py  # 主题管理（颜色/字体）
│   │   ├── logger.py         # 日志系统
│   │   ├── simulation_thread.py # 仿真线程
│   │   └── node_executor.py  # 节点执行引擎
│   └── assets/               # 资源文件
├── models/                    # 机器人模型
│   ├── base.py               # 模型基类
│   └── unitree/              # Unitree机器人
│       ├── unitree_model.py # Unitree控制逻辑
│       ├── unitree_mujoco/  # MuJoCo仿真包
│       └── unitree_sdk2_python/ # SDK包
├── nodes/                     # 功能节点定义
│   ├── base_node.py          # 节点基类
│   ├── action_nodes.py       # 动作节点
│   ├── logic_nodes.py        # 逻辑节点
│   └── sensor_nodes.py       # 传感器节点
├── utils/                     # 工具模块
│   ├── logger.py             # 日志工具
│   └── path_helper.py        # 路径工具
└── requirements.txt           # 依赖列表
```

## 安装

### 1. 安装Python依赖

```bash
pip install -r requirements.txt
```

### 2. 安装Unitree SDK（可选）

将Unitree SDK放置到以下目录：
- `./models/unitree/unitree_sdk2_python/`
- `./models/unitree/unitree_mujoco/`

### 3. 配置路径

编辑 `config/system.ini` 文件，确保路径配置正确。

## 运行

```bash
python main.py
```

## 功能特性

### 已实现
- ✅ **配置管理系统** - 支持INI和JSON配置文件
- ✅ **线程安全的数据管理器** - DataManager支持并发读写
- ✅ **主题管理系统** - 支持Light/Dark主题切换，颜色和字体配置
- ✅ **日志系统** - 集成CmdLogWidget，支持实时日志显示和打字机效果
- ✅ **Unitree机器人模型** - MuJoCo仿真集成
- ✅ **仿真线程管理** - 独立线程运行仿真，不阻塞UI
- ✅ **基础UI框架** - 左侧日志窗口 + 右侧工作区
- ✅ **代码生成器** - 自动生成Python代码
- ✅ **节点系统框架** - 支持动作、逻辑、传感器节点

### 待实现（需要从原celebrimbor.py提取）
- ⏳ 完整的图形编辑器（GraphScene, GraphView）
- ⏳ 模块拖拽系统（ModulePalette, ModuleCard）
- ⏳ 节点连接和执行
- ⏳ 完整的动作库
- ⏳ 传感器数据可视化

## 扩展新机器人

### 1. 创建模型类

在 `models/your_robot/` 目录下创建 `your_robot_model.py`:

```python
from models.base import BaseRobotModel

class YourRobotModel(BaseRobotModel):
    def __init__(self, robot_type: str):
        super().__init__(robot_type)
        # 初始化代码
    
    def initialize(self) -> bool:
        # 实现初始化逻辑
        pass
    
    def load_model(self) -> bool:
        # 实现模型加载
        pass
    
    def run_action(self, action_name: str, **kwargs) -> bool:
        # 实现动作执行
        pass
    
    def get_available_actions(self) -> List[str]:
        # 返回可用动作列表
        pass
    
    def get_sensor_data(self) -> Dict[str, Any]:
        # 返回传感器数据
        pass
    
    def stop(self):
        # 实现停止逻辑
        pass
```

### 2. 注册模型

在 `models/__init__.py` 中注册:

```python
from .your_robot import YourRobotModel
register_model("your_robot", YourRobotModel)
```

### 3. 配置路径

在 `config/system.ini` 中添加路径配置。

## 配置说明

### system.ini

- `[PATH]`: 路径配置
- `[SIMULATION]`: 仿真参数
- `[MUJOCO]`: MuJoCo设置
- `[UI]`: 界面配置
- `[DEBUG]`: 调试选项

### user.ini

- `[PREFERENCES]`: 用户偏好设置（主题等）
- `[RECENT]`: 最近使用记录
- `[CUSTOM]`: 自定义配置

### ui.ini (新增)

- `[Font]`: 字体配置（family, size_mini, size_small等）
- `[Light]`: Light主题颜色配置
- `[Dark]`: Dark主题颜色配置

## 核心功能使用

### 数据管理器 (DataManager)

```python
from bin.core.data_manager import load_data, get_value, up_data

# 加载配置文件
load_data('config/system.ini')

# 读取配置值
robot = get_value('config/system.ini', 'SIMULATION', 'default_robot', 'go2')

# 更新配置
up_data('config/user.ini', section='PREFERENCES', key='theme', value='dark')
```

### 主题管理器

```python
from bin.core.theme_manager import get_color, get_font_size, set_theme

# 设置主题
set_theme('dark')

# 获取颜色
bg_color = get_color('bg', '#1e1e1e')
text_color = get_color('text_primary', '#ffffff')

# 获取字体大小
normal_size = get_font_size('size_normal', 12)
```

### 日志系统

```python
from bin.core.logger import log_info, log_success, log_error, log_debug

# 记录日志（会显示在左侧日志窗口）
log_info("程序启动")
log_success("操作成功")
log_warning("警告信息")
log_error("错误信息")

# 打字机效果
log_info("正在加载...", typer=True)
```

## 开发指南

### 添加新节点

1. 在 `nodes/` 目录下创建节点类
2. 继承 `BaseNode` 基类
3. 实现 `execute()` 方法
4. 在 `nodes/node_registry.py` 中注册

### 日志

```python
from utils.logger import get_logger
logger = get_logger()

logger.info("信息")
logger.warning("警告")
logger.error("错误")
```

## 故障排除

### MuJoCo无法启动

- 检查环境变量 `MUJOCO_GL` 设置
- 确认MuJoCo版本 >= 3.0.0
- Ubuntu用户可能需要安装 `libglfw3` 和 `libglew-dev`

### Unitree SDK导入失败

- 确认SDK路径配置正确
- 检查Python路径是否包含SDK目录

### 图形编辑器功能缺失

当前版本为简化版本，完整的图形编辑器功能需要：
1. 从原 `celebrimbor.py` 提取 `GraphScene` 类
2. 提取 `GraphView` 类
3. 提取 `ModulePalette` 和 `ModuleCard` 类
4. 在 `bin/components/` 目录下创建对应文件

## 许可证

待定

## 贡献

欢迎提交Issue和Pull Request

## 联系方式

待定
