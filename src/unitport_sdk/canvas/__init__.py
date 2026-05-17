"""unitport_sdk.canvas — 画布 widget 库 / Canvas widget library.

为 Studio 画布节点提供统一的视觉 paint helper（端口圆点、range slider、
choice dropdown、edit button、polarity badge）+ 节点作者糖（NodeBuilder
metaclass）+ 端口类型颜色字典。

设计原则：
- 全部以 **纯 paint 函数 + 小工具类** 形式提供——绝不导出 QWidget，更不
  套 ``QGraphicsProxyWidget``（RELEASE 项目硬规）。
- 调用方负责 mouse 事件、值状态、LOD 决策；本模块只画。
- 颜色 **必须从 ``system.ini[Theme]`` 取**（``Config.get_color(slot)``）。
  禁止在本目录下任何文件中定义 ``_DEFAULT_THEME`` / inline hex 字典 / 直接
  ``QColor("#xxxxxx")`` —— 视觉新增槽位先加进 ``system.ini[Theme]`` 再用。
  仅 ``port_palette.py``（端口类型→色映射，逻辑数据非主题）保留独立色表。

公开 API：
    get_port_color, PORT_TYPE_COLORS              — 端口类型 → QColor 映射
    draw_dot_port                                 — 端口圆点
    draw_range_slider, RangeSliderHitTest         — 单/双手柄 range slider
    draw_choice_dropdown                          — 下拉选择框（带 ▾ 箭头）
    draw_edit_button                              — pencil/code/browse 按钮
    draw_polarity_badge                           — circle/diamond/square 极性徽章
    make_port_visual                              — PortSpec → 视觉描述 dict
    NodeBuilder                                   — 节点 metaclass，INPUT_TYPES 风格
"""

from __future__ import annotations

from .badge import draw_polarity_badge
from .badges import Node_ClipRefIndicator, Node_ModulePolarityBadge
from .choice_box import draw_choice_dropdown
from .composites import (
    Node_AMPResultsPanel,
    Node_BufferSizePickerWidget,
    Node_FPSSliderWidget,
    Node_MotionFilePickerWidget,
    Node_PreviewButtonWidget,
    Node_RegistryModuleEditor,
    Node_SaveListButton,
    Node_TrainingItemEditor,
    Node_TrainingItemRow,
)
from .dialogs import (
    Node_StageOverrideDialog,
    Node_StageSwitchCriteriaDialog,
    Node_StageSwitchEditorWidget,
)
from .edit_button import draw_edit_button
from .inputs import (
    Node_CodeEdit,
    Node_IndexButton,
    Node_ModuleValueButton,
    Node_NodeInputer,
)
from .node_builder import NodeBuilder
from .popups import (
    Node_AnchoredPopup,
    Node_CodeEditorPopup,
    Node_DataInput,
    Node_EditorPopupHost,
)
from .pickers import (
    Node_DropdownChoicePicker,
    Node_ExportBundlePicker,
    Node_MainPicker,
    Node_MotionLibraryPicker,
    Node_ObsComponentsPicker,
    Node_RichChoiceCard,
    Node_RichChoicePicker,
    Node_RichChoicePopup,
    Node_SceneTypePicker,
    Node_StartPointChoicePicker,
    Node_TaskTypePicker,
)
from .port_dot import draw_dot_port
from .port_item import PORT_HIT_R, PORT_R, Node_TrainingNodePort
from .port_palette import PORT_TYPE_COLORS, default_color_hex, get_port_color, known_types
from .port_visual import make_port_visual
from .range_slider import RangeSliderHitTest, draw_range_slider
from .rows import (
    Node_BehaviorSetting,
    Node_ComboSetting,
    Node_ConditionSet,
    Node_InputSetting,
    Node_NodeRow,
    Node_NodeTableRowsWidget,
    Node_OutputSet,
    Node_PortInputRow,
    Node_RegistryModuleRow,
    Node_ScriptInAndOut,
    Node_ScriptSetting,
)
from .sliders import (
    Node_CurvePreviewArea,
    Node_DualHandleRangeSlider,
    Node_ModuleCurveSlider,
    Node_NodeSlider,
    Node_RangeHandleSlider,
)
from .widgets import (
    LaviCodeEditor,
    LaviDoubleSpinBox,
    LaviLineEdit,
    LaviSpinBox,
    setCodeEditor,
    setDoubleSpinBox,
    setLineEdit,
    setSpinBox,
)


__all__ = [
    # palette
    "PORT_TYPE_COLORS",
    "default_color_hex",
    "get_port_color",
    "known_types",
    # paint helpers
    "draw_dot_port",
    "draw_range_slider",
    "RangeSliderHitTest",
    "draw_choice_dropdown",
    "draw_edit_button",
    "draw_polarity_badge",
    # factories
    "make_port_visual",
    "NodeBuilder",
    # themed input widgets
    "LaviLineEdit",
    "LaviSpinBox",
    "LaviDoubleSpinBox",
    "LaviCodeEditor",
    "setLineEdit",
    "setSpinBox",
    "setDoubleSpinBox",
    "setCodeEditor",
    # geometry constants
    "PORT_R",
    "PORT_HIT_R",
    # Node_* — pickers
    "Node_RichChoiceCard",
    "Node_RichChoicePopup",
    "Node_RichChoicePicker",
    "Node_TaskTypePicker",
    "Node_ObsComponentsPicker",
    "Node_DropdownChoicePicker",
    "Node_SceneTypePicker",
    "Node_ExportBundlePicker",
    "Node_StartPointChoicePicker",
    "Node_MotionLibraryPicker",
    "Node_MainPicker",
    # Node_* — sliders
    "Node_RangeHandleSlider",
    "Node_DualHandleRangeSlider",
    "Node_NodeSlider",
    "Node_ModuleCurveSlider",
    "Node_CurvePreviewArea",
    # Node_* — inputs
    "Node_IndexButton",
    "Node_NodeInputer",
    "Node_CodeEdit",
    "Node_ModuleValueButton",
    # Node_* — popups（frameless 就地编辑）
    "Node_AnchoredPopup",
    "Node_DataInput",
    "Node_CodeEditorPopup",
    "Node_EditorPopupHost",
    # Node_* — rows
    "Node_NodeRow",
    "Node_PortInputRow",
    "Node_ConditionSet",
    "Node_OutputSet",
    "Node_InputSetting",
    "Node_ComboSetting",
    "Node_BehaviorSetting",
    "Node_ScriptSetting",
    "Node_ScriptInAndOut",
    "Node_NodeTableRowsWidget",
    "Node_RegistryModuleRow",
    # Node_* — badges
    "Node_ModulePolarityBadge",
    "Node_ClipRefIndicator",
    # Node_* — port item
    "Node_TrainingNodePort",
    # Node_* — composites
    "Node_SaveListButton",
    "Node_BufferSizePickerWidget",
    "Node_MotionFilePickerWidget",
    "Node_FPSSliderWidget",
    "Node_PreviewButtonWidget",
    "Node_AMPResultsPanel",
    "Node_TrainingItemRow",
    "Node_TrainingItemEditor",
    "Node_RegistryModuleEditor",
    # Node_* — dialogs
    "Node_StageOverrideDialog",
    "Node_StageSwitchCriteriaDialog",
    "Node_StageSwitchEditorWidget",
]
