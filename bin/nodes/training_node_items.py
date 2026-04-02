#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training_node_items.py — Custom node cards for the Training Ground canvas.

Provides:
  TrainingNodePort  (QGraphicsEllipseItem) — type-coloured port dot
  TrainingNodeItem  (QGraphicsRectItem)    — complete Training Ground node card
                                             with interactive param widgets

Both classes follow the GraphScene data() port/node protocol so existing
wiring infrastructure (_get_node_port, _create_connection) works out of the
box.

Port data() protocol:
  data(0)  = "port"
  data(1)  = "in" | "out"
  data(2)  = connections list  (mutable, managed by GraphScene)
  data(3)  = slot_name  (str)
  data(20) = {"channel":"data", "data_type":str, "dot_kind":"sub_dot",
               "max_connections":int}

Node data() protocol:
  data(10) = node_id      (str)
  data(11) = display_name (str)
  data(12) = node_type    (str)
  data(13) = logic_node   (BaseNode | None)
"""
from __future__ import annotations

import json
import pathlib
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

from PySide6.QtCore import QPoint, QPointF, QRectF, QSize, Qt, QThread, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QDoubleValidator,
    QFont,
    QFontMetrics,
    QIcon,
    QLinearGradient,
    QPainter,
    QPen,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsProxyWidget,
    QGraphicsRectItem,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSizePolicy,
    QStyle,
    QStyleOptionSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from src.system.core.theme_manager import get_color_for_theme
from src.system.service.checkpoint_registry import CheckpointRegistry
from src.system.training.robot_family import resolve_robot_family
from src.system.training.task_module_registry import (
    TaskModuleItem,
    reward_registry,
    termination_registry,
)


def get_color(color_key: str, fallback: str = "#FFFFFF") -> str:
    """Training Ground widgets intentionally stay on the dark palette for now."""
    return get_color_for_theme(color_key, "dark", fallback)


# ---------------------------------------------------------------------------
# Type registry — one unique colour per training data-flow type
# ---------------------------------------------------------------------------

def _training_port_types() -> Dict[str, Dict[str, str]]:
    return {
        "robot_spec":         {"color": get_color("training_port_robot_spec", "#FF8C42"), "label": "robot_spec"},
        "physics_config":     {"color": get_color("training_port_physics_config", "#4FC3F7"), "label": "physics_config"},
        "rewards":            {"color": get_color("training_port_rewards", "#38BDF8"), "label": "rewards"},
        "terminations":       {"color": get_color("training_port_terminations", "#F472B6"), "label": "terminations"},
        "task_config":        {"color": get_color("training_port_task_config", "#81C784"), "label": "task_config"},
        "obs_action_config":  {"color": get_color("training_port_obs_action_config", "#FFD54F"), "label": "obs_action_config"},
        "domain_rand_config": {"color": get_color("training_port_domain_rand_config", "#CE93D8"), "label": "domain_rand_config"},
        "env_config":         {"color": get_color("training_port_env_config", "#4DB6AC"), "label": "env_config"},
        "algo_config":        {"color": get_color("training_port_algo_config", "#F48FB1"), "label": "algo_config"},
        "eval_config":        {"color": get_color("training_port_eval_config", "#A5D6A7"), "label": "eval_config"},
        "train_result":       {"color": get_color("training_port_train_result", "#FF7043"), "label": "train_result"},
        "bundle_path":        {"color": get_color("training_port_bundle_path", "#90A4AE"), "label": "bundle_path"},
        "vis_check":          {"color": get_color("training_port_vis_check", "#80DEEA"), "label": "vis_check"},
        "scene_config":            {"color": get_color("training_port_scene_config", "#FFD180"), "label": "scene_config"},
        "base_asset":              {"color": get_color("training_port_base_asset",   "#CE93D8"), "label": "base_asset"},
        "reference_motion_config": {"color": get_color("training_port_reference_motion", "#FFB300"), "label": "reference_motion_config"},
        "init_pose_config":        {"color": get_color("training_port_init_pose", "#A5D6A7"),          "label": "init_pose_config"},
        "joint_config":            {"color": get_color("training_port_joint_config", "#BCAAA4"),        "label": "joint_config"},
        "int":                     {"color": get_color("training_port_int", "#FDD835"),                  "label": "int"},
    }

# ---------------------------------------------------------------------------
# Layer → visual style
# ---------------------------------------------------------------------------

_NODE_LAYER: Dict[str, str] = {
    "robot_mjcf":        "R",
    "physics_config":    "A",
    "rewards":           "A",
    "terminations":      "A",
    "task_config":       "A",
    "domain_rand":       "A",
    "obs_action_config": "A",
    "env_assembler":     "B",
    "algo_config":       "B",
    "train":             "C",
    "eval_config":       "D",
    "export":            "D",
    "vis_check":         "D",
    "scene_config":      "A",
    "base_asset":        "C",
    "reference_motion":  "A",
    "init_pose":         "A",
}

# Each layer has: bg (node body), title (title bar), text (title label)
def _layer_colors() -> Dict[str, Dict[str, str]]:
    return {
        "R": {
            "bg": get_color("training_layer_robot_bg", "#2A1B12"),
            "title": get_color("training_layer_robot_title", "#9A4F1F"),
            "text": get_color("training_layer_robot_text", "#FFD1B0"),
        },
        "A": {
            "bg": get_color("training_layer_a_bg", "#1B2E1B"),
            "title": get_color("training_layer_a_title", "#2D6A2D"),
            "text": get_color("training_layer_a_text", "#A5D6A7"),
        },
        "B": {
            "bg": get_color("training_layer_b_bg", "#1A1F2E"),
            "title": get_color("training_layer_b_title", "#1A3A6B"),
            "text": get_color("training_layer_b_text", "#90CAF9"),
        },
        "C": {
            "bg": get_color("training_layer_c_bg", "#2E1B1B"),
            "title": get_color("training_layer_c_title", "#6B1A1A"),
            "text": get_color("training_layer_c_text", "#FFCDD2"),
        },
        "D": {
            "bg": get_color("training_layer_d_bg", "#2A1B2E"),
            "title": get_color("training_layer_d_title", "#4A1B6B"),
            "text": get_color("training_layer_d_text", "#E1BEE7"),
        },
    }


def _fallback_style() -> Dict[str, str]:
    return {
        "bg": get_color("training_node_bg", get_color("card_bg", "#1e1e1e")),
        "title": get_color("training_node_title_bg", "#333333"),
        "text": get_color("training_node_title_text", get_color("text_primary", "#e5e7eb")),
    }


def _style_for_node(node_type: str) -> Dict[str, str]:
    layer = _NODE_LAYER.get(node_type, "A")
    return dict(_layer_colors().get(layer, _fallback_style()))

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

NODE_W: int = 268          # total node width (px)
TITLE_H: int = 30          # title bar height
PORT_ROW_H: int = 26       # height of an input/output port row
SEP_H: int = 8             # vertical separator between sections
H_PAD: int = 14            # horizontal padding
PORT_MARGIN: int = 10      # distance from node edge to port-dot centre
PORT_R: int = 6            # port dot radius

LABEL_COL: int = 116       # label column width (px) — wide enough for longest display_names
WIDGET_X: int = H_PAD + LABEL_COL   # = 130 — widget starts here

# ── Standardised widget dimensions (from ui.ini [TrainingNode]) ──────
def _read_training_node_ini() -> Tuple[int, int]:
    """Read NODE_STANDARD_W / NODE_STANDARD_H from ui.ini, with hardcoded fallback."""
    try:
        import configparser
        ini = configparser.ConfigParser()
        ini.read(str(pathlib.Path(__file__).parent.parent.parent / "src" / "config" / "ui.ini"), encoding="utf-8")
        w = ini.getint("TrainingNode", "NODE_STANDARD_W", fallback=134)
        h = ini.getint("TrainingNode", "NODE_STANDARD_H", fallback=24)
        return w, h
    except Exception:
        return 134, 24

NODE_STANDARD_W, NODE_STANDARD_H = _read_training_node_ini()

WIDGET_W: int = NODE_STANDARD_W     # functional widget max width (dropdowns, sliders, inputs)
PARAM_ROW_H: int = NODE_STANDARD_H  # default height of a parameter row

# ---------------------------------------------------------------------------
# Node UI rows — loaded once from NODE设计.json
# ---------------------------------------------------------------------------

_JSON_PATH = pathlib.Path(__file__).parent.parent.parent / "knowledge_base" / "NODE设计.json"

# Maps node_type key (e.g. "robot_mjcf") → list of ui_row dicts
NODE_UI_ROWS: Dict[str, List[dict]] = {}

_NODE_ID_TO_TYPE: Dict[str, str] = {
    "RobotMJCFNode":          "robot_mjcf",
    "PhysicsConfigNode":      "physics_config",
    "RewardsNode":            "rewards",
    "TerminationsNode":       "terminations",
    "TaskConfigNode":         "task_config",
    "DomainRandNode":         "domain_rand",
    "ObsActionConfigNode":    "obs_action_config",
    "EnvAssemblerNode":       "env_assembler",
    "AlgorithmConfigNode":    "algo_config",
    "TrainNode":              "train",
    "EvalConfigNode":         "eval_config",
    "ExportNode":             "export",
    "VisCheckNode":           "vis_check",
    "SceneConfigNode":        "scene_config",
    "BaseAssetNode":          "base_asset",
    "ReferenceMotionNode":      "reference_motion",
    "InitPoseNode":             "init_pose",
    "MultiGatedRewardNode":     "multigated_reward",
}

try:
    with _JSON_PATH.open("r", encoding="utf-8") as _f:
        _design_data = json.load(_f)
    for _node_def in _design_data.get("nodes", []):
        _nid = _node_def["id"]
        _ntype = _NODE_ID_TO_TYPE.get(_nid, _nid.lower())
        NODE_UI_ROWS[_ntype] = [r for r in _node_def.get("ui_rows", []) if r.get("kind") == "param"]
except Exception:
    pass  # graceful degradation — no widgets if JSON is missing

NODE_UI_ROWS.setdefault("train", []).append(
    {
        "kind": "param",
        "key": "__review__",
        "display_name": "",
        "widget": "action_button",
        "button_text": "Review",
        "button_action": "review_setup",
        "row_height": 24,
        "full_width_widget": True,
        "tooltip": "Open a MuJoCo review viewer using the current training setup.",
    }
)
NODE_UI_ROWS.setdefault("export", []).append(
    {
        "kind": "param",
        "key": "__review_export__",
        "display_name": "",
        "widget": "action_button",
        "button_text": "Review",
        "button_action": "review_export",
        "row_height": 24,
        "full_width_widget": True,
        "tooltip": "Run one real episode from the exported runtime bundle in MuJoCo.",
    }
)
NODE_UI_ROWS.setdefault("scene_config", []).append(
    {
        "kind": "param",
        "key": "__preview__",
        "display_name": "",
        "widget": "action_button",
        "button_text": "Preview",
        "button_action": "scene_preview",
        "row_height": 24,
        "full_width_widget": True,
        "tooltip": "Open a MuJoCo scene preview and drop the robot under the current gravity setting.",
    }
)

NODE_UI_ROWS.setdefault("rewards", []).append(
    {
        "kind": "param",
        "key": "__reward_preset__",
        "display_name": "Preset",
        "widget": "module_preset_picker",
        "registry": "rewards",
        "data_key": "reward_terms",
        "row_height": 24,
        "tooltip": "Save the current reward configuration as a named preset, or load a saved one.",
    }
)
NODE_UI_ROWS.setdefault("terminations", []).append(
    {
        "kind": "param",
        "key": "__termination_preset__",
        "display_name": "Preset",
        "widget": "module_preset_picker",
        "registry": "terminations",
        "data_key": "termination_conditions",
        "row_height": 24,
        "tooltip": "Save the current termination configuration as a named preset, or load a saved one.",
    }
)

NODE_UI_ROWS["reference_motion"] = [
    {
        "kind": "param",
        "key": "motion_source",
        "display_name": "Source",
        "widget": "dropdown",
        "choices": [
            "",
            "generate:standing",
            "generate:walk",
            "loco:UnitreeGo2:walk",
            "loco:UnitreeGo2:trot",
            "loco:UnitreeA1:walk",
        ],
        "default": "",
        "row_height": 24,
        "tooltip": "Select a motion source: procedural generator or loco-mujoco dataset. Leave empty to use a custom .npy file below.",
    },
    {
        "kind": "param",
        "key": "motion_file",
        "display_name": "Load Motion",
        "widget": "motion_library_picker",
        "row_height": 24,
        "tooltip": "Pick a registered reference motion, or <Add> to import a new .npy file. Overridden when Source is set.",
    },
    {
        "kind": "param",
        "key": "phase_mode",
        "display_name": "Phase Mode",
        "widget": "dropdown",          # handled specially in _make_param_widget
        "choices": ["loop", "once"],
        "default": "loop",
        "row_height": 24,
    },
    {
        "kind": "param",
        "key": "motion_fps",
        "display_name": "Motion FPS",
        "widget": "fps_slider",
        "row_height": 24,
        "tooltip": "Capture rate of the .npy file (Hz). 0 = 1 frame per control step (no resampling).",
    },
    {
        "kind": "param",
        "key": "random_start_phase",
        "display_name": "Random Start",
        "widget": "toggle",
        "default": "false",
        "row_height": 24,
        "tooltip": "Randomise starting frame each episode to improve generalisation.",
    },
    {
        "kind": "param",
        "key": "tracking_weight",
        "display_name": "Weight",
        "widget": "slider_float",
        "min": 0.0,
        "max": 10.0,
        "step": 0.1,
        "decimals": 1,
        "default": "1.0",
        "row_height": 24,
        "tooltip": "Reward scale for the reference tracking term (auto-enabled when connected).",
    },
    {
        "kind": "param",
        "key": "tracking_sigma",
        "display_name": "Sigma",
        "widget": "slider_float",
        "min": 0.1,
        "max": 50.0,
        "step": 0.5,
        "decimals": 1,
        "default": "5.0",
        "row_height": 24,
        "tooltip": "Sharpness of the Gaussian similarity: exp(-sigma * ||q_cur - q_ref||²).",
    },
    # ── Imitation Learning (BC) section ──────────────────────────────
    {
        "kind": "separator",
        "row_height": 6,
    },
    {
        "kind": "param",
        "key": "imitation_enabled",
        "display_name": "Imitation Learning",
        "widget": "toggle",
        "default": "true",
        "row_height": 24,
        "tooltip": (
            "Enable behavioral cloning (BC) pre-training before RL.\n"
            "Phase 1: supervised action matching on reference trajectories.\n"
            "Phase 2: BC + RL blended loss with decaying BC coefficient.\n"
            "Standard practice per DeepMimic / AMP / DAPG."
        ),
    },
    {
        "kind": "param",
        "key": "bc_epochs",
        "display_name": "BC Epochs",
        "widget": "slider_float",
        "min": 1,
        "max": 200,
        "step": 1,
        "decimals": 0,
        "default": "50",
        "row_height": 24,
        "condition": {"key": "imitation_enabled", "value": "true"},
        "tooltip": "Number of supervised BC training epochs (Phase 1). More epochs = better initial policy but longer pre-training.",
    },
    {
        "kind": "param",
        "key": "bc_learning_rate",
        "display_name": "BC LR",
        "widget": "text_input",
        "default": "1e-3",
        "row_height": 24,
        "condition": {"key": "imitation_enabled", "value": "true"},
        "tooltip": "Adam learning rate for BC supervised loss. Typical range: 1e-4 to 3e-3.",
    },
    {
        "kind": "param",
        "key": "bc_loss_type",
        "display_name": "BC Loss",
        "widget": "dropdown",
        "choices": ["mse", "huber"],
        "default": "mse",
        "row_height": 24,
        "condition": {"key": "imitation_enabled", "value": "true"},
        "tooltip": "Loss function for BC. MSE (L2) is standard; Huber (smooth-L1) is more robust to outliers.",
    },
    {
        "kind": "param",
        "key": "bc_blend_steps",
        "display_name": "Blend Steps",
        "widget": "text_input",
        "default": "200000",
        "row_height": 24,
        "condition": {"key": "imitation_enabled", "value": "true"},
        "tooltip": (
            "Phase 2 duration: number of RL timesteps with decaying BC auxiliary loss.\n"
            "The BC coefficient λ decays linearly from λ₀ to 0 over this many steps.\n"
            "Set to 0 to skip Phase 2 (jump directly from pure BC to pure RL)."
        ),
    },
    {
        "kind": "param",
        "key": "bc_blend_coef_start",
        "display_name": "Blend λ₀",
        "widget": "slider_float",
        "min": 0.0,
        "max": 2.0,
        "step": 0.05,
        "decimals": 2,
        "default": "0.5",
        "row_height": 24,
        "condition": {"key": "imitation_enabled", "value": "true"},
        "tooltip": "Starting BC coefficient for Phase 2. Actor loss = RL_loss + λ(t) · BC_loss.",
    },
    {
        "kind": "param",
        "key": "demo_num_trajectories",
        "display_name": "Demo Trajs",
        "widget": "slider_float",
        "min": 5,
        "max": 200,
        "step": 5,
        "decimals": 0,
        "default": "50",
        "row_height": 24,
        "condition": {"key": "imitation_enabled", "value": "true"},
        "tooltip": "Number of reference trajectories to collect for BC dataset. More = better coverage but slower collection.",
    },
    {
        "kind": "param",
        "key": "demo_noise_std",
        "display_name": "Demo Noise",
        "widget": "slider_float",
        "min": 0.0,
        "max": 0.1,
        "step": 0.005,
        "decimals": 3,
        "default": "0.0",
        "row_height": 24,
        "condition": {"key": "imitation_enabled", "value": "true"},
        "tooltip": "Gaussian noise σ added to reference action during demo collection (DAgger-like). 0 = deterministic replay.",
    },
    {
        "kind": "param",
        "key": "auto_inject_ref_obs",
        "display_name": "Auto Ref Obs",
        "widget": "toggle",
        "default": "true",
        "row_height": 24,
        "condition": {"key": "imitation_enabled", "value": "true"},
        "tooltip": (
            "Automatically inject reference_joint_positions, reference_joint_velocities, "
            "and phase_sin_cos into the observation space. Disable only if you want to "
            "manually configure obs components."
        ),
    },
    # ── Preview button (always last) ─────────────────────────────────
    {
        "kind": "param",
        "key": "preview",
        "display_name": "",
        "widget": "preview_button",
        "row_height": 24,
        "full_width_widget": True,
    },
]


NODE_UI_ROWS["init_pose"] = [
    {
        "kind": "param",
        "key": "mode",
        "display_name": "Mode",
        "widget": "dropdown",
        "choices": ["default", "reference_frame_0", "keyframe", "custom"],
        "default": "default",
        "row_height": 24,
        "tooltip": (
            "default — GO2 standing pose + noise\n"
            "reference_frame_0 — first frame of connected Reference Motion\n"
            "keyframe — named MJCF keyframe\n"
            "custom — explicit 12-joint array"
        ),
    },
    {
        "kind": "param",
        "key": "noise_scale",
        "display_name": "Noise Scale",
        "widget": "slider_float",
        "min": 0.0,
        "max": 0.3,
        "step": 0.005,
        "decimals": 3,
        "default": "0.05",
        "row_height": 24,
        "tooltip": "Per-joint Gaussian perturbation added at reset (rad). 0 = deterministic.",
    },
    {
        "kind": "param",
        "key": "base_height",
        "display_name": "Base Height",
        "widget": "slider_float",
        "min": -1.0,
        "max": 1.2,
        "step": 0.01,
        "decimals": 2,
        "default": "-1.0",
        "row_height": 24,
        "tooltip": "Robot Z at reset (m). -1 = auto (0.32 m for GO2 default/custom; from keyframe/ref otherwise).",
    },
    {
        "kind": "param",
        "key": "keyframe_name",
        "display_name": "Keyframe",
        "widget": "text",
        "default": "home",
        "row_height": 24,
        "tooltip": "MJCF keyframe name to use when mode = 'keyframe'.",
        "condition": {"key": "mode", "value": "keyframe"},
    },
    {
        "kind": "param",
        "key": "custom_qpos",
        "display_name": "Custom qpos",
        "widget": "json_editor",
        "default": "[]",
        "row_height": 24,
        "tooltip": "JSON array of 12 joint angles (rad) used when mode = 'custom'.",
        "condition": {"key": "mode", "value": "custom"},
    },
    {
        "kind": "param",
        "key": "__init_pose_preview__",
        "display_name": "",
        "widget": "action_button",
        "button_text": "Preview Pose",
        "button_action": "init_pose_preview",
        "row_height": 24,
        "full_width_widget": True,
        "tooltip": "Open MuJoCo viewer showing the robot frozen in this init pose configuration.",
    },
]


NODE_UI_ROWS["multigated_reward"] = [
    # Gate logic only — stage reward terms come from connected RewardsNodes (stage_0 / stage_1 ports)
    {
        "kind": "param",
        "key": "min_ep_window",
        "display_name": "Min Episodes",
        "widget": "int_spinbox",
        "min": 1,
        "max": 10000,
        "step": 5,
        "default": "10",
        "row_height": 24,
        "tooltip": (
            "Minimum number of episodes to complete in the current stage before\n"
            "the gate may open.  Acts as a readiness guard regardless of step count."
        ),
    },
    {
        "kind": "param",
        "key": "max_step_stage0",
        "display_name": "Hard Timeout",
        "widget": "int_spinbox",
        "min": 0,
        "max": 10000000,
        "step": 10000,
        "default": "0",
        "row_height": 24,
        "tooltip": (
            "Force the gate open at this global step even if no other condition is met.\n"
            "0 = disabled (gate opens only via performance)."
        ),
    },
    {
        "kind": "param",
        "key": "reward_threshold_ratio",
        "display_name": "Stability Ratio",
        "widget": "slider_float",
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
        "decimals": 2,
        "default": "0.75",
        "row_height": 24,
        "tooltip": (
            "Stability guard: rolling episode mean >= best_ever × this ratio.\n"
            "0.75 = performance must stay within 25% of the all-time best.\n"
            "Set to 0 to disable (advance purely on plateau detection)."
        ),
    },
    {
        "kind": "param",
        "key": "plateau_window",
        "display_name": "Plateau Window",
        "widget": "int_spinbox",
        "min": 2,
        "max": 500,
        "step": 5,
        "default": "10",
        "row_height": 24,
        "tooltip": (
            "Number of recent stage-episodes used to compute the reward slope\n"
            "for plateau detection.  Should be <= Min Episodes."
        ),
    },
    {
        "kind": "param",
        "key": "plateau_eps",
        "display_name": "Plateau ε",
        "widget": "slider_float",
        "min": 0.0,
        "max": 0.1,
        "step": 0.001,
        "decimals": 4,
        "default": "0.005",
        "row_height": 24,
        "tooltip": (
            "Normalised slope threshold for plateau detection.\n"
            "|slope| / |mean_reward| < ε  →  reward has converged → advance stage.\n"
            "Smaller = require a flatter curve before declaring convergence."
        ),
    },
    {
        "kind": "param",
        "key": "blend_steps",
        "display_name": "Blend Steps",
        "widget": "int_spinbox",
        "min": 0,
        "max": 100000,
        "step": 500,
        "default": "3000",
        "row_height": 24,
        "tooltip": "Env steps over which reward weights are linearly interpolated at gate opening.",
    },
    {
        "kind": "param",
        "key": "stage_behavior",
        "display_name": "Stage Behavior",
        "widget": "dropdown",
        "choices": ["replace", "accumulate"],
        "default": "replace",
        "row_height": 24,
        "tooltip": (
            "replace — Stage 1 weights replace the base reward weights.\n"
            "accumulate — Stage 1 weights are stacked on top of Stage 0 weights."
        ),
    },
    {
        "kind": "param",
        "key": "ep_reward_window",
        "display_name": "Ep Window",
        "widget": "int_spinbox",
        "min": 1,
        "max": 200,
        "step": 5,
        "default": "20",
        "row_height": 24,
        "tooltip": "Number of recent episodes used for rolling mean reward (stability guard).",
    },
    # ── Total-steps AUTO calculation ──────────────────────────────────────────
    {
        "kind": "separator",
        "row_height": 6,
    },
    {
        "kind": "param",
        "key": "stage1_ratio",
        "display_name": "S1 Budget ×",
        "widget": "slider_float",
        "min": 0.1,
        "max": 10.0,
        "step": 0.1,
        "decimals": 1,
        "default": "1.5",
        "row_height": 24,
        "tooltip": (
            "Stage 1 budget multiplier for AUTO total-steps calculation.\n"
            "Recommended total = Hard Timeout × (1 + this value).\n"
            "e.g. Hard Timeout = 200 k, ratio = 1.5 → recommended total = 500 k.\n"
            "Output via the 'total_steps' port → connect to Algorithm Config."
        ),
    },
]


def _override_export_bundle_name_row() -> None:
    rows = NODE_UI_ROWS.setdefault("export", [])
    for row in rows:
        if row.get("key") != "bundle_name":
            continue
        row["display_name"] = "Checkpoint"
        row["widget"] = "export_bundle_picker"
        row["default"] = ""
        row["placeholder"] = ""
        row["tooltip"] = (
            "Choose an existing checkpoint to update, or select <NEW> to register "
            "a new export name before writing the bundle."
        )
        row.pop("button_text", None)
        row.pop("button_width", None)
        row.pop("button_action", None)
        row.pop("button_tooltip", None)
        break


_override_export_bundle_name_row()

NODE_UI_ROWS["base_asset"] = [
    {
        "kind": "param",
        "key": "start_point",
        "display_name": "Start Point",
        "widget": "start_point_picker",
        "default": "__new__",
        "tooltip": "Choose whether training starts new, from the latest export artifact, or from a registered training asset.",
    },
    {
        "kind": "param",
        "key": "load_mode",
        "display_name": "Load Mode",
        "widget": "dropdown",
        "choices": ["scratch", "resume_sb3", "warm_start_actor"],
        "default": "scratch",
        "tooltip": "scratch: train from random init.\nresume_sb3: restore weights + optimizer state.\nwarm_start_actor: copy actor weights only, reset optimizer.",
        "condition": {
            "key": "start_point",
            "op": "!=",
            "value": "__new__",
        },
    },
]

# ---------------------------------------------------------------------------
# Dark widget stylesheet
# ---------------------------------------------------------------------------

def _widget_style() -> str:
    text = get_color("training_widget_text", get_color("text_primary", "#d1d5db"))
    input_bg = get_color("training_widget_input_bg", get_color("input_bg", "#1c1c1c"))
    border = get_color("training_widget_border", get_color("input_border", "#3a3a3a"))
    focus = get_color("training_widget_focus_border", get_color("connection_hover", "#4f7ecc"))
    popup_sel = get_color("training_widget_popup_selected_bg", get_color("input_popup_selected_bg", "#2d4a7a"))
    btn_bg = get_color("training_widget_button_bg", get_color("button_bg", "#252525"))
    btn_hover = get_color("training_widget_button_hover_bg", get_color("button_hover", "#2e2e2e"))
    btn_border = get_color("training_widget_button_border", border)
    btn_text = get_color("training_widget_button_text", get_color("text_secondary", "#9ca3af"))
    toggle_bg = get_color("training_widget_toggle_bg", "#14532d")
    toggle_text = get_color("training_widget_toggle_text", "#86efac")
    toggle_border = get_color("training_widget_toggle_border", "#22c55e")
    pressed_bg = get_color("training_widget_button_pressed_bg", "#1a3d20")
    slider_groove = get_color("training_widget_slider_groove", "#303030")
    slider_fill = get_color("training_widget_slider_fill", "#F6D393")
    slider_unfilled = get_color("training_widget_slider_unfilled", slider_groove)
    slider_handle = get_color("training_widget_slider_handle", "#5a6470")
    slider_handle_hover = get_color("training_widget_slider_handle_hover", "#7a8490")
    slider_handle_hover_border = get_color("training_widget_slider_handle_hover_border", "#22c55e")
    label_text = get_color("training_widget_label_text", get_color("text_secondary", "#9ca3af"))
    return f"""
QWidget {{ background: transparent; color: {text}; font-size: 11px; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {input_bg}; color: {text};
    border: 1px solid {border}; border-radius: 3px;
    padding: 0px 3px;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {focus};
}}
QComboBox::drop-down {{ width: 14px; border: none; }}
QComboBox QAbstractItemView {{
    background: {input_bg}; color: {text};
    border: 1px solid {border}; selection-background-color: {popup_sel};
}}
QPushButton {{
    background: {btn_bg}; color: {btn_text};
    border: 1px solid {btn_border}; border-radius: 3px;
    padding: 0px 5px; font-size: 10px;
}}
QPushButton:hover {{ background: {btn_hover}; border-color: {btn_border}; }}
QPushButton:checked {{ background: {toggle_bg}; color: {toggle_text}; border-color: {toggle_border}; }}
QPushButton:pressed {{ background: {pressed_bg}; }}
QSlider::groove:horizontal {{
    background: transparent; height: 4px; border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background: {slider_fill}; height: 4px; border-radius: 2px;
}}
QSlider::add-page:horizontal {{
    background: {slider_unfilled}; height: 4px; border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {slider_handle}; width: 12px; height: 12px;
    border-radius: 6px; margin: -4px 0;
}}
QSlider::handle:horizontal:hover {{
    background: {slider_handle_hover};
    width: 8px; height: 8px; border-radius: 6px; margin: -4px 0;
    border: 2px solid {slider_handle_hover_border};
}}
QLabel {{ background: transparent; color: {label_text}; font-size: 10px; }}
"""


# ---------------------------------------------------------------------------
# Task Type selector
# ---------------------------------------------------------------------------

_TASK_TYPE_META: Dict[str, Dict[str, str]] = {
    "velocity_tracking": {
        "title": "Velocity Tracking",
        "desc": "Learn stable walking while following target velocity commands.",
    },
    "stand_balance": {
        "title": "Stand Balance",
        "desc": "Hold posture and reject disturbances with minimal extra motion.",
    },
    "turn_in_place": {
        "title": "Turn In Place",
        "desc": "Rotate around the body center without drifting across the floor.",
    },
    "recovery": {
        "title": "Recovery",
        "desc": "Recover from unstable states and return to a safe stance.",
    },
    "waypoint_following": {
        "title": "Waypoint Following",
        "desc": "Track ordered targets with smooth and stable locomotion.",
    },
}

_OBS_COMPONENT_META: Dict[str, Dict[str, str]] = {
    "joint_pos": {"title": "Joint Position", "desc": "Current joint angles for all actuated joints."},
    "joint_vel": {"title": "Joint Velocity", "desc": "Angular velocity of each actuated joint."},
    "imu": {"title": "IMU", "desc": "Body orientation and inertial measurements from the torso frame."},
    "command": {"title": "Command", "desc": "Target motion command provided by the current task."},
    "previous_action": {"title": "Previous Action", "desc": "Last action sent to the policy for short-term temporal context."},
    "base_lin_vel": {"title": "Base Linear Velocity", "desc": "Robot base linear velocity in the body or world frame."},
    "base_ang_vel": {"title": "Base Angular Velocity", "desc": "Robot base angular velocity for balance and turning cues."},
    "gravity_vec": {"title": "Gravity Vector", "desc": "Projected gravity direction used for orientation awareness."},
}


def _default_choice_meta(choice: str) -> Dict[str, str]:
    title = str(choice or "").replace("_", " ").title()
    return {
        "title": title,
        "desc": f"Select {title.lower()} for this setting.",
    }


def _module_registry_meta(registry: Dict[str, TaskModuleItem]) -> Dict[str, Dict[str, str]]:
    return {
        key: {"title": item.title, "desc": item.desc}
        for key, item in registry.items()
    }


class _ModulePolarityBadge(QWidget):
    """Small shape badge describing reward polarity."""

    def __init__(self, polarity: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._polarity = str(polarity or "").strip().lower()
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(18, 18)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        tooltip_map = {
            "reward": "Reward",
            "bidirectional": "Bidirectional",
            "penalty": "Penalty",
        }
        self.setToolTip(tooltip_map.get(self._polarity, ""))

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect().adjusted(3, 3, -3, -3)
        color_map = {
            "reward": QColor("#22c55e"),
            "bidirectional": QColor("#3b82f6"),
            "penalty": QColor("#ef4444"),
        }
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(color_map.get(self._polarity, QColor("#9ca3af"))))

        if self._polarity == "reward":
            painter.drawEllipse(rect)
            return
        if self._polarity == "bidirectional":
            diamond = [
                QPoint(rect.center().x(), rect.top()),
                QPoint(rect.right(), rect.center().y()),
                QPoint(rect.center().x(), rect.bottom()),
                QPoint(rect.left(), rect.center().y()),
            ]
            painter.drawPolygon(diamond)
            return
        painter.drawRect(rect)


class DataInput(QFrame):
    """Reusable popup input with ✔/❌ buttons and float range validation.

    A compact inline popup that appears next to the triggering widget.
    Validates float input within [min_value, max_value]; shows red border
    on invalid input.  The ❌ button is followed by addStretch to keep
    the whole component left-aligned.

    Usage::

        popup = DataInput(current_value=1.0, min_value=0.0, max_value=10.0)
        popup.show_at(global_pos, row_height_px)
        # After popup closes, call popup.accepted_value()
    """

    def __init__(
        self,
        *,
        current_value: float,
        min_value: float,
        max_value: float,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(
            parent,
            Qt.WindowType.Popup
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.NoDropShadowWindowHint,
        )
        self.setObjectName("dataInputPopup")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._accepted_value = float(current_value)
        self._min_value = float(min_value)
        self._max_value = float(max_value)

        border = get_color("training_widget_border", "#3d3d3d")
        bg = get_color("training_widget_input_bg", "#1A1A1A")
        text = get_color("training_widget_text", "#cccccc")
        btn_bg = get_color("training_widget_button_bg", get_color("button_bg", "#252525"))
        hover = get_color("training_button_hover_bg", get_color("button_hover", "#2e2e2e"))

        self.setStyleSheet(
            f"""
            #dataInputPopup {{
                background: transparent;
                border: none;
            }}
            QLineEdit[dataInput="true"] {{
                background: {bg};
                color: {text};
                border: 1px solid {border};
                border-radius: 4px;
                padding: 0px 4px;
            }}
            QLineEdit[dataInput="true"][invalid="true"] {{
                border-color: #f87171;
            }}
            QPushButton[dataInputBtn="true"] {{
                background: {btn_bg};
                color: {text};
                border: 1px solid {border};
                border-radius: 4px;
                padding: 0px 2px;
            }}
            QPushButton[dataInputBtn="true"]:hover {{
                background: {hover};
            }}
            """
        )

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(1)

        self._input = QLineEdit(self)
        self._input.setProperty("dataInput", True)
        self._input.setText(f"{current_value:.6g}")
        self._input.selectAll()
        self._input.setValidator(QDoubleValidator(min_value, max_value, 6, self))
        self._input.setFixedWidth(96)
        row.addWidget(self._input, 0)

        self._ok_btn = QPushButton("✔", self)
        self._ok_btn.setProperty("dataInputBtn", True)
        self._ok_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._ok_btn.clicked.connect(self._accept_if_valid)
        row.addWidget(self._ok_btn, 0)

        self._cancel_btn = QPushButton("❌", self)
        self._cancel_btn.setProperty("dataInputBtn", True)
        self._cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._cancel_btn.clicked.connect(self.close)
        row.addWidget(self._cancel_btn, 0)

        row.addStretch(1)

        self._input.returnPressed.connect(self._accept_if_valid)

    def _accept_if_valid(self) -> None:
        raw = str(self._input.text() or "").strip()
        try:
            value = float(raw)
        except (TypeError, ValueError):
            self._mark_invalid(True)
            return
        if value < self._min_value or value > self._max_value:
            self._mark_invalid(True)
            return
        self._mark_invalid(False)
        self._accepted_value = value
        self.close()

    def accepted_value(self) -> float:
        return float(self._accepted_value)

    def show_at(self, pos: QPoint, row_height: int = 50) -> None:
        # row_height is in screen pixels (already zoom-scaled).
        # Scale all sub-widgets proportionally so the popup tracks the zoom.
        h = max(24, row_height)
        font_px = max(12, round(h * 0.45))
        font = QFont()
        font.setPixelSize(font_px)
        input_w = max(96, round(h * 3.2))
        self._input.setFixedSize(input_w, h)
        self._input.setFont(font)
        self._ok_btn.setFixedSize(h, h)
        self._ok_btn.setFont(font)
        self._cancel_btn.setFixedSize(h, h)
        self._cancel_btn.setFont(font)
        self.setFixedHeight(h)
        self.adjustSize()
        y_offset = max(0, (row_height - h) // 2)
        self.move(QPoint(pos.x(), pos.y() + y_offset))
        self.show()
        self.raise_()
        self._input.setFocus()
        self._input.selectAll()

    def _mark_invalid(self, invalid: bool) -> None:
        self._input.setProperty("invalid", bool(invalid))
        self.style().unpolish(self._input)
        self.style().polish(self._input)
        self._input.setFocus()
        self._input.selectAll()


class IndexButton(QPushButton):
    """Numeric display button that spawns a :class:`DataInput` popup on click.

    Shows the current value with configurable decimal places.  When the user
    clicks, a ``DataInput`` popup appears for direct value entry.  On
    acceptance the button updates its display and emits ``valueChanged``.

    Parameters
    ----------
    current_value:
        Initial numeric value.
    min_value, max_value:
        Range bounds passed to DataInput for validation.
    decimals:
        Display decimal places (default 3).
    """

    valueChanged = Signal(float)

    def __init__(
        self,
        *,
        current_value: float = 0.0,
        min_value: float = 0.0,
        max_value: float = 1.0,
        decimals: int = 3,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._value = float(current_value)
        self._min = float(min_value)
        self._max = float(max_value)
        self._decimals = int(decimals)
        self._popup: Optional[DataInput] = None

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumWidth(34)
        self.setFlat(True)
        self._refresh_text()
        self.clicked.connect(self._toggle_popup)

    def value(self) -> float:
        return self._value

    def set_value(self, v: float) -> None:
        self._value = float(v)
        self._refresh_text()

    def set_range(self, lo: float, hi: float) -> None:
        self._min = float(lo)
        self._max = float(hi)

    def _refresh_text(self) -> None:
        self.setText(f"{self._value:.{self._decimals}f}")

    # ── popup lifecycle ───────────────────────────────────────────────

    def _toggle_popup(self) -> None:
        if self._popup is not None:
            self._popup.close()
            self._popup = None
            return
        popup = DataInput(
            current_value=self._value,
            min_value=self._min,
            max_value=self._max,
        )
        self._popup = popup
        popup.destroyed.connect(self._clear_popup)
        popup.destroyed.connect(lambda *_a, p=popup: self._apply(p))
        pos, row_h = self._popup_anchor()
        popup.show_at(pos, row_h)

    def _apply(self, popup: DataInput) -> None:
        v = popup.accepted_value()
        if abs(v - self._value) < 1e-9:
            return
        self._value = v
        self._refresh_text()
        self.valueChanged.emit(v)

    def _clear_popup(self, *_a) -> None:
        self._popup = None

    def _popup_anchor(self) -> Tuple[QPoint, int]:
        """Return (global_pos, row_height_px) for popup positioning."""
        try:
            host = self.window()
            proxy = host.graphicsProxyWidget() if host is not None else None
            if proxy is not None:
                scene = proxy.scene()
                views = scene.views() if scene is not None else []
                if views:
                    view = views[0]
                    scale = view.transform().m11()
                    row_h = round(self.height() * scale)
                    tl = self.mapTo(host, QPoint(0, 0))
                    br = self.mapTo(host, QPoint(self.width(), 0))
                    scene_br = proxy.mapToScene(QPointF(br))
                    scene_tl = proxy.mapToScene(QPointF(tl))
                    anchor = QPointF(scene_br.x() + 2, scene_tl.y())
                    view_pt = view.mapFromScene(anchor)
                    return view.viewport().mapToGlobal(view_pt), row_h
        except Exception:
            pass
        return self.mapToGlobal(QPoint(self.width() + 2, 0)), self.height()


class _RangeHandleSlider(QWidget):
    """Dual-handle range slider with rectangular handles.

    Handles are small rounded rectangles: height = 2 * ``_HANDLE_R``,
    width = ``_HANDLE_R``, corner radius = 2 px.
    """

    rangeChanged = Signal(float, float)

    _HANDLE_R = 4
    _TRACK_H  = 4
    _MARGIN   = 7

    def __init__(
        self,
        minimum: float,
        maximum: float,
        lo: float,
        hi: float,
        step: float = 0.01,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._min  = float(minimum)
        self._max  = float(maximum)
        self._step = max(float(step), 1e-12)
        self._lo   = self._snap(max(self._min, min(lo, self._max)))
        self._hi   = self._snap(max(self._lo,  min(hi, self._max)))

        self._dragging: Optional[str] = None
        self._hovered:  Optional[str] = None

        self.setFixedHeight(22)
        self.setMinimumWidth(60)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)

    def _snap(self, v: float) -> float:
        return round(v / self._step) * self._step

    def _val_to_x(self, val: float) -> float:
        m = self._MARGIN
        span = self._max - self._min
        if span == 0:
            return float(m)
        frac = (val - self._min) / span
        return m + frac * (self.width() - 2 * m)

    def _x_to_val(self, x: float) -> float:
        m = self._MARGIN
        track_w = self.width() - 2 * m
        if track_w <= 0:
            return self._min
        frac = max(0.0, min(1.0, (x - m) / track_w))
        return self._snap(self._min + frac * (self._max - self._min))

    def _hit_handle(self, x: float) -> Optional[str]:
        lo_x = self._val_to_x(self._lo)
        hi_x = self._val_to_x(self._hi)
        r = self._HANDLE_R + 5
        d_lo = abs(x - lo_x)
        d_hi = abs(x - hi_x)
        in_lo = d_lo <= r
        in_hi = d_hi <= r
        if in_lo and in_hi:
            return "lo" if d_lo <= d_hi else "hi"
        if in_lo:
            return "lo"
        if in_hi:
            return "hi"
        return None

    def mousePressEvent(self, ev) -> None:
        self._dragging = self._hit_handle(ev.position().x())
        ev.accept()

    def mouseMoveEvent(self, ev) -> None:
        x = ev.position().x()
        if self._dragging == "lo":
            new = min(self._x_to_val(x), self._hi)
            if new != self._lo:
                self._lo = new
                self.rangeChanged.emit(self._lo, self._hi)
                self.update()
        elif self._dragging == "hi":
            new = max(self._x_to_val(x), self._lo)
            if new != self._hi:
                self._hi = new
                self.rangeChanged.emit(self._lo, self._hi)
                self.update()
        else:
            hov = self._hit_handle(x)
            if hov != self._hovered:
                self._hovered = hov
                self.update()
        ev.accept()

    def mouseReleaseEvent(self, ev) -> None:
        self._dragging = None
        ev.accept()

    def leaveEvent(self, ev) -> None:
        self._hovered = None
        self.update()
        super().leaveEvent(ev)

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w  = self.width()
        cy = self.height() / 2.0
        m  = self._MARGIN
        th = self._TRACK_H

        lo_x = self._val_to_x(self._lo)
        hi_x = self._val_to_x(self._hi)

        groove = QColor(get_color("training_widget_slider_groove", "#303030"))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(groove))
        p.drawRoundedRect(QRectF(m, cy - th / 2, w - 2 * m, th), th / 2, th / 2)

        fill = QColor(get_color("training_widget_slider_fill", "#F6D393"))
        p.setBrush(QBrush(fill))
        p.drawRoundedRect(QRectF(lo_x, cy - th / 2, max(0.0, hi_x - lo_x), th), th / 2, th / 2)

        # Rectangular handles: height = 2*R, width = R, corner radius = 2
        r = float(self._HANDLE_R)
        hw = r          # width  = HANDLE_R
        hh = r * 2.0    # height = 2 * HANDLE_R
        for handle, hx in (("lo", lo_x), ("hi", hi_x)):
            active = (self._hovered == handle) or (self._dragging == handle)
            if active:
                col = QColor(get_color("training_widget_slider_handle_hover", "#7a8490"))
                bdr = QColor(get_color("training_widget_slider_handle_hover_border", "#22c55e"))
                p.setPen(QPen(bdr, 1.5))
            else:
                col = QColor(get_color("training_widget_slider_handle", "#5a6470"))
                p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(col))
            p.drawRoundedRect(QRectF(hx - hw / 2, cy - hh / 2, hw, hh), 2.0, 2.0)

        p.end()

    def set_values(self, lo: float, hi: float) -> None:
        self._lo = self._snap(max(self._min, min(lo, self._max)))
        self._hi = self._snap(max(self._lo, min(hi, self._max)))
        self.update()

    def lo_value(self) -> float:
        return self._lo

    def hi_value(self) -> float:
        return self._hi


class NodeSlider(QWidget):
    """Unified slider component with standard and range variants.

    **Standard** (``mode="standard"``)::

        [ QSlider ──────── ][ IndexButton ]

    Emits ``valueChanged(float)`` on slider drag or IndexButton edit.

    **Range** (``mode="range"``)::

        [ IndexButton_lo ][ RangeSlider ─── ][ IndexButton_hi ]

    Emits ``rangeChanged(float, float)`` on slider drag or IndexButton edit.
    """

    valueChanged = Signal(float)
    rangeChanged = Signal(float, float)

    def __init__(
        self,
        *,
        mode: str = "standard",
        minimum: float = 0.0,
        maximum: float = 1.0,
        step: float = 0.01,
        decimals: int = 2,
        current: float = 0.0,
        lo: float = 0.0,
        hi: float = 1.0,
        snap_value: Optional[float] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._mode = mode
        self._min = float(minimum)
        self._max = float(maximum)
        self._step = float(step)
        self._decimals = int(decimals)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        hl = QHBoxLayout(self)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(4)

        if mode == "range":
            self._lo_btn = IndexButton(
                current_value=lo, min_value=minimum, max_value=maximum,
                decimals=decimals, parent=self,
            )
            self._lo_btn.setFixedWidth(36)
            self._hi_btn = IndexButton(
                current_value=hi, min_value=minimum, max_value=maximum,
                decimals=decimals, parent=self,
            )
            self._hi_btn.setFixedWidth(36)
            self._range_slider = _RangeHandleSlider(
                minimum=minimum, maximum=maximum, lo=lo, hi=hi, step=step,
            )

            self._range_slider.rangeChanged.connect(self._on_range_slide)
            self._lo_btn.valueChanged.connect(self._on_lo_btn)
            self._hi_btn.valueChanged.connect(self._on_hi_btn)

            hl.addWidget(self._lo_btn, 0)
            hl.addWidget(self._range_slider, 1)
            hl.addWidget(self._hi_btn, 0)
        else:
            n_steps = max(1, round((maximum - minimum) / step))
            snap_slot = None
            if snap_value is not None:
                snap_slot = round((snap_value - minimum) / step)

            self._qslider = _ModuleCurveSlider(
                snap_value=snap_slot, parent=self,
            )
            self._qslider.setCursor(Qt.CursorShape.PointingHandCursor)
            self._qslider.setRange(0, n_steps)

            self._idx_btn = IndexButton(
                current_value=current, min_value=minimum, max_value=maximum,
                decimals=decimals, parent=self,
            )

            cur_slot = round((current - minimum) / step)
            self._qslider.setValue(max(0, min(n_steps, cur_slot)))

            self._n_steps = n_steps
            self._qslider.valueChanged.connect(self._on_std_slide)
            self._idx_btn.valueChanged.connect(self._on_std_btn)

            hl.addWidget(self._qslider, 1)
            hl.addWidget(self._idx_btn, 0)

    # ── standard mode ─────────────────────────────────────────────────

    def _on_std_slide(self, slot: int) -> None:
        v = round(self._min + slot * self._step, 6)
        self._idx_btn.blockSignals(True)
        self._idx_btn.set_value(v)
        self._idx_btn.blockSignals(False)
        self.valueChanged.emit(v)

    def _on_std_btn(self, v: float) -> None:
        slot = round((v - self._min) / self._step)
        self._qslider.blockSignals(True)
        self._qslider.setValue(max(0, min(self._n_steps, slot)))
        self._qslider.blockSignals(False)
        self.valueChanged.emit(v)

    # ── range mode ────────────────────────────────────────────────────

    def _on_range_slide(self, lo: float, hi: float) -> None:
        self._lo_btn.blockSignals(True)
        self._hi_btn.blockSignals(True)
        self._lo_btn.set_value(lo)
        self._hi_btn.set_value(hi)
        self._lo_btn.blockSignals(False)
        self._hi_btn.blockSignals(False)
        self.rangeChanged.emit(lo, hi)

    def _on_lo_btn(self, v: float) -> None:
        hi = self._range_slider.hi_value()
        v = min(v, hi)
        self._range_slider.set_values(v, hi)
        self._lo_btn.blockSignals(True)
        self._lo_btn.set_value(v)
        self._lo_btn.blockSignals(False)
        self.rangeChanged.emit(v, hi)

    def _on_hi_btn(self, v: float) -> None:
        lo = self._range_slider.lo_value()
        v = max(v, lo)
        self._range_slider.set_values(lo, v)
        self._hi_btn.blockSignals(True)
        self._hi_btn.set_value(v)
        self._hi_btn.blockSignals(False)
        self.rangeChanged.emit(lo, v)

    # ── public API ────────────────────────────────────────────────────

    def set_value(self, v: float) -> None:
        """Set value for standard mode."""
        self._idx_btn.set_value(v)
        slot = round((v - self._min) / self._step)
        self._qslider.blockSignals(True)
        self._qslider.setValue(max(0, min(self._n_steps, slot)))
        self._qslider.blockSignals(False)

    def set_range_values(self, lo: float, hi: float) -> None:
        """Set values for range mode."""
        self._range_slider.set_values(lo, hi)
        self._lo_btn.set_value(self._range_slider.lo_value())
        self._hi_btn.set_value(self._range_slider.hi_value())


# Legacy alias — internal code that still references the old name
_ModuleValuePopup = DataInput



class NodeRow(QWidget):
    """Universal row container for Training Ground node cards.

    Layout::

        [ input_zone (opt) ][ function_zone (title + widget) ][ output_zone (opt) ]

    * ``input_zone`` / ``output_zone`` are ``NODE_STANDARD_H × NODE_STANDARD_H``
      square placeholders.  The actual ``TrainingNodePort`` dots are positioned
      on top by ``_reflow_layout``; these zones just reserve the visual space.
    * When an external connection feeds into ``input_zone``, the function_zone
      widgets are **disabled** (greyed out) and their display is overridden with
      the upstream value (or ``<overwrote>`` if no value is available yet).

    Parameters
    ----------
    input_slot:
        Slot name for the input port zone (e.g. ``"total_steps"``).
        ``None`` means no input zone.
    output_slot:
        Slot name for the output port zone.  ``None`` means no output zone.
    title:
        Label text rendered in the function zone's left column.
    widget:
        The functional QWidget (slider, input, dropdown, etc.) placed to the
        right of the title.  ``None`` for label-only rows.
    """

    _OVERWRITE_PLACEHOLDER = "<overwrote>"

    def __init__(
        self,
        *,
        input_slot: Optional[str] = None,
        output_slot: Optional[str] = None,
        title: str = "",
        widget: Optional[QWidget] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.input_slot = input_slot
        self.output_slot = output_slot
        self._title = title
        self._func_widget = widget
        self._connected = False
        self._original_tooltip = ""
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        hl = QHBoxLayout(self)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(0)

        sz = NODE_STANDARD_H

        # ── input zone (square placeholder for port dot) ──────────────
        if input_slot is not None:
            self._input_zone = QWidget(self)
            self._input_zone.setFixedSize(sz, sz)
            self._input_zone.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
            hl.addWidget(self._input_zone, 0)
        else:
            self._input_zone = None

        # ── function zone (title label + stretch + widget, right-aligned) ─
        func = QWidget(self)
        func.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        func_hl = QHBoxLayout(func)
        func_hl.setContentsMargins(0, 0, 0, 0)
        func_hl.setSpacing(4)

        if title:
            lbl = QLabel(title, func)
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            lbl_w = LABEL_COL - sz - 4 if input_slot else LABEL_COL
            lbl.setFixedWidth(max(0, lbl_w))
            func_hl.addWidget(lbl, 0)

        func_hl.addStretch(1)

        if widget is not None:
            self._original_tooltip = widget.toolTip()
            widget.setMaximumWidth(WIDGET_W)
            func_hl.addWidget(widget, 0)

        hl.addWidget(func, 1)

        # ── output zone (square placeholder for port dot) ─────────────
        if output_slot is not None:
            self._output_zone = QWidget(self)
            self._output_zone.setFixedSize(sz, sz)
            self._output_zone.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
            hl.addWidget(self._output_zone, 0)
        else:
            self._output_zone = None


    # ── connection state ──────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._connected

    def set_connected(self, connected: bool, upstream_value: Optional[str] = None) -> None:
        """Called by the parent TrainingNodeItem when the input port's connection changes."""
        self._connected = connected
        w = self._func_widget
        if w is None:
            return

        if connected:
            display = upstream_value if upstream_value else self._OVERWRITE_PLACEHOLDER
            self._push_display(w, display)
            w.setEnabled(False)
            w.setToolTip(
                "Value set by connected upstream node.\n"
                "Disconnect the port to edit manually."
            )
        else:
            w.setEnabled(True)
            w.setToolTip(self._original_tooltip)


    @staticmethod
    def _push_display(w: QWidget, display: str) -> None:
        """Try every known way to push *display* into the widget's visual."""
        # QSpinBox inside a container (timestep_input)
        from PySide6.QtWidgets import QSpinBox
        for child in w.findChildren(QSpinBox):
            try:
                child.setValue(int(float(display)))
                return
            except (TypeError, ValueError):
                child.setSpecialValueText(display)
                child.setValue(child.minimum())
                return
        # NodeSlider / IndexButton
        if hasattr(w, "set_value"):
            try:
                w.set_value(float(display))
                return
            except (TypeError, ValueError):
                pass
        # QLineEdit
        if hasattr(w, "setText"):
            w.setText(str(display))
            return
        # Generic setValue
        if hasattr(w, "setValue"):
            try:
                w.setValue(int(float(display)))
            except (TypeError, ValueError):
                pass

    # ── public helpers ────────────────────────────────────────────────

    @property
    def func_widget(self) -> Optional[QWidget]:
        return self._func_widget

    def has_input_zone(self) -> bool:
        return self._input_zone is not None

    def has_output_zone(self) -> bool:
        return self._output_zone is not None


class _TextInputPopup(QFrame):
    """
    Compact inline text-input popup — same positioning idiom as _ModuleValuePopup.

    Usage::

        popup = _TextInputPopup(placeholder="Enter name…")
        popup.set_validator(lambda s: "Error msg" or None)
        popup.accepted.connect(callback)
        pos, row_h = some_widget._text_anchor_global()
        popup.show_at(pos, row_h)
    """

    accepted = Signal(str)

    def __init__(self, *, placeholder: str = "", parent=None) -> None:
        super().__init__(
            parent,
            Qt.WindowType.Popup
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.NoDropShadowWindowHint,
        )
        self.setObjectName("textInputPopup")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._validate_fn: Optional[Callable[[str], Optional[str]]] = None
        self._did_accept = False

        border = get_color("training_widget_border", "#3d3d3d")
        bg     = get_color("training_widget_input_bg", "#1A1A1A")
        text   = get_color("training_widget_text", "#cccccc")
        btn_bg = get_color("training_widget_button_bg", get_color("button_bg", "#252525"))
        hover  = get_color("training_button_hover_bg", get_color("button_hover", "#2e2e2e"))
        err_c  = get_color("training_widget_error", "#f87171")

        self.setStyleSheet(
            f"""
            #textInputPopup {{ background: transparent; border: none; }}
            QLineEdit[textInputField="true"] {{
                background: {bg}; color: {text};
                border: 1px solid {border}; border-radius: 4px; padding: 0px 4px;
            }}
            QLineEdit[textInputField="true"][invalid="true"] {{
                border-color: {err_c};
            }}
            QPushButton[textInputBtn="true"] {{
                background: {btn_bg}; color: {text};
                border: 1px solid {border}; border-radius: 4px; padding: 0px 2px;
            }}
            QPushButton[textInputBtn="true"]:hover {{ background: {hover}; }}
            QLabel[textInputError="true"] {{
                color: {err_c}; font-size: 10px;
                background: {bg}; border: 1px solid {border};
                border-radius: 3px; padding: 1px 4px;
            }}
            """
        )

        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(2)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(1)
        col.addLayout(row)

        self._input = QLineEdit(self)
        self._input.setProperty("textInputField", True)
        self._input.setPlaceholderText(placeholder)
        row.addWidget(self._input, 1)

        self._ok_btn = QPushButton("✔", self)
        self._ok_btn.setProperty("textInputBtn", True)
        self._ok_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._ok_btn.clicked.connect(self._accept_if_valid)
        row.addWidget(self._ok_btn, 0)

        self._cancel_btn = QPushButton("❌", self)
        self._cancel_btn.setProperty("textInputBtn", True)
        self._cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._cancel_btn.clicked.connect(self.close)
        row.addWidget(self._cancel_btn, 0)

        self._error_lbl = QLabel("", self)
        self._error_lbl.setProperty("textInputError", True)
        self._error_lbl.setVisible(False)
        col.addWidget(self._error_lbl)

        self._input.returnPressed.connect(self._accept_if_valid)
        self._input.textEdited.connect(lambda _: self._clear_error())

    def set_validator(self, fn: Callable[[str], Optional[str]]) -> None:
        """fn(name) → error string, or None if valid."""
        self._validate_fn = fn

    def _accept_if_valid(self) -> None:
        name = self._input.text().strip()
        if not name:
            self._mark_invalid("Name is required.")
            return
        if self._validate_fn is not None:
            err = self._validate_fn(name)
            if err is not None:
                self._mark_invalid(err)
                return
        self._did_accept = True
        self.accepted.emit(name)
        self.close()

    def _mark_invalid(self, msg: str = "") -> None:
        self._input.setProperty("invalid", True)
        self.style().unpolish(self._input)
        self.style().polish(self._input)
        if msg:
            self._error_lbl.setText(msg)
            self._error_lbl.setVisible(True)
            self.adjustSize()
        self._input.setFocus()
        self._input.selectAll()

    def _clear_error(self) -> None:
        self._input.setProperty("invalid", False)
        self.style().unpolish(self._input)
        self.style().polish(self._input)
        self._error_lbl.setVisible(False)
        self.adjustSize()

    def show_at(self, pos: QPoint, row_height: int = 32) -> None:
        h = min(50, max(20, row_height))
        scale = h / 50.0
        font_px = max(8, round(16 * scale))
        from PySide6.QtGui import QFont as _QFont
        font = _QFont()
        font.setPixelSize(font_px)
        self._input.setFixedHeight(h)
        self._input.setFixedWidth(round(140 * scale))
        self._input.setFont(font)
        self._ok_btn.setFixedSize(h, h)
        self._ok_btn.setFont(font)
        self._cancel_btn.setFixedSize(h, h)
        self._cancel_btn.setFont(font)
        self.adjustSize()
        y_offset = max(0, (row_height - h) // 2)
        self.move(QPoint(pos.x(), pos.y() + y_offset))
        self.show()
        self.raise_()
        self._input.setFocus()


class _ModuleValueButton(QPushButton):
    """Transparent numeric readout button used by registry slider rows."""

    def __init__(
        self,
        *,
        title: str,
        current_value: float,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._current_value = float(current_value)
        self.setProperty("moduleValueButton", True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumWidth(60)
        self.setFlat(True)
        self.set_value(current_value)

    def set_value(self, value: float) -> None:
        self._current_value = float(value)
        self.setText(f"{self._current_value:.3f}")


class _ModuleCurveSlider(QSlider):
    """Track-click slider with a small midpoint snap zone for module rows."""

    def __init__(
        self,
        *,
        snap_value: Optional[int] = None,
        snap_radius_px: int = 10,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(Qt.Orientation.Horizontal, parent)
        self._snap_value = snap_value
        self._snap_radius_px = max(0, int(snap_radius_px))

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        handle_rect = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider,
            opt,
            QStyle.SubControl.SC_SliderHandle,
            self,
        )
        if handle_rect.contains(event.position().toPoint()):
            super().mousePressEvent(event)
            return

        groove_rect = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider,
            opt,
            QStyle.SubControl.SC_SliderGroove,
            self,
        )
        pos_x = int(round(event.position().x()))
        pos_x = max(groove_rect.left(), min(groove_rect.right(), pos_x))
        center_x = groove_rect.center().x()

        if self._snap_value is not None and abs(pos_x - center_x) <= self._snap_radius_px:
            self.setValue(int(self._snap_value))
            event.accept()
            return

        span = max(1, groove_rect.width())
        relative = pos_x - groove_rect.left()
        new_value = QStyle.sliderValueFromPosition(
            self.minimum(),
            self.maximum(),
            relative,
            span,
            opt.upsideDown,
        )
        self.setValue(int(new_value))
        event.accept()


class _RegistryModuleRow(QWidget):
    """Reusable registry row: [index, name, NodeSlider] with split-value mapping."""

    def __init__(
        self,
        *,
        item: TaskModuleItem,
        index_text: str,
        index_width: int,
        name_width: int,
        slider_min_width: int,
        current_value: float,
        write_value,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._item = item
        self._write_value = write_value
        self.setProperty("moduleRow", True)
        self.setFixedHeight(32)
        self._split_value = self._resolve_split_value()

        hl = QHBoxLayout(self)
        hl.setContentsMargins(4, 2, 4, 2)
        hl.setSpacing(4)

        if item.kind == "reward":
            index_label = _ModulePolarityBadge(item.polarity, self)
            index_label.setFixedSize(index_width, index_width)
        else:
            index_label = QLabel(index_text, self)
            index_label.setProperty("moduleIndex", True)
            index_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            index_label.setFixedWidth(index_width)

        name_label = QLabel(item.title, self)
        name_label.setProperty("moduleName", True)
        name_label.setFixedWidth(name_width)
        name_label.setToolTip(item.desc)
        name_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        snap = self._split_value if self._split_value is not None else None
        self._node_slider = NodeSlider(
            mode="standard",
            minimum=item.min_value,
            maximum=item.max_value,
            step=item.step,
            decimals=3,
            current=current_value,
            snap_value=snap,
            parent=self,
        )
        self._node_slider.setMinimumWidth(slider_min_width)
        self._node_slider.valueChanged.connect(self._on_value)

        hl.addWidget(index_label, 0, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        hl.addWidget(name_label, 0)
        hl.addWidget(self._node_slider, 1)

    def _on_value(self, v: float) -> None:
        self._write_value(round(v, 6))

    def _resolve_split_value(self) -> Optional[float]:
        min_v = float(self._item.min_value)
        max_v = float(self._item.max_value)
        if min_v >= 0.0 and max_v > 2.0:
            return 2.0
        if max_v <= 0.0 and min_v < -2.0:
            return -2.0
        return None


class _RichChoiceCard(QFrame):
    clicked = Signal(str)

    ICON_SIZE = 56

    def __init__(
        self,
        value: str,
        *,
        title: str,
        desc: str,
        selected: bool,
        leading_mode: str,
        icon_path: Optional[pathlib.Path] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._value = value
        self._leading_mode = leading_mode
        self._title_color = get_color("training_widget_text", get_color("text_primary", "#d1d5db"))
        self._desc_color = get_color("training_widget_label_text", get_color("text_secondary", "#9ca3af"))
        self._selected_title_color = get_color(
            "training_tabs_tab_selected_text",
            get_color("text_on_accent", get_color("accent_text", "#ffffff")),
        )
        self._selected_desc_color = get_color(
            "text_on_accent",
            get_color("training_tabs_tab_selected_text", "#ffffff"),
        )
        self.setObjectName("richChoiceCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setProperty("selected", selected)

        row = QHBoxLayout(self)
        row.setContentsMargins(10, 8, 10, 8)
        row.setSpacing(12)

        if leading_mode == "checkbox":
            self._checkbox = QCheckBox(self)
            self._checkbox.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            self._checkbox.setChecked(selected)
            self._checkbox.setFixedWidth(22)
            row.addWidget(self._checkbox, 0, Qt.AlignmentFlag.AlignTop)
        else:
            icon_label = QLabel()
            icon_label.setFixedSize(self.ICON_SIZE, self.ICON_SIZE)
            if icon_path is not None:
                icon = QIcon(str(icon_path))
            else:
                icon = QIcon()
            if not icon.isNull():
                icon_label.setPixmap(icon.pixmap(QSize(self.ICON_SIZE, self.ICON_SIZE)))
            else:
                icon_label.setText(value[:1].upper())
                icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            row.addWidget(icon_label, 0, Qt.AlignmentFlag.AlignTop)

        title_label = QLabel(title)
        title_label.setProperty("taskTitle", True)
        desc_label = QLabel(desc)
        desc_label.setProperty("taskDesc", True)
        desc_label.setWordWrap(True)
        self._title_label = title_label
        self._desc_label = desc_label

        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(2)
        text_col.addWidget(title_label)
        text_col.addWidget(desc_label)

        row.addLayout(text_col, 1)
        self.adjustSize()
        self.set_selected(selected)

    def set_selected(self, selected: bool) -> None:
        self.setProperty("selected", bool(selected))
        checkbox = getattr(self, "_checkbox", None)
        if checkbox is not None:
            checkbox.setChecked(bool(selected))
        if hasattr(self, "_title_label"):
            self._title_label.setStyleSheet(
                f"color: {self._selected_title_color if selected else self._title_color};"
                " font-size: 13px; font-weight: 600; background: transparent;"
            )
        if hasattr(self, "_desc_label"):
            self._desc_label.setStyleSheet(
                f"color: {self._selected_desc_color if selected else self._desc_color};"
                " font-size: 11px; background: transparent;"
            )
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._value)
            event.accept()
            return
        super().mousePressEvent(event)


class _RichChoicePopup(QFrame):
    selection_changed = Signal(list)
    MAX_HEIGHT = 320

    def __init__(
        self,
        choices: List[str],
        current_values: List[str],
        *,
        meta_map: Dict[str, Dict[str, str]],
        leading_mode: str = "icon",
        multi_select: bool = False,
        parent=None,
    ):
        super().__init__(parent, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.setObjectName("richChoicePopup")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setMinimumWidth(320)
        self.setMaximumHeight(self.MAX_HEIGHT)
        self._choices = [str(choice) for choice in choices]
        self._selected = [value for value in current_values if value in self._choices]
        self._leading_mode = leading_mode
        self._multi_select = multi_select
        self._meta_map = meta_map
        self._cards: Dict[str, _RichChoiceCard] = {}

        bg = get_color("training_widget_input_bg", get_color("input_bg", "#1c1c1c"))
        border = get_color("training_widget_border", get_color("input_border", "#3a3a3a"))
        text = get_color("training_widget_text", get_color("text_primary", "#d1d5db"))
        hover = get_color("training_widget_popup_selected_bg", "#2d4a7a")
        selected_bg = get_color("training_widget_focus_border", "#4f7ecc")
        selected_text = get_color(
            "training_tabs_tab_selected_text",
            get_color("text_on_accent", get_color("accent_text", "#ffffff")),
        )
        selected_desc = get_color(
            "text_on_accent",
            get_color("training_tabs_tab_selected_text", "#ffffff"),
        )
        muted = get_color("training_widget_label_text", get_color("text_secondary", "#9ca3af"))
        accent = get_color("training_widget_focus_border", "#4f7ecc")
        check_green = get_color("training_widget_toggle_border", "#22c55e")

        self.setStyleSheet(
            f"""
            #richChoicePopup {{
                background: {bg};
                border: 1px solid {border};
                border-radius: 8px;
            }}
            #richChoiceCard {{
                background: transparent;
                border: 1px solid transparent;
                border-radius: 8px;
            }}
            #richChoiceCard:hover {{
                background: {hover};
                border-color: {accent};
            }}
            #richChoiceCard[selected="true"] {{
                background: {selected_bg};
                border-color: {accent};
            }}
            QLabel[taskTitle="true"] {{
                color: {text};
                font-size: 13px;
                font-weight: 600;
            }}
            QLabel[taskTitle="true"][selected="true"] {{
                color: {selected_text};
            }}
            QLabel[taskDesc="true"] {{
                color: {muted};
                font-size: 11px;
            }}
            QLabel[taskDesc="true"][selected="true"] {{
                color: {selected_desc};
            }}
            QScrollArea {{
                background: transparent;
                border: none;
            }}
            QScrollBar:vertical {{
                background: transparent;
                width: 10px;
                margin: 2px 0 2px 0;
            }}
            QScrollBar::handle:vertical {{
                background: {border};
                border-radius: 5px;
                min-height: 24px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: transparent;
                border: none;
                height: 0px;
            }}
            QCheckBox {{
                background: transparent;
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border-radius: 4px;
                border: 1px solid {border};
                background: transparent;
            }}
            QCheckBox::indicator:checked {{
                background: {check_green};
                border-color: {check_green};
            }}
            QCheckBox::indicator:unchecked {{
                background: transparent;
                border-color: {border};
            }}
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(0)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        layout.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)

        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(8)
        content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        for choice in self._choices:
            meta = self._meta_map.get(choice, {})
            card = _RichChoiceCard(
                choice,
                title=meta.get("title", choice.replace("_", " ").title()),
                desc=meta.get("desc", ""),
                selected=choice in self._selected,
                leading_mode=self._leading_mode,
                icon_path=self._icon_path(choice) if self._leading_mode == "icon" else None,
                parent=self,
            )
            card.clicked.connect(self._select)
            self._cards[choice] = card
            content_layout.addWidget(card)

        content_layout.addStretch(1)

    @staticmethod
    def _icon_path(task_type: str) -> pathlib.Path:
        from src.system.core.theme_manager import _get_icon_dir
        return _get_icon_dir() / f"icon_{task_type}.svg"

    def _select(self, value: str) -> None:
        if self._multi_select:
            if value in self._selected:
                self._selected = [item for item in self._selected if item != value]
            else:
                self._selected.append(value)
            self._sync_card_selection()
            self.selection_changed.emit(list(self._selected))
            return

        self._selected = [value]
        self._sync_card_selection()
        self.selection_changed.emit([value])
        try:
            self.close()
        except RuntimeError:
            pass  # C++ object already deleted during window teardown

    def _sync_card_selection(self) -> None:
        selected_set = set(self._selected)
        for value, card in self._cards.items():
            card.set_selected(value in selected_set)


class _RichChoicePicker(QPushButton):
    currentTextChanged = Signal(str)
    selectionChanged = Signal(list)

    def __init__(
        self,
        choices: List[str],
        *,
        current: str = "",
        current_values: Optional[List[str]] = None,
        meta_map: Optional[Dict[str, Dict[str, str]]] = None,
        leading_mode: str = "icon",
        multi_select: bool = False,
        choices_provider: Optional[Callable[[], Tuple[List[str], Dict[str, Dict[str, str]]]]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._choices = [str(c) for c in choices]
        self._multi_select = multi_select
        self._meta_map = meta_map or {}
        self._leading_mode = leading_mode
        self._choices_provider = choices_provider
        self._popup = None
        default_text = current if current in self._choices else (self._choices[0] if self._choices else "")
        default_values = [value for value in (current_values or []) if value in self._choices]
        self._current_text = default_text
        self._current_values = default_values
        if self._multi_select and not self._current_values:
            self._current_values = []
        self._raw_label: str = ""
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        # Prevent the QGraphicsProxyWidget from auto-expanding when text changes
        self.setMaximumWidth(WIDGET_W)
        self.clicked.connect(self._show_popup)
        self._apply_label()

    def set_choices(
        self,
        choices: List[str],
        *,
        meta_map: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> None:
        self._choices = [str(choice) for choice in choices]
        if meta_map is not None:
            self._meta_map = dict(meta_map)
        if self._multi_select:
            self._current_values = [value for value in self._current_values if value in self._choices]
        elif self._current_text not in self._choices:
            self._current_text = self._choices[0] if self._choices else ""
        self._apply_label()

    def currentText(self) -> str:
        return self._current_text

    def findText(self, text: str) -> int:
        try:
            return self._choices.index(text)
        except ValueError:
            return -1

    def setCurrentIndex(self, index: int) -> None:
        if 0 <= index < len(self._choices):
            self.setCurrentText(self._choices[index])

    def setCurrentText(self, text: str) -> None:
        if self._multi_select:
            self.setCurrentValues([text] if text else [])
            return
        if text not in self._choices:
            return
        changed = text != self._current_text
        self._current_text = text
        self._apply_label()
        if changed:
            self.currentTextChanged.emit(text)
            self.selectionChanged.emit([text])

    def currentValues(self) -> List[str]:
        if self._multi_select:
            return list(self._current_values)
        return [self._current_text] if self._current_text else []

    def setCurrentValues(self, values: List[str]) -> None:
        normalized = [value for value in values if value in self._choices]
        changed = normalized != self._current_values
        self._current_values = normalized
        self._apply_label()
        if changed:
            self.selectionChanged.emit(list(self._current_values))

    def _apply_label(self) -> None:
        if self._multi_select:
            count = len(self._current_values)
            if count == 0:
                raw = "Select…"
            elif count == 1:
                value = self._current_values[0]
                meta = self._meta_map.get(value, {})
                raw = meta.get("title", value.replace("_", " ").title())
            else:
                raw = f"{count} selected"
        else:
            meta = self._meta_map.get(self._current_text, {})
            raw = meta.get("title", self._current_text.replace("_", " ").title())
        self._raw_label = raw
        self._update_elided_text()

    def _update_elided_text(self) -> None:
        avail = (self.width() or WIDGET_W) - 22   # subtract ~22px for arrow indicator + padding
        elided = QFontMetrics(self.font()).elidedText(
            self._raw_label, Qt.TextElideMode.ElideRight, max(30, avail)
        )
        # Use QPushButton.setText directly to avoid re-entrant _apply_label calls
        QPushButton.setText(self, elided)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._raw_label:
            self._update_elided_text()

    def _show_popup(self) -> None:
        if self._choices_provider is not None:
            choices, meta_map = self._choices_provider()
            self.set_choices(choices, meta_map=meta_map)
        if not self._choices:
            return
        popup = _RichChoicePopup(
            self._choices,
            self.currentValues(),
            meta_map=self._meta_map,
            leading_mode=self._leading_mode,
            multi_select=self._multi_select,
            parent=None,
        )
        self._popup = popup
        popup.selection_changed.connect(self._on_popup_selection_changed)
        popup.destroyed.connect(self._clear_popup_ref)
        popup.move(self._popup_anchor_global())
        popup.show()

    def _on_popup_selection_changed(self, values: List[str]) -> None:
        if self._multi_select:
            self.setCurrentValues(values)
            return
        self.setCurrentText(values[0] if values else "")

    def _clear_popup_ref(self, *_args) -> None:
        self._popup = None

    def _popup_anchor_global(self) -> QPoint:
        """Anchor the popup to the right edge of the containing node card."""
        try:
            host = self.window()
            proxy = host.graphicsProxyWidget() if host is not None else None
            if proxy is not None:
                parent_item = proxy.parentItem()
                scene = proxy.scene()
                views = scene.views() if scene is not None else []
                if parent_item is not None and views:
                    view = views[0]
                    node_rect = parent_item.sceneBoundingRect()
                    anchor_scene = QPointF(node_rect.right() + 2, node_rect.top() + proxy.pos().y())
                    anchor_view = view.mapFromScene(anchor_scene)
                    return view.viewport().mapToGlobal(anchor_view)
        except Exception:
            pass
        return self.mapToGlobal(QPoint(self.width() + 2, 0))

    def _text_anchor_global(self) -> Tuple[QPoint, int]:
        """Return (global_pos, row_height_px) for _TextInputPopup anchoring.

        Positions the text popup to the right of this button at the button's
        screen Y, with row_height scaled to the canvas zoom level — exactly the
        same idiom as _RegistryModuleRow._popup_anchor_global().
        """
        try:
            host = self.window()
            proxy = host.graphicsProxyWidget() if host is not None else None
            if proxy is not None:
                scene = proxy.scene()
                views = scene.views() if scene is not None else []
                if views:
                    view = views[0]
                    scale = view.transform().m11()
                    row_h = round(self.height() * scale)
                    btn_tl = self.mapTo(host, QPoint(0, 0))
                    btn_br = self.mapTo(host, QPoint(self.width(), 0))
                    tl_scene = proxy.mapToScene(QPointF(btn_tl))
                    br_scene = proxy.mapToScene(QPointF(btn_br))
                    anchor_scene = QPointF(max(tl_scene.x(), br_scene.x()) + 2, tl_scene.y())
                    anchor_view = view.mapFromScene(anchor_scene)
                    return view.viewport().mapToGlobal(anchor_view), row_h
        except Exception:
            pass
        return self.mapToGlobal(QPoint(self.width() + 2, 0)), self.height()


class _TaskTypePicker(_RichChoicePicker):
    def __init__(self, choices: List[str], current: str = "", parent=None):
        super().__init__(
            choices,
            current=current,
            meta_map=_TASK_TYPE_META,
            leading_mode="icon",
            multi_select=False,
            parent=parent,
        )


class _ObsComponentsPicker(_RichChoicePicker):
    selectionTextChanged = Signal(str)

    def __init__(self, choices: List[str], current_values: List[str], parent=None):
        super().__init__(
            choices,
            current_values=current_values,
            meta_map=_OBS_COMPONENT_META,
            leading_mode="checkbox",
            multi_select=True,
            parent=parent,
        )
        self.selectionChanged.connect(self._emit_selection_text)

    def _emit_selection_text(self, values: List[str]) -> None:
        self.selectionTextChanged.emit(" ".join(values))


class _DropdownChoicePicker(_RichChoicePicker):
    def __init__(self, choices: List[str], current: str = "", parent=None):
        meta_map = {choice: _default_choice_meta(choice) for choice in choices}
        super().__init__(
            choices,
            current=current,
            meta_map=meta_map,
            leading_mode="checkbox",
            multi_select=False,
            parent=parent,
        )


class _SceneTypePicker(_RichChoicePicker):
    selectionValueChanged = Signal(str)

    def __init__(
        self,
        *,
        current_scene_type: str,
        current_scene_path: str,
        runtime_scene_xml: str,
        parent=None,
    ):
        self._runtime_scene_xml = str(runtime_scene_xml or "").strip()
        self._runtime_scene_type = "custom" if self._runtime_scene_xml else "flat"
        runtime_choice = "runtime_scene"

        choices = [runtime_choice, "flat", "terrain", "custom"]
        runtime_title = (
            "Custom (Runtime)" if self._runtime_scene_type == "custom" else "Flat (Runtime)"
        )
        runtime_subtitle = self._runtime_scene_xml or "Inherited from Mission runtime scene"
        meta_map = {
            runtime_choice: {
                "title": runtime_title,
                "subtitle": runtime_subtitle,
            },
            "flat": _default_choice_meta("flat"),
            "terrain": _default_choice_meta("terrain"),
            "custom": _default_choice_meta("custom"),
        }
        current_choice = str(current_scene_type or "").strip() or "flat"
        current_path = str(current_scene_path or "").strip()
        if (
            (self._runtime_scene_type == "flat" and current_choice == "flat" and not current_path)
            or (self._runtime_scene_type == "custom" and current_choice == "custom" and current_path == self._runtime_scene_xml)
        ):
            current_choice = runtime_choice

        super().__init__(
            choices,
            current=current_choice,
            meta_map=meta_map,
            leading_mode="checkbox",
            multi_select=False,
            parent=parent,
        )
        self.currentTextChanged.connect(self._emit_selection_value)

    def runtime_scene_xml(self) -> str:
        return self._runtime_scene_xml

    def runtime_scene_type(self) -> str:
        return self._runtime_scene_type

    def _emit_selection_value(self, choice: str) -> None:
        if choice == "runtime_scene":
            self.selectionValueChanged.emit(self._runtime_scene_type)
            return
        self.selectionValueChanged.emit(choice)


class _RegistryModuleEditor(QWidget):
    height_changed = Signal(int)
    width_changed = Signal(int)
    SORT_REGISTRY = {
        "registry": "registry",
        "title_asc": "title_asc",
    }

    def __init__(
        self,
        registry: Dict[str, TaskModuleItem],
        initial_raw: str,
        write_fn,
        selector_title: str = "Title",
        sort_mode: str = "registry",
        family_provider: Optional[Callable[[], str]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._full_row_widget = True
        self._registry = dict(registry)
        self._order = list(self._registry.keys())
        self._write_fn = write_fn
        self._selector_title = str(selector_title or "Title").strip() or "Title"
        self._sort_mode = self.SORT_REGISTRY.get(str(sort_mode or "").strip().lower(), "registry")
        self._family_provider = family_provider
        self._values = self._parse_values(initial_raw)
        self._selected = [key for key in self._ordered_keys(self._registry) if key in self._values]
        self._preferred_height = PARAM_ROW_H
        self._preferred_width = WIDGET_W

        border = get_color("training_widget_border", "#3d3d3d")
        input_bg = get_color("training_widget_input_bg", "#1A1A1A")
        row_bg = input_bg
        text = get_color("training_widget_text", "#cccccc")
        muted = get_color("training_widget_label_text", "#aaaaaa")

        self.setStyleSheet(
            f"""
            QLabel[moduleHeader="true"] {{
                color: {muted};
                font-size: 11px;
                background: transparent;
            }}
            QWidget[moduleTableHeader="true"] {{
                background: {input_bg};
                border: none;
                border-radius: 0px;
            }}
            #moduleTableWrap {{
                background: {input_bg};
                border: 1px solid {border};
                border-radius: 6px;
            }}
            QWidget[moduleRow="true"] {{
                background: {row_bg};
                border: none;
                border-radius: 0px;
            }}
            QLabel[moduleIndex="true"] {{
                color: {muted};
                font-size: 11px;
                background: transparent;
            }}
            QLabel[moduleName="true"] {{
                color: {text};
                font-size: 11px;
                background: transparent;
            }}
            QPushButton[moduleValueButton="true"] {{
                color: {text};
                background: transparent;
                border: none;
                padding: 2px 4px;
                text-align: right;
            }}
            QPushButton[moduleValueButton="true"]:hover {{
                background: rgba(255, 255, 255, 0.08);
                border-radius: 4px;
            }}
            QLabel[moduleColumnHeader="true"] {{
                color: {muted};
                font-size: 10px;
                font-weight: 600;
                background: transparent;
            }}
            QFrame[moduleColumnDivider="true"] {{
                background: {border};
                min-width: 1px;
                max-width: 1px;
                border: none;
            }}
            QFrame[moduleDivider="true"] {{
                background: {border};
                min-height: 1px;
                max-height: 1px;
                border: none;
            }}
            """
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        top_row = QWidget(self)
        top_layout = QHBoxLayout(top_row)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(4)

        header = QLabel(f"{self._selector_title}:")
        header.setProperty("moduleHeader", True)
        top_layout.addStretch(1)
        top_layout.addWidget(header, 0)

        self._selector = _RichChoicePicker(
            self._selector_choices()[0],
            current_values=self._selected,
            meta_map=self._selector_choices()[1],
            leading_mode="checkbox",
            multi_select=True,
            choices_provider=self._selector_choices,
            parent=self,
        )
        self._selector.setFixedSize(WIDGET_W, max(20, PARAM_ROW_H - 4))
        self._selector.selectionChanged.connect(self._on_selection_changed)
        top_layout.addWidget(self._selector, 0)
        root.addWidget(top_row)

        self._content_wrap = QFrame(self)
        self._content_wrap.setObjectName("moduleTableWrap")
        self._content = QWidget(self._content_wrap)
        wrap_layout = QVBoxLayout(self._content_wrap)
        wrap_layout.setContentsMargins(2, 2, 2, 2)
        wrap_layout.setSpacing(0)

        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(0)
        wrap_layout.addWidget(self._content)
        root.addWidget(self._content_wrap)

        self._rebuild_content()

    def extra_widget_style(self) -> str:
        border = get_color("training_widget_border", "#3d3d3d")
        input_bg = get_color("training_widget_input_bg", "#1A1A1A")
        text = get_color("training_widget_text", "#cccccc")
        muted = get_color("training_widget_label_text", "#aaaaaa")
        return f"""
        QLabel[moduleHeader="true"] {{
            color: {muted};
            font-size: 11px;
            background: transparent;
        }}
        #moduleTableWrap {{
            background: {input_bg};
            border: 1px solid {border};
            border-radius: 6px;
        }}
        QWidget[moduleRow="true"] {{
            background: {input_bg};
            border: none;
            border-radius: 0px;
        }}
        QLabel[moduleIndex="true"] {{
            color: {muted};
            font-size: 11px;
            background: transparent;
        }}
        QLabel[moduleName="true"] {{
            color: {text};
            font-size: 11px;
            background: transparent;
        }}
        QPushButton[moduleValueButton="true"] {{
            color: {text};
            background: transparent;
            border: none;
            padding: 2px 4px;
            text-align: right;
        }}
        QPushButton[moduleValueButton="true"]:hover {{
            background: rgba(255, 255, 255, 0.08);
            border-radius: 4px;
        }}
        QFrame[moduleDivider="true"] {{
            background: {border};
            min-height: 1px;
            max-height: 1px;
            border: none;
        }}
        """

    @staticmethod
    def _parse_values(raw: str) -> Dict[str, float]:
        try:
            data = json.loads(str(raw or "{}"))
            if isinstance(data, dict):
                return {str(k): float(v) for k, v in data.items()}
        except Exception:
            pass
        return {}

    def preferred_row_height(self) -> int:
        return max(PARAM_ROW_H, int(self._preferred_height))

    def preferred_row_width(self) -> int:
        return max(WIDGET_W, int(self._preferred_width))

    def sizeHint(self) -> QSize:
        return QSize(self.preferred_row_width(), self.preferred_row_height())

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def _index_column_width(self) -> int:
        if any(item.kind == "reward" for item in self._registry.values()):
            return 18
        count = max(1, len([key for key in self._selected if key in self._registry]))
        probe = QLabel(str(count))
        probe.setProperty("moduleIndex", True)
        return max(18, probe.sizeHint().width() + 6)

    def _name_column_width(self) -> int:
        titles = [self._registry[key].title for key in self._selected if key in self._registry]
        if not titles:
            titles = [item.title for item in self._registry.values()]
        width = 72
        for title in titles:
            probe = QLabel(title)
            probe.setProperty("moduleName", True)
            width = max(width, probe.sizeHint().width() + 8)
        return min(width, 104)

    @staticmethod
    def _slider_min_width(index_w: int, name_w: int) -> int:
        row_fixed = 8 + index_w + 4 + name_w + 4 + 2 + 44
        current_slider_w = max(72, (NODE_W - H_PAD * 2) - row_fixed)
        return max(148, int(round(current_slider_w * 1.75)))

    @classmethod
    def _content_target_width(cls, index_w: int, name_w: int) -> int:
        slider_min_w = cls._slider_min_width(index_w, name_w)
        return 8 + index_w + 4 + name_w + 4 + slider_min_w + 2 + 44 + 8

    def _on_selection_changed(self, values: List[str]) -> None:
        selected = [key for key in self._ordered_keys(self._registry) if key in values]
        self._selected = selected
        self._values = {
            key: self._values.get(key, self._registry[key].default)
            for key in selected
            if key in self._registry
        }
        self._write_back()
        self._rebuild_content()

    def _resolved_family(self) -> str:
        try:
            family = str(self._family_provider() if self._family_provider is not None else "").strip().lower()
        except Exception:
            family = ""
        return family or "generic_locomotion"

    def _ordered_keys(self, registry: Dict[str, TaskModuleItem]) -> List[str]:
        keys = list(registry.keys())
        if self._sort_mode != "title_asc":
            return [key for key in self._order if key in registry]
        return sorted(
            keys,
            key=lambda key: (
                str(registry[key].title or "").strip().lower(),
                str(key).lower(),
            ),
        )

    def _filtered_registry(self) -> Dict[str, TaskModuleItem]:
        family = self._resolved_family()
        allowed: Dict[str, TaskModuleItem] = {}
        for key in self._order:
            item = self._registry.get(key)
            if item is None:
                continue
            if key in self._selected or family in item.applicable_families:
                allowed[key] = item
        return allowed

    def _selector_choices(self) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
        filtered = self._filtered_registry()
        ordered = self._ordered_keys(filtered)
        return ordered, _module_registry_meta(filtered)

    def _write_back(self) -> None:
        payload = {key: self._values[key] for key in self._selected if key in self._values}
        self._write_fn(json.dumps(payload, ensure_ascii=False))

    def load_from_raw(self, raw: str) -> None:
        """Reload editor from a JSON string (called by preset picker)."""
        values = self._parse_values(raw)
        self._values = values
        self._selected = [key for key in self._ordered_keys(self._registry) if key in values]
        self._selector.setCurrentValues(self._selected)
        self._rebuild_content()
        self._write_back()

    def _on_row_value_changed(self, item_key: str, value: float) -> None:
        self._values[item_key] = round(float(value), 6)
        self._write_back()

    def _rebuild_content(self) -> None:
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        selected_keys = [key for key in self._selected if key in self._registry]
        index_w = self._index_column_width()
        name_w = self._name_column_width()
        slider_min_w = self._slider_min_width(index_w, name_w)
        row_count = 0
        divider_count = 0

        for idx, key in enumerate(self._selected, start=1):
            item = self._registry.get(key)
            if item is None:
                continue
            row_count += 1
            current_value = float(self._values.get(key, item.default))
            row = _RegistryModuleRow(
                item=item,
                index_text=str(idx),
                index_width=index_w,
                name_width=name_w,
                slider_min_width=slider_min_w,
                current_value=current_value,
                write_value=lambda real, item_key=key: self._on_row_value_changed(item_key, real),
                parent=self._content,
            )
            self._content_layout.addWidget(row)

            if key != selected_keys[-1]:
                divider = QFrame(self._content)
                divider.setProperty("moduleDivider", True)
                self._content_layout.addWidget(divider)
                divider_count += 1

        if self._content_layout.count() == 0:
            empty = QLabel("No modules selected")
            empty.setProperty("moduleHeader", True)
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setFixedHeight(28)
            self._content_layout.addWidget(empty)
            content_h = 28
        else:
            content_h = row_count * 32 + divider_count

        self._content_layout.activate()
        wrap_margins = self._content_wrap.layout().contentsMargins()
        wrap_h = content_h + wrap_margins.top() + wrap_margins.bottom()
        self._content.setFixedHeight(content_h)
        self._content_wrap.setFixedHeight(wrap_h)

        root_layout = self.layout()
        if root_layout is not None:
            root_layout.activate()
            margins = root_layout.contentsMargins()
            total_h = (
                margins.top()
                + margins.bottom()
                + self._content_wrap.height()
                + self._selector.height()
                + root_layout.spacing()
            )
        else:
            total_h = wrap_h

        self._preferred_height = max(PARAM_ROW_H, int(total_h))
        self._preferred_width = max(WIDGET_W, self._content_target_width(index_w, name_w))
        self.setMinimumHeight(self._preferred_height)
        self.setMaximumHeight(self._preferred_height)
        self.updateGeometry()
        self.height_changed.emit(self._preferred_height)
        self.width_changed.emit(self._preferred_width)


# ---------------------------------------------------------------------------
# _make_spin_widget — ◀ value ▶ control replacing native spinbox arrows
# ---------------------------------------------------------------------------

def _make_spin_widget(
    min_v: int, max_v: int, step: int, initial: int, write_fn
) -> QWidget:
    """
    Return a horizontal ◀ [QSpinBox] ▶ widget.

    The native spinbox arrow buttons are hidden; the ◀ / ▶ QPushButtons
    decrement / increment by *step* each click and display as left/right arrows.
    """
    from PySide6.QtWidgets import QAbstractSpinBox

    container = QWidget()
    container.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    hl = QHBoxLayout(container)
    hl.setContentsMargins(0, 0, 0, 0)
    hl.setSpacing(2)

    btn_l = QPushButton("◀")
    btn_l.setFixedWidth(20)
    btn_l.setFixedHeight(20)
    btn_l.setCursor(Qt.CursorShape.PointingHandCursor)

    spin = QSpinBox()
    spin.setRange(min_v, max_v)
    spin.setSingleStep(step)
    spin.setValue(initial)
    spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
    spin.setAlignment(Qt.AlignmentFlag.AlignCenter)

    btn_r = QPushButton("▶")
    btn_r.setFixedWidth(20)
    btn_r.setFixedHeight(20)
    btn_r.setCursor(Qt.CursorShape.PointingHandCursor)

    btn_l.clicked.connect(lambda: spin.setValue(spin.value() - step))
    btn_r.clicked.connect(lambda: spin.setValue(spin.value() + step))
    spin.valueChanged.connect(lambda v: write_fn(str(v)))

    hl.addWidget(btn_l)
    hl.addWidget(spin, 1)
    hl.addWidget(btn_r)
    return container


# ---------------------------------------------------------------------------
# TrainingNodePort
# ---------------------------------------------------------------------------

class TrainingNodePort(QGraphicsEllipseItem):
    """
    Type-coloured sub_dot port for a Training Ground node.

    Maintains the full GraphScene port data() protocol so that
    _get_node_port / _create_connection work without modification.
    """

    def __init__(
        self,
        parent: "TrainingNodeItem",
        slot_name: str,
        io: str,
        data_type: str,
        required: bool = True,
    ) -> None:
        r = PORT_R
        super().__init__(-r, -r, r * 2, r * 2, parent)

        self._slot_name = slot_name
        self._io = io
        self._data_type = data_type
        self._required = required

        hex_c = _training_port_types().get(data_type, {}).get("color", get_color("training_port_fallback", "#9ca3af"))
        self._type_color = QColor(hex_c)

        # ── GraphScene port protocol ──────────────────────────────────
        self.setData(0, "port")
        self.setData(1, io)
        self.setData(2, [])
        self.setData(3, slot_name)
        self.setData(20, {
            "channel": "data",
            "data_type": data_type,
            "dot_kind": "sub_dot",
            "max_connections": 8 if io == "out" else 1,
            "type_color": hex_c,
        })

        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setZValue(4)  # above overlay (z=3) and proxies (z=2)

        self._apply_visual("normal")

    # ------------------------------------------------------------------
    # Visual state
    # ------------------------------------------------------------------

    def _apply_visual(self, state: str) -> None:
        c = QColor(self._type_color)

        if state == "normal":
            rim = QColor(c)
            rim.setAlpha(180 if self._required else 120)
            fill = QColor(c)
            fill.setAlpha(90 if self._required else 50)
            self.setPen(QPen(rim, 1.5))
            self.setBrush(QBrush(fill))

        elif state == "hover":
            self.setPen(QPen(QColor(get_color("training_port_hover_border", "#ffffff")), 2.0))
            c.setAlpha(230)
            self.setBrush(QBrush(c))

        elif state in ("connected", "active"):
            glow = c.lighter(160)
            self.setPen(QPen(glow, 2.0))
            c.setAlpha(200)
            self.setBrush(QBrush(c))

        elif state in ("valid",):
            self.setPen(QPen(QColor(get_color("training_port_valid_border", "#22c55e")), 3.0))
            self.setBrush(QBrush(QColor(get_color("training_port_valid_fill", "#0f2e1a"))))

        elif state in ("invalid", "incompatible"):
            gray = QColor(get_color("training_port_invalid_border", "#555555"))
            self.setPen(QPen(gray.lighter(110), 1.0))
            self.setBrush(QBrush(QColor(get_color("training_port_invalid_fill", "#2a2a2a"))))

    def hoverEnterEvent(self, event) -> None:
        self._apply_visual("hover")
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        conns = self.data(2)
        self._apply_visual("connected" if conns else "normal")
        super().hoverLeaveEvent(event)


# ---------------------------------------------------------------------------
# Dual-handle range slider (reusable)
# ---------------------------------------------------------------------------

class _DualHandleRangeSlider(QWidget):
    """
    Reusable dual-handle horizontal range slider.

    Paints a groove track with:
      - unfilled ends (slider_groove color)
      - highlighted range region between the two handles (slider_fill color)
      - two round handles (slider_handle color; active handle gets hover border)

    Layout hint:   [lo_label] [_DualHandleRangeSlider] [hi_label]
                                    ↑            ↑
                                lo handle     hi handle

    Signal:
        rangeChanged(float, float)  — emitted on every drag step
    """

    rangeChanged = Signal(float, float)

    _HANDLE_R = 4    # handle radius (px) — intentionally smaller than QSlider handle
    _TRACK_H  = 4    # groove height (px)
    _MARGIN   = 7    # left/right guard so handle circle stays inside widget

    def __init__(
        self,
        minimum: float,
        maximum: float,
        lo: float,
        hi: float,
        step: float = 0.01,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._min  = float(minimum)
        self._max  = float(maximum)
        self._step = max(float(step), 1e-12)
        self._lo   = self._snap(max(self._min, min(lo, self._max)))
        self._hi   = self._snap(max(self._lo,  min(hi, self._max)))

        self._dragging: Optional[str] = None   # "lo" | "hi"
        self._hovered:  Optional[str] = None   # "lo" | "hi"

        self.setFixedHeight(22)
        self.setMinimumWidth(60)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)

    # ── helpers ───────────────────────────────────────────────────────

    def _snap(self, v: float) -> float:
        return round(v / self._step) * self._step

    def _val_to_x(self, val: float) -> float:
        m = self._MARGIN
        span = self._max - self._min
        if span == 0:
            return float(m)
        frac = (val - self._min) / span
        return m + frac * (self.width() - 2 * m)

    def _x_to_val(self, x: float) -> float:
        m = self._MARGIN
        track_w = self.width() - 2 * m
        if track_w <= 0:
            return self._min
        frac = max(0.0, min(1.0, (x - m) / track_w))
        return self._snap(self._min + frac * (self._max - self._min))

    def _hit_handle(self, x: float) -> Optional[str]:
        """Return which handle (if any) is within click radius of *x*."""
        lo_x = self._val_to_x(self._lo)
        hi_x = self._val_to_x(self._hi)
        r = self._HANDLE_R + 5   # generous hit area despite small visual size
        d_lo = abs(x - lo_x)
        d_hi = abs(x - hi_x)
        in_lo = d_lo <= r
        in_hi = d_hi <= r
        if in_lo and in_hi:
            return "lo" if d_lo <= d_hi else "hi"
        if in_lo:
            return "lo"
        if in_hi:
            return "hi"
        return None

    # ── Qt events ─────────────────────────────────────────────────────

    def mousePressEvent(self, ev) -> None:
        self._dragging = self._hit_handle(ev.position().x())
        ev.accept()

    def mouseMoveEvent(self, ev) -> None:
        x = ev.position().x()
        if self._dragging == "lo":
            new_lo = min(self._x_to_val(x), self._hi)
            if new_lo != self._lo:
                self._lo = new_lo
                self.rangeChanged.emit(self._lo, self._hi)
                self.update()
        elif self._dragging == "hi":
            new_hi = max(self._x_to_val(x), self._lo)
            if new_hi != self._hi:
                self._hi = new_hi
                self.rangeChanged.emit(self._lo, self._hi)
                self.update()
        else:
            new_hov = self._hit_handle(x)
            if new_hov != self._hovered:
                self._hovered = new_hov
                self.update()
        ev.accept()

    def mouseReleaseEvent(self, ev) -> None:
        self._dragging = None
        ev.accept()

    def leaveEvent(self, ev) -> None:
        self._hovered = None
        self.update()
        super().leaveEvent(ev)

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w  = self.width()
        cy = self.height() / 2.0
        m  = self._MARGIN
        th = self._TRACK_H

        lo_x = self._val_to_x(self._lo)
        hi_x = self._val_to_x(self._hi)

        # ── groove (full track behind fill) ───────────────────────────
        groove = QColor(get_color("training_widget_slider_groove", "#303030"))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(groove))
        p.drawRoundedRect(
            QRectF(m, cy - th / 2, w - 2 * m, th),
            th / 2, th / 2,
        )

        # ── range fill (between handles) ──────────────────────────────
        fill = QColor(get_color("training_widget_slider_fill", "#F6D393"))
        p.setBrush(QBrush(fill))
        p.drawRoundedRect(
            QRectF(lo_x, cy - th / 2, max(0.0, hi_x - lo_x), th),
            th / 2, th / 2,
        )

        # ── handles ───────────────────────────────────────────────────
        r = float(self._HANDLE_R)
        for handle, hx in (("lo", lo_x), ("hi", hi_x)):
            active = (self._hovered == handle) or (self._dragging == handle)
            if active:
                col    = QColor(get_color("training_widget_slider_handle_hover", "#7a8490"))
                border = QColor(get_color("training_widget_slider_handle_hover_border", "#22c55e"))
                p.setPen(QPen(border, 2.0))
            else:
                col = QColor(get_color("training_widget_slider_handle", "#5a6470"))
                p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(col))
            p.drawEllipse(QPointF(hx, cy), r, r)

        p.end()

    # ── public API ────────────────────────────────────────────────────

    def set_values(self, lo: float, hi: float) -> None:
        """Programmatically set both handles (used by load_parameters)."""
        self._lo = self._snap(max(self._min, min(lo, self._max)))
        self._hi = self._snap(max(self._lo,  min(hi, self._max)))
        self.update()

    def lo_value(self) -> float:
        return self._lo

    def hi_value(self) -> float:
        return self._hi


# ---------------------------------------------------------------------------
# Robot-type row with MuJoCo asset-availability indicator
# ---------------------------------------------------------------------------

def _list_registered_robots() -> "List[str]":
    """Return model_ids of all registered robots that have MuJoCo asset rules.

    This is the canonical source of truth for the Robot Type dropdown.
    """
    try:
        from src.system.models.mujoco_asset_registry import registered_mujoco_asset_rules
        return sorted({rule.model_id for rule in registered_mujoco_asset_rules()})
    except Exception:
        return ["go2", "spot"]  # minimal fallback


def _query_robot_asset(robot_type: str):
    """
    Return (found: bool, source: str, scene_path: str) for *robot_type*.

    Looks up (brand_id, model_id) from the mujoco_asset_registry's registered
    rules, then resolves to an actual MJCF scene file.  Results are cached.
    """
    _cache = _query_robot_asset._cache  # type: ignore[attr-defined]
    if robot_type in _cache:
        return _cache[robot_type]
    try:
        from src.system.models.mujoco_asset_registry import (
            registered_mujoco_asset_rules,
            resolve_mujoco_asset,
        )
        # Build brand map dynamically from the canonical registry.
        brand_info = None
        for rule in registered_mujoco_asset_rules():
            if rule.model_id == robot_type:
                brand_info = (rule.brand_id, rule.model_id)
                break
        if brand_info:
            loc = resolve_mujoco_asset(brand_info[0], brand_info[1])
            if loc is not None:
                result = (True, loc.source, str(loc.scene_path))
            else:
                result = (False, "", "")
        else:
            result = (False, "", "")
    except Exception:
        result = (False, "", "")
    _cache[robot_type] = result
    return result

_query_robot_asset._cache = {}  # type: ignore[attr-defined]


def _make_robot_type_row(
    choices: list,
    current: str,
    write_fn,
) -> "QWidget":
    """
    Build a QWidget containing [dropdown  |  ✔/❌ indicator].

    The indicator is a small QLabel that:
      - shows ✔ (green) when resolve_mujoco_asset finds a scene for the
        selected robot_type, ❌ (red/muted) otherwise
      - has a tooltip describing the asset source and path on hover
      - updates immediately when the dropdown selection changes
    """
    from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel
    from PySide6.QtCore import Qt

    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)

    picker = _DropdownChoicePicker(choices, current)
    indicator = QLabel()
    indicator.setFixedWidth(20)
    indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
    indicator.setToolTipDuration(8000)

    def _update_indicator(robot_type: str) -> None:
        found, source, path = _query_robot_asset(robot_type)
        if found:
            indicator.setText("✔")
            source_label = (
                "mujoco_menagerie" if source == "mujoco_menagerie"
                else f"custom ({source})"
            )
            indicator.setToolTip(
                f"Asset found — {source_label}\n{path}"
            )
            indicator.setStyleSheet("color: #4CAF50; font-size: 11px;")
        else:
            indicator.setText("❌")
            indicator.setToolTip(
                f"No MuJoCo scene found for '{robot_type}'.\n"
                "Training will fall back to the built-in minimal scene; add a scene to\n"
                "runtime/simulation/mujoco/menagerie/<robot_dir>/ or set MJCF Overwrite."
            )
            indicator.setStyleSheet("color: #888; font-size: 11px;")

    def _on_change(val: str) -> None:
        write_fn(val)
        _update_indicator(val)

    picker.currentTextChanged.connect(_on_change)
    _update_indicator(current)

    layout.addWidget(picker, stretch=1)
    layout.addWidget(indicator, stretch=0)
    return container


def _list_export_checkpoint_ids() -> List[str]:
    try:
        entries = CheckpointRegistry().discover()
    except Exception:
        return []
    names = sorted(
        {
            str(entry.policy_id).strip()
            for entry in entries
            if getattr(entry, "is_valid", True) and str(entry.policy_id).strip()
        },
        key=str.lower,
    )
    return names


def _list_training_asset_entries() -> List[object]:
    try:
        from src.system.training.training_asset_registry import TrainingAssetRegistry

        return [
            entry for entry in TrainingAssetRegistry().list_assets()
            if getattr(entry, "is_valid", True) and str(getattr(entry, "asset_id", "") or "").strip()
        ]
    except Exception:
        return []


def _resolve_start_point_token(params: Dict[str, str]) -> str:
    start_point = str(params.get("start_point", "") or "").strip()
    if start_point:
        return start_point
    asset_id = str(params.get("asset_id", "") or "").strip()
    if asset_id:
        return f"asset:{asset_id}"
    return _StartPointChoicePicker.NEW_TOKEN


class _ExportBundlePicker(_RichChoicePicker):
    valueChanged = Signal(str)

    NEW_LABEL = "<NEW>"

    def __init__(self, current_value: str = "", parent=None):
        self._known_names = _list_export_checkpoint_ids()
        self._current_value = str(current_value or "").strip()
        if self._current_value == "trained_policy" and self._current_value not in self._known_names:
            self._current_value = ""
        self._suspend_events = False
        choices, meta_map = self._choice_state()
        super().__init__(
            choices,
            current=self._current_value or self.NEW_LABEL,
            meta_map=meta_map,
            leading_mode="checkbox",
            multi_select=False,
            parent=parent,
        )

    def current_value(self) -> str:
        return self._current_value

    def _choice_state(self) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
        items = [self.NEW_LABEL] + list(self._known_names)
        if self._current_value and self._current_value not in items:
            items.append(self._current_value)
        meta_map = {
            self.NEW_LABEL: {
                "title": self.NEW_LABEL,
                "subtitle": "Create and register a new checkpoint name.",
            }
        }
        for item in items:
            if item == self.NEW_LABEL:
                continue
            meta_map[item] = _default_choice_meta(item)
        return items, meta_map

    def _rebuild_items(self) -> None:
        items, meta_map = self._choice_state()
        self._suspend_events = False
        self.set_choices(items, meta_map=meta_map)
        self._restore_previous_selection()

    def _restore_previous_selection(self) -> None:
        self._suspend_events = True
        self.setCurrentText(self._current_value or self.NEW_LABEL)
        self._suspend_events = False

    def _set_current_value(self, value: str) -> None:
        value = str(value or "").strip()
        self._current_value = value
        if value and value not in self._known_names:
            self._known_names.append(value)
            self._known_names.sort(key=str.lower)
        self._rebuild_items()
        self.valueChanged.emit(value)

    def _on_popup_selection_changed(self, values: List[str]) -> None:
        if self._suspend_events:
            return
        text = values[0] if values else ""
        if text == self.NEW_LABEL:
            # Defer until popup is fully closed to avoid Qt Popup focus-loss interference.
            from PySide6.QtCore import QTimer
            QTimer.singleShot(0, self._run_new_name_dialog)
            return
        self._set_current_value(text)

    def _run_new_name_dialog(self) -> None:
        known = list(self._known_names)

        def _validate(name: str) -> Optional[str]:
            if "/" in name or "\\" in name:
                return "Cannot contain '/' or '\\'"
            if name in known:
                return f"'{name}' already exists"
            return None

        popup = _TextInputPopup(placeholder="Checkpoint name…")
        popup.set_validator(_validate)
        popup.accepted.connect(self._set_current_value)
        popup.destroyed.connect(
            lambda *_: self._restore_previous_selection() if not popup._did_accept else None
        )
        pos, row_h = self._text_anchor_global()
        popup.show_at(pos, row_h)


class _StartPointChoicePicker(_RichChoicePicker):
    valueChanged = Signal(str)

    NEW_TOKEN = "__new__"
    LATEST_EXPORT_TOKEN = "__latest_export__"

    def __init__(self, current_value: str = "", parent=None):
        self._current_value = str(current_value or "").strip() or self.NEW_TOKEN
        choices, meta_map = self._choice_state()
        super().__init__(
            choices,
            current=self._current_value if self._current_value in choices else self.NEW_TOKEN,
            meta_map=meta_map,
            leading_mode="checkbox",
            multi_select=False,
            choices_provider=self._choice_state,
            parent=parent,
        )

    def _choice_state(self) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
        choices = [self.NEW_TOKEN, self.LATEST_EXPORT_TOKEN]
        meta_map = {
            self.NEW_TOKEN: {
                "title": "New",
                "desc": "Start training from scratch with random initialisation.",
            },
            self.LATEST_EXPORT_TOKEN: {
                "title": "Latest Export",
                "desc": "Resume from the newest training artifact exported by this workspace.",
            },
        }
        for entry in _list_training_asset_entries():
            token = f"asset:{entry.asset_id}"
            choices.append(token)
            meta_map[token] = {
                "title": entry.label(),
                "desc": f"Use training asset '{entry.asset_id}' as the start point.",
            }
        return choices, meta_map

    def current_value(self) -> str:
        return self._current_value

    def _on_popup_selection_changed(self, values: List[str]) -> None:
        value = values[0] if values else self.NEW_TOKEN
        if value not in self._choices:
            value = self.NEW_TOKEN
        self._current_value = value
        self.setCurrentText(value)
        self.valueChanged.emit(value)


# ---------------------------------------------------------------------------
# Module preset helpers (shared by Rewards / Terminations nodes)
# ---------------------------------------------------------------------------

def _preset_file_path(kind: str) -> pathlib.Path:
    from src.system.core.utils.path_helper import get_project_root
    return pathlib.Path(get_project_root()) / "config" / f"{kind}_presets.json"


def _load_module_presets(kind: str) -> Dict[str, dict]:
    """Return {name: values_dict} for saved presets of *kind*."""
    try:
        with _preset_file_path(kind).open("r", encoding="utf-8") as _fp:
            data = json.load(_fp)
        return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, dict)}
    except Exception:
        return {}


def _save_module_preset(kind: str, name: str, values: dict) -> None:
    path = _preset_file_path(kind)
    presets = _load_module_presets(kind)
    presets[name] = dict(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as _fp:
        json.dump(presets, _fp, indent=2)


class SaveListButton(QWidget):
    """Preset dropdown + Save button for Rewards / Terminations nodes.

    Layout::

        [ ▾ dropdown button (shows selected preset name) ][ Save ]

    Behaviour:

    * Dropdown lists saved presets plus a ``<New>`` entry (default).
    * Selecting a preset loads it and updates the dropdown text.
    * ``<New>`` is selected by default — dropdown shows ``<New>``.
    * Clicking **Save** while ``<New>`` is active → inline ``DataInput``
      appears on the same row to name the new preset.  After confirmation
      the preset is written and the dropdown switches to the new name.
    * Clicking **Save** while a custom preset is selected → silently
      overwrites that preset on disk with the current configuration.
    """

    preset_loaded = Signal(str)   # emits JSON string of loaded values

    _NEW_LABEL = "<New>"

    def __init__(self, kind: str, get_current_fn: Callable[[], str], parent=None):
        super().__init__(parent)
        self._kind = kind
        self._get_current_fn = get_current_fn
        self._selected_preset: Optional[str] = None  # None ≡ <New>
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMaximumWidth(WIDGET_W)

        hl = QHBoxLayout(self)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(2)

        # ── dropdown (reuses _RichChoicePicker) ───────────────────────
        choices, meta = self._choice_state()
        self._picker = _RichChoicePicker(
            choices,
            current=self._NEW_LABEL,
            meta_map=meta,
            leading_mode="checkbox",
            multi_select=False,
            choices_provider=self._choice_state,
            parent=self,
        )
        self._picker.setMaximumWidth(WIDGET_W)
        self._picker.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._suspend = False

        # Override the base class handler to route through our logic
        try:
            self._picker.selectionChanged.disconnect()
        except RuntimeError:
            pass
        self._picker.selectionChanged.connect(self._on_selection)

        # ── save button ───────────────────────────────────────────────
        self._save_btn = QPushButton("Save", self)
        self._save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._save_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        self._save_btn.clicked.connect(self._on_save)

        hl.addWidget(self._picker, 1)
        hl.addWidget(self._save_btn, 0)

    # ── choice list builder ───────────────────────────────────────────

    def _choice_state(self) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
        presets = _load_module_presets(self._kind)
        items = [self._NEW_LABEL] + sorted(presets.keys(), key=str.lower)
        meta: Dict[str, Dict[str, str]] = {
            self._NEW_LABEL: {"title": self._NEW_LABEL},
        }
        for name, vals in presets.items():
            n = len(vals) if isinstance(vals, dict) else 0
            meta[name] = {"title": name, "subtitle": f"{n} term(s)"}
        return items, meta

    # ── selection handling ────────────────────────────────────────────

    def _on_selection(self, values: List[str]) -> None:
        if self._suspend:
            return
        text = values[0] if values else ""
        if text == self._NEW_LABEL:
            self._selected_preset = None
            return
        # Load the chosen preset
        presets = _load_module_presets(self._kind)
        if text in presets:
            self._selected_preset = text
            try:
                self.preset_loaded.emit(json.dumps(presets[text]))
            except Exception:
                pass

    # ── save handling ─────────────────────────────────────────────────

    def _on_save(self) -> None:
        raw = self._get_current_fn()
        try:
            values: dict = json.loads(raw) if raw else {}
        except Exception:
            values = {}
        if not values:
            return

        if self._selected_preset is not None:
            # Overwrite existing preset
            _save_module_preset(self._kind, self._selected_preset, values)
            return

        # <New> is active — ask for a name via _TextInputPopup
        def _validate(name: str) -> Optional[str]:
            if "/" in name or "\\" in name:
                return "Cannot contain '/' or '\\'"
            return None

        kind = self._kind

        def _on_accepted(name: str) -> None:
            _save_module_preset(kind, name, values)
            # Switch dropdown to the newly saved preset
            self._selected_preset = name
            self._suspend = True
            choices, meta = self._choice_state()
            self._picker.set_choices(choices, meta_map=meta)
            if self._picker.findText(name) >= 0:
                self._picker.setCurrentText(name)
            self._suspend = False

        popup = _TextInputPopup(placeholder="Preset name…")
        popup.set_validator(_validate)
        popup.accepted.connect(_on_accepted)
        pos, row_h = self._picker._text_anchor_global()
        popup.show_at(pos, row_h)


# Legacy alias for internal references
_ModulePresetPicker = SaveListButton


# ---------------------------------------------------------------------------
# Buffer Size — AUTO / Manual toggle
# ---------------------------------------------------------------------------

class _BufferSizePickerWidget(QWidget):
    """
    Toggle widget for SAC/TD3 replay buffer size.

    Single-row layout: [AUTO/Manual toggle]  [value label or plain QLineEdit]

    • AUTO   — read-only label shows total_timesteps ÷ 10 rounded to 10 k.
    • Manual — plain QLineEdit (no arrow buttons) next to the toggle.
               Width is trimmed so toggle + field never exceeds WIDGET_W.

    Writes both ``buffer_size_mode`` and ``buffer_size`` to node parameters
    on every state change so the backend always has both up to date.
    """

    # btn_w + spacing + field leaves room in WIDGET_W (134 px)
    _BTN_W   = 52
    _SPACING = 4
    _FIELD_W = WIDGET_W - _BTN_W - _SPACING   # ≈ 78 px

    def __init__(
        self,
        get_all_params: Callable[[], dict],
        write_params_fn: Callable[[dict], None],
        row: dict,
        parent=None,
    ):
        super().__init__(parent)
        self._get_all_params = get_all_params
        self._write_params = write_params_fn
        self._step_v = int(row.get("step", 10_000))
        self._min_v  = int(row.get("min",  10_000))
        self._max_v  = int(row.get("max",  10_000_000))

        params = self._get_all_params()
        raw_mode = str(params.get("buffer_size_mode", "auto")).strip().lower()
        self._mode = raw_mode if raw_mode in ("auto", "manual") else "auto"
        try:
            self._manual_value = int(float(params.get("buffer_size", "1000000")))
        except (ValueError, TypeError):
            self._manual_value = 1_000_000

        row_h = max(20, PARAM_ROW_H - 4)

        hl = QHBoxLayout(self)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(self._SPACING)

        # ── Toggle button ────────────────────────────────────────────
        self._toggle_btn = QPushButton()
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.setFixedWidth(self._BTN_W)
        self._toggle_btn.setFixedHeight(row_h)
        self._toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle_btn.clicked.connect(self._on_toggle_clicked)
        hl.addWidget(self._toggle_btn, 0)

        # ── AUTO read-only label ──────────────────────────────────────
        self._auto_label = QLabel()
        self._auto_label.setFixedWidth(self._FIELD_W)
        self._auto_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        hl.addWidget(self._auto_label, 0)

        # ── Manual plain text field (no arrows) ───────────────────────
        from PySide6.QtGui import QIntValidator
        self._manual_edit = QLineEdit()
        self._manual_edit.setFixedWidth(self._FIELD_W)
        self._manual_edit.setFixedHeight(row_h)
        self._manual_edit.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._manual_edit.setValidator(
            QIntValidator(self._min_v, self._max_v, self._manual_edit)
        )
        self._manual_edit.setText(str(self._manual_value))
        self._manual_edit.editingFinished.connect(self._on_manual_edited)
        hl.addWidget(self._manual_edit, 0)

        self.setFixedWidth(WIDGET_W)
        self._apply_mode(emit=False)

    # ------------------------------------------------------------------

    def _compute_auto_value(self) -> int:
        try:
            total = int(float(self._get_all_params().get("total_timesteps", "1000000")))
        except (ValueError, TypeError):
            total = 1_000_000
        raw = max(self._min_v, total // 10)
        rounded = round(raw / self._step_v) * self._step_v
        return min(self._max_v, max(self._min_v, rounded))

    @staticmethod
    def _fmt(n: int) -> str:
        return f"{n:,}"

    def _apply_mode(self, emit: bool = True) -> None:
        is_auto = self._mode == "auto"
        self._toggle_btn.setChecked(is_auto)
        self._toggle_btn.setText("AUTO" if is_auto else "Manual")
        self._auto_label.setVisible(is_auto)
        self._manual_edit.setVisible(not is_auto)
        if is_auto:
            val = self._compute_auto_value()
            self._auto_label.setText(self._fmt(val))
            if emit:
                self._write_params({"buffer_size_mode": "auto", "buffer_size": str(val)})
        else:
            if emit:
                self._write_params({
                    "buffer_size_mode": "manual",
                    "buffer_size": str(self._manual_value),
                })

    def _on_toggle_clicked(self, checked: bool) -> None:
        if not checked and self._mode == "auto":
            # AUTO → Manual: seed field with the current computed value
            computed = self._compute_auto_value()
            self._manual_value = computed
            self._manual_edit.setText(str(computed))
        self._mode = "auto" if checked else "manual"
        self._apply_mode()

    def _on_manual_edited(self) -> None:
        text = self._manual_edit.text().strip()
        try:
            val = max(self._min_v, min(self._max_v, int(text)))
        except (ValueError, TypeError):
            val = self._manual_value
        self._manual_value = val
        self._manual_edit.setText(str(val))
        self._write_params({"buffer_size_mode": "manual", "buffer_size": str(val)})


# ---------------------------------------------------------------------------
# Reference Motion — specialised widgets
# ---------------------------------------------------------------------------

class _MotionLibraryPicker(_RichChoicePicker):
    """
    Rich-choice picker for the reference motion library.

    Shows:
      • <Add>  — browse & import a new .npy into custom_motions/
      • ★ name — files from the same robot model (pinned first)
      •   name — other files from the same category

    Emits ``valueChanged(str)`` with the full absolute path, or ``""`` when
    <Add> is pending / nothing is selected.
    """

    valueChanged = Signal(str)
    ADD_LABEL = "<Add>"

    def __init__(
        self,
        current_path: str,
        robot_type: str = "",
        parent=None,
    ) -> None:
        self._current_path = str(current_path or "")
        self._robot_type = str(robot_type or "").lower().strip()
        self._suspend = False

        choices, meta, c2p, p2c = self._build_state()
        self._choice_to_path = c2p
        self._path_to_choice = p2c

        initial = p2c.get(self._current_path, self.ADD_LABEL)

        super().__init__(
            choices,
            current=initial,
            meta_map=meta,
            leading_mode="checkbox",
            multi_select=False,
            parent=parent,
        )
        self.currentTextChanged.connect(self._on_changed)

    # ------------------------------------------------------------------
    # State builder
    # ------------------------------------------------------------------

    def _build_state(self):
        """Return (choices, meta_map, choice→path, path→choice)."""
        from src.system.training.motion_library import list_entries, get_category

        cat = get_category(self._robot_type) if self._robot_type else None
        entries = list_entries(category=cat)

        # Sort: same model first, then category / model / name
        rt = self._robot_type
        entries_sorted = sorted(
            entries,
            key=lambda e: (0 if e.robot_model == rt else 1,
                           e.category, e.robot_model, e.name.lower()),
        )

        # Detect duplicate names so we can add [model] disambiguation
        name_count = Counter(e.name for e in entries_sorted)

        choices = [self.ADD_LABEL]
        meta: Dict[str, Dict[str, str]] = {
            self.ADD_LABEL: {
                "title": self.ADD_LABEL,
                "desc": "Browse for a .npy file and import it into the motion library.",
            }
        }
        c2p: Dict[str, str] = {}
        p2c: Dict[str, str] = {}

        for e in entries_sorted:
            if name_count[e.name] > 1:
                label = f"{e.name}  [{e.robot_model}]"
            else:
                label = e.name

            star = "★ " if e.robot_model == rt else ""
            meta[label] = {
                "title": f"{star}{e.name}",
                "desc": f"{e.category}  ·  {e.robot_model}",
            }
            full = str(e.path)
            c2p[label] = full
            p2c[full] = label
            choices.append(label)

        return choices, meta, c2p, p2c

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_changed(self, label: str) -> None:
        if self._suspend:
            return
        if label == self.ADD_LABEL:
            self._do_add()
            return
        path = self._choice_to_path.get(label, "")
        if path:
            self._current_path = path
            self.valueChanged.emit(path)

    def _do_add(self) -> None:
        from src.system.training.motion_library import import_file
        import pathlib as _pl

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Reference Motion",
            "",
            "NumPy Files (*.npy);;All Files (*)",
        )
        if not path:
            self._revert()
            return

        try:
            entry = import_file(
                _pl.Path(path),
                robot_type=self._robot_type or "generic",
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Import Failed",
                f"Could not import motion file:\n{exc}",
            )
            self._revert()
            return

        # Rebuild and select new entry
        choices, meta, c2p, p2c = self._build_state()
        self._choice_to_path = c2p
        self._path_to_choice = p2c
        self._suspend = True
        self.set_choices(choices, meta_map=meta)
        self._suspend = False

        new_label = p2c.get(str(entry.path), "")
        if new_label:
            self._current_path = str(entry.path)
            self.setCurrentText(new_label)
            self.valueChanged.emit(self._current_path)
        else:
            self._revert()

    def _revert(self) -> None:
        """Restore the previously selected entry without emitting."""
        self._suspend = True
        saved = self._path_to_choice.get(self._current_path, self.ADD_LABEL)
        self.setCurrentText(saved)
        self._suspend = False


class _MotionFilePickerWidget(QWidget):
    """Read-only filename label + Browse button + shape/fps info label."""

    valueChanged = Signal(str)

    def __init__(self, current_path: str, write_fn: Callable, parent=None) -> None:
        super().__init__(parent)
        self._path = str(current_path or "")
        self._write_fn = write_fn
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        vl = QVBoxLayout(self)
        vl.setContentsMargins(0, 2, 0, 2)
        vl.setSpacing(2)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)

        self._name_edit = QLineEdit()
        self._name_edit.setReadOnly(True)
        self._name_edit.setPlaceholderText("no file selected")
        self._name_edit.setMinimumWidth(0)

        self._browse_btn = QPushButton("…")
        self._browse_btn.setFixedWidth(26)
        self._browse_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._browse_btn.setToolTip("Browse for a .npy reference motion file")
        self._browse_btn.clicked.connect(self._on_browse)

        row.addWidget(self._name_edit, 1)
        row.addWidget(self._browse_btn, 0)

        self._info_lbl = QLabel("")
        self._info_lbl.setObjectName("motionFileInfo")
        self._info_lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        vl.addLayout(row)
        vl.addWidget(self._info_lbl)

        self._update_display(self._path)

    # ------------------------------------------------------------------

    def preferred_row_height(self) -> int:
        return 46

    def _on_browse(self) -> None:
        start_dir = str(pathlib.Path(self._path).parent) if self._path else ""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Reference Motion File",
            start_dir,
            "NumPy Files (*.npy);;All Files (*)",
        )
        if path:
            self._update_display(path)
            self._write_fn(path)
            self.valueChanged.emit(path)

    def _update_display(self, path: str) -> None:
        self._path = path
        if path:
            self._name_edit.setText(pathlib.Path(path).name)
            self._load_info(path)
        else:
            self._name_edit.clear()
            self._info_lbl.clear()

    def _load_info(self, path: str) -> None:
        try:
            import numpy as np
            arr = np.load(path, mmap_mode="r")
            if arr.ndim == 2:
                T, J = arr.shape
                self._info_lbl.setText(f"{T} frames  ·  {J} joints")
            else:
                self._info_lbl.setText(f"shape: {list(arr.shape)}")
        except Exception:
            self._info_lbl.setText("(could not read file)")


class _FPSSliderWidget(QWidget):
    """Integer FPS slider (0 – 200) with an Auto button that reads the npy file."""

    valueChanged = Signal(str)

    _MAX_FPS = 200

    def __init__(self, current: str, logic_node: Any, write_fn: Callable, parent=None) -> None:
        super().__init__(parent)
        self._logic_node = logic_node
        self._write_fn = write_fn
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        try:
            init = min(self._MAX_FPS, max(0, round(float(current))))
        except (ValueError, TypeError):
            init = 0

        hl = QHBoxLayout(self)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(4)

        self._node_slider = NodeSlider(
            mode="standard",
            minimum=0.0, maximum=float(self._MAX_FPS),
            step=1.0, decimals=0,
            current=float(init),
        )
        self._node_slider.valueChanged.connect(self._on_slide)

        self._auto_btn = QPushButton("Auto")
        self._auto_btn.setFixedWidth(36)
        self._auto_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._auto_btn.setToolTip(
            "Read the motion file and project control rate to suggest a matching FPS.\n"
            "Sets FPS = control_frequency_hz (50 Hz default for Go2),\n"
            "ensuring the trajectory plays at its native speed."
        )
        self._auto_btn.clicked.connect(self._on_auto)

        hl.addWidget(self._node_slider, 1)
        hl.addWidget(self._auto_btn, 0)

    # ------------------------------------------------------------------

    def _on_slide(self, val: float) -> None:
        v = int(round(val))
        self._write_fn(str(float(v)))
        self.valueChanged.emit(str(float(v)))

    def _on_auto(self) -> None:
        params = getattr(self._logic_node, "parameters", None) or {}
        path = str(params.get("motion_file", "") or "")
        if not path:
            self._auto_btn.setToolTip("⚠ Set a Motion File first, then click Auto.")
            return

        try:
            import numpy as np
            arr = np.load(path, mmap_mode="r")
            T = int(arr.shape[0])
        except Exception:
            self._auto_btn.setToolTip(f"⚠ Could not read file: {pathlib.Path(path).name}")
            return

        control_hz = 50.0
        for key in ("control_hz", "control_frequency_hz"):
            raw = params.get(key)
            if raw is not None:
                try:
                    control_hz = float(raw)
                    break
                except (TypeError, ValueError):
                    pass

        suggested = min(self._MAX_FPS, max(1, round(control_hz)))
        duration_s = T / suggested if suggested > 0 else 0.0

        self._auto_btn.setToolTip(
            f"File: {pathlib.Path(path).name}\n"
            f"Frames: {T}  ·  Control rate: {control_hz:.0f} Hz\n"
            f"Suggested FPS: {suggested}  →  plays in {duration_s:.2f} s"
        )
        self._node_slider.set_value(float(suggested))
        self._write_fn(str(float(suggested)))
        self.valueChanged.emit(str(float(suggested)))


# ---------------------------------------------------------------------------
# Motion Preview — thread + button widget
# ---------------------------------------------------------------------------

def _find_scene_xml_for_preview(robot_type: str) -> Optional[str]:
    """Return the best flat-scene XML path for a MuJoCo preview of *robot_type*."""
    try:
        from src.system.training.unitree_gym_env import (
            _resolve_registered_scene_xml,
            _find_go2_scene_xml,
        )
        xml = _resolve_registered_scene_xml(str(robot_type or "go2").lower(), "flat")
        if xml:
            return xml
        return _find_go2_scene_xml("flat")
    except Exception:
        return None


class _MotionPreviewThread(QThread):
    """Background thread: physics-based reference motion preview with HUD.

    Runs a live MuJoCo simulation where a PD controller tracks the
    reference joint targets each control step — exactly what the RL
    agent will be asked to do during training.  A text overlay shows
    real-time base velocity, tracking error, and phase so the user can
    verify the motion is physically plausible and matches their intended
    task commands before committing to a multi-hour training run.
    """

    stopped = Signal()
    error   = Signal(str)

    # PD gains matching Unitree Go2 hardware defaults and Isaac Gym standard.
    _KP = 70.0   # Nm/rad
    _KD = 1.5    # Nm·s/rad

    def __init__(
        self,
        scene_xml: str,
        npy_path: str,
        fps: float,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._scene_xml  = scene_xml
        self._npy_path   = npy_path
        self._fps        = max(1.0, float(fps) if fps > 0 else 30.0)
        self._stop_flag  = False

    def request_stop(self) -> None:
        self._stop_flag = True

    def run(self) -> None:
        import time
        try:
            import mujoco
            import mujoco.viewer as _viewer
            import numpy as np
        except ImportError as exc:
            self.error.emit(f"MuJoCo not available: {exc}")
            self.stopped.emit()
            return

        try:
            frames = np.load(self._npy_path)
            if frames.ndim != 2:
                self.error.emit(f"Expected 2-D array, got shape {frames.shape}")
                self.stopped.emit()
                return

            model = mujoco.MjModel.from_xml_path(self._scene_xml)
            data  = mujoco.MjData(model)

            # ── Detect free-joint and actuated DOFs ──────────────────
            jnt_offset = 0
            for i in range(model.njnt):
                if model.joint(i).type == mujoco.mjtJoint.mjJNT_FREE:
                    jnt_offset = 7
                    break

            n_actuated = model.nq - jnt_offset
            n_act = min(n_actuated, model.nu)
            n_cols = int(frames.shape[1])

            if n_cols >= jnt_offset + n_actuated:
                joint_frames = frames[:, jnt_offset: jnt_offset + n_actuated].copy()
            elif n_cols >= n_actuated:
                joint_frames = frames[:, :n_actuated].copy()
            else:
                joint_frames = frames.copy()

            n_frames = int(joint_frames.shape[0])
            n_joints = min(int(joint_frames.shape[1]), n_act)

            # Precompute reference joint velocities (finite-difference)
            ref_vels = np.zeros_like(joint_frames)
            if n_frames > 1:
                dt_ref = 1.0 / self._fps
                ref_vels[:-1] = (joint_frames[1:] - joint_frames[:-1]) / dt_ref
                ref_vels[-1] = ref_vels[-2]

            # Per-actuator gear ratio for ctrl → torque conversion
            raw_gear = np.asarray(model.actuator_gear[:n_act, 0], dtype=np.float32)
            gear = np.where(raw_gear > 0.0, raw_gear, 1.0)

            # Build qpos→ctrl permutation index.
            # qpos joints and ctrl actuators may be in different order
            # (e.g. Go2: qpos=FL,FR,RL,RR but ctrl=FR,FL,RR,RL).
            # For each actuator i, find which qpos joint index it drives.
            qpos_idx_for_ctrl = np.arange(n_act, dtype=np.intp)  # identity fallback
            for aid in range(n_act):
                trnid = int(model.actuator_trnid[aid, 0])
                qpos_adr = int(model.jnt_qposadr[trnid])
                qpos_idx_for_ctrl[aid] = qpos_adr - jnt_offset  # relative to joint block

            # ── Simulation timing ────────────────────────────────────
            sim_dt = model.opt.timestep
            control_dt = 1.0 / self._fps
            decimation = max(1, int(round(control_dt / sim_dt)))

            # ── Initial pose: first reference frame ──────────────────
            mujoco.mj_resetData(model, data)
            if jnt_offset >= 7:
                data.qpos[2] = 0.35                          # standing height
                data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]       # upright quat
            n = min(n_joints, model.nq - jnt_offset)
            data.qpos[jnt_offset: jnt_offset + n] = joint_frames[0, :n]
            mujoco.mj_forward(model, data)

            frame_idx = 0
            wall_start = time.monotonic()
            prev_base_pos = np.array(data.qpos[:3], dtype=np.float64)

            with _viewer.launch_passive(model, data) as viewer:
                viewer.cam.distance  = 2.5
                viewer.cam.elevation = -20.0

                while viewer.is_running() and not self._stop_flag:
                    ref_idx = frame_idx % n_frames
                    q_ref = joint_frames[ref_idx, :n_joints]
                    v_ref = ref_vels[ref_idx, :n_joints]

                    # ── PD control per actuator ───────────────────
                    # q_ref / v_ref are in qpos order.  For each actuator
                    # we pick the matching qpos joint via qpos_idx_for_ctrl.
                    q_cur_all = np.asarray(data.qpos[jnt_offset: jnt_offset + n_actuated], dtype=np.float32)
                    vel_offset = 6 if jnt_offset == 7 else jnt_offset
                    v_cur_all = np.asarray(data.qvel[vel_offset: vel_offset + n_actuated], dtype=np.float32)
                    for ai in range(n_act):
                        qi = int(qpos_idx_for_ctrl[ai])
                        if qi < n_joints:
                            err_p = q_ref[qi] - q_cur_all[qi]
                            err_v = v_ref[qi] - v_cur_all[qi]
                        else:
                            err_p = 0.0
                            err_v = 0.0
                        data.ctrl[ai] = (self._KP * err_p + self._KD * err_v) / float(gear[ai])

                    # ── Step physics ─────────────────────────────────
                    for _ in range(decimation):
                        mujoco.mj_step(model, data)

                    # ── Measure base velocity ────────────────────────
                    base_pos = np.array(data.qpos[:3], dtype=np.float64)
                    base_vel = (base_pos - prev_base_pos) / control_dt
                    prev_base_pos = base_pos.copy()

                    base_ang_vel = data.qvel[3:6] if model.nv >= 6 else [0, 0, 0]

                    # Joint tracking error (rad RMS)
                    q_cur_subset = q_cur_all[:n_joints]
                    track_err = float(np.sqrt(np.mean((q_ref - q_cur_subset) ** 2)))

                    # Base height
                    base_z = float(data.qpos[2]) if model.nq >= 3 else 0.0

                    # ── HUD overlay ──────────────────────────────────
                    # set_texts format: (font, gridpos, text1, text2)
                    # font=None → default, gridpos=None → top-left
                    try:
                        viewer.set_texts([
                            (None, None, "Mode", "Physics PD Tracking"),
                            (None, None, "Phase", f"{ref_idx}/{n_frames} ({100*ref_idx/n_frames:.0f}%)"),
                            (None, None, "Base Vx", f"{base_vel[0]:+.3f} m/s"),
                            (None, None, "Base Vy", f"{base_vel[1]:+.3f} m/s"),
                            (None, None, "Yaw rate", f"{float(base_ang_vel[2]):+.3f} rad/s"),
                            (None, None, "Height", f"{base_z:.3f} m"),
                            (None, None, "Track err", f"{track_err:.4f} rad RMS"),
                            (None, None, "FPS", f"{self._fps:.0f} Hz"),
                        ])
                    except Exception:
                        pass  # HUD not critical — don't crash the preview

                    viewer.sync()
                    frame_idx += 1

                    # Real-time pacing
                    expected = wall_start + frame_idx * control_dt
                    sleep_t = expected - time.monotonic()
                    if sleep_t > 0:
                        time.sleep(sleep_t)

        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.stopped.emit()


class _PreviewButtonWidget(QWidget):
    """▶ Preview / ■ Stop toggle button for the ReferenceMotionNode."""

    def __init__(
        self,
        get_npy_path: Callable[[], str],
        get_fps: Callable[[], float],
        get_robot_type: Callable[[], str],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._get_npy        = get_npy_path
        self._get_fps        = get_fps
        self._get_robot_type = get_robot_type
        self._thread: Optional[_MotionPreviewThread] = None

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        hl = QHBoxLayout(self)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(0)

        self._btn = QPushButton("▶  Preview")
        self._btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn.setToolTip(
            "Open a MuJoCo physics simulation with PD tracking.\n"
            "Shows real-time base velocity, tracking error, and height\n"
            "to verify the motion is physically plausible."
        )
        self._btn.clicked.connect(self._toggle)
        hl.addWidget(self._btn)

    # ------------------------------------------------------------------

    def _toggle(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            self._thread.request_stop()
            return

        npy = self._get_npy()
        if not npy:
            self._btn.setToolTip("⚠ Select a motion file first, then click Preview.")
            return

        xml = _find_scene_xml_for_preview(self._get_robot_type())
        if not xml:
            self._btn.setToolTip("⚠ Could not locate a MuJoCo scene XML for this robot type.")
            return

        fps = self._get_fps()
        self._thread = _MotionPreviewThread(xml, npy, fps)
        self._thread.stopped.connect(self._on_stopped)
        self._thread.error.connect(self._on_error)
        self._thread.start()

        self._btn.setText("■  Stop")
        self._btn.setToolTip("Close the MuJoCo viewer.")

    def _on_stopped(self) -> None:
        self._btn.setText("▶  Preview")
        self._btn.setToolTip(
            "Open a MuJoCo physics simulation with PD tracking.\n"
            "Shows real-time base velocity, tracking error, and height\n"
            "to verify the motion is physically plausible."
        )

    def _on_error(self, msg: str) -> None:
        self._btn.setText("▶  Preview")
        self._btn.setToolTip(f"⚠ Preview failed: {msg}")


# ---------------------------------------------------------------------------
# TrainingNodeItem
# ---------------------------------------------------------------------------

class TrainingNodeItem(QGraphicsRectItem):
    """
    Full Training Ground node card with interactive param widgets.

    Visual layout (top → bottom):
    ┌─ TITLE BAR ──────────────────────────────────────────┐
    │ ● in_slot  (PORT_ROW_H per visible input port)       │
    │ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ │
    │ Label  [Widget]   (ui_row.row_height, default 24 px) │
    │ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ │
    │                  out_slot ●  (PORT_ROW_H per output) │
    └──────────────────────────────────────────────────────┘

    Param rows with a ``condition`` field are shown/hidden dynamically
    when the referenced master widget value changes.
    """

    def __init__(self, node_id: str, node_type: str, display_name: str) -> None:
        super().__init__(0, 0, NODE_W, TITLE_H)

        self._node_id = node_id
        self._node_type = node_type
        self._display_name = display_name
        self._logic_node = None
        self._runtime_scene_xml = ""
        self._robot_type_hint = ""
        self._node_width = NODE_W

        # Refs used by the Preview button to read sibling widget values
        self._ref_motion_picker_widget: Optional[_MotionLibraryPicker] = None
        self._ref_fps_slider_widget: Optional[_FPSSliderWidget] = None
        # Refs used by preset pickers to refresh the module editor
        self._reward_editor_widget: Optional[_RegistryModuleEditor] = None
        self._termination_editor_widget: Optional[_RegistryModuleEditor] = None

        self._style: Dict[str, str] = _style_for_node(node_type)

        # Row state
        self._input_order: List[str] = []
        self._output_order: List[str] = []
        self._optional_inputs: set = set()
        self._hidden_input_slots: set = set()

        # Port registry: "slot:io" → TrainingNodePort
        self._ports: Dict[str, TrainingNodePort] = {}

        # Overlay items for disabled NodeRow rows (z=3, above proxies, below ports)
        self._row_overlays: List[QGraphicsRectItem] = []

        # Param proxy registry: [(ui_row, QGraphicsProxyWidget, row_height_px)]
        self._param_proxies: List[Tuple[dict, QGraphicsProxyWidget, int]] = []

        # Paint-time cache for label positions: [(ui_row, y_start, row_h)]
        self._param_row_y_cache: List[Tuple[dict, float, int]] = []

        # ── GraphScene node protocol ──────────────────────────────────
        self.setData(10, node_id)
        self.setData(11, display_name)
        self.setData(12, node_type)
        self.setData(13, None)

        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.setZValue(0)

        self.setPen(QPen(QColor(get_color("training_node_border", get_color("border", "#2d2d2d"))), 1))
        self.setBrush(QBrush(QColor(self._style["bg"])))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def attach_logic_node(self, logic_node) -> None:
        """Bind a logic node and rebuild all port items + proxy widgets."""
        self._logic_node = logic_node
        self.setData(13, logic_node)
        self._rebuild_geometry()

    def apply_theme(self) -> None:
        """Refresh node card, ports, and embedded widgets from ui.ini colors."""
        self._style = _style_for_node(self._node_type)
        self.setPen(QPen(QColor(get_color("training_node_border", get_color("border", "#2d2d2d"))), 1))
        self.setBrush(QBrush(QColor(self._style["bg"])))
        for port in self._ports.values():
            hex_c = _training_port_types().get(port._data_type, {}).get("color", get_color("training_port_fallback", "#9ca3af"))
            port._type_color = QColor(hex_c)
            port._apply_visual("connected" if (port.data(2) or []) else "normal")
        for _row, proxy, _rh in self._param_proxies:
            widget = proxy.widget()
            if widget is not None:
                extra = widget.extra_widget_style() if hasattr(widget, "extra_widget_style") else ""
                widget.setStyleSheet(_widget_style() + extra)
        self.update()

    def get_port(self, slot_name: str, io: str) -> Optional[TrainingNodePort]:
        """Return the port item for the given slot name and direction."""
        return self._ports.get(f"{slot_name}:{io}")

    # ------------------------------------------------------------------
    # TrainNode double-click trigger
    # ------------------------------------------------------------------

    def mouseDoubleClickEvent(self, event) -> None:
        """Double-clicking a TrainNode emits train_requested on the scene."""
        if self._node_type == "train":
            scene = self.scene()
            if scene is not None and hasattr(scene, "train_requested"):
                scene.train_requested.emit(self._build_job_spec())
            event.accept()
            return
        try:
            super().mouseDoubleClickEvent(event)
        except TypeError:
            pass

    def _build_job_spec(self) -> dict:
        """
        Build the job-spec dict for train_requested by reading AlgorithmConfigNode
        parameters reachable through the algo_config input port connection.
        Falls back to sensible defaults when the port is unconnected.
        """
        spec: dict = {
            "policy_id_out": "trained_policy",
            "total_timesteps": 1_000_000,
            "algorithm": "PPO",
        }
        algo_port = self._ports.get("algo_config:in")
        if algo_port:
            for conn in (algo_port.data(2) or []):
                try:
                    algo_logic = conn.out_port.parentItem().data(13)
                    if algo_logic is not None:
                        p = algo_logic.parameters or {}
                        spec["policy_id_out"] = str(
                            p.get("policy_id_out", spec["policy_id_out"]))
                        spec["total_timesteps"] = int(
                            p.get("total_timesteps", spec["total_timesteps"]))
                        spec["algorithm"] = str(
                            p.get("algorithm", spec["algorithm"]))
                    break
                except Exception:
                    pass
        return spec

    def _resolve_canvas_robot_type(self) -> str:
        if self._logic_node is not None and self._node_type == "robot_mjcf":
            return str((self._logic_node.parameters or {}).get("robot_type", "")).strip()
        scene = self.scene()
        if scene is None:
            return ""
        for item in scene.items():
            if item is self:
                continue
            try:
                if item.data(12) != "robot_mjcf":
                    continue
                logic = item.data(13)
                params = getattr(logic, "parameters", {}) or {}
                robot_type = str(params.get("robot_type", "")).strip()
                if robot_type:
                    return robot_type
            except Exception:
                continue
        return ""

    def _resolve_canvas_robot_family(self) -> str:
        return resolve_robot_family(self._resolve_canvas_robot_type())

    # ------------------------------------------------------------------
    # Serialization API  (Step 3)
    # ------------------------------------------------------------------

    def get_parameters(self) -> dict:
        """Read-only snapshot of logic_node.parameters."""
        if self._logic_node is None:
            return {}
        return dict(self._logic_node.parameters or {})

    def load_parameters(self, params: dict) -> None:
        """Push saved parameter values back into the node, then rebuild proxy widgets."""
        if self._logic_node is None or not isinstance(params, dict):
            return
        if self._logic_node.parameters is None:
            self._logic_node.parameters = {}
        self._logic_node.parameters.update(params)
        self._rebuild_geometry(rebuild_ports=False)

    def set_runtime_scene_xml(self, runtime_scene_xml: str) -> None:
        self._runtime_scene_xml = str(runtime_scene_xml or "").strip()
        if self._node_type == "scene_config" and self._logic_node is not None:
            self._rebuild_geometry(rebuild_ports=False)

    def set_robot_type_hint(self, robot_type: str) -> None:
        """Hint the current robot type so the motion library picker can filter."""
        self._robot_type_hint = str(robot_type or "").lower().strip()
        if self._node_type == "reference_motion" and self._logic_node is not None:
            self._rebuild_geometry(rebuild_ports=False)

    def serialize(self) -> dict:
        """Return a JSON-serializable snapshot of this node."""
        p = self.pos()
        return {
            "id": self._node_id,
            "node_type": self._node_type,
            "display_name": self._display_name,
            "pos": [p.x(), p.y()],
            "parameters": self.get_parameters(),
        }

    @staticmethod
    def deserialize(data: dict, scene) -> "TrainingNodeItem":
        """Reconstruct a TrainingNodeItem from serialize() output.

        Uses ``display_name`` (e.g. "Algorithm Config") — not the internal
        ``node_type`` key (e.g. "algo_config") — because TrainingGraphScene.
        create_node() keys on display names.
        """
        from PySide6.QtCore import QPointF
        display_name = data.get("display_name", data["node_type"])
        x, y = data.get("pos", [0.0, 0.0])
        item = scene.create_node(display_name, QPointF(x, y))
        if item is not None:
            item.load_parameters(data.get("parameters", {}))
        return item

    # ------------------------------------------------------------------
    # Geometry rebuild
    # ------------------------------------------------------------------

    def _clear_param_proxies(self) -> None:
        for _row, proxy, _rh in self._param_proxies:
            try:
                widget = proxy.widget()
            except Exception:
                widget = None
            if widget is not None:
                try:
                    widget.deleteLater()
                except Exception:
                    pass
            try:
                proxy.setParentItem(None)
            except Exception:
                pass
        self._param_proxies.clear()
        self._param_row_y_cache.clear()

    def _rebuild_geometry(self, rebuild_ports: bool = True) -> None:
        """Rebuild node geometry; keep existing ports when only params changed."""
        if rebuild_ports:
            for child in list(self.childItems()):
                child.setParentItem(None)
            self._ports.clear()
            self._clear_param_proxies()
        else:
            self._clear_param_proxies()

        if self._logic_node is None:
            return

        hidden: set = getattr(self._logic_node, "_HIDDEN_PORTS", set())
        optional: set = getattr(self._logic_node, "_OPTIONAL_INPUTS", set())
        self._optional_inputs = optional

        all_inputs = self._logic_node.inputs or {}
        visible_inputs = {k: v for k, v in all_inputs.items() if k not in hidden}
        outputs = self._logic_node.outputs or {}

        self._input_order = list(visible_inputs.keys())
        self._output_order = list(outputs.keys())
        # Only create hidden ports that are merged into NodeRow param rows
        _noderow_slots: set = set()
        if self._node_type == "algo_config":
            _noderow_slots = {"total_steps"}
        self._hidden_input_slots = (set(hidden) & set(all_inputs.keys())) & _noderow_slots

        # Port type map: slot_name → data_type (may differ for aliased ports like stage_*)
        _port_types = {"in": {}, "out": {}}
        try:
            _port_types = self._logic_node.get_port_types()
        except Exception:
            pass
        _in_types  = _port_types.get("in",  {}) or {}
        _out_types = _port_types.get("out", {}) or {}

        # Include hidden input ports in expected keys so they are created
        expected_port_keys = {
            *(f"{slot_name}:in" for slot_name in list(visible_inputs.keys()) + list(self._hidden_input_slots)),
            *(f"{slot_name}:out" for slot_name in self._output_order),
        }
        if not rebuild_ports and set(self._ports.keys()) != set(expected_port_keys):
            self._rebuild_geometry(rebuild_ports=True)
            return

        # ── Input port items (visible rows) ───────────────────────────
        for slot_name in self._input_order:
            req = slot_name not in optional
            key = f"{slot_name}:in"
            data_type = _in_types.get(slot_name, slot_name)
            port = self._ports.get(key)
            if port is None:
                port = TrainingNodePort(self, slot_name, "in", data_type, required=req)
                self._ports[key] = port
            else:
                port._required = req
                port._apply_visual("connected" if (port.data(2) or []) else "normal")

        # ── Hidden input port items (merged into NodeRow param rows) ──
        for slot_name in self._hidden_input_slots:
            key = f"{slot_name}:in"
            data_type = _in_types.get(slot_name, slot_name)
            req = slot_name not in optional
            port = self._ports.get(key)
            if port is None:
                port = TrainingNodePort(self, slot_name, "in", data_type, required=req)
                self._ports[key] = port
            else:
                port._required = req
                port._apply_visual("connected" if (port.data(2) or []) else "normal")

        # ── Output port items ─────────────────────────────────────────
        for slot_name in self._output_order:
            key = f"{slot_name}:out"
            data_type = _out_types.get(slot_name, slot_name)
            port = self._ports.get(key)
            if port is None:
                port = TrainingNodePort(self, slot_name, "out", data_type, required=True)
                self._ports[key] = port
            else:
                port._required = True
                port._apply_visual("connected" if (port.data(2) or []) else "normal")

        # ── Param proxy widgets ───────────────────────────────────────
        # Map: param_key → input_slot_name for NodeRow wrapping.
        # When a param key appears here, its widget is wrapped in a NodeRow
        # with the specified input_slot, merging the port dot into the param row.
        _noderow_input_map: Dict[str, str] = {}
        if self._node_type == "algo_config":
            _noderow_input_map["total_timesteps"] = "total_steps"

        ui_rows = NODE_UI_ROWS.get(self._node_type, [])
        for row in ui_rows:
            if row.get("kind", "param") != "param":
                continue
            row_h = int(row.get("row_height", PARAM_ROW_H))
            widget = self._make_param_widget(row)
            if widget is not None:
                # Wrap in NodeRow if this param has a merged input port
                param_key = row.get("key", "")
                input_slot = _noderow_input_map.get(param_key)
                if input_slot is not None:
                    display = row.get("display_name", param_key)
                    nr = NodeRow(
                        input_slot=input_slot,
                        title=display,
                        widget=widget,
                    )
                    nr._full_row_widget = True  # take full width so input_zone sits at left edge
                    row = dict(row)  # copy — mark full_width so paint skips external label
                    row["full_width_widget"] = True
                    widget = nr

                extra = widget.extra_widget_style() if hasattr(widget, "extra_widget_style") else ""
                widget.setStyleSheet(_widget_style() + extra)
                proxy = QGraphicsProxyWidget(self)
                proxy.setWidget(widget)
                proxy.setZValue(2)
                if hasattr(widget, "preferred_row_height"):
                    try:
                        row_h = max(row_h, int(widget.preferred_row_height()))
                    except Exception:
                        pass
                self._param_proxies.append((row, proxy, row_h))
                if hasattr(widget, "height_changed"):
                    widget.height_changed.connect(
                        lambda new_h, p=proxy: self._update_proxy_row_height(p, new_h)
                    )
                if hasattr(widget, "width_changed"):
                    widget.width_changed.connect(
                        lambda _new_w, p=proxy: self._update_proxy_row_width(p)
                    )

        # Initial condition pass + layout
        self._refresh_conditions(reflow=False)
        self._reflow_layout()

    # ------------------------------------------------------------------
    # Widget factory
    # ------------------------------------------------------------------

    def _make_param_widget(self, row: dict) -> Optional[QWidget]:
        """Create and return a QWidget for a single ui_row."""
        if row.get("kind", "param") != "param":
            return None
        key = row["key"]
        w_type = row.get("widget", "text_input")
        logic = self._logic_node

        def get_val() -> str:
            return str((logic.parameters or {}).get(key, str(row.get("default", ""))))

        def write_val(v: str, _key: str = key) -> None:
            if logic is not None and logic.parameters is not None:
                logic.parameters[_key] = v
            self._refresh_conditions()
            self._notify_scene_param_changed(_key, v)

        def write_params(updates: Dict[str, str]) -> None:
            if logic is not None and logic.parameters is not None:
                logic.parameters.update(updates)
            self._refresh_conditions()

        # ── motion_library_picker ─────────────────────────────────────
        if w_type == "motion_library_picker":
            w = _MotionLibraryPicker(
                get_val(),
                robot_type=self._robot_type_hint,
            )
            w.valueChanged.connect(write_val)
            self._ref_motion_picker_widget = w
            return w

        # ── motion_file_picker ────────────────────────────────────────
        if w_type == "motion_file_picker":
            w = _MotionFilePickerWidget(get_val(), write_val)
            w.valueChanged.connect(write_val)
            return w

        # ── fps_slider ────────────────────────────────────────────────
        if w_type == "fps_slider":
            w = _FPSSliderWidget(get_val(), logic, write_val)
            w.valueChanged.connect(write_val)
            self._ref_fps_slider_widget = w
            return w

        # ── preview_button ────────────────────────────────────────────
        if w_type == "preview_button":
            def _get_npy() -> str:
                # Source 1: explicit .npy from the library picker
                p = self._ref_motion_picker_widget
                if p is not None and p._current_path:
                    return p._current_path

                # Source 2: resolve motion_source (dropdown) at preview time
                # This handles "generate:walk", "generate:standing", "loco:..." etc.
                if self._logic_node is not None:
                    params = self._logic_node.parameters or {}
                    # Check motion_file param directly (may have been set by picker
                    # but picker widget not yet created during deserialise)
                    mf = str(params.get("motion_file", "") or "").strip()
                    if mf and pathlib.Path(mf).is_file():
                        return mf
                    ms = str(params.get("motion_source", "") or "").strip()
                    if ms:
                        try:
                            from src.system.nodes.sys_nodes.training_nodes import (
                                ReferenceMotionNode,
                            )
                            resolved = ReferenceMotionNode._resolve_motion_source(ms)
                            if resolved and pathlib.Path(resolved).is_file():
                                return resolved
                        except Exception:
                            pass
                return ""

            def _get_fps() -> float:
                s = self._ref_fps_slider_widget
                if s is None:
                    return 30.0
                try:
                    return float(s._slider.value()) or 30.0
                except Exception:
                    return 30.0

            return _PreviewButtonWidget(_get_npy, _get_fps,
                                        lambda: self._robot_type_hint)

        # ── start_point_picker ───────────────────────────────────────
        if w_type == "start_point_picker":
            token = _resolve_start_point_token(logic.parameters or {})
            if str((logic.parameters or {}).get("start_point", "") or "").strip() != token:
                write_params({"start_point": token})
            w = _StartPointChoicePicker(token)

            def _on_start_point_changed(choice: str) -> None:
                choice = str(choice or "").strip() or _StartPointChoicePicker.NEW_TOKEN
                current_load_mode = str((logic.parameters or {}).get("load_mode", "scratch") or "scratch").strip()
                if choice == _StartPointChoicePicker.NEW_TOKEN:
                    write_params({
                        "start_point": choice,
                        "asset_id": "",
                        "checkpoint_file": "",
                        "load_mode": "scratch",
                    })
                    return

                if choice == _StartPointChoicePicker.LATEST_EXPORT_TOKEN:
                    load_mode = current_load_mode if current_load_mode != "scratch" else "resume_sb3"
                    write_params({
                        "start_point": choice,
                        "asset_id": "",
                        "checkpoint_file": "",
                        "load_mode": load_mode,
                    })
                    return

                asset_id = choice.split(":", 1)[1] if choice.startswith("asset:") else ""
                checkpoint_file = ""
                load_mode = current_load_mode
                try:
                    from src.system.training.training_asset_registry import TrainingAssetRegistry

                    entry = TrainingAssetRegistry().get(asset_id)
                    checkpoint_file = str(getattr(entry, "primary_checkpoint", "") or "")
                    if load_mode == "scratch":
                        load_mode = "resume_sb3" if getattr(entry, "framework", "") == "sb3" else "warm_start_actor"
                except Exception:
                    if load_mode == "scratch":
                        load_mode = "resume_sb3"

                write_params({
                    "start_point": choice,
                    "asset_id": asset_id,
                    "checkpoint_file": checkpoint_file,
                    "load_mode": load_mode,
                })

            w.valueChanged.connect(_on_start_point_changed)
            return w

        # ── module_preset_picker ─────────────────────────────────────
        if w_type == "module_preset_picker":
            kind = str(row.get("registry", "rewards")).strip().lower()
            data_key = str(row.get("data_key", "reward_terms")).strip()

            def _get_module_data(_dk: str = data_key) -> str:
                return str((logic.parameters or {}).get(_dk, "{}"))

            w = _ModulePresetPicker(kind, _get_module_data)

            def _on_preset_loaded(json_str: str, _dk: str = data_key, _kind: str = kind) -> None:
                write_params({_dk: json_str})
                editor_ref = (
                    self._reward_editor_widget if _kind == "rewards"
                    else self._termination_editor_widget
                )
                if editor_ref is not None:
                    editor_ref.load_from_raw(json_str)

            w.preset_loaded.connect(_on_preset_loaded)
            return w

        # ── text_input ────────────────────────────────────────────────
        if w_type == "text_input":
            w = QLineEdit()
            w.setPlaceholderText(str(row.get("placeholder", "")))
            w.setText(get_val())
            w.textChanged.connect(write_val)
            return w

        # ── int_spinbox ───────────────────────────────────────────────
        if w_type == "int_spinbox":
            try:
                initial = int(float(get_val()))
            except (ValueError, TypeError):
                initial = int(row.get("default", 0))
            return _make_spin_widget(
                min_v=int(row.get("min", 0)),
                max_v=int(row.get("max", 999_999)),
                step=int(row.get("step", 1)),
                initial=initial,
                write_fn=write_val,
            )

        # ── float_input ───────────────────────────────────────────────
        if w_type == "float_input":
            w = QLineEdit()
            validator = QDoubleValidator(
                float(row.get("min", -1e9)),
                float(row.get("max", 1e9)),
                int(row.get("decimals", 6)),
            )
            w.setValidator(validator)
            w.setText(get_val())
            w.textChanged.connect(write_val)
            return w

        # ── scientific_input ──────────────────────────────────────────
        if w_type == "scientific_input":
            w = QLineEdit()
            w.setPlaceholderText("e.g. 3e-4")
            w.setText(get_val())
            w.textChanged.connect(write_val)
            return w

        # ── export_bundle_picker ─────────────────────────────────────
        if w_type == "export_bundle_picker":
            w = _ExportBundlePicker(get_val())
            if not w.current_value() and get_val():
                write_val("")
            w.valueChanged.connect(write_val)
            return w

        # ── dropdown ──────────────────────────────────────────────────
        if w_type == "dropdown":
            # phase_mode on ReferenceMotionNode — rich descriptions
            if key == "phase_mode" and self._node_type == "reference_motion":
                meta_map = {
                    "loop": {
                        "title": "Loop",
                        "desc": (
                            "Cycle through frames endlessly. Best for periodic gaits "
                            "like walking or trotting."
                        ),
                    },
                    "once": {
                        "title": "Once",
                        "desc": (
                            "Play through frames once, then hold the last frame. "
                            "Best for one-shot motions like stand-up or jump."
                        ),
                    },
                }
                w = _RichChoicePicker(
                    ["loop", "once"],
                    current=get_val(),
                    meta_map=meta_map,
                    leading_mode="checkbox",
                    multi_select=False,
                )
                w.currentTextChanged.connect(write_val)
                return w

            if key == "task_type":
                w = _TaskTypePicker(
                    [str(choice) for choice in row.get("choices", [])],
                    get_val(),
                )
                w.currentTextChanged.connect(write_val)
                return w

            # robot_type on RobotMJCFNode — dynamic from mujoco_asset_registry
            if key == "robot_type" and self._node_type == "robot_mjcf":
                return _make_robot_type_row(
                    choices=_list_registered_robots(),
                    current=get_val(),
                    write_fn=write_val,
                )

            if key == "scene_type" and self._node_type == "scene_config":
                w = _SceneTypePicker(
                    current_scene_type=get_val(),
                    current_scene_path=str((logic.parameters or {}).get("custom_scene_path", "")),
                    runtime_scene_xml=self._runtime_scene_xml,
                )

                def _on_scene_type_changed(value: str) -> None:
                    write_val(value)
                    if w.currentText() == "runtime_scene":
                        write_val(
                            w.runtime_scene_xml() if w.runtime_scene_type() == "custom" else "",
                            "custom_scene_path",
                        )
                    elif value != "custom":
                        write_val("", "custom_scene_path")

                w.selectionValueChanged.connect(_on_scene_type_changed)
                return w

            w = _DropdownChoicePicker(
                [str(choice) for choice in row.get("choices", [])],
                get_val(),
            )
            w.currentTextChanged.connect(write_val)
            return w

        # ── toggle ────────────────────────────────────────────────────
        if w_type == "toggle":
            w = QPushButton()
            w.setCheckable(True)
            is_on = get_val().lower() in ("true", "1", "yes")
            w.setChecked(is_on)
            w.setText("ON" if is_on else "OFF")

            def _on_toggle(checked: bool, btn: QPushButton = w, fn=write_val) -> None:
                btn.setText("ON" if checked else "OFF")
                fn("true" if checked else "false")

            w.toggled.connect(_on_toggle)
            return w

        # ── slider_float ──────────────────────────────────────────────
        if w_type == "slider_float":
            min_v = float(row.get("min", 0.0))
            max_v = float(row.get("max", 1.0))
            step  = float(row.get("step", 0.01))
            dec   = int(row.get("decimals", 2))
            try:
                cur_v = float(get_val())
            except (ValueError, TypeError):
                cur_v = min_v

            ns = NodeSlider(
                mode="standard",
                minimum=min_v, maximum=max_v, step=step, decimals=dec,
                current=cur_v,
            )

            def _on_sf(v: float, fn=write_val) -> None:
                fn(str(round(v, 6)))

            ns.valueChanged.connect(_on_sf)
            return ns

        # ── range_pair ────────────────────────────────────────────────
        if w_type == "range_pair":
            container = QWidget()
            container.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
            hl = QHBoxLayout(container)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.setSpacing(2)

            rng = row.get("range", [0.0, 1.0])
            step = float(row.get("step", 0.1))
            decimals = int(row.get("decimals", 2))

            lo = QDoubleSpinBox()
            hi = QDoubleSpinBox()
            for sb in (lo, hi):
                sb.setRange(float(rng[0]), float(rng[1]))
                sb.setSingleStep(step)
                sb.setDecimals(decimals)
                sb.setMaximumWidth(58)

            try:
                parts = json.loads(str(get_val()))
                lo.setValue(float(parts[0]))
                hi.setValue(float(parts[1]))
            except Exception:
                lo.setValue(float(rng[0]))
                hi.setValue(float(rng[1]))

            dash = QLabel("–")
            dash.setMaximumWidth(10)

            def _on_range(lo_: QDoubleSpinBox = lo, hi_: QDoubleSpinBox = hi,
                          fn=write_val) -> None:
                fn(f"[{lo_.value()}, {hi_.value()}]")

            lo.valueChanged.connect(lambda v: _on_range())
            hi.valueChanged.connect(lambda v: _on_range())

            hl.addWidget(lo, 1)
            hl.addWidget(dash, 0)
            hl.addWidget(hi, 1)
            return container

        # ── range_slider ──────────────────────────────────────────────
        # Full-row layout (240 px): [name | lo_val | slider | hi_val]
        # _full_row_widget=True tells TrainingNodeItem to skip drawing the
        # external label and give the widget the full row width.
        if w_type == "range_slider":
            rng          = row.get("range", [0.0, 1.0])
            step         = float(row.get("step", 0.01))
            decimals     = int(row.get("decimals", 2))
            display_name = row.get("display_name", row.get("key", ""))

            try:
                parts = json.loads(str(get_val()))
                lo_v  = float(parts[0])
                hi_v  = float(parts[1])
            except Exception:
                lo_v = float(rng[0])
                hi_v = float(rng[1])

            container = QWidget()
            container.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
            container._full_row_widget = True

            hl = QHBoxLayout(container)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.setSpacing(2)

            name_lbl = QLabel(display_name)
            name_lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            name_lbl.setFixedWidth(88)

            ns = NodeSlider(
                mode="range",
                minimum=float(rng[0]), maximum=float(rng[1]),
                step=step, decimals=decimals,
                lo=lo_v, hi=hi_v,
            )

            def _on_rs(lo_: float, hi_: float,
                       dec: int = decimals, fn=write_val) -> None:
                fn(f"[{round(lo_, dec + 2)}, {round(hi_, dec + 2)}]")

            ns.rangeChanged.connect(_on_rs)

            hl.addWidget(name_lbl, 0)
            hl.addWidget(ns, 1)
            return container

        # ── timestep_input ────────────────────────────────────────────
        if w_type == "timestep_input":
            try:
                initial = int(float(get_val()))
            except (ValueError, TypeError):
                initial = int(row.get("default", 0))
            return _make_spin_widget(
                min_v=int(row.get("min", 0)),
                max_v=int(row.get("max", 100_000_000)),
                step=int(row.get("step", 10_000)),
                initial=initial,
                write_fn=write_val,
            )

        # ── buffer_size_picker ────────────────────────────────────────
        if w_type == "buffer_size_picker":
            def _get_all_params() -> dict:
                return dict(logic.parameters or {})

            return _BufferSizePickerWidget(
                get_all_params=_get_all_params,
                write_params_fn=write_params,
                row=row,
            )

        # ── tag_list_input ────────────────────────────────────────────
        if w_type == "tag_list_input":
            w = QLineEdit()
            w.setPlaceholderText(str(row.get("placeholder", "e.g. 256 256")))
            # Convert "[256, 256]" → "256 256" for display
            cur = get_val()
            try:
                items = json.loads(cur)
                if isinstance(items, list):
                    cur = " ".join(str(x) for x in items)
            except Exception:
                pass
            w.setText(cur)

            def _on_taglist(text: str, fn=write_val) -> None:
                parts = text.replace(",", " ").split()
                try:
                    fn(str([int(p) for p in parts if p]))
                except ValueError:
                    fn(text)

            w.textChanged.connect(_on_taglist)
            return w

        # ── multiselect_tags ──────────────────────────────────────────
        if w_type == "multiselect_tags":
            if key == "obs_components":
                w = _ObsComponentsPicker(
                    [str(option) for option in row.get("options", [])],
                    get_val().split(),
                )
                w.selectionTextChanged.connect(write_val)
                return w
            options: List[str] = [str(o) for o in row.get("options", [])]
            cur_set = set(get_val().split())
            btn = QPushButton(f"{len(cur_set & set(options))} selected — Edit…")

            def _on_multiselect(
                checked: bool = False,
                btn_: QPushButton = btn,
                opts: List[str] = options,
                fn=write_val,
                _row: dict = row,
                _logic=logic,
                _key: str = key,
            ) -> None:
                dialog = QDialog()
                dialog.setWindowTitle(_row.get("display_name", "Select"))
                dialog.setModal(True)
                vl = QVBoxLayout(dialog)

                list_w = QListWidget()
                list_w.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
                existing = set(str((_logic.parameters or {}).get(_key, "")).split())
                for opt in opts:
                    item = QListWidgetItem(opt)
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(
                        Qt.CheckState.Checked if opt in existing
                        else Qt.CheckState.Unchecked
                    )
                    list_w.addItem(item)
                vl.addWidget(list_w)

                buttons = QDialogButtonBox(
                    QDialogButtonBox.StandardButton.Ok |
                    QDialogButtonBox.StandardButton.Cancel
                )
                buttons.accepted.connect(dialog.accept)
                buttons.rejected.connect(dialog.reject)
                vl.addWidget(buttons)
                dialog.resize(200, 220)

                if dialog.exec() == QDialog.DialogCode.Accepted:
                    selected = [
                        list_w.item(i).text()
                        for i in range(list_w.count())
                        if list_w.item(i).checkState() == Qt.CheckState.Checked
                    ]
                    fn(" ".join(selected))
                    btn_.setText(f"{len(selected)} selected — Edit…")

            btn.clicked.connect(_on_multiselect)
            return btn

        # ── json_kv_editor ────────────────────────────────────────────
        if w_type == "json_kv_editor":
            try:
                init_d = json.loads(get_val())
            except Exception:
                init_d = {}
            btn = QPushButton(f"{len(init_d)} terms — Edit…")

            def _on_kv_edit(
                checked: bool = False,
                btn_: QPushButton = btn,
                fn=write_val,
                _row: dict = row,
                _logic=logic,
                _key: str = key,
            ) -> None:
                dialog = QDialog()
                dialog.setWindowTitle(_row.get("display_name", "Edit"))
                dialog.setModal(True)
                vl = QVBoxLayout(dialog)

                table = QTableWidget()
                table.setColumnCount(2)
                table.setHorizontalHeaderLabels(["Key", "Value"])
                table.horizontalHeader().setStretchLastSection(True)
                try:
                    data = json.loads(str((_logic.parameters or {}).get(_key, "{}")))
                except Exception:
                    data = {}
                table.setRowCount(len(data))
                for r_i, (k_v, v_v) in enumerate(data.items()):
                    table.setItem(r_i, 0, QTableWidgetItem(str(k_v)))
                    table.setItem(r_i, 1, QTableWidgetItem(str(v_v)))

                btn_row = QWidget()
                btn_hl = QHBoxLayout(btn_row)
                btn_hl.setContentsMargins(0, 0, 0, 0)
                add_btn = QPushButton("+ Add")
                rem_btn = QPushButton("– Remove")
                add_btn.clicked.connect(lambda: table.insertRow(table.rowCount()))
                rem_btn.clicked.connect(lambda: table.removeRow(table.currentRow()))
                btn_hl.addWidget(add_btn)
                btn_hl.addWidget(rem_btn)
                btn_hl.addStretch()
                vl.addWidget(table)
                vl.addWidget(btn_row)

                buttons = QDialogButtonBox(
                    QDialogButtonBox.StandardButton.Ok |
                    QDialogButtonBox.StandardButton.Cancel
                )
                buttons.accepted.connect(dialog.accept)
                buttons.rejected.connect(dialog.reject)
                vl.addWidget(buttons)
                dialog.resize(300, 250)

                if dialog.exec() == QDialog.DialogCode.Accepted:
                    result: dict = {}
                    for r_i in range(table.rowCount()):
                        k_item = table.item(r_i, 0)
                        v_item = table.item(r_i, 1)
                        if k_item and k_item.text():
                            try:
                                result[k_item.text()] = float(v_item.text()) if v_item else 0.0
                            except (ValueError, TypeError):
                                result[k_item.text()] = v_item.text() if v_item else ""
                    fn(json.dumps(result))
                    btn_.setText(f"{len(result)} terms — Edit…")

            btn.clicked.connect(_on_kv_edit)
            return btn

        # ── module_registry_editor ────────────────────────────────────
        if w_type == "module_registry_editor":
            registry_kind = str(row.get("registry", "")).strip().lower()
            registry = reward_registry() if registry_kind == "rewards" else termination_registry()
            editor = _RegistryModuleEditor(
                registry,
                get_val(),
                write_val,
                selector_title=str(row.get("selector_title", "Title")),
                sort_mode=str(row.get("sort_mode", "title_asc")),
                family_provider=self._resolve_canvas_robot_family,
            )
            if registry_kind == "rewards":
                self._reward_editor_widget = editor
            elif registry_kind == "terminations":
                self._termination_editor_widget = editor
            return editor

        # ── json_editor ───────────────────────────────────────────────
        if w_type == "json_editor":
            btn = QPushButton("{…} — Edit…")

            def _on_json_edit(
                checked: bool = False,
                btn_: QPushButton = btn,
                fn=write_val,
                _row: dict = row,
                _logic=logic,
                _key: str = key,
            ) -> None:
                dialog = QDialog()
                dialog.setWindowTitle(_row.get("display_name", "Edit JSON"))
                dialog.setModal(True)
                vl = QVBoxLayout(dialog)

                editor = QPlainTextEdit()
                raw = str((_logic.parameters or {}).get(_key, "{}"))
                try:
                    editor.setPlainText(json.dumps(json.loads(raw), indent=2))
                except Exception:
                    editor.setPlainText(raw)
                vl.addWidget(editor)

                status_lbl = QLabel("")
                vl.addWidget(status_lbl)

                buttons = QDialogButtonBox(
                    QDialogButtonBox.StandardButton.Ok |
                    QDialogButtonBox.StandardButton.Cancel
                )

                def _try_accept(ed=editor, lbl=status_lbl, d=dialog) -> None:
                    text = ed.toPlainText()
                    try:
                        json.loads(text)
                        d.accept()
                    except json.JSONDecodeError as exc:
                        lbl.setText(f"Invalid JSON: {exc}")

                buttons.accepted.connect(_try_accept)
                buttons.rejected.connect(dialog.reject)
                vl.addWidget(buttons)
                dialog.resize(320, 280)

                if dialog.exec() == QDialog.DialogCode.Accepted:
                    fn(editor.toPlainText())

            btn.clicked.connect(_on_json_edit)
            return btn

        # ── path_browse ───────────────────────────────────────────────
        if w_type == "path_browse":
            container = QWidget()
            container.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
            hl = QHBoxLayout(container)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.setSpacing(2)

            path_edit = QLineEdit()
            path_edit.setReadOnly(True)
            path_edit.setPlaceholderText("(none)")
            path_edit.setText(get_val())

            browse_btn = QPushButton("…")
            browse_btn.setFixedWidth(24)

            mode = row.get("mode", "file")
            filt = row.get("filter", "")

            def _on_browse(
                checked: bool = False,
                edit: QLineEdit = path_edit,
                fn=write_val,
                m: str = mode,
                f: str = filt,
            ) -> None:
                if m == "dir":
                    path = QFileDialog.getExistingDirectory(None, "Select Directory")
                else:
                    path, _ = QFileDialog.getOpenFileName(None, "Select File", "", f)
                if path:
                    edit.setText(path)
                    fn(path)

            browse_btn.clicked.connect(_on_browse)
            path_edit.textChanged.connect(write_val)
            hl.addWidget(path_edit, 1)
            hl.addWidget(browse_btn, 0)
            return container

        # ── action_button ────────────────────────────────────────────
        if w_type == "action_button":
            btn = QPushButton(str(row.get("button_text", "Action")))
            btn.setCursor(Qt.PointingHandCursor)
            btn.setToolTip(str(row.get("tooltip", "") or ""))
            action_name = str(row.get("button_action", "") or "").strip()

            def _on_action(_checked: bool = False, action: str = action_name) -> None:
                scene = self.scene()
                if scene is None:
                    return
                if action == "review_setup" and hasattr(scene, "review_requested"):
                    scene.review_requested.emit(self._build_job_spec())
                if action == "review_export" and hasattr(scene, "export_review_requested"):
                    scene.export_review_requested.emit(self._build_job_spec())
                if action == "scene_preview" and hasattr(scene, "scene_preview_requested"):
                    scene.scene_preview_requested.emit(self.get_parameters())
                if action == "init_pose_preview" and hasattr(scene, "init_pose_preview_requested"):
                    scene.init_pose_preview_requested.emit(self.get_parameters())

            btn.clicked.connect(_on_action)
            return btn

        # Unknown widget type — fall back to plain text input
        w = QLineEdit()
        w.setText(get_val())
        w.textChanged.connect(write_val)
        return w

    # ------------------------------------------------------------------
    # Condition handling
    # ------------------------------------------------------------------

    def _refresh_conditions(self, reflow: bool = True) -> None:
        """Show/hide proxy widgets based on their condition field."""
        if self._logic_node is None:
            return
        params = self._logic_node.parameters or {}
        changed = False

        for row, proxy, _rh in self._param_proxies:
            cond = row.get("condition")
            if cond is None:
                if not proxy.isVisible():
                    proxy.setVisible(True)
                    changed = True
                continue

            cond_key = cond["key"]
            cond_op = cond.get("op", "==")
            cur = str(params.get(cond_key, ""))

            if cond_op == "==":
                visible = cur == str(cond.get("value", ""))
            elif cond_op == "in":
                visible = cur in [str(v) for v in cond.get("values", [])]
            elif cond_op == "!=":
                visible = cur != str(cond.get("value", ""))
            else:
                visible = True

            if proxy.isVisible() != visible:
                proxy.setVisible(visible)
                changed = True

        if reflow and changed:
            self._reflow_layout()

        # Also refresh port-driven param states (e.g. total_steps → disable widget)
        self._refresh_port_driven_params()

    def _notify_scene_param_changed(self, key: str, value: str) -> None:
        """Bubble parameter writes up to TrainingGraphScene.node_param_changed."""
        scene = self.scene()
        if scene is not None and hasattr(scene, "_on_node_param_changed"):
            scene._on_node_param_changed(self._node_type, key, str(value))

    def _refresh_port_driven_params(self) -> None:
        """Disable / re-enable param widgets whose NodeRow input_zone is connected.

        Scans all param proxies for NodeRow instances.  When a NodeRow's
        input port is connected, its function_zone is disabled and displays
        the upstream value.  When disconnected, it re-enables.
        Also handles legacy non-NodeRow widgets for backward compatibility.
        """
        # ── Generic NodeRow scan ──────────────────────────────────────
        from src.system.core.logger import log_debug
        for _row, proxy, _rh in self._param_proxies:
            widget = proxy.widget()
            if not isinstance(widget, NodeRow) or not widget.has_input_zone():
                continue
            port = self._ports.get(f"{widget.input_slot}:in")
            if port is None:
                log_debug(f"[NodeRow] Port '{widget.input_slot}:in' not found in ports")
                continue

            conns = port.data(2) or []
            connected = bool(conns)
            log_debug(f"[NodeRow] slot={widget.input_slot}, connected={connected}, conns={len(conns)}")

            upstream_value: Optional[str] = None
            if connected:
                try:
                    conn = conns[0]
                    upstream_item = conn.out_port.parentItem()
                    upstream_logic = upstream_item.data(13) if upstream_item else None
                    if upstream_logic is not None:
                        # Try known computation methods
                        if hasattr(upstream_logic, "compute_recommended_total_steps"):
                            v = upstream_logic.compute_recommended_total_steps()
                            if v and int(v) > 0:
                                upstream_value = str(int(v))
                        # Fallback: try execute() output
                        if upstream_value is None:
                            out_slot = conn.out_port.data(3) or ""
                            result = upstream_logic.execute({})
                            val = result.get(out_slot)
                            if val is not None:
                                upstream_value = str(val)
                except Exception:
                    pass

                # Write upstream value into logic_node parameters
                if upstream_value and self._logic_node is not None:
                    param_key = _row.get("key", "")
                    if param_key:
                        params = self._logic_node.parameters or {}
                        params[param_key] = upstream_value

            widget.set_connected(connected, upstream_value)

        # Rebuild overlay rects for all connected NodeRows
        self._rebuild_row_overlays()

    def _rebuild_row_overlays(self) -> None:
        """Create / destroy semi-transparent overlay rects for disabled NodeRows.

        Each overlay is a child QGraphicsRectItem at z=3 (above proxy z=2,
        below port dots z=4) spanning the full node width at the row's y.
        """
        # Remove old overlays
        for ov in self._row_overlays:
            ov.setParentItem(None)
        self._row_overlays.clear()

        node_w = self._node_width

        for row_def, row_y, row_h in self._param_row_y_cache:
            # Find the proxy for this row
            proxy = None
            for r, p, _rh in self._param_proxies:
                if r is row_def:
                    proxy = p
                    break
            if proxy is None:
                continue
            widget = proxy.widget()
            if not isinstance(widget, NodeRow) or not widget.is_connected:
                continue

            ov = QGraphicsRectItem(0, row_y, node_w, row_h, self)
            ov.setPen(QPen(Qt.PenStyle.NoPen))
            ov.setBrush(QBrush(QColor(0, 0, 0, 90)))
            ov.setZValue(3)
            self._row_overlays.append(ov)

        self.update()

    def _update_proxy_row_height(self, proxy: QGraphicsProxyWidget, new_height: int) -> None:
        target = max(PARAM_ROW_H, int(new_height))
        for idx, (row, item_proxy, row_h) in enumerate(self._param_proxies):
            if item_proxy is proxy:
                if row_h != target:
                    self._param_proxies[idx] = (row, item_proxy, target)
                    self._reflow_layout()
                return

    def _update_proxy_row_width(self, proxy: QGraphicsProxyWidget) -> None:
        for _row, item_proxy, _row_h in self._param_proxies:
            if item_proxy is proxy:
                self._reflow_layout()
                return

    # ------------------------------------------------------------------
    # Layout engine
    # ------------------------------------------------------------------

    def _reflow_layout(self) -> None:
        """
        Reposition all port items + visible param proxy widgets and resize
        the node rect to match current visible content.
        """
        has_in = bool(self._input_order)
        visible_params = [
            (row, proxy, rh)
            for (row, proxy, rh) in self._param_proxies
            if proxy.isVisible()
        ]
        has_params = bool(visible_params)
        sep_in_params = SEP_H if (has_in and has_params) else 0
        sep_params_out = SEP_H if (has_in or has_params) else 0

        # Compute total height
        h = (
            TITLE_H
            + len(self._input_order) * PORT_ROW_H
            + sep_in_params
            + sum(rh for _, _, rh in visible_params)
            + sep_params_out
            + len(self._output_order) * PORT_ROW_H
        )
        node_w = NODE_W
        for _row, proxy, _rh in visible_params:
            widget = proxy.widget()
            if widget is None or not hasattr(widget, "preferred_row_width"):
                continue
            try:
                pref_w = max(0, int(widget.preferred_row_width()))
            except Exception:
                continue
            full_width = bool(getattr(widget, "_full_row_widget", False))
            if full_width:
                node_w = max(node_w, pref_w + H_PAD * 2)
            else:
                node_w = max(node_w, WIDGET_X + pref_w + 4)
        self._node_width = node_w
        self.setRect(0, 0, node_w, h)

        y = float(TITLE_H)

        # ── Position input port items ─────────────────────────────────
        for slot_name in self._input_order:
            port = self._ports.get(f"{slot_name}:in")
            if port:
                port.setPos(PORT_MARGIN, y + PORT_ROW_H * 0.5)
            y += PORT_ROW_H

        if sep_in_params:
            y += sep_in_params

        # ── Position param proxy widgets + build label cache ──────────
        self._param_row_y_cache.clear()
        for row, proxy, rh in self._param_proxies:
            if not proxy.isVisible():
                continue
            self._param_row_y_cache.append((row, y, rh))
            widget = proxy.widget()
            full_width = bool(getattr(widget, "_full_row_widget", False)) if widget is not None else False
            widget_h = max(20, rh - 4) if full_width else min(rh - 4, 20)
            widget_y = y + 2 if full_width else y + (rh - widget_h) * 0.5
            widget_x = H_PAD if full_width else WIDGET_X
            widget_w = node_w - H_PAD * 2 if full_width else max(WIDGET_W, node_w - WIDGET_X - 4)
            proxy.setPos(widget_x, widget_y)
            proxy.resize(widget_w, widget_h)

            # Position hidden port dots on NodeRow input/output zones
            if isinstance(widget, NodeRow):
                if widget.has_input_zone() and widget.input_slot:
                    port = self._ports.get(f"{widget.input_slot}:in")
                    if port:
                        port.setPos(PORT_MARGIN, y + rh * 0.5)
                if widget.has_output_zone() and widget.output_slot:
                    port = self._ports.get(f"{widget.output_slot}:out")
                    if port:
                        port.setPos(node_w - PORT_MARGIN, y + rh * 0.5)

            y += rh

        y += sep_params_out

        # ── Position output port items ────────────────────────────────
        for slot_name in self._output_order:
            port = self._ports.get(f"{slot_name}:out")
            if port:
                port.setPos(node_w - PORT_MARGIN, y + PORT_ROW_H * 0.5)
            y += PORT_ROW_H

        self.update()

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Group-drag: when this node is selected and dragged, co-move all
    # other selected TrainingNodeItems by the same delta.
    # ------------------------------------------------------------------

    def mouseMoveEvent(self, event) -> None:
        if self.isSelected():
            delta = event.scenePos() - event.lastScenePos()
            scene = self.scene()
            if scene is not None and (delta.x() != 0 or delta.y() != 0):
                for item in scene.selectedItems():
                    if item is not self and isinstance(item, TrainingNodeItem):
                        item.moveBy(delta.x(), delta.y())
        super().mouseMoveEvent(event)

    # ── Phase 7 A5: SkillManifest badge helpers ─────────────────────

    _ACTION_BADGE = {
        "joint_position": ("JP", "#3B82F6"),
        "joint_torque": ("JT", "#EF4444"),
        "cartesian_delta": ("CD", "#8B5CF6"),
        "end_effector_6d": ("EE", "#F59E0B"),
        "hybrid": ("HY", "#10B981"),
    }

    def _get_manifest_badge_text(self) -> Optional[Tuple[str, str]]:
        """Return (badge_text, badge_color) from the logic node's SkillManifest.

        Returns None when no manifest data is available.
        """
        logic = self._logic_node
        if logic is None:
            return None
        summary_fn = getattr(logic, "skill_manifest_summary", None)
        if not callable(summary_fn):
            return None
        try:
            summary = summary_fn()
            if not summary:
                return None
            ast = summary.get("action_space_type", "")
            if ast in self._ACTION_BADGE:
                return self._ACTION_BADGE[ast]
            if ast:
                return (ast[:3].upper(), "#6B7280")
        except Exception:
            pass
        return None

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        r = self.rect()
        W, H = r.width(), r.height()
        style = self._style
        selected = self.isSelected()

        # ── Node background ───────────────────────────────────────────
        bg = QColor(style["bg"])
        border_c = QColor(get_color("training_node_border", get_color("border", "#2d2d2d")))
        painter.setPen(QPen(border_c, 1.0))
        painter.setBrush(QBrush(bg))
        painter.drawRoundedRect(r, 6.0, 6.0)

        # ── Title bar (gradient, top-rounded) ─────────────────────────
        tc = QColor(style["title"])
        grad = QLinearGradient(0, 0, 0, TITLE_H)
        grad.setColorAt(0.0, tc.lighter(145))
        grad.setColorAt(1.0, tc)
        painter.save()
        painter.setClipRect(QRectF(0, 0, W, TITLE_H))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(grad))
        painter.drawRoundedRect(r, 6.0, 6.0)
        painter.restore()

        # Title / body divider
        painter.setPen(QPen(tc.darker(150), 1))
        painter.drawLine(0, TITLE_H, int(W), TITLE_H)

        # ── Title text ────────────────────────────────────────────────
        tf = QFont()
        tf.setPixelSize(12)
        tf.setWeight(QFont.Weight.Bold)
        painter.setFont(tf)
        painter.setPen(QColor(style["text"]))
        painter.drawText(
            QRectF(H_PAD, 0.0, W - H_PAD * 2 - 40, TITLE_H),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            self._display_name,
        )

        # ── Phase 7 A5: SkillManifest badge (top-right corner) ───────
        manifest_badge = self._get_manifest_badge_text()
        if manifest_badge:
            badge_text, badge_color = manifest_badge
            badge_font = QFont()
            badge_font.setPixelSize(8)
            badge_font.setWeight(QFont.Weight.Bold)
            painter.setFont(badge_font)
            fm = QFontMetrics(badge_font)
            badge_w = fm.horizontalAdvance(badge_text) + 8
            badge_h = 14
            badge_x = W - H_PAD - badge_w
            badge_y = (TITLE_H - badge_h) / 2.0
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(badge_color))
            painter.drawRoundedRect(QRectF(badge_x, badge_y, badge_w, badge_h), 3.0, 3.0)
            painter.setPen(QColor("#ffffff"))
            painter.drawText(
                QRectF(badge_x, badge_y, badge_w, badge_h),
                Qt.AlignmentFlag.AlignCenter,
                badge_text,
            )

        # ── Row font ──────────────────────────────────────────────────
        bf = QFont()
        bf.setPixelSize(11)
        painter.setFont(bf)

        y = float(TITLE_H)

        # ── Input port rows ───────────────────────────────────────────
        for slot_name in self._input_order:
            row_rect = QRectF(0, y, W, PORT_ROW_H)
            _in_port = self._ports.get(f"{slot_name}:in")
            _in_dtype = _in_port._data_type if _in_port else slot_name
            tc2 = QColor(_training_port_types().get(_in_dtype, {}).get("color", get_color("training_port_fallback", "#9ca3af")))
            tint = QColor(tc2)
            tint.setAlpha(18)
            painter.fillRect(row_rect, tint)

            req = slot_name not in self._optional_inputs
            label = slot_name if req else f"{slot_name}  (opt)"
            painter.setPen(QColor(tc2))
            lx = PORT_MARGIN + PORT_R + 7
            painter.drawText(
                QRectF(lx, y, W - lx - 4, PORT_ROW_H),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                label,
            )
            y += PORT_ROW_H

        # Separator between inputs and params
        has_in = bool(self._input_order)
        has_params = bool(self._param_row_y_cache)
        if has_in and has_params:
            sep_y = int(y + SEP_H * 0.4)
            painter.setPen(QPen(QColor(get_color("training_node_separator", "#2e2e2e")), 1))
            painter.drawLine(int(H_PAD), sep_y, int(W - H_PAD), sep_y)
            y += SEP_H

        # ── Param rows — draw label in left column ────────────────────
        lf = QFont()
        lf.setPixelSize(11)
        painter.setFont(lf)

        # Build a quick key→is_full_width lookup from widget attributes
        _full_width_keys: set = set()
        for _row, _proxy, _rh in self._param_proxies:
            w = _proxy.widget()
            if w is not None and getattr(w, "_full_row_widget", False):
                _full_width_keys.add(_row.get("key", ""))

        for row_def, row_y, row_h in self._param_row_y_cache:
            # Subtle row tint
            tint = QColor(QColor(style["title"]))
            tint.setAlpha(12)
            painter.fillRect(QRectF(0, row_y, W, row_h), tint)

            if row_def.get("full_width_widget", False) or row_def.get("key", "") in _full_width_keys:
                continue

            # Label text (left column, right-aligned to label edge)
            display_name = row_def.get("display_name", row_def.get("key", ""))
            painter.setPen(QColor(get_color("training_node_param_label", get_color("text_secondary", "#9ca3af"))))
            painter.drawText(
                QRectF(H_PAD, row_y, LABEL_COL - 4, row_h),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                display_name,
            )

        # Separator before outputs
        if has_in or has_params:
            # y should now be at end of last param row
            if self._param_row_y_cache:
                last_row_y, last_row_h = self._param_row_y_cache[-1][1], self._param_row_y_cache[-1][2]
                y = last_row_y + last_row_h
            sep_y = int(y + SEP_H * 0.4)
            painter.setPen(QPen(QColor(get_color("training_node_separator", "#2e2e2e")), 1))
            painter.drawLine(int(H_PAD), sep_y, int(W - H_PAD), sep_y)
            y += SEP_H

        # ── Output port rows ──────────────────────────────────────────
        for slot_name in self._output_order:
            row_rect = QRectF(0, y, W, PORT_ROW_H)
            oc = QColor(_training_port_types().get(slot_name, {}).get("color", get_color("training_port_fallback", "#9ca3af")))
            tint = QColor(oc)
            tint.setAlpha(18)
            painter.fillRect(row_rect, tint)

            painter.setPen(QColor(oc))
            text_right = W - PORT_MARGIN - PORT_R - 7
            painter.drawText(
                QRectF(4.0, y, text_right - 4, PORT_ROW_H),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                slot_name,
            )
            y += PORT_ROW_H

        # ── Selection highlight (white dashed border) ─────────────────
        if selected:
            sel_pen = QPen(QColor(255, 255, 255, 210), 1.5, Qt.PenStyle.DashLine)
            painter.setPen(sel_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(r, 6.0, 6.0)
