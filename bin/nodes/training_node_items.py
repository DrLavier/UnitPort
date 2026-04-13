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
    QToolTip,
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
    il_reward_registry,
    il_obs_registry,
    il_termination_registry,
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
        "total_reward":       {"color": get_color("training_port_rewards", "#38BDF8"), "label": "total_reward"},
        "terminations":       {"color": get_color("training_port_terminations", "#F472B6"), "label": "terminations"},
        "task_config":        {"color": get_color("training_port_task_config", "#81C784"), "label": "task_config"},
        "obs_action_config":  {"color": get_color("training_port_obs_action_config", "#FFD54F"), "label": "obs_action_config"},
        "domain_rand_config": {"color": get_color("training_port_domain_rand_config", "#BA68C8"), "label": "domain_rand_config"},
        "env_config":         {"color": get_color("training_port_env_config", "#4DB6AC"), "label": "env_config"},
        "algo_config":        {"color": get_color("training_port_algo_config", "#F48FB1"), "label": "algo_config"},
        "eval_config":        {"color": get_color("training_port_eval_config", "#A5D6A7"), "label": "eval_config"},
        "train_pipe":         {"color": get_color("training_port_train_pipe", "#FF7043"), "label": "train_pipe"},
        "bundle_path":        {"color": get_color("training_port_bundle_path", "#90A4AE"), "label": "bundle_path"},
        "vis_check":          {"color": get_color("training_port_vis_check", "#80DEEA"), "label": "vis_check"},
        "scene_config":            {"color": get_color("training_port_scene_config", "#FFD180"), "label": "scene_config"},
        "checkpoint":              {"color": get_color("training_port_checkpoint",   "#CE93D8"), "label": "checkpoint"},
        "reference_motion_config": {"color": get_color("training_port_reference_motion", "#FFB300"), "label": "reference_motion_config"},
        "init_pose_config":        {"color": get_color("training_port_init_pose", "#A5D6A7"),          "label": "init_pose_config"},
        "joint_config":            {"color": get_color("training_port_joint_config", "#BCAAA4"),        "label": "joint_config"},
        "int":                     {"color": get_color("training_port_int", "#FDD835"),                  "label": "int"},
        # Isaac Lab ports (legacy)
        "isaac_lab_task_config":   {"color": get_color("training_port_il_task", "#26C6DA"),              "label": "isaac_lab_task_config"},
        "isaac_lab_result":        {"color": get_color("training_port_il_result", "#00ACC1"),            "label": "isaac_lab_result"},
        "export_config":           {"color": get_color("training_port_export_config", "#00897B"),        "label": "export_config"},
        # Isaac Lab granular node ports (IL_A – IL_F)
        "sim_context":        {"color": "#78909C", "label": "sim_context"},
        "terrain":            {"color": "#8D6E63", "label": "terrain"},
        "height_scanner":     {"color": "#A1887F", "label": "height_scanner"},
        "robot_entity":       {"color": "#EF5350", "label": "robot_entity"},
        "joint_info":         {"color": "#EF9A9A", "label": "joint_info"},
        "body_info":          {"color": "#FFCDD2", "label": "body_info"},
        "actuator_config":    {"color": "#FF8A65", "label": "actuator_config"},
        "contact_sensor":     {"color": "#CE93D8", "label": "contact_sensor"},
        "obs_vector":         {"color": "#0288D1", "label": "obs_vector"},
        "action_entity":      {"color": "#66BB6A", "label": "action_entity"},
        "velocity_command":   {"color": "#26C6DA", "label": "velocity_command"},
        "total_reward":       {"color": "#F57C00", "label": "total_reward"},
        "termination_config": {"color": "#E57373", "label": "termination_config"},
        "policy":             {"color": "#7E57C2", "label": "policy"},
        "training_metrics":   {"color": "#9575CD", "label": "training_metrics"},
        "exported_model":     {"color": "#4DB6AC", "label": "exported_model"},
        "command_source":     {"color": "#26C6DA", "label": "velocity_command"},
        "action_source":      {"color": "#66BB6A", "label": "action_entity"},
        "body_mapping":       {"color": "#FFCDD2", "label": "body_mapping"},
        # Unified 6-section pipes — colors match their owning node layer.
        # §1 Actor block uses the IL_A title color (#00695C dark teal) so
        # the wire matches the Robot / Actor Setting / IL Actuator Config
        # node bodies. §2 Scene uses the "A" layer title color (#2D6A2D
        # dark green) matching the Rewards node.
        "robot_pipe":         {"color": "#00695C", "label": "robot_pipe"},
        "actor_pipe":         {"color": "#00695C", "label": "actor_pipe"},
        "joint_init":         {"color": "#26A69A", "label": "joint_init"},
        "scene_pipe":         {"color": "#2D6A2D", "label": "scene_pipe"},
        "command_pipe":       {"color": "#1565C0", "label": "command_pipe"},
        "reference_motion_pipe": {"color": "#7E57C2", "label": "reference_motion"},
    }

# ---------------------------------------------------------------------------
# Layer → visual style
# ---------------------------------------------------------------------------

_NODE_LAYER: Dict[str, str] = {
    # §1 Actor block — shares the IL_A teal palette.
    "robot":               "IL_A",
    "actor_setting":       "IL_A",
    "joint_init":          "IL_A",
    # §2 Scene block
    "play_ground_setting": "S",
    # § Training Commands — IL_B (slightly bluer teal) so commands
    # read as a sibling of the actor block but visually distinct.
    "training_commands":   "IL_B",
    # SB3 core pipeline
    "physics_config":      "A",
    "rewards":             "A",
    "terminations":        "A",
    "task_config":         "A",
    "domain_rand":         "A",
    "obs_action_config":   "A",
    "env_assembler":       "B",
    "algo_config":         "B",
    "train":               "C",
    "eval_config":         "D",
    "export":              "D",
    "vis_check":           "D",
    "base_asset":          "C",
    "reference_motion":    "A",
    "init_pose":           "A",
    # Isaac Lab granular nodes (remaining after cleanup)
    "il_observation":      "IL_B",
    "il_policy_network":   "IL_F",
    "il_ppo_trainer":      "IL_F",
    # Legacy alias — canvases saved pre-merge still carry amp_ppo_trainer.
    "amp_ppo_trainer":     "IL_F",
}

# Each layer has: bg (node body), title (title bar), text (title label)
def _layer_colors() -> Dict[str, Dict[str, str]]:
    return {
        # §1 Robot / Actor / JointInit — teal (blue-green).
        # Functional-area color rule: §1 = teal, §2 = dark forest green, …
        "R": {
            "bg":    get_color("training_layer_robot_bg",    "#0A2A2B"),
            "title": get_color("training_layer_robot_title", "#0F766E"),
            "text":  get_color("training_layer_robot_text",  "#99F6E4"),
        },
        # §2 Scene (Play Ground) — same dark forest green as the
        # Rewards node's layer-A palette so §2 matches the visual
        # weight users already associate with green nodes.
        "S": {
            "bg":    get_color("training_layer_scene_bg",    "#1B2E1B"),
            "title": get_color("training_layer_scene_title", "#2D6A2D"),
            "text":  get_color("training_layer_scene_text",  "#A5D6A7"),
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
        "IL": {
            "bg": get_color("training_layer_il_bg", "#0D2B2B"),
            "title": get_color("training_layer_il_title", "#00796B"),
            "text": get_color("training_layer_il_text", "#80CBC4"),
        },
        # Isaac Lab granular layers (IL_A – IL_F)
        "IL_A": {
            "bg": get_color("training_layer_il_a_bg", "#1B2A2A"),
            "title": get_color("training_layer_il_a_title", "#00695C"),
            "text": get_color("training_layer_il_a_text", "#B2DFDB"),
        },
        "IL_B": {
            "bg": get_color("training_layer_il_b_bg", "#1A237E"),
            "title": get_color("training_layer_il_b_title", "#1565C0"),
            "text": get_color("training_layer_il_b_text", "#BBDEFB"),
        },
        "IL_C": {
            "bg": get_color("training_layer_il_c_bg", "#004D40"),
            "title": get_color("training_layer_il_c_title", "#00897B"),
            "text": get_color("training_layer_il_c_text", "#B2DFDB"),
        },
        "IL_D": {
            "bg": get_color("training_layer_il_d_bg", "#3E2723"),
            "title": get_color("training_layer_il_d_title", "#E65100"),
            "text": get_color("training_layer_il_d_text", "#FFE0B2"),
        },
        "IL_E": {
            "bg": get_color("training_layer_il_e_bg", "#311B92"),
            "title": get_color("training_layer_il_e_title", "#7B1FA2"),
            "text": get_color("training_layer_il_e_text", "#E1BEE7"),
        },
        "IL_F": {
            "bg": get_color("training_layer_il_f_bg", "#1A1A2E"),
            "title": get_color("training_layer_il_f_title", "#4527A0"),
            "text": get_color("training_layer_il_f_text", "#D1C4E9"),
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
# Node UI rows — runtime config (bin/nodes/training_node_ui.json)
# ---------------------------------------------------------------------------

_UI_JSON_PATH = pathlib.Path(__file__).parent / "training_node_ui.json"

# Maps node_type key (e.g. "robot_mjcf") → list of ui_row dicts
NODE_UI_ROWS: Dict[str, List[dict]] = {}

try:
    with _UI_JSON_PATH.open("r", encoding="utf-8") as _f:
        NODE_UI_ROWS = json.load(_f)
except FileNotFoundError:
    raise FileNotFoundError(
        f"Training node UI config not found: {_UI_JSON_PATH}\n"
        "This file is required for node rendering."
    )
except Exception as _exc:
    raise RuntimeError(
        f"Failed to load training node UI config: {_UI_JSON_PATH}\n{_exc}"
    )

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
NODE_UI_ROWS.setdefault("il_policy_exporter", []).append(
    {
        "kind": "param",
        "key": "__review_il_export__",
        "display_name": "",
        "widget": "action_button",
        "button_text": "Review",
        "button_action": "review_il_export",
        "row_height": 24,
        "full_width_widget": True,
        "tooltip": "Replay the exported Isaac Lab policy in an Isaac Sim viewport (play.py).",
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
    # Phase_2 of AMP_design.yaml §3 — unified Reference Motion picker.
    # The picker drives BOTH motion_file and motion_source params; the
    # legacy "Source" dropdown was merged into this single entry. The
    # picker writes motion_source for symbolic entries (generate:/loco:/
    # pack:) and motion_file for single files, mirroring the backend
    # precedence rule motion_source > motion_file.
    {
        "kind": "param",
        "key": "motion_file",
        "display_name": "Reference Motion",
        "widget": "motion_library_picker",
        "row_height": 24,
        "tooltip": (
            "Pick a single clip, a procedural generator (⚙), a loco-mujoco "
            "dataset (📦), or an entire community pack (📦 pack:…) — "
            "packs are the natural granularity for AMP discriminator "
            "training. <Import .npy to library> drops a hand-recorded clip "
            "into the SB3 motion library."
        ),
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
        # Phase_2 of AMP_design.yaml §3 — tracking reward is irrelevant
        # when the user is feeding the motion straight into the AMP
        # discriminator. Hide the knob in that case to prevent confusion.
        "condition": {"key": "consumer_mode", "op": "!=", "value": "amp"},
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
        "condition": {"key": "consumer_mode", "op": "!=", "value": "amp"},
    },
    # ── Phase_2 (AMP_design.yaml §3.reference_motion) additive fields ──
    # These 4 knobs control how the motion is CONSUMED (tracking reward
    # vs AMP discriminator). Default values preserve the pre-phase_2
    # behaviour (tracking only), so old canvases keep working unchanged.
    {
        "kind": "separator",
        "row_height": 6,
    },
    {
        "kind": "param",
        "key": "format_id",
        "display_name": "Format",
        "widget": "dropdown",
        "choices": ["unitport_npy", "amp_legged_gym"],
        "default": "unitport_npy",
        "row_height": 24,
        "tooltip": (
            "unitport_npy = legacy tracking .npy files from the SB3 motion "
            "library.\n"
            "amp_legged_gym = DeepMimic-style .txt/.json clips from "
            "community packages (AMP_for_hardware, MetalHead, rl_amp). "
            "Required when Consumer Mode is 'amp' or 'both'."
        ),
    },
    {
        "kind": "param",
        "key": "consumer_mode",
        "display_name": "Consumer Mode",
        "widget": "dropdown",
        "choices": ["tracking", "amp", "both"],
        "default": "tracking",
        "row_height": 24,
        "tooltip": (
            "tracking  — feed the motion through the legacy PPO + "
            "reference-tracking reward (the pre-phase_2 path).\n"
            "amp       — route transitions through the AMP discriminator "
            "(requires an AMP PPO Trainer downstream).\n"
            "both      — tracking reward + AMP discriminator simultaneously."
        ),
    },
    {
        "kind": "param",
        "key": "amp_obs_fields",
        "display_name": "AMP Obs Fields",
        "widget": "multiselect_tags",
        "storage": "csv",
        "options": [
            "joint_pos",
            "toe_pos_local",
            "base_lin_vel_local",
            "base_ang_vel_local",
            "joint_vel",
            "root_height",
        ],
        "option_meta": {
            "joint_pos": {
                "title": "Joint Position",
                "desc": "Current joint angles for all actuated DoFs.",
            },
            "toe_pos_local": {
                "title": "Toe Position (local)",
                "desc": "Foot/toe positions in the base frame via forward kinematics.",
            },
            "base_lin_vel_local": {
                "title": "Base Linear Vel (local)",
                "desc": "Base linear velocity expressed in the body frame.",
            },
            "base_ang_vel_local": {
                "title": "Base Angular Vel (local)",
                "desc": "Base angular velocity expressed in the body frame.",
            },
            "joint_vel": {
                "title": "Joint Velocity",
                "desc": "Angular velocity of each actuated joint.",
            },
            "root_height": {
                "title": "Root Height",
                "desc": "Scalar height of the robot base above the terrain.",
            },
        },
        "default": "",
        "row_height": 24,
        "tooltip": (
            "Select which observation fields feed the AMP discriminator. "
            "Empty = use the full default set (all 6 fields). "
            "Only consulted when Consumer Mode is 'amp' or 'both'."
        ),
        "condition": {"key": "consumer_mode", "op": "!=", "value": "tracking"},
    },
    # amp_reward_coef removed from reference_motion — the authoritative
    # knob lives on the amp_ppo_trainer node.  Keeping the field here
    # caused a confusing duplicate that could silently override the
    # trainer's value downstream.
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
    # ── Preview physics toggle ───────────────────────────────────────
    # ON  → _MotionPreviewThread runs PD tracking (the RL agent's view).
    # OFF → pure kinematic replay — reference pose dropped into qpos,
    #       showing the motion exactly as authored regardless of gains.
    {
        "kind": "param",
        "key": "preview_physics",
        "display_name": "Preview physic",
        "widget": "toggle",
        "row_height": 24,
    },
    # ── Preview button (always last) ─────────────────────────────────
    {
        "kind": "param",
        "key": "preview",
        "display_name": "Preview",
        "widget": "preview_button",
        "row_height": 24,
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
        "key": "rsi_prob",
        "display_name": "RSI Prob",
        "widget": "slider_float",
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
        "decimals": 2,
        "default": "1.0",
        "row_height": 24,
        "tooltip": (
            "Probability of using reference-state initialization at reset.\n"
            "1.0 = always reference, 0.0 = always default pose.\n"
            "Only effective when mode = 'reference_frame_0'."
        ),
        "condition": {"key": "mode", "value": "reference_frame_0"},
    },
    {
        "kind": "param",
        "key": "rsi_sample_mode",
        "display_name": "RSI Sample",
        "widget": "dropdown",
        "choices": ["frame_0", "uniform_phase"],
        "default": "frame_0",
        "row_height": 24,
        "tooltip": (
            "frame_0 — always use the first motion frame (deterministic)\n"
            "uniform_phase — sample a random frame each reset (APEX-style)"
        ),
        "condition": {"key": "mode", "value": "reference_frame_0"},
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


def _override_il_export_bundle_name_row() -> None:
    """Mirror the SB3 export node: replace IL Policy Exporter's plain
    text bundle_name input with the same ``export_bundle_picker`` widget
    (``<NEW>`` + an existing-checkpoint dropdown sourced from
    ``CheckpointRegistry``). After the IL training run finishes the
    backend auto-imports the exported bundle into the same registry, so
    SB3 and IL bundles share one namespace and one picker UI."""
    rows = NODE_UI_ROWS.setdefault("il_policy_exporter", [])
    for row in rows:
        if row.get("key") != "bundle_name":
            continue
        row["display_name"] = "Checkpoint"
        row["widget"] = "export_bundle_picker"
        row["default"] = ""
        row["placeholder"] = ""
        row["tooltip"] = (
            "Choose an existing checkpoint to update, or select <NEW> to register "
            "a new export name before the Isaac Lab bundle is auto-imported."
        )
        break


_override_il_export_bundle_name_row()

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
        "choices": ["scratch", "resume", "warm_start_actor"],
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


class NodeInputer(QPushButton):
    """Button that displays a text value; click to edit via a popup.

    Replaces bare ``QLineEdit`` inside node cards.  The button never
    enters an editable focus state on the graphics scene — instead it
    spawns a :class:`_TextInputPopup` (text mode) or :class:`DataInput`
    (float mode) popup on click, avoiding the input-lock issue that
    ``QLineEdit`` causes inside ``QGraphicsProxyWidget``.

    Parameters
    ----------
    current_text:
        Initial display text.
    placeholder:
        Placeholder shown in the popup input when empty.
    mode:
        ``"text"`` — free-form string (uses ``_TextInputPopup``).
        ``"float"`` — numeric with optional range (uses ``DataInput``).
        ``"scientific"`` — like float but accepts scientific notation.
    min_value, max_value:
        Range bounds for ``"float"`` / ``"scientific"`` mode.
    decimals:
        Display decimal places for ``"float"`` mode.
    """

    textChanged = Signal(str)

    def __init__(
        self,
        *,
        current_text: str = "",
        placeholder: str = "",
        mode: str = "text",
        min_value: float = -1e15,
        max_value: float = 1e15,
        decimals: int = 6,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._text = str(current_text)
        self._placeholder = placeholder
        self._mode = mode
        self._min = float(min_value)
        self._max = float(max_value)
        self._decimals = int(decimals)
        self._popup = None

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMaximumWidth(WIDGET_W)
        self.setFlat(True)
        self._refresh_display()
        self.clicked.connect(self._open_popup)

    # ── public API ────────────────────────────────────────────────────

    def text(self) -> str:
        return self._text

    def setText(self, t: str) -> None:            # noqa: N802  (Qt naming)
        self._text = str(t)
        self._refresh_display()

    # ── display ───────────────────────────────────────────────────────

    def _refresh_display(self) -> None:
        display = self._text if self._text else (self._placeholder or "—")
        avail = (self.width() or WIDGET_W) - 16
        elided = QFontMetrics(self.font()).elidedText(
            display, Qt.TextElideMode.ElideRight, max(30, avail),
        )
        QPushButton.setText(self, elided)

    def resizeEvent(self, event) -> None:         # noqa: N802
        super().resizeEvent(event)
        self._refresh_display()

    # ── popup lifecycle ───────────────────────────────────────────────

    def _open_popup(self) -> None:
        if self._popup is not None:
            self._popup.close()
            self._popup = None
            return

        if self._mode == "float":
            try:
                cur_v = float(self._text)
            except (ValueError, TypeError):
                cur_v = 0.0
            popup = DataInput(
                current_value=cur_v,
                min_value=self._min,
                max_value=self._max,
            )
            self._popup = popup
            popup.destroyed.connect(self._clear_popup)
            popup.destroyed.connect(lambda *_a, p=popup: self._apply_float(p))
            pos, row_h = self._popup_anchor()
            popup.show_at(pos, row_h)
        else:
            # text / scientific
            popup = _TextInputPopup(placeholder=self._placeholder or "Enter value…")
            popup._input.setText(self._text)
            popup._input.selectAll()
            if self._mode == "scientific":
                popup._input.setPlaceholderText("e.g. 3e-4")
            self._popup = popup
            popup.accepted.connect(self._apply_text)
            popup.destroyed.connect(self._clear_popup)
            pos, row_h = self._popup_anchor()
            popup.show_at(pos, row_h)

    def _apply_float(self, popup: DataInput) -> None:
        v = popup.accepted_value()
        new_text = f"{v:.{self._decimals}g}"
        if new_text != self._text:
            self._text = new_text
            self._refresh_display()
            self.textChanged.emit(self._text)

    def _apply_text(self, value: str) -> None:
        if value != self._text:
            self._text = value
            self._refresh_display()
            self.textChanged.emit(self._text)
        if self._popup is not None:
            self._popup.close()

    def _clear_popup(self, *_a) -> None:
        self._popup = None

    def _popup_anchor(self) -> Tuple[QPoint, int]:
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
    """Compact inline text-input popup — uses DataInput styling for consistency.

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
        self.setObjectName("dataInputPopup")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._validate_fn: Optional[Callable[[str], Optional[str]]] = None
        self._did_accept = False

        border = get_color("training_widget_border", "#3d3d3d")
        bg = get_color("training_widget_input_bg", "#1A1A1A")
        text = get_color("training_widget_text", "#cccccc")
        btn_bg = get_color("training_widget_button_bg", get_color("button_bg", "#252525"))
        hover = get_color("training_button_hover_bg", get_color("button_hover", "#2e2e2e"))
        err_c = get_color("training_widget_error", "#f87171")

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
                border-color: {err_c};
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
            QLabel[dataInputError="true"] {{
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
        self._input.setProperty("dataInput", True)
        self._input.setPlaceholderText(placeholder)
        row.addWidget(self._input, 1)

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

        self._error_lbl = QLabel("", self)
        self._error_lbl.setProperty("dataInputError", True)
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

    def show_at(self, pos: QPoint, row_height: int = 50) -> None:
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
        algorithm_provider: Optional[Callable[[], str]] = None,
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
        self._algorithm_provider = algorithm_provider
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

    def _resolved_algorithm(self) -> str:
        try:
            alg = str(self._algorithm_provider() if self._algorithm_provider is not None else "").strip().upper()
        except Exception:
            alg = ""
        return alg or "ALL"

    def _filtered_registry(self) -> Dict[str, TaskModuleItem]:
        family = self._resolved_family()
        algorithm = self._resolved_algorithm()
        allowed: Dict[str, TaskModuleItem] = {}
        for key in self._order:
            item = self._registry.get(key)
            if item is None:
                continue
            # Already-selected items always remain visible so the user
            # can remove them manually (avoid silently dropping weights).
            if key in self._selected:
                allowed[key] = item
                continue
            if family not in item.applicable_families:
                continue
            # Algorithm compatibility: "ALL" in item.algorithms means
            # universally applicable; otherwise the canvas algorithm
            # must be listed, or the canvas algorithm is "ALL" (unknown).
            item_algs = getattr(item, "algorithms", None)
            if item_algs:
                if "ALL" not in item_algs and algorithm != "ALL" and algorithm not in item_algs:
                    continue
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

    # Slots that accept unlimited incoming edges (fan-in)
    _FAN_IN_SLOTS: set = {"actuator_configs"}

    def __init__(
        self,
        parent: "TrainingNodeItem",
        slot_name: str,
        io: str,
        data_type: str,
        required: bool = True,
        max_connections: Optional[int] = None,
    ) -> None:
        if max_connections is None:
            if io == "in" and slot_name in self._FAN_IN_SLOTS:
                max_connections = -1
            elif io == "out":
                max_connections = 8
            else:
                max_connections = 1

        self._is_fan_in = (max_connections is not None and max_connections < 0)
        # Fan-in ports are 1px larger in radius
        r = PORT_R + 1 if self._is_fan_in else PORT_R
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
            "max_connections": max_connections,
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

    def paint(self, painter: QPainter, option, widget=None) -> None:
        super().paint(painter, option, widget)
        if self._is_fan_in:
            font = QFont("Segoe UI", 7, QFont.Weight.Bold)
            painter.setFont(font)
            painter.setPen(QPen(QColor("#ffffff"), 0))
            painter.drawText(self.boundingRect(), Qt.AlignCenter, "F")

    def _tooltip_label(self) -> str:
        """Build a human-readable tooltip for this port."""
        dt = self._data_type or "any"
        if self._is_fan_in:
            return f"Fan-In  [{dt}]"
        return dt

    def hoverEnterEvent(self, event) -> None:
        self._apply_visual("hover")
        self.setToolTip(self._tooltip_label())
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        conns = self.data(2)
        self._apply_visual("connected" if conns else "normal")
        self.setToolTip("")
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


def _list_active_project_run_checkpoints() -> List[Tuple[str, str, str]]:
    """Scan the active project's ``training/runs/*/model_*.pt`` tree.

    Returns a list of ``(run_slug, ckpt_name, abs_path)`` tuples, sorted
    with most-recent runs first and the highest iteration checkpoint per
    run first (``model_1499.pt`` before ``model_100.pt``).

    Returns an empty list when no project is active (e.g. node is being
    rendered in a preview where the project session hasn't been set up)
    or when the runs tree does not exist yet.
    """
    try:
        from src.system.core.project_session import get_active_project_root
        root = get_active_project_root()
    except Exception:
        return []
    if root is None:
        return []
    runs_dir = pathlib.Path(root) / "training" / "runs"
    if not runs_dir.is_dir():
        return []

    entries: List[Tuple[str, str, str]] = []
    try:
        run_dirs = sorted(
            (d for d in runs_dir.iterdir() if d.is_dir()),
            key=lambda d: d.name,
            reverse=True,
        )
    except Exception:
        return []

    def _iter_key(name: str) -> int:
        # ``model_1499.pt`` → 1499, ``model_.pt`` → -1
        stem = pathlib.Path(name).stem
        tail = stem.rsplit("_", 1)[-1]
        try:
            return int(tail)
        except ValueError:
            return -1

    for run_dir in run_dirs:
        try:
            ckpts = sorted(
                run_dir.glob("model_*.pt"),
                key=lambda p: _iter_key(p.name),
                reverse=True,
            )
        except Exception:
            continue
        for pt in ckpts:
            entries.append((run_dir.name, pt.name, str(pt.resolve())))
    return entries


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

        # Run-dir checkpoints from the active project. Listed one entry
        # per .pt (so users can pick a specific iteration), labeled with
        # run slug + iteration so the dropdown is self-explanatory.
        for run_slug, ckpt_name, abs_path in _list_active_project_run_checkpoints():
            token = f"run:{abs_path}"
            choices.append(token)
            # Extract iter number from ``model_1499.pt`` → ``iter 1499``
            stem = pathlib.Path(ckpt_name).stem
            tail = stem.rsplit("_", 1)[-1]
            iter_label = f"iter {tail}" if tail.isdigit() else ckpt_name
            meta_map[token] = {
                "title": f"{run_slug}  ·  {iter_label}",
                "subtitle": ckpt_name,
                "desc": (
                    f"Run-dir checkpoint: {abs_path}\n"
                    f"Use with load_mode='resume' to continue the run, "
                    f"or 'warm_start_actor' to seed a new run (e.g. "
                    f"AMP-PPO from a PPO checkpoint)."
                ),
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
    return pathlib.Path(get_project_root()) / "src" / "config" / f"{kind}_presets.json"


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
        except (RuntimeError, TypeError):
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
# Stage Editor — multi-stage training pipeline widget
# ---------------------------------------------------------------------------

class _StageOverrideDialog(QDialog):
    """Modal dialog for editing a single stage's overrides and criteria."""

    # Keys that can be overridden per-stage, grouped for UI layout
    _OVERRIDE_GROUPS = [
        ("Training Mode", [
            ("training_mode", "Training Mode", "dropdown", ["PPO", "AMP_PPO"]),
        ]),
        ("AMP Parameters", [
            ("amp_reward_coef", "AMP Reward Coef", "float", (0.0, 10.0, 0.1)),
            ("task_reward_lerp", "Task Reward Lerp", "float", (0.0, 1.0, 0.05)),
            ("disc_lr", "Disc LR", "text", None),
            ("disc_grad_penalty", "Disc Grad Penalty", "float", (0.0, 50.0, 1.0)),
            ("disc_label_smoothing", "Disc Label Smooth", "float", (0.0, 1.0, 0.05)),
        ]),
        ("PPO Hyperparams", [
            ("learning_rate", "Learning Rate", "text", None),
            ("entropy_coef", "Entropy Coef", "text", None),
            ("clip_param", "Clip Param", "float", (0.0, 1.0, 0.05)),
        ]),
        ("Domain Randomization", [
            ("domain_rand_enabled", "DR Enabled", "bool", None),
        ]),
        ("Lerp Schedule", [
            ("lerp_schedule", "Lerp Schedule JSON", "json", None),
        ]),
    ]

    def __init__(self, stage_data: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Stage: {stage_data.get('display_name', 'Edit Stage')}")
        self.setModal(True)
        self.resize(420, 520)

        self._stage = dict(stage_data)
        self._overrides = dict(stage_data.get("overrides", {}))
        self._checks: Dict[str, QCheckBox] = {}
        self._widgets: Dict[str, QWidget] = {}

        root = QVBoxLayout(self)
        root.setSpacing(6)

        # ── Stage metadata ──
        meta_frame = QFrame()
        meta_layout = QVBoxLayout(meta_frame)
        meta_layout.setContentsMargins(4, 4, 4, 4)
        meta_layout.setSpacing(4)

        # Stage ID
        hl = QHBoxLayout()
        hl.addWidget(QLabel("ID:"))
        self._id_edit = QLineEdit(str(stage_data.get("stage_id", "")))
        self._id_edit.setPlaceholderText("e.g. 0_ppo_warmup")
        hl.addWidget(self._id_edit)
        meta_layout.addLayout(hl)

        # Display name
        hl2 = QHBoxLayout()
        hl2.addWidget(QLabel("Name:"))
        self._name_edit = QLineEdit(str(stage_data.get("display_name", "")))
        self._name_edit.setPlaceholderText("e.g. PPO Warmup")
        hl2.addWidget(self._name_edit)
        meta_layout.addLayout(hl2)

        # Iterations
        hl3 = QHBoxLayout()
        hl3.addWidget(QLabel("Iterations:"))
        self._iter_spin = QSpinBox()
        self._iter_spin.setRange(1, 999999)
        self._iter_spin.setValue(int(stage_data.get("iterations", 1000)))
        hl3.addWidget(self._iter_spin)
        meta_layout.addLayout(hl3)

        root.addWidget(meta_frame)

        # ── Overrides (checkbox-gated) ──
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(4, 4, 4, 4)
        scroll_layout.setSpacing(2)

        for group_name, items in self._OVERRIDE_GROUPS:
            grp_label = QLabel(f"── {group_name} ──")
            grp_label.setStyleSheet("color: #888; font-weight: bold; font-size: 11px;")
            scroll_layout.addWidget(grp_label)

            for key, label, wtype, opts in items:
                row_hl = QHBoxLayout()
                row_hl.setSpacing(4)

                cb = QCheckBox(label)
                cb.setChecked(key in self._overrides)
                self._checks[key] = cb
                row_hl.addWidget(cb)

                if wtype == "dropdown":
                    w = QComboBox()
                    for choice in (opts or []):
                        w.addItem(str(choice))
                    if key in self._overrides:
                        idx = w.findText(str(self._overrides[key]))
                        if idx >= 0:
                            w.setCurrentIndex(idx)
                elif wtype == "float":
                    w = QDoubleSpinBox()
                    lo, hi, step = opts if opts else (0.0, 100.0, 0.1)
                    w.setRange(lo, hi)
                    w.setSingleStep(step)
                    w.setDecimals(6)
                    if key in self._overrides:
                        try:
                            w.setValue(float(self._overrides[key]))
                        except (TypeError, ValueError):
                            pass
                elif wtype == "bool":
                    w = QCheckBox("Enabled")
                    if key in self._overrides:
                        v = self._overrides[key]
                        w.setChecked(
                            str(v).strip().lower() in ("true", "1", "yes")
                        )
                elif wtype == "json":
                    w = QLineEdit()
                    w.setPlaceholderText('{"start":0.75,"end":0.3,"over_iters":2000}')
                    if key in self._overrides:
                        v = self._overrides[key]
                        w.setText(
                            json.dumps(v) if isinstance(v, dict) else str(v)
                        )
                else:  # text
                    w = QLineEdit()
                    if key in self._overrides:
                        w.setText(str(self._overrides[key]))

                self._widgets[key] = w
                w.setEnabled(key in self._overrides)
                cb.toggled.connect(w.setEnabled)
                row_hl.addWidget(w)
                scroll_layout.addLayout(row_hl)

        scroll.setWidget(scroll_content)
        root.addWidget(scroll)

        # ── Advance criteria ──
        adv_frame = QFrame()
        adv_layout = QVBoxLayout(adv_frame)
        adv_layout.setContentsMargins(4, 4, 4, 4)
        adv_layout.setSpacing(4)
        adv_label = QLabel("── Advance Criteria ──")
        adv_label.setStyleSheet("color: #888; font-weight: bold; font-size: 11px;")
        adv_layout.addWidget(adv_label)

        criteria = stage_data.get("advance_criteria", {})
        if isinstance(criteria, str):
            try:
                criteria = json.loads(criteria)
            except Exception:
                criteria = {}

        # Mode
        hl_mode = QHBoxLayout()
        hl_mode.addWidget(QLabel("Mode:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["auto", "manual", "hybrid"])
        idx = self._mode_combo.findText(str(stage_data.get("advance_mode", "auto")))
        if idx >= 0:
            self._mode_combo.setCurrentIndex(idx)
        hl_mode.addWidget(self._mode_combo)
        adv_layout.addLayout(hl_mode)

        # Min iterations
        hl_mi = QHBoxLayout()
        hl_mi.addWidget(QLabel("Min Iters:"))
        self._min_iter_spin = QSpinBox()
        self._min_iter_spin.setRange(0, 999999)
        self._min_iter_spin.setValue(int(criteria.get("min_iterations", 0)))
        hl_mi.addWidget(self._min_iter_spin)
        adv_layout.addLayout(hl_mi)

        # Min mean episode length
        hl_mel = QHBoxLayout()
        hl_mel.addWidget(QLabel("Min Ep Len:"))
        self._min_ep_spin = QSpinBox()
        self._min_ep_spin.setRange(0, 999999)
        self._min_ep_spin.setValue(int(criteria.get("min_mean_episode_length", 0)))
        hl_mel.addWidget(self._min_ep_spin)
        adv_layout.addLayout(hl_mel)

        root.addWidget(adv_frame)

        # ── Buttons ──
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def get_result(self) -> dict:
        """Return the edited stage dict."""
        overrides = {}
        for key, cb in self._checks.items():
            if not cb.isChecked():
                continue
            w = self._widgets[key]
            if isinstance(w, QComboBox):
                overrides[key] = w.currentText()
            elif isinstance(w, QDoubleSpinBox):
                overrides[key] = w.value()
            elif isinstance(w, QCheckBox):
                overrides[key] = w.isChecked()
            elif isinstance(w, QLineEdit):
                text = w.text().strip()
                # Try to parse as number or JSON
                try:
                    overrides[key] = json.loads(text)
                except (json.JSONDecodeError, ValueError):
                    overrides[key] = text
            else:
                overrides[key] = str(w.text()) if hasattr(w, "text") else ""

        criteria = {}
        mi = self._min_iter_spin.value()
        if mi > 0:
            criteria["min_iterations"] = mi
        mel = self._min_ep_spin.value()
        if mel > 0:
            criteria["min_mean_episode_length"] = mel

        return {
            "stage_id": self._id_edit.text().strip() or f"stage_{0}",
            "display_name": self._name_edit.text().strip() or "Untitled",
            "iterations": self._iter_spin.value(),
            "overrides": overrides,
            "advance_mode": self._mode_combo.currentText(),
            "advance_criteria": criteria,
            "eval_interval": 50,
        }


class _StageEditorWidget(QWidget):
    """Dynamic stage list editor for the ILPPOTrainerNode.

    Displays a compact list of training stages with add/remove/edit controls.
    Each stage shows: [index] [name] [iterations] [Edit] [x]
    Stores the full stage list as JSON in the node parameter.
    """

    height_changed = Signal(int)

    _STAGE_ROW_H = 28
    _HEADER_H = 28
    _FOOTER_H = 24
    _SPACING = 2

    def __init__(self, initial_value: str, write_fn, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._full_row_widget = True  # tells _reflow_layout to use full width + dynamic height
        self._write_fn = write_fn
        self._stages: List[dict] = []
        self._preferred_height = PARAM_ROW_H

        # Parse initial value
        try:
            parsed = json.loads(initial_value) if isinstance(initial_value, str) else initial_value
            if isinstance(parsed, list):
                self._stages = parsed
        except Exception:
            pass

        # Root layout
        self._root_layout = QVBoxLayout(self)
        self._root_layout.setContentsMargins(0, 0, 0, 0)
        self._root_layout.setSpacing(0)

        # Header: label + add button
        header = QWidget()
        header.setFixedHeight(self._HEADER_H)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(4, 0, 4, 0)
        hl.setSpacing(4)
        lbl = QLabel("Stages")
        lbl.setStyleSheet("font-weight: bold; font-size: 11px;")
        hl.addWidget(lbl)
        hl.addStretch()
        preset_btn = QPushButton("☰")
        preset_btn.setFixedSize(22, 20)
        preset_btn.setToolTip("Load stage preset")
        preset_btn.setCursor(Qt.PointingHandCursor)
        preset_btn.clicked.connect(self._on_load_preset)
        hl.addWidget(preset_btn)
        add_btn = QPushButton("+")
        add_btn.setFixedSize(22, 20)
        add_btn.setToolTip("Add new stage")
        add_btn.setCursor(Qt.PointingHandCursor)
        add_btn.clicked.connect(self._on_add_stage)
        hl.addWidget(add_btn)
        self._root_layout.addWidget(header)

        # Content area for stage rows
        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(self._SPACING)
        self._root_layout.addWidget(self._content)

        # Footer: total iterations
        self._footer = QLabel("")
        self._footer.setFixedHeight(self._FOOTER_H)
        self._footer.setStyleSheet("color: #888; font-size: 10px; padding-left: 4px;")
        self._root_layout.addWidget(self._footer)

        self._rebuild_rows()

    def _rebuild_rows(self) -> None:
        """Rebuild all stage row widgets from self._stages."""
        # Clear existing
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for idx, stage in enumerate(self._stages):
            row = self._make_stage_row(idx, stage)
            self._content_layout.addWidget(row)

        # Update footer
        total = sum(int(s.get("iterations", 0)) for s in self._stages)
        self._footer.setText(f"Total: {total} iters  |  {len(self._stages)} stage(s)")

        # Recalculate height
        n = len(self._stages)
        content_h = n * (self._STAGE_ROW_H + self._SPACING) if n > 0 else 0
        total_h = self._HEADER_H + content_h + self._FOOTER_H + 4
        self._preferred_height = max(PARAM_ROW_H, int(total_h))
        self._content.setFixedHeight(max(0, content_h))
        self.setMinimumHeight(self._preferred_height)
        self.setMaximumHeight(self._preferred_height)
        self.updateGeometry()
        self.height_changed.emit(self._preferred_height)

    def _make_stage_row(self, idx: int, stage: dict) -> QWidget:
        """Create a single stage row widget."""
        row = QWidget()
        row.setFixedHeight(self._STAGE_ROW_H)
        hl = QHBoxLayout(row)
        hl.setContentsMargins(4, 1, 4, 1)
        hl.setSpacing(3)

        # Index badge
        idx_lbl = QLabel(f"{idx}")
        idx_lbl.setFixedWidth(16)
        idx_lbl.setAlignment(Qt.AlignCenter)
        idx_lbl.setStyleSheet(
            "background: #444; color: #ccc; border-radius: 3px; "
            "font-size: 10px; font-weight: bold;"
        )
        hl.addWidget(idx_lbl)

        # Display name (truncated)
        name = str(stage.get("display_name", stage.get("stage_id", f"Stage {idx}")))
        name_lbl = QLabel(name)
        name_lbl.setToolTip(name)
        name_lbl.setStyleSheet("font-size: 11px;")
        hl.addWidget(name_lbl, stretch=1)

        # Iterations
        iters = int(stage.get("iterations", 0))
        iter_lbl = QLabel(f"{iters}it")
        iter_lbl.setFixedWidth(38)
        iter_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        iter_lbl.setStyleSheet("color: #999; font-size: 10px;")
        hl.addWidget(iter_lbl)

        # Edit button
        edit_btn = QPushButton("✎")
        edit_btn.setFixedSize(22, 20)
        edit_btn.setToolTip("Edit stage overrides")
        edit_btn.setCursor(Qt.PointingHandCursor)
        edit_btn.clicked.connect(lambda _=False, i=idx: self._on_edit_stage(i))
        hl.addWidget(edit_btn)

        # Delete button
        del_btn = QPushButton("×")
        del_btn.setFixedSize(22, 20)
        del_btn.setToolTip("Remove stage")
        del_btn.setCursor(Qt.PointingHandCursor)
        del_btn.clicked.connect(lambda _=False, i=idx: self._on_remove_stage(i))
        hl.addWidget(del_btn)

        return row

    def _on_add_stage(self) -> None:
        """Add a new empty stage."""
        idx = len(self._stages)
        new_stage = {
            "stage_id": f"stage_{idx}",
            "display_name": f"Stage {idx}",
            "iterations": 1000,
            "overrides": {},
            "advance_mode": "auto",
            "advance_criteria": {},
            "eval_interval": 50,
        }
        # Open editor immediately
        dlg = _StageOverrideDialog(new_stage)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._stages.append(dlg.get_result())
            self._write_back()
            self._rebuild_rows()

    def _on_edit_stage(self, idx: int) -> None:
        """Open the stage override editor dialog."""
        if idx < 0 or idx >= len(self._stages):
            return
        dlg = _StageOverrideDialog(self._stages[idx])
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._stages[idx] = dlg.get_result()
            self._write_back()
            self._rebuild_rows()

    def _on_remove_stage(self, idx: int) -> None:
        """Remove a stage by index."""
        if idx < 0 or idx >= len(self._stages):
            return
        self._stages.pop(idx)
        self._write_back()
        self._rebuild_rows()

    def _on_load_preset(self) -> None:
        """Show preset selection menu and load chosen stages."""
        from PySide6.QtWidgets import QMenu
        menu = QMenu()

        # Built-in presets
        presets = self._discover_presets()
        if presets:
            for name, path in presets:
                action = menu.addAction(f"📋 {name}")
                action.setData(path)
        else:
            no_action = menu.addAction("(no presets found)")
            no_action.setEnabled(False)

        menu.addSeparator()
        import_action = menu.addAction("📂 Import from YAML…")

        chosen = menu.exec(self.mapToGlobal(self.rect().bottomLeft()))
        if chosen is None:
            return

        if chosen is import_action:
            path, _ = QFileDialog.getOpenFileName(
                None, "Import Stage Preset", "",
                "YAML files (*.yaml *.yml);;All files (*)"
            )
            if path:
                stages = self._load_stages_from_yaml(path)
                if stages:
                    self._stages = stages
                    self._write_back()
                    self._rebuild_rows()
        elif chosen.data():
            stages = self._load_stages_from_yaml(str(chosen.data()))
            if stages:
                self._stages = stages
                self._write_back()
                self._rebuild_rows()

    @staticmethod
    def _discover_presets() -> List[tuple]:
        """Find available stage preset YAML files."""
        import pathlib
        presets = []
        # Knowledge base presets
        kb_dir = pathlib.Path(__file__).parent.parent.parent / "knowledge_base"
        for f in sorted(kb_dir.glob("*STAGES*.yaml")) if kb_dir.is_dir() else []:
            presets.append((f.stem, str(f)))
        # Custom mods presets
        cm_dir = pathlib.Path(__file__).parent.parent.parent / "custom_mods" / "training" / "stage_presets"
        if cm_dir.is_dir():
            for f in sorted(cm_dir.glob("*.yaml")):
                presets.append((f.stem, str(f)))
        return presets

    @staticmethod
    def _load_stages_from_yaml(path: str) -> List[dict]:
        """Parse a YAML file and extract stage definitions.

        Supports the RSI-AMP_STAGES.yaml format where stages are under
        the top-level 'stages' key, each with stage_id, display_name,
        iterations, overrides, advance_mode, advance_criteria, etc.
        """
        try:
            import yaml
        except ImportError:
            # Fallback: try to parse as JSON
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return data
            except Exception:
                pass
            return []

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception:
            return []

        if not isinstance(data, dict):
            return []

        raw_stages = data.get("stages", [])
        if not isinstance(raw_stages, list):
            return []

        result = []
        for s in raw_stages:
            if not isinstance(s, dict):
                continue
            # Build the stage dict matching TrainingStage.from_dict format
            stage = {
                "stage_id": str(s.get("id", s.get("stage_id", ""))),
                "display_name": str(s.get("display_name", s.get("id", "Untitled"))),
                "iterations": int(s.get("iterations", 1000)),
                "advance_mode": str(s.get("advance_criteria", {}).get("auto", {}) and "auto" or "auto"),
                "eval_interval": 50,
            }

            # Build overrides from stage-level keys
            overrides = {}
            if "training_mode" in s:
                overrides["training_mode"] = str(s["training_mode"])
            if "domain_rand_enabled" in s:
                overrides["domain_rand_enabled"] = bool(s["domain_rand_enabled"])
            # AMP params
            for key in ("amp_reward_coef", "task_reward_lerp", "disc_lr",
                        "disc_grad_penalty", "disc_label_smoothing"):
                if key in s:
                    overrides[key] = s[key]
            # Lerp schedule
            if "lerp_schedule" in s and isinstance(s["lerp_schedule"], dict):
                overrides["lerp_schedule"] = s["lerp_schedule"]
            # PPO params
            for key in ("learning_rate", "entropy_coef", "clip_param"):
                if key in s:
                    overrides[key] = s[key]

            stage["overrides"] = overrides

            # Advance criteria
            criteria = {}
            adv = s.get("advance_criteria", {})
            if isinstance(adv, dict):
                auto = adv.get("auto", {})
                if isinstance(auto, dict):
                    if "min_iterations" in auto:
                        criteria["min_iterations"] = int(auto["min_iterations"])
                    if "min_mean_episode_length" in auto:
                        criteria["min_mean_episode_length"] = int(auto["min_mean_episode_length"])
                    if "min_mean_reward" in auto:
                        criteria["min_mean_reward"] = float(auto["min_mean_reward"])
                    if "disc_accuracy_range" in auto:
                        criteria["disc_accuracy_range"] = auto["disc_accuracy_range"]
                stage["advance_mode"] = "auto" if auto else "manual"
                if adv.get("manual"):
                    stage["advance_mode"] = "hybrid" if auto else "manual"
            stage["advance_criteria"] = criteria

            result.append(stage)

        return result

    def _write_back(self) -> None:
        """Persist the current stages list to the node parameter."""
        self._write_fn(json.dumps(self._stages))

    def preferred_row_height(self) -> int:
        return self._preferred_height


# ---------------------------------------------------------------------------
# Reference Motion — specialised widgets
# ---------------------------------------------------------------------------

class _MotionLibraryPicker(_RichChoicePicker):
    """
    Unified Reference Motion picker.

    Single dropdown that exposes every way the user can supply a reference
    motion to a ReferenceMotionNode. Five entry classes:

      • 📁  SB3 library files          (custom_mods/training/motions/<cat>/<model>/*.npy)
            ★-pinned for the current robot_type.
      • 📁  Single archive clips       (custom_mods/archives/<pkg>/datasets/.../*.{txt,json})
      • ⚙   Procedural generators      ("generate:standing", "generate:walk")
      • 📦  Loco-mujoco datasets       ("loco:UnitreeGo2:walk", …)
      • 📦  Archive packs              ("pack:AMP_for_hardware:mocap_motions", …)
            Each pack expands at training-launch time to ALL files in its
            mocap subdir — the natural granularity for AMP discriminator
            training (the disc learns the joint distribution of a clip pool,
            not a single file). See AMP_design.yaml §3.

    Plus an ``<Import .npy to library>`` action for SB3 users who still
    want to drop a hand-recorded clip into the SB3 library.

    The picker writes back via ``write_params({motion_file, motion_source})``
    so it can correctly distinguish "single file" picks (write motion_file,
    clear motion_source) from "symbolic" picks (write motion_source, clear
    motion_file). The downstream ReferenceMotionConfig already understands
    both fields with the documented "motion_source overrides motion_file"
    precedence rule.
    """

    ADD_LABEL = "<Import .npy to library>"

    #: Built-in symbolic sources that have always been available via the
    #: legacy motion_source dropdown. Same set the SB3 backend already
    #: knows how to resolve via ReferenceMotionNode._resolve_motion_source.
    _GENERATOR_ENTRIES = (
        ("generate:standing", "generate:standing", "Procedural standing-pose reference (single frame)."),
        ("generate:walk",     "generate:walk",     "Procedural sinusoidal walking reference."),
    )
    _LOCO_ENTRIES = (
        ("loco:UnitreeGo2:walk", "loco:UnitreeGo2:walk", "loco-mujoco dataset · Unitree Go2 walking trajectory."),
        ("loco:UnitreeGo2:trot", "loco:UnitreeGo2:trot", "loco-mujoco dataset · Unitree Go2 trotting trajectory."),
        ("loco:UnitreeA1:walk",  "loco:UnitreeA1:walk",  "loco-mujoco dataset · Unitree A1 walking trajectory."),
    )

    def __init__(
        self,
        current_motion_file: str,
        current_motion_source: str = "",
        robot_type: str = "",
        write_params: Optional[Callable[[Dict[str, str]], None]] = None,
        parent=None,
    ) -> None:
        self._current_motion_file = str(current_motion_file or "")
        self._current_motion_source = str(current_motion_source or "")
        self._robot_type = str(robot_type or "").lower().strip()
        self._write_params = write_params
        self._suspend = False

        choices, meta, label_to_payload, payload_to_label = self._build_state()
        self._label_to_payload = label_to_payload
        self._payload_to_label = payload_to_label

        initial = self._initial_label()

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

    def _initial_label(self) -> str:
        """Pick the label that matches the node's current persisted state.

        Source field has precedence over motion_file (matches the backend
        resolution rule in ``ReferenceMotionNode.execute()``).
        """
        if self._current_motion_source:
            label = self._payload_to_label.get(
                ("source", self._current_motion_source)
            )
            if label:
                return label
        if self._current_motion_file:
            label = self._payload_to_label.get(
                ("file", self._current_motion_file)
            )
            if label:
                return label
        return self.ADD_LABEL

    def _build_state(self):
        """Return (choices, meta_map, label→(kind, value), (kind, value)→label).

        ``kind`` is one of ``"file"``, ``"source"``, ``"add"``. ``value`` is
        the absolute path (file kind), the symbolic identifier (source kind),
        or ``""`` (add kind).
        """
        from src.system.training.motion_library import (
            list_entries, get_category, list_motion_packs,
        )

        cat = get_category(self._robot_type) if self._robot_type else None
        entries = list_entries(category=cat)
        packs = list_motion_packs()

        # Sort SB3+archive single-file entries: same model first, then by
        # category/model/name.
        rt = self._robot_type
        entries_sorted = sorted(
            entries,
            key=lambda e: (0 if e.robot_model == rt else 1,
                           e.category, e.robot_model, e.name.lower()),
        )
        name_count = Counter(e.name for e in entries_sorted)

        choices: List[str] = []
        meta: Dict[str, Dict[str, str]] = {}
        label_to_payload: Dict[str, tuple] = {}
        payload_to_label: Dict[tuple, str] = {}

        # ── 1. ⚙ Procedural generators ──
        for label, src_id, desc in self._GENERATOR_ENTRIES:
            ui_label = f"⚙ {label}"
            choices.append(ui_label)
            meta[ui_label] = {"title": ui_label, "desc": desc}
            label_to_payload[ui_label] = ("source", src_id)
            payload_to_label[("source", src_id)] = ui_label

        # ── 2. 📦 Loco-mujoco datasets ──
        for label, src_id, desc in self._LOCO_ENTRIES:
            ui_label = f"📦 {label}"
            choices.append(ui_label)
            meta[ui_label] = {"title": ui_label, "desc": desc}
            label_to_payload[ui_label] = ("source", src_id)
            payload_to_label[("source", src_id)] = ui_label

        # ── 3. 📦 Archive packs (the AMP-natural granularity) ──
        for pack in packs:
            ui_label = f"📦 {pack.package} / {pack.subdir}"
            desc = (
                f"{pack.file_count} clips  ·  recommended for AMP "
                f"discriminator training (the disc learns this entire "
                f"motion distribution as the expert pool)."
            )
            choices.append(ui_label)
            meta[ui_label] = {"title": ui_label, "desc": desc}
            label_to_payload[ui_label] = ("source", pack.pack_id)
            payload_to_label[("source", pack.pack_id)] = ui_label

        # ── 4. 📁 Single-file entries (SB3 library + individual archive clips) ──
        for e in entries_sorted:
            if name_count[e.name] > 1:
                ui_label = f"📁 {e.name}  [{e.robot_model}]"
            else:
                ui_label = f"📁 {e.name}"
            star = "★ " if e.robot_model == rt else ""
            meta[ui_label] = {
                "title": f"{star}{e.name}",
                "desc": f"{e.category}  ·  {e.robot_model}  ·  {e.path.name}",
            }
            full = str(e.path)
            choices.append(ui_label)
            label_to_payload[ui_label] = ("file", full)
            payload_to_label[("file", full)] = ui_label

        # ── 5. <Add> import button ──
        choices.append(self.ADD_LABEL)
        meta[self.ADD_LABEL] = {
            "title": self.ADD_LABEL,
            "desc": "Browse for a .npy file and import it into the SB3 motion library.",
        }
        label_to_payload[self.ADD_LABEL] = ("add", "")

        return choices, meta, label_to_payload, payload_to_label

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _emit(self, motion_file: str, motion_source: str) -> None:
        """Push the new (file, source) pair to the node parameters."""
        self._current_motion_file = motion_file
        self._current_motion_source = motion_source
        if self._write_params is not None:
            self._write_params({
                "motion_file":   motion_file,
                "motion_source": motion_source,
            })

    def _on_changed(self, label: str) -> None:
        if self._suspend:
            return
        payload = self._label_to_payload.get(label)
        if payload is None:
            return
        kind, value = payload
        if kind == "add":
            self._do_add()
            return
        if kind == "file":
            # Single file → write motion_file, CLEAR motion_source so the
            # backend resolution rule (source > file) doesn't shadow us.
            self._emit(motion_file=value, motion_source="")
        elif kind == "source":
            # Symbolic identifier (generate:/loco:/pack:) → write
            # motion_source, CLEAR motion_file so the legacy SB3 path
            # doesn't accidentally hit a stale .npy.
            self._emit(motion_file="", motion_source=value)

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
        choices, meta, l2p, p2l = self._build_state()
        self._label_to_payload = l2p
        self._payload_to_label = p2l
        self._suspend = True
        self.set_choices(choices, meta_map=meta)
        self._suspend = False

        new_label = p2l.get(("file", str(entry.path)), "")
        if new_label:
            self.setCurrentText(new_label)
            self._emit(motion_file=str(entry.path), motion_source="")
        else:
            self._revert()

    def _revert(self) -> None:
        """Restore the previously selected entry without emitting."""
        self._suspend = True
        saved = self._initial_label()
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
        """Detect FPS from whatever the unified picker produced.

        Sources, in priority order (matches the picker's writeback rule):

          1. ``motion_source = "pack:<pkg>:<subdir>"`` →
                expand the pack, peek at the first file's FrameDuration.
          2. ``motion_source = "generate:..."`` / ``"loco:..."`` →
                no file metadata; suggest control_hz fallback (50 Hz).
          3. ``motion_file = <abs/path/to/clip.npy>`` →
                np.load and assume fps = control_hz (legacy SB3 path).
          4. ``motion_file = <abs/path/to/clip.txt|.json>`` (amp_legged_gym) →
                read FrameDuration from the JSON header. fps = 1 / dt.
                This is the *real* native fps of an AMP mocap clip and is
                much more accurate than the control_hz fallback.
        """
        params = getattr(self._logic_node, "parameters", None) or {}
        src = str(params.get("motion_source", "") or "").strip()
        mf = str(params.get("motion_file", "") or "").strip()

        target_path: Optional[str] = None
        kind: str = "unknown"      # "amp_json" | "npy" | "synthetic"

        # ── (1) pack: → first file in the pack ──
        if src.startswith("pack:"):
            try:
                from src.system.training.motion_library import expand_motion_pack
                files = expand_motion_pack(src)
                if files:
                    target_path = str(files[0])
                    kind = "amp_json"
            except Exception as exc:
                self._auto_btn.setToolTip(
                    f"⚠ Could not expand pack {src!r}: {exc}"
                )
                return

        # ── (2) generate:/loco: → synthetic, no file metadata ──
        elif src.startswith(("generate:", "loco:")):
            kind = "synthetic"

        # ── (3)/(4) explicit file path → infer kind from extension ──
        elif mf:
            target_path = mf
            ext = pathlib.Path(mf).suffix.lower()
            if ext in (".txt", ".json"):
                kind = "amp_json"
            elif ext == ".npy":
                kind = "npy"
            else:
                kind = "unknown"

        # Resolve control_hz from spec params (used as fallback default)
        control_hz = 50.0
        for key in ("control_hz", "control_frequency_hz"):
            raw = params.get(key)
            if raw is not None:
                try:
                    control_hz = float(raw)
                    break
                except (TypeError, ValueError):
                    pass

        if kind == "amp_json" and target_path:
            # Real AMP mocap header — extract FrameDuration directly.
            try:
                import json
                with open(target_path, "r", encoding="utf-8") as f:
                    j = json.load(f)
                frame_duration = float(j.get("FrameDuration", 0) or 0)
                n_frames = len(j.get("Frames", []) or [])
            except Exception as exc:
                self._auto_btn.setToolTip(
                    f"⚠ Could not parse {pathlib.Path(target_path).name}: {exc}"
                )
                return
            if frame_duration <= 0 or n_frames < 2:
                self._auto_btn.setToolTip(
                    f"⚠ {pathlib.Path(target_path).name} has invalid "
                    f"FrameDuration={frame_duration} or n_frames={n_frames}."
                )
                return
            suggested = min(self._MAX_FPS, max(1, round(1.0 / frame_duration)))
            duration_s = n_frames * frame_duration
            label = pathlib.Path(target_path).name
            extra = ""
            if src.startswith("pack:"):
                extra = f"\nPack: {src} (showing first member)"
            self._auto_btn.setToolTip(
                f"File: {label}{extra}\n"
                f"Frames: {n_frames}  ·  FrameDuration: {frame_duration:.4f}s\n"
                f"Native FPS: {suggested}  →  clip length {duration_s:.2f}s"
            )

        elif kind == "npy" and target_path:
            # Legacy SB3 .npy: no embedded fps, fall back to control_hz.
            try:
                import numpy as np
                arr = np.load(target_path, mmap_mode="r")
                T = int(arr.shape[0])
            except Exception:
                self._auto_btn.setToolTip(
                    f"⚠ Could not read file: {pathlib.Path(target_path).name}"
                )
                return
            suggested = min(self._MAX_FPS, max(1, round(control_hz)))
            duration_s = T / suggested if suggested > 0 else 0.0
            self._auto_btn.setToolTip(
                f"File: {pathlib.Path(target_path).name}\n"
                f"Frames: {T}  ·  Control rate: {control_hz:.0f} Hz\n"
                f"Suggested FPS: {suggested}  →  plays in {duration_s:.2f} s"
            )

        elif kind == "synthetic":
            # generate: / loco: are resolved at runtime — no file to peek at.
            suggested = min(self._MAX_FPS, max(1, round(control_hz)))
            self._auto_btn.setToolTip(
                f"Source {src!r}: synthetic / loco-mujoco trajectory has no "
                f"embedded FPS metadata.\n"
                f"Suggested FPS = control_hz = {suggested}."
            )

        else:
            self._auto_btn.setToolTip(
                "⚠ No motion source set. Pick a clip / pack first, then click Auto."
            )
            return

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


def _guess_motion_format_id(path: str) -> str:
    """Map a motion file path to the loader format_id used by ``get_loader``.

    .npy → legacy unitport_npy; .txt/.json → amp_legged_gym DeepMimic layout.
    Unknown extensions fall back to unitport_npy so pre-phase_2 behavior is
    preserved when the picker is in legacy mode.
    """
    ext = pathlib.Path(str(path or "")).suffix.lower()
    if ext == ".npy":
        return "unitport_npy"
    if ext in (".txt", ".json"):
        return "amp_legged_gym"
    return "unitport_npy"


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
        motion_path: str,
        fps: float,
        format_id: str = "",
        mode: str = "physics",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._scene_xml   = scene_xml
        self._motion_path = motion_path
        self._fps_hint    = max(1.0, float(fps) if fps > 0 else 30.0)
        self._format_id   = format_id or _guess_motion_format_id(motion_path)
        self._mode        = mode if mode in ("physics", "kinematic") else "physics"
        self._stop_flag   = False

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
            # ── Load motion through the unified loader pipeline ──
            # unitport_npy → .npy joint frames; amp_legged_gym → .txt/.json
            # with full root+joint DeepMimic layout. Both produce a
            # MotionClip with joint_pos (and for amp, root_pos/root_quat).
            try:
                from src.system.training.motion.loader import get_loader, LoaderError
                loader = get_loader(self._format_id, fps=self._fps_hint) \
                    if self._format_id == "unitport_npy" \
                    else get_loader(self._format_id)
                clip = loader.load(self._motion_path)
            except LoaderError as exc:
                self.error.emit(f"Loader error: {exc}")
                self.stopped.emit()
                return
            except Exception as exc:
                self.error.emit(f"Failed to load motion: {exc}")
                self.stopped.emit()
                return

            frames = np.asarray(clip.joint_pos, dtype=np.float32)
            if frames.ndim != 2 or frames.shape[0] < 2:
                self.error.emit(
                    f"MotionClip {clip.name!r} has invalid joint_pos shape {frames.shape}"
                )
                self.stopped.emit()
                return

            # Clip's own fps is authoritative when the loader parsed it
            # from file metadata (amp FrameDuration). Fall back to the UI
            # slider only for unitport_npy which carries no timing header.
            self._fps = float(clip.fps) if clip.fps > 0 else self._fps_hint

            has_root = (
                clip.root_pos is not None
                and clip.root_quat is not None
                and clip.root_pos.shape[0] == frames.shape[0]
            )

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

            # MotionClip.joint_pos is already DOF-only (no root block), so
            # unlike the old raw-.npy path we don't need to strip a free-joint
            # prefix — but we still accept legacy npy files that included it.
            if n_cols >= jnt_offset + n_actuated and self._format_id == "unitport_npy":
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
                if has_root:
                    data.qpos[0:3] = np.asarray(clip.root_pos[0], dtype=np.float64)
                    # MotionClip stores quat as (x,y,z,w); MuJoCo wants (w,x,y,z).
                    qx, qy, qz, qw = [float(v) for v in clip.root_quat[0]]
                    data.qpos[3:7] = [qw, qx, qy, qz]
                else:
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

                    # ── Kinematic replay: no physics, just drop the ──
                    # reference pose straight into qpos. This shows the
                    # motion as authored regardless of PD gain/gravity.
                    if self._mode == "kinematic":
                        if jnt_offset >= 7 and has_root:
                            data.qpos[0:3] = np.asarray(clip.root_pos[ref_idx], dtype=np.float64)
                            qx, qy, qz, qw = [float(v) for v in clip.root_quat[ref_idx]]
                            data.qpos[3:7] = [qw, qx, qy, qz]
                        data.qpos[jnt_offset: jnt_offset + n_joints] = q_ref
                        mujoco.mj_forward(model, data)

                        try:
                            viewer.set_texts([
                                (None, None, "Mode", "Kinematic Replay"),
                                (None, None, "Clip", clip.name),
                                (None, None, "Phase", f"{ref_idx}/{n_frames} ({100*ref_idx/n_frames:.0f}%)"),
                                (None, None, "FPS", f"{self._fps:.0f} Hz"),
                            ])
                        except Exception:
                            pass
                        viewer.sync()
                        frame_idx += 1
                        expected = wall_start + frame_idx * control_dt
                        sleep_t = expected - time.monotonic()
                        if sleep_t > 0:
                            time.sleep(sleep_t)
                        continue

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
                            (None, None, "Clip", clip.name),
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
    """▶ Preview / ■ Stop toggle button for the ReferenceMotionNode.

    Layout: ``[Clip picker (RichChoicePicker)] [▶]``.

    When the active motion source is a single file, the picker is hidden
    and the button previews that one file. When the source is a
    ``pack:<package>:<subdir>`` identifier, the picker lists every clip
    in the pack (labeled by file stem) and the user can flip through
    them and preview each clip individually — this matters for AMP
    packs where the 📦 entry is the *training-time* granularity but
    users still want eyeballs on individual members before committing.

    Physics-vs-kinematic mode lives on its own row (``preview_physics``
    toggle) and is read from ``logic_node.parameters`` at click time.
    """

    def __init__(
        self,
        get_candidates: Callable[[], List[tuple]],
        get_fps: Callable[[], float],
        get_robot_type: Callable[[], str],
        get_scene_xml_override: Optional[Callable[[], str]] = None,
        get_mode: Optional[Callable[[], str]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        # get_candidates() → list of (label, absolute_path) tuples.
        # Single-file source → 1-element list; pack source → N-element list.
        self._get_candidates = get_candidates
        self._get_fps        = get_fps
        self._get_robot_type = get_robot_type
        # Optional: resolve an explicit MJCF path (e.g. from the selected
        # IL Robot Asset's asset_id). Takes precedence over the generic
        # robot_type → registered-scene fallback.
        self._get_scene_xml_override = get_scene_xml_override
        # Optional: return "physics" or "kinematic" — read from the
        # sibling preview_physics toggle row.
        self._get_mode = get_mode
        self._thread: Optional[_MotionPreviewThread] = None
        self._candidates: List[tuple] = []
        # choice_label (file stem, possibly disambiguated) → absolute path
        self._clip_path_by_choice: Dict[str, str] = {}

        # Lock the whole widget to the standard node widget column so the
        # row stays inside NODE_W like every other parameter row.
        self.setFixedWidth(WIDGET_W)
        self.setFixedHeight(max(20, PARAM_ROW_H - 4))
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        hl = QHBoxLayout(self)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(4)

        # ── Clip picker — project's custom _RichChoicePicker ────────
        # QComboBox misbehaves inside QGraphicsProxyWidget (popup anchor
        # drifts, styling lost), so every other dropdown on training
        # nodes uses _RichChoicePicker. Match the convention.
        #
        # choices_provider runs every time the popup opens, re-querying
        # the parent node's current motion_source. That's how we pick up
        # changes made by the sibling Reference Motion picker without
        # depending on the proxy being rebuilt.
        self._clip_picker = _RichChoicePicker(
            [],
            current="",
            leading_mode="checkbox",
            multi_select=False,
            choices_provider=self._provide_choices,
        )
        self._clip_picker.setToolTip("Pick which clip in the pack to preview.")
        self._clip_picker.setVisible(False)
        # Cap width inside the row so the ▶ button always fits.
        self._clip_picker.setMaximumWidth(WIDGET_W - 22 - 4)
        self._clip_picker.setFixedHeight(max(18, PARAM_ROW_H - 6))
        hl.addWidget(self._clip_picker, 1)

        # ── Play / Stop icon button ──────────────────────────────────
        self._btn = QPushButton("▶")
        self._btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn.setToolTip(
            "Open a MuJoCo physics simulation with PD tracking.\n"
            "Shows real-time base velocity, tracking error, and height\n"
            "to verify the motion is physically plausible."
        )
        self._btn.setFixedWidth(22)
        self._btn.setFixedHeight(max(18, PARAM_ROW_H - 6))
        self._btn.clicked.connect(self._toggle)
        hl.addWidget(self._btn, 0)

        self._refresh_candidates()

    # ------------------------------------------------------------------

    def _provide_choices(self) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
        """Build (labels, meta) from the parent node's current candidates.

        This is the ``choices_provider`` wired into ``_RichChoicePicker``
        — it runs every time the clip popup opens, so the list always
        reflects the live motion_source even after the sibling Reference
        Motion picker has changed it. Side effects: updates
        ``self._candidates`` and ``self._clip_path_by_choice`` so
        ``_current_path()`` can map the picker's string selection back
        to an absolute file path.
        """
        try:
            cands = list(self._get_candidates() or [])
        except Exception:
            cands = []
        # Disambiguate same-stem clips from different subdirs with a
        # trailing "(n)" suffix so the picker's string keys stay unique.
        labels: List[str] = []
        seen: Dict[str, int] = {}
        path_by_choice: Dict[str, str] = {}
        meta_map: Dict[str, Dict[str, str]] = {}
        for raw_label, path in cands:
            base = str(raw_label)
            if base in seen:
                seen[base] += 1
                label = f"{base} ({seen[base]})"
            else:
                seen[base] = 1
                label = base
            labels.append(label)
            path_by_choice[label] = str(path)
            meta_map[label] = {"title": label, "desc": str(path)}
        self._candidates = cands
        self._clip_path_by_choice = path_by_choice
        return labels, meta_map

    def _refresh_candidates(self) -> None:
        """Synchronously rebuild the picker's choices + visibility.

        Called at construction, on ▶ click, and from the parent node's
        ``write_params`` whenever motion_file/motion_source mutates.
        The popup's own just-in-time refresh (via ``choices_provider``)
        covers the click-the-dropdown case; this method covers the
        ambient "change the source, see the button relabel itself" case.
        """
        labels, meta_map = self._provide_choices()

        # Remember prior selection so an unrelated source change doesn't
        # stomp a deliberate pack clip pick. Only fall back to labels[0]
        # when the previous choice is no longer in the list.
        prev = self._clip_picker.currentText()
        self._clip_picker.set_choices(labels, meta_map=meta_map)
        if prev and prev in labels:
            self._clip_picker.setCurrentText(prev)
        elif labels:
            self._clip_picker.setCurrentText(labels[0])

        # Show the picker only when the source is a pack (>1 clip);
        # single-file sources don't need it and the extra widget would
        # just steal row width.
        self._clip_picker.setVisible(len(self._candidates) > 1)

    def _current_path(self) -> str:
        if not self._candidates:
            return ""
        if self._clip_picker.isVisible():
            choice = self._clip_picker.currentText()
            if choice and choice in self._clip_path_by_choice:
                return self._clip_path_by_choice[choice]
        return str(self._candidates[0][1])

    def _toggle(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            self._thread.request_stop()
            return

        # Always re-scan candidates on click so picker changes since the
        # widget was built are reflected without a panel rebuild.
        self._refresh_candidates()

        path = self._current_path()
        if not path:
            self._btn.setToolTip("⚠ Select a motion file or pack first, then click Preview.")
            return

        xml = ""
        if self._get_scene_xml_override is not None:
            try:
                xml = str(self._get_scene_xml_override() or "")
            except Exception:
                xml = ""
        if not xml:
            xml = _find_scene_xml_for_preview(self._get_robot_type()) or ""
        if not xml:
            self._btn.setToolTip("⚠ Could not locate a MuJoCo scene XML for this robot type.")
            return

        fps = self._get_fps()
        mode = "physics"
        if self._get_mode is not None:
            try:
                mode = str(self._get_mode() or "physics")
            except Exception:
                mode = "physics"
        self._thread = _MotionPreviewThread(
            xml, path, fps,
            format_id=_guess_motion_format_id(path),
            mode=mode,
        )
        self._thread.stopped.connect(self._on_stopped)
        self._thread.error.connect(self._on_error)
        self._thread.start()

        self._btn.setText("■")
        self._btn.setToolTip("Close the MuJoCo viewer.")

    def _on_stopped(self) -> None:
        self._btn.setText("▶")
        self._btn.setToolTip(
            "Open a MuJoCo physics simulation with PD tracking.\n"
            "Shows real-time base velocity, tracking error, and height\n"
            "to verify the motion is physically plausible."
        )

    def _on_error(self, msg: str) -> None:
        self._btn.setText("▶")
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
        # Ref to the Preview button widget so write_params can push
        # clip-list refreshes into it when the motion source changes.
        self._ref_preview_button_widget: Optional[_PreviewButtonWidget] = None
        # Refs used by preset pickers to refresh the module editor
        self._reward_editor_widget: Optional[_RegistryModuleEditor] = None
        self._termination_editor_widget: Optional[_RegistryModuleEditor] = None

        self._style: Dict[str, str] = _style_for_node(node_type)

        # SafetyReview overlay status — set by the workspace layer via
        # :meth:`set_safety_status`. Values: None / "warning" / "error".
        # Drives a coloured border overlay in :meth:`paint`. Cleared
        # automatically when SafetyReview re-runs with no issues on
        # this node.
        self._safety_status: Optional[str] = None

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

    # Swappable node pairs: node_type → (swap_target_type, swap_target_display_name)
    _SWAP_TARGETS: Dict[str, Tuple[str, str]] = {
        "il_velocity_cmd":  ("il_gamepad_input", "IL Gamepad Input"),
        "il_gamepad_input": ("il_velocity_cmd",  "IL Velocity Command"),
    }

    def contextMenuEvent(self, event) -> None:
        swap = self._SWAP_TARGETS.get(self._node_type)
        if not swap:
            super().contextMenuEvent(event)
            return
        from PySide6.QtWidgets import QMenu
        menu = QMenu()
        target_type, target_display = swap
        act = menu.addAction(f"Replace with {target_display}")
        chosen = menu.exec(event.screenPos())
        if chosen == act:
            scene = self.scene()
            if scene is not None and hasattr(scene, "_swap_node_type"):
                scene._swap_node_type(self, target_type, target_display)
        event.accept()

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
        # Prefer the unified Robot node's asset_id (which IS the robot_type
        # for registry-resolved assets). Fall back to the deprecated
        # robot_mjcf node's robot_type param for backwards compatibility.
        if self._logic_node is not None:
            if self._node_type == "robot":
                aid = str((self._logic_node.parameters or {}).get("asset_id", "")).strip()
                if aid:
                    return aid
            if self._node_type == "robot_mjcf":
                return str((self._logic_node.parameters or {}).get("robot_type", "")).strip()
        scene = self.scene()
        if scene is None:
            return ""
        for item in scene.items():
            if item is self:
                continue
            try:
                nt = item.data(12)
                if nt not in ("robot", "robot_mjcf"):
                    continue
                logic = item.data(13)
                params = getattr(logic, "parameters", {}) or {}
                if nt == "robot":
                    aid = str(params.get("asset_id", "")).strip()
                    if aid:
                        return aid
                else:
                    robot_type = str(params.get("robot_type", "")).strip()
                    if robot_type:
                        return robot_type
            except Exception:
                continue
        return ""

    def _resolve_canvas_robot_family(self) -> str:
        return resolve_robot_family(self._resolve_canvas_robot_type())

    def _resolve_canvas_algorithm(self) -> str:
        """Infer the RL algorithm from whichever trainer node sits on the
        same canvas.  Returns uppercase string: "PPO", "AMP", "SAC", etc.
        Falls back to "ALL" when no trainer node is found so nothing is
        filtered out."""
        _TRAINER_ALG_MAP = {
            "amp_ppo_trainer": "AMP",
            "il_ppo_trainer": "PPO",
            "train": "PPO",           # SB3 default
        }
        scene = self.scene()
        if scene is None:
            return "ALL"
        for item in scene.items():
            if item is self:
                continue
            try:
                ntype = item.data(12)
                if ntype in _TRAINER_ALG_MAP:
                    return _TRAINER_ALG_MAP[ntype]
                logic = item.data(13)
                if logic is not None:
                    p = getattr(logic, "parameters", {}) or {}
                    alg = str(p.get("algorithm", "")).strip().upper()
                    if alg in ("PPO", "SAC", "TD3", "AMP"):
                        return alg
            except Exception:
                continue
        return "ALL"

    # ------------------------------------------------------------------
    # Body IR validator widget
    # ------------------------------------------------------------------

    def _make_body_ir_validator(self, key, get_val, write_val, logic):
        """Build the body-mapping validator for the Joints Mapping node.

        - Each IR role is one body part (``Foot FL``, ``Thigh FR``, …).
        - Only shows roles **needed** by the rewards/terminations/DR on
          the current canvas (scans for ``{ir:*}`` in ``il_params``).
        - Uses ``_RichChoicePicker`` with a ``None`` option.
        - Defers ``_rebuild`` via ``QTimer`` to avoid deleting a widget
          from within its own signal handler (Qt crash).
        """
        from PySide6.QtCore import QTimer
        from src.system.training.body_ir import (
            BodyIRMapper, split_compound_categories, AUTO_FROM_ASSET_CATEGORIES,
        )

        MAX_W = NODE_W - H_PAD * 2
        ROW_H = 20
        HDR_H = 14
        BTN_H = 18
        COL_ST = 14
        COL_IR = 62
        COL_BD = MAX_W - COL_ST - COL_IR - 12

        border = get_color("training_widget_border", "#3d3d3d")
        input_bg = get_color("training_widget_input_bg", "#1A1A1A")
        text_c = get_color("training_widget_text", "#cccccc")
        muted = get_color("training_widget_label_text", "#aaaaaa")
        ok_c = "#4CAF50"
        err_c = "#F44336"
        skip_c = "#777777"

        _self_ref = self

        container = QWidget()
        container._full_row_widget = True
        container._preferred_height = PARAM_ROW_H

        root = QVBoxLayout(container)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(1)

        table_wrap = QFrame(container)
        table_wrap.setStyleSheet(
            f"background: {input_bg}; border: 1px solid {border}; border-radius: 4px;")
        tw_layout = QVBoxLayout(table_wrap)
        tw_layout.setContentsMargins(2, 2, 2, 2)
        tw_layout.setSpacing(0)
        root.addWidget(table_wrap)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setFixedHeight(BTN_H)
        refresh_btn.setStyleSheet(f"color: {text_c}; font-size: 9px;")
        root.addWidget(refresh_btn)

        _row_widgets: list = []
        _mapper_ref: list = [None]
        _rebuild_scheduled: list = [False]

        _height_cbs: list = []

        def _emit_h(h: int):
            container._preferred_height = h
            container.setMinimumHeight(h)
            container.setMaximumHeight(h)
            container.updateGeometry()
            for fn in _height_cbs:
                try:
                    fn(h)
                except Exception:
                    pass

        class _HSig:
            def connect(self_, fn):
                _height_cbs.append(fn)
            def emit(self_, h):
                _emit_h(h)

        container.height_changed = _HSig()
        container.preferred_row_height = lambda: max(PARAM_ROW_H, int(container._preferred_height))

        # ── resolve robot asset ──
        def _resolve_asset():
            scene = _self_ref.scene()
            if scene is None:
                return [], "quadruped"
            # Prefer the unified Robot node; fall back to the deprecated
            # il_robot_asset for transitional canvases.
            # When this widget is hosted on the Robot node itself, read
            # the asset_id directly off _self_ref.
            if getattr(_self_ref, "_node_type", "") == "robot":
                try:
                    logic = _self_ref._logic_node
                    p = getattr(logic, "parameters", {}) or {}
                    aid = str(p.get("asset_id", "")).strip()
                    if aid:
                        from src.system.training.robot_assets import get_asset, rescan
                        rescan()
                        asset = get_asset(aid)
                        bodies: list = []
                        if asset.base_link:
                            bodies.append(asset.base_link)
                        for jn in (asset.joint_names or []):
                            if jn not in bodies:
                                bodies.append(jn)
                        for fn in (asset.foot_link_names or []):
                            if fn not in bodies:
                                bodies.append(fn)
                        family = getattr(asset, "family", "quadruped") or "quadruped"
                        return bodies, family
                except Exception:
                    pass
            for item in scene.items():
                if item is _self_ref:
                    continue
                try:
                    nt = item.data(12)
                    if nt not in ("robot", "il_robot_asset"):
                        continue
                    nl = item.data(13)
                    if nl is None:
                        continue
                    p = getattr(nl, "parameters", {}) or {}
                    aid = str(p.get("asset_id", "")).strip()
                    if not aid:
                        continue
                    from src.system.training.robot_assets import get_asset, rescan
                    rescan()
                    asset = get_asset(aid)
                    bodies: list = []
                    if asset.base_link:
                        bodies.append(asset.base_link)
                    for jn in (asset.joint_names or []):
                        if jn not in bodies:
                            bodies.append(jn)
                    for fn in (asset.foot_link_names or []):
                        if fn not in bodies:
                            bodies.append(fn)
                    # Infer foot links from calf joints if parser missed them
                    if not asset.foot_link_names:
                        import re as _re
                        from src.system.training.body_ir import joint_name_to_link_name
                        for jn in (asset.joint_names or []):
                            ln = joint_name_to_link_name(jn)
                            if "calf" in ln.lower() or "shank" in ln.lower():
                                cand = _re.sub(r"(?i)calf|shank", "foot", ln)
                                if cand != ln and cand not in bodies:
                                    bodies.append(cand)
                    family = getattr(asset, "family", "quadruped") or "quadruped"
                    return bodies, family
                except Exception:
                    continue
            return [], "quadruped"

        # Hosting on the Robot node means "show the full structural view"
        # — every body of the selected asset with its auto-resolved IR
        # role, read-only. When hosted elsewhere (legacy joints_mapping),
        # the old needed-filter / picker behaviour applies.
        _display_only = (getattr(_self_ref, "_node_type", "") == "robot")

        # ── scan canvas for needed categories ──
        def _needed_categories() -> set:
            """Parse {ir:*} placeholders from rewards + terminations on canvas."""
            # Display-only mode on the Robot node: no filtering — we
            # surface every role the mapper built so the user sees the
            # complete structural view of the asset.
            if _display_only:
                return set()
            import re as _re
            from src.system.training.task_module_registry import IL_REWARD_REGISTRY
            needed: set = set()
            scene = _self_ref.scene()
            if scene is None:
                return needed
            for item in scene.items():
                if item is _self_ref:
                    continue
                try:
                    ntype = item.data(12)
                    if ntype != "rewards":
                        continue
                    nl = item.data(13)
                    if nl is None:
                        continue
                    p = getattr(nl, "parameters", {}) or {}
                    if p.get("backend", "sb3") != "isaac_lab":
                        continue
                    terms = json.loads(p.get("reward_terms", "{}"))
                    for rk in terms:
                        entry = IL_REWARD_REGISTRY.get(rk)
                        if entry and entry.il_params:
                            for m in _re.finditer(r"\{ir:(\w+)\}", entry.il_params):
                                for cat in split_compound_categories(m.group(1)):
                                    needed.add(cat)
                except Exception:
                    continue
            # base is always needed (mass randomisation, illegal_contact)
            needed.add("base")
            return needed

        # ── rebuild ──
        def _rebuild():
            _rebuild_scheduled[0] = False
            # Guard: _rebuild is wired to scene-wide signals (reward edits on
            # other nodes trigger a re-scan here).  If the containing layout
            # or table wrapper has been destroyed — e.g. the joints_mapping
            # node was removed, or the scene is being torn down — touching
            # tw_layout raises ``RuntimeError: Internal C++ object already
            # deleted``.  Bail out cleanly in that case.
            try:
                tw_layout.count()
            except RuntimeError:
                return
            for w in list(_row_widgets):
                try:
                    tw_layout.removeWidget(w)
                    w.deleteLater()
                except RuntimeError:
                    pass
            _row_widgets.clear()

            body_names, family = _resolve_asset()
            needed = _needed_categories()

            raw = str((logic.parameters or {}).get(key, ""))
            if raw:
                try:
                    mapper = BodyIRMapper.from_dict(json.loads(raw))
                    if (set(mapper.body_names) != set(body_names) and body_names) \
                            or mapper.family != family:
                        mapper = BodyIRMapper.from_body_list(body_names, family)
                except Exception:
                    mapper = BodyIRMapper.from_body_list(body_names, family)
            else:
                mapper = BodyIRMapper.from_body_list(body_names, family)
            _mapper_ref[0] = mapper

            if not body_names:
                msg = "Select an asset_id" if _display_only else "Connect robot_entity input"
                empty = QLabel(msg)
                empty.setFixedHeight(ROW_H)
                empty.setAlignment(Qt.AlignCenter)
                empty.setStyleSheet(f"color: {err_c}; font-size: 10px; background: transparent;")
                _row_widgets.append(empty)
                tw_layout.addWidget(empty)
                _write(mapper)
                table_wrap.setFixedHeight(ROW_H + 4)
                _emit_h(ROW_H + 4 + BTN_H + 2)
                return

            # Header — new column order: [Current Body, Role, Status]
            hdr = QWidget(table_wrap)
            hdr.setFixedHeight(HDR_H)
            hl = QHBoxLayout(hdr)
            hl.setContentsMargins(4, 0, 4, 0)
            hl.setSpacing(4)
            for txt, w in [("Current", COL_BD), ("Role", COL_IR), ("", COL_ST)]:
                lb = QLabel(txt)
                lb.setFixedWidth(w)
                lb.setStyleSheet(
                    f"color: {muted}; font-size: 8px; font-weight: 600; "
                    f"background: transparent; border: none;"
                )
                hl.addWidget(lb)
            _row_widgets.append(hdr)
            tw_layout.addWidget(hdr)

            # Display-only mode on the Robot node: show every role the
            # mapper produced — this is the full structural view of the
            # asset. Legacy joints_mapping mode filters by rewards-needed
            # categories and hides auto-populated roles.
            if _display_only:
                visible_roles = list(mapper.roles)
            else:
                visible_roles = [r for r in mapper.roles
                                 if (r.category in needed or not r.category)
                                 and not r.auto_from_asset]
            if not visible_roles:
                info = QLabel("No body roles needed")
                info.setFixedHeight(ROW_H)
                info.setAlignment(Qt.AlignCenter)
                info.setStyleSheet(f"color: {muted}; font-size: 9px; background: transparent;")
                _row_widgets.append(info)
                tw_layout.addWidget(info)
                _write(mapper)
                table_wrap.setFixedHeight(HDR_H + ROW_H + 4)
                _emit_h(HDR_H + ROW_H + 4 + BTN_H + 2)
                return

            for idx, role in enumerate(visible_roles):
                rw = QWidget(table_wrap)
                rw.setFixedHeight(ROW_H)
                rl = QHBoxLayout(rw)
                rl.setContentsMargins(4, 0, 4, 0)
                rl.setSpacing(4)
                # Subtle zebra striping for the table feel — no per-cell
                # borders anywhere, only the outer table_wrap frame.
                zebra = input_bg if (idx % 2) else "transparent"
                rw.setStyleSheet(
                    f"QWidget {{ background: {zebra}; border: none; }}"
                )

                from src.system.training.body_ir import joint_name_to_link_name

                # Column 1 — Current Body
                if _display_only:
                    body_txt = (role.body or "—")
                    lbl = QLabel(body_txt)
                    lbl.setFixedWidth(COL_BD)
                    lbl.setFixedHeight(ROW_H - 2)
                    lbl.setStyleSheet(
                        f"color: {text_c}; font-size: 9px; "
                        f"background: transparent; border: none; padding: 0 2px;"
                    )
                    rl.addWidget(lbl)
                else:
                    # USD body picker — choices use link names (what the
                    # compiled config emits), not raw joint names.
                    link_names = list(dict.fromkeys(
                        joint_name_to_link_name(b) for b in body_names))
                    choices = ["None"] + link_names
                    current = role.body if role.body else "None"
                    picker = _RichChoicePicker(
                        choices,
                        current=current,
                        parent=rw,
                    )
                    picker.setFixedSize(COL_BD, ROW_H - 2)

                    def _on_pick(selected, _rid=role.role_id):
                        if not selected:
                            return
                        val = selected[0] if isinstance(selected, list) else selected
                        cur = _mapper_ref[0]
                        cur.override(_rid, None if val == "None" else val)
                        _write(cur)
                        _schedule_rebuild()

                    picker.selectionChanged.connect(_on_pick)
                    rl.addWidget(picker)

                # Column 2 — Role label
                rlbl = QLabel(role.label)
                rlbl.setFixedWidth(COL_IR)
                rlbl.setStyleSheet(
                    f"color: {text_c}; font-size: 9px; "
                    f"background: transparent; border: none;"
                )
                rl.addWidget(rlbl)

                # Column 3 — Status
                if role.resolved:
                    ic, ic_c = "OK", ok_c
                elif role.required:
                    ic, ic_c = "!!", err_c
                else:
                    ic, ic_c = "--", skip_c
                sl = QLabel(ic)
                sl.setFixedWidth(COL_ST)
                sl.setAlignment(Qt.AlignCenter)
                sl.setStyleSheet(
                    f"color: {ic_c}; font-size: 9px; font-weight: bold; "
                    f"background: transparent; border: none;"
                )
                rl.addWidget(sl)

                _row_widgets.append(rw)
                tw_layout.addWidget(rw)

            _write(mapper)

            table_h = HDR_H + len(visible_roles) * ROW_H + 4
            table_wrap.setFixedHeight(table_h)
            _emit_h(table_h + BTN_H + 2)

        def _schedule_rebuild():
            """Defer _rebuild to next event loop tick (avoids deleting
            the picker widget from within its own signal handler)."""
            if not _rebuild_scheduled[0]:
                _rebuild_scheduled[0] = True
                QTimer.singleShot(0, _rebuild)

        def _write(mapper):
            write_val(json.dumps(mapper.to_dict(), ensure_ascii=False))

        # ── auto-refresh when upstream nodes change ──
        def _on_canvas_param_changed(node_type: str, param_key: str, _value: str):
            """Rebuild when robot asset or reward selection changes."""
            if (node_type in ("robot", "il_robot_asset") and param_key == "asset_id") \
                    or (node_type == "rewards" and param_key == "reward_terms"):
                _schedule_rebuild()

        def _connect_scene_signal():
            # Guard against shiboken-deleted C++ items: if the owning
            # TrainingNodeItem was torn down between QTimer.singleShot
            # scheduling and firing (common during startup pre-warm),
            # ``_self_ref.scene()`` raises RuntimeError. Treat it as
            # "nothing to connect" — the widget is gone anyway.
            try:
                scene = _self_ref.scene()
            except RuntimeError:
                return
            if scene is not None and hasattr(scene, "node_param_changed"):
                try:
                    scene.node_param_changed.connect(_on_canvas_param_changed)
                except Exception:
                    pass

        QTimer.singleShot(0, _rebuild)
        # Connect scene signal after scene is ready
        QTimer.singleShot(50, _connect_scene_signal)
        refresh_btn.clicked.connect(lambda: _schedule_rebuild())

        return container

    # ------------------------------------------------------------------
    # Controller status panel (TrainingCommands.controller_status)
    # ------------------------------------------------------------------

    def _make_controller_status_panel(self, key, get_val, write_val, logic):
        """Read-only sidebar-mirror for the Training Commands node.

        Displays, live:

          * Current :class:`GlobalInputManager` mode (Gamepad / Keyboard
            / Off) and ``invert_vy`` flag — the same state the sidebar
            Controller panel edits. The user changes those in the
            sidebar; this widget only reflects them.
          * The fixed gamepad binding convention (Left Stick Y → vx,
            Left Stick X → vy, Right Stick X → vyaw).
          * The effective velocity ranges derived from the ``gain_*``
            sliders on the same node. Training sampler and deployment
            CommandBus both read these same four gains — the panel is
            the visual proof of that coupling.

        Click the Refresh button to re-query :class:`GlobalInputManager`
        (cheap — it's a singleton).
        """
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

        _self_ref = self

        border = get_color("training_widget_border", "#3d3d3d")
        input_bg = get_color("training_widget_input_bg", "#1A1A1A")
        text_c = get_color("training_widget_text", "#cccccc")
        muted = get_color("training_widget_label_text", "#888888")
        ok_c = "#4CAF50"
        warn_c = "#FFB74D"

        MAX_W = NODE_W - H_PAD * 2
        HDR_H = 14
        ROW_H = 18
        BTN_H = 16

        container = QWidget()
        container._full_row_widget = True
        container._preferred_height = PARAM_ROW_H

        root = QVBoxLayout(container)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(1)

        table_wrap = QFrame(container)
        table_wrap.setStyleSheet(
            f"background: {input_bg}; border: 1px solid {border}; border-radius: 4px;"
        )
        tw_layout = QVBoxLayout(table_wrap)
        tw_layout.setContentsMargins(4, 3, 4, 3)
        tw_layout.setSpacing(0)
        root.addWidget(table_wrap)

        _row_widgets: list = []
        _height_cbs: list = []

        def _emit_h(h: int):
            container._preferred_height = h
            container.setMinimumHeight(h)
            container.setMaximumHeight(h)
            container.updateGeometry()
            for fn in _height_cbs:
                try:
                    fn(h)
                except Exception:
                    pass

        class _HSig:
            def connect(self_, fn):
                _height_cbs.append(fn)
            def emit(self_, h):
                _emit_h(h)

        container.height_changed = _HSig()
        container.preferred_row_height = lambda: max(PARAM_ROW_H, int(container._preferred_height))

        def _read_controller_state():
            """Return (mode_label, invert_vy, mode_ok)."""
            try:
                from src.system.runtime.global_input_manager import get_global_input_manager
                mgr = get_global_input_manager()
                mode = str(getattr(mgr, "mode", "") or "")
                if not mode:
                    # Fallback: query source name
                    src = getattr(mgr, "active_source", None)
                    mode = type(src).__name__ if src is not None else "Off"
                invert = bool(getattr(mgr, "invert_vy", False))
                return (mode.capitalize(), invert, True)
            except Exception:
                return ("Unknown", False, False)

        def _read_gains():
            """Derive the current (fwd, bwd, lat, yaw) from sibling params."""
            p = (logic.parameters or {}) if logic is not None else {}
            def _f(k, d):
                try:
                    return float(p.get(k, d))
                except (TypeError, ValueError):
                    return d
            return (
                _f("gain_lin_forward", 2.0),
                _f("gain_lin_backward", 1.0),
                _f("gain_lin_lateral", 0.5),
                _f("gain_yaw", 1.5),
            )

        def _read_mapping():
            """Return a compact label describing the current stick→vel curve."""
            p = (logic.parameters or {}) if logic is not None else {}
            mode = str(p.get("mapping_mode", "linear") or "linear").strip().lower()
            if mode == "deadzone":
                try:
                    dz = float(p.get("deadzone", 0.1) or 0.1)
                except (TypeError, ValueError):
                    dz = 0.1
                return f"deadzone({dz:.2f})"
            if mode == "exponential":
                try:
                    exp = float(p.get("curve_exponent", 2.0) or 2.0)
                except (TypeError, ValueError):
                    exp = 2.0
                return f"exp(p={exp:.2f})"
            return "linear"

        def _row(widgets):
            rw = QWidget(table_wrap)
            rw.setFixedHeight(ROW_H)
            rl = QHBoxLayout(rw)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(4)
            for w in widgets:
                rl.addWidget(w)
            return rw

        def _lbl(text, color=text_c, width=None, bold=False, align=None):
            lb = QLabel(text)
            weight = "700" if bold else "400"
            lb.setStyleSheet(
                f"color: {color}; font-size: 9px; font-weight: {weight}; "
                f"background: transparent; border: none;"
            )
            if width is not None:
                lb.setFixedWidth(width)
            if align is not None:
                lb.setAlignment(align)
            return lb

        def _rebuild():
            # Startup pre-warm / canvas tear-down guard: if the owning
            # TrainingNodeItem has been destroyed between QTimer.singleShot
            # scheduling and firing, every Qt-child access below raises
            # RuntimeError. Check ``tw_layout.count()`` as a cheap liveness
            # probe and bail out silently — nothing to rebuild on a dead
            # widget.
            try:
                tw_layout.count()
            except RuntimeError:
                return
            for w in list(_row_widgets):
                try:
                    tw_layout.removeWidget(w)
                    w.deleteLater()
                except RuntimeError:
                    pass
            _row_widgets.clear()

            mode, invert, ok = _read_controller_state()
            fwd, bwd, lat, yaw = _read_gains()
            map_label = _read_mapping()

            # Header row: mode + invert_vy
            mode_ok_color = ok_c if ok and mode.lower() in ("gamepad", "keyboard") else warn_c
            hdr = _row([
                _lbl("Mode", muted, width=36, bold=True),
                _lbl(mode, mode_ok_color, bold=True),
                _lbl("·  Invert Y", muted),
                _lbl("ON" if invert else "OFF",
                     ok_c if not invert else warn_c, bold=True),
            ])
            _row_widgets.append(hdr)
            tw_layout.addWidget(hdr)

            # Mapping row — surface the current stick→vel curve so the
            # status panel also mirrors deployment behaviour at a glance.
            map_row = _row([
                _lbl("Curve", muted, width=36, bold=True),
                _lbl(map_label, ok_c, bold=True),
            ])
            _row_widgets.append(map_row)
            tw_layout.addWidget(map_row)

            # Separator
            sep = QFrame(table_wrap)
            sep.setFixedHeight(1)
            sep.setStyleSheet(f"background: {border}; border: none;")
            _row_widgets.append(sep)
            tw_layout.addWidget(sep)

            # Bindings + derived ranges (3 rows, one per axis)
            binding_rows = [
                ("L-Y",  "→ Lin X",  f"[-{bwd:.2f}, +{fwd:.2f}]",  "m/s"),
                ("L-X",  "→ Lin Y",  f"±{lat:.2f}",                "m/s"),
                ("R-X",  "→ Yaw Z",  f"±{yaw:.2f}",                "rad/s"),
            ]
            for stick, axis, rng, unit in binding_rows:
                row_w = _row([
                    _lbl(stick,  muted,   width=28, bold=True),
                    _lbl(axis,   text_c,  width=48),
                    _lbl(rng,    ok_c,    bold=True),
                    _lbl(unit,   muted),
                ])
                _row_widgets.append(row_w)
                tw_layout.addWidget(row_w)

            # +1 extra row for the Curve header line
            total_h = HDR_H + 1 + ROW_H * (len(binding_rows) + 1) + 6
            table_wrap.setFixedHeight(total_h)
            _emit_h(total_h + BTN_H + 2)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setFixedHeight(BTN_H)
        # Match the standard node input surface: input_bg fill + border
        # frame + rounded corners + text_c label. Same look as the
        # node's edit rows so the button blends with the rest of the
        # Training Commands panel instead of sticking out as a plain
        # Qt-default chrome button.
        refresh_btn.setStyleSheet(
            f"QPushButton {{ color: {text_c}; background: {input_bg}; "
            f"border: 1px solid {border}; border-radius: 3px; "
            f"font-size: 8px; padding: 0 6px; }}"
            f"QPushButton:hover {{ border: 1px solid #666; }}"
            f"QPushButton:pressed {{ background: #111; }}"
        )
        refresh_btn.clicked.connect(lambda: QTimer.singleShot(0, _rebuild))
        root.addWidget(refresh_btn)

        # Rebuild on sibling param edits so the derived-range / curve
        # read-out tracks the user's changes in real time.
        _watch_keys = {
            "gain_lin_forward", "gain_lin_backward",
            "gain_lin_lateral", "gain_yaw",
            "mapping_mode", "deadzone", "curve_exponent",
        }

        def _on_canvas_param_changed(node_type: str, param_key: str, _value: str):
            if node_type == "training_commands" and param_key in _watch_keys:
                QTimer.singleShot(0, _rebuild)

        def _connect_scene_signal():
            # Guard against shiboken-deleted C++ items: if the owning
            # TrainingNodeItem was torn down between QTimer.singleShot
            # scheduling and firing (common during startup pre-warm),
            # ``_self_ref.scene()`` raises RuntimeError. Treat it as
            # "nothing to connect" — the widget is gone anyway.
            try:
                scene = _self_ref.scene()
            except RuntimeError:
                return
            if scene is not None and hasattr(scene, "node_param_changed"):
                try:
                    scene.node_param_changed.connect(_on_canvas_param_changed)
                except Exception:
                    pass

        QTimer.singleShot(0, _rebuild)
        QTimer.singleShot(50, _connect_scene_signal)
        return container

    # ------------------------------------------------------------------
    # Active file table (Robot.active_override)
    # ------------------------------------------------------------------

    def _make_active_file_table(self, key, get_val, write_val, logic):
        """Flat 3-row file-type table for the Robot node.

        Rows: ``MJCF`` / ``USD`` / ``URDF``. Each row shows:

          * file type label (fixed)
          * resolved path (truncated, or ``—`` when not in registry)
          * status tag: ``Loaded`` when registered, ``—`` otherwise
          * ``✔`` marker on the currently-active file

        Clicking a Loaded row writes that file type into the
        ``active_override`` param. Clicking the active row again reverts
        to ``auto`` (backend-derived). Unavailable rows are click-locked.

        Re-scans on every refresh (asset_id change, canvas load, and
        the explicit refresh button).
        """
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QFrame, QLabel, QPushButton, QVBoxLayout, QHBoxLayout

        _self_ref = self

        border = get_color("training_widget_border", "#3d3d3d")
        input_bg = get_color("training_widget_input_bg", "#1A1A1A")
        text_c = get_color("training_widget_text", "#cccccc")
        muted = get_color("training_widget_label_text", "#aaaaaa")
        ok_c = "#4CAF50"
        skip_c = "#555555"

        MAX_W = NODE_W - H_PAD * 2
        ROW_H = 20
        HDR_H = 14
        # Three-column table: Type | Status | ✔
        COL_CHK = 22
        COL_STATUS = 60
        COL_TYPE = MAX_W - COL_CHK - COL_STATUS - 16

        container = QWidget()
        container._full_row_widget = True
        container._preferred_height = PARAM_ROW_H

        root = QVBoxLayout(container)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(1)

        table_wrap = QFrame(container)
        table_wrap.setStyleSheet(
            f"background: {input_bg}; border: 1px solid {border}; border-radius: 4px;"
        )
        tw_layout = QVBoxLayout(table_wrap)
        tw_layout.setContentsMargins(2, 2, 2, 2)
        tw_layout.setSpacing(0)
        root.addWidget(table_wrap)

        _height_cbs: list = []

        def _emit_h(h: int):
            container._preferred_height = h
            container.setMinimumHeight(h)
            container.setMaximumHeight(h)
            container.updateGeometry()
            for fn in _height_cbs:
                try:
                    fn(h)
                except Exception:
                    pass

        class _HSig:
            def connect(self_, fn):
                _height_cbs.append(fn)
            def emit(self_, h):
                _emit_h(h)

        container.height_changed = _HSig()
        container.preferred_row_height = lambda: max(PARAM_ROW_H, int(container._preferred_height))

        _row_widgets: list = []

        def _resolve_files():
            """Pull current asset files from the registry via this node's asset_id."""
            aid = str((logic.parameters or {}).get("asset_id", "")).strip()
            if not aid:
                return None
            try:
                from src.system.training.robot_assets import get_asset, rescan
                rescan()
                return get_asset(aid)
            except Exception:
                return None

        def _normalize_path(p: Any) -> str:
            """Normalize path separators to forward slashes for display.
            Nucleus URLs (``nucleus:Robots/...``) and http URLs are
            returned unchanged."""
            if p is None:
                return ""
            s = str(p)
            if "://" in s or s.startswith("nucleus:"):
                return s
            return s.replace("\\", "/")

        def _current_active(asset) -> str:
            """Resolve which file is the ✔ row.

            'auto' selects the best file for the current training
            backend:
              - SB3 (MuJoCo): MJCF > USD > URDF
              - Isaac Lab (Isaac Sim): USD > URDF > MJCF
            An explicit override ('mjcf'/'usd'/'urdf') is honoured
            unconditionally.
            """
            raw = str(get_val() or "auto").strip().lower()
            if raw in ("mjcf", "usd", "urdf"):
                return raw
            if asset is None:
                return ""
            # Determine backend from the scene (experiment-bound).
            backend = "sb3"
            try:
                scene = _self_ref.scene()
                if scene is not None:
                    backend = str(getattr(scene, "_current_backend", "sb3") or "sb3")
            except Exception:
                pass
            if backend == "isaac_lab":
                # Isaac Sim prefers USD > URDF > MJCF
                if asset.usd_path is not None or getattr(asset, "usd_url", ""):
                    return "usd"
                if asset.urdf_path is not None:
                    return "urdf"
                if asset.mjcf_path is not None:
                    return "mjcf"
            else:
                # MuJoCo prefers MJCF > USD > URDF
                if asset.mjcf_path is not None:
                    return "mjcf"
                if asset.usd_path is not None or getattr(asset, "usd_url", ""):
                    return "usd"
                if asset.urdf_path is not None:
                    return "urdf"
            return ""

        def _rebuild():
            # Startup pre-warm / canvas tear-down guard: if the owning
            # TrainingNodeItem has been destroyed between QTimer.singleShot
            # scheduling and firing, every Qt-child access below raises
            # RuntimeError. Check ``tw_layout.count()`` as a cheap liveness
            # probe and bail out silently — nothing to rebuild on a dead
            # widget.
            try:
                tw_layout.count()
            except RuntimeError:
                return
            for w in list(_row_widgets):
                try:
                    tw_layout.removeWidget(w)
                    w.deleteLater()
                except RuntimeError:
                    pass
            _row_widgets.clear()

            asset = _resolve_files()
            active = _current_active(asset)

            # Header — Type | Status | ✔ (path moves to row tooltip)
            hdr = QWidget(table_wrap)
            hdr.setFixedHeight(HDR_H)
            hl = QHBoxLayout(hdr)
            hl.setContentsMargins(4, 0, 4, 0)
            hl.setSpacing(4)
            for txt, w in [
                ("Type", COL_TYPE),
                ("Status", COL_STATUS),
                ("", COL_CHK),
            ]:
                lb = QLabel(txt)
                lb.setFixedWidth(w)
                lb.setStyleSheet(
                    f"color: {muted}; font-size: 8px; font-weight: 600; "
                    f"background: transparent; border: none;"
                )
                hl.addWidget(lb)
            _row_widgets.append(hdr)
            tw_layout.addWidget(hdr)

            rows_spec = [
                ("MJCF", "mjcf", getattr(asset, "mjcf_path", None) if asset else None),
                ("USD",  "usd",  (
                    getattr(asset, "usd_path", None)
                    or getattr(asset, "usd_url", "")
                ) if asset else None),
                ("URDF", "urdf", getattr(asset, "urdf_path", None) if asset else None),
            ]

            for idx, (label, file_key, path_value) in enumerate(rows_spec):
                row = QWidget(table_wrap)
                row.setFixedHeight(ROW_H)
                rl = QHBoxLayout(row)
                rl.setContentsMargins(4, 0, 4, 0)
                rl.setSpacing(4)

                loaded = bool(path_value)
                is_active = loaded and file_key == active
                row_bg = input_bg if (idx % 2) else "transparent"
                row.setStyleSheet(f"QWidget {{ background: {row_bg}; border: none; }}")

                # Row-wide tooltip (normalized path) — hover anywhere
                # on the row. Children inherit via explicit setToolTip.
                tip = _normalize_path(path_value) if loaded else "Not in registry"
                row.setToolTip(tip)

                type_lbl = QLabel(label)
                type_lbl.setFixedWidth(COL_TYPE)
                type_lbl.setStyleSheet(
                    f"color: {text_c if loaded else skip_c}; font-size: 9px; "
                    f"font-weight: 600; background: transparent; border: none;"
                )
                type_lbl.setToolTip(tip)
                rl.addWidget(type_lbl)

                status_lbl = QLabel("Loaded" if loaded else "—")
                status_lbl.setFixedWidth(COL_STATUS)
                status_lbl.setStyleSheet(
                    f"color: {ok_c if loaded else skip_c}; font-size: 9px; "
                    f"font-weight: 600; background: transparent; border: none;"
                )
                status_lbl.setToolTip(tip)
                rl.addWidget(status_lbl)

                chk_lbl = QLabel("✔" if is_active else "")
                chk_lbl.setFixedWidth(COL_CHK)
                chk_lbl.setAlignment(Qt.AlignCenter)
                chk_lbl.setStyleSheet(
                    f"color: {ok_c}; font-size: 11px; font-weight: 900; "
                    f"background: transparent; border: none;"
                )
                chk_lbl.setToolTip(tip)
                rl.addWidget(chk_lbl)

                # Click-to-select (only when Loaded)
                if loaded:
                    def _on_click(_evt=None, _key=file_key, _active=is_active):
                        if _active:
                            write_val("auto")
                        else:
                            write_val(_key)
                        QTimer.singleShot(0, _rebuild)
                    row.mousePressEvent = _on_click  # type: ignore[assignment]
                    row.setCursor(Qt.PointingHandCursor)

                _row_widgets.append(row)
                tw_layout.addWidget(row)

            table_h = HDR_H + len(rows_spec) * ROW_H + 4
            table_wrap.setFixedHeight(table_h)
            _emit_h(table_h + 2)

        def _on_canvas_param_changed(node_type: str, param_key: str, _value: str):
            if node_type == "robot" and param_key == "asset_id":
                QTimer.singleShot(0, _rebuild)

        def _connect_scene_signal():
            # Guard against shiboken-deleted C++ items: if the owning
            # TrainingNodeItem was torn down between QTimer.singleShot
            # scheduling and firing (common during startup pre-warm),
            # ``_self_ref.scene()`` raises RuntimeError. Treat it as
            # "nothing to connect" — the widget is gone anyway.
            try:
                scene = _self_ref.scene()
            except RuntimeError:
                return
            if scene is not None and hasattr(scene, "node_param_changed"):
                try:
                    scene.node_param_changed.connect(_on_canvas_param_changed)
                except Exception:
                    pass

        QTimer.singleShot(0, _rebuild)
        QTimer.singleShot(50, _connect_scene_signal)
        return container

    # ------------------------------------------------------------------
    # Scene dropdown (PlayGroundSetting.scene_id)
    # ------------------------------------------------------------------

    def _make_scene_dropdown(self, key, get_val, write_val, write_params, logic):
        """Registry-backed scene picker for :class:`PlayGroundSettingNode`.

        Choices are pulled from :mod:`scene_registry` on every popup
        open and filtered by:

          * ``backend`` — sniffed from the canvas Train node
            (``train`` / ``il_ppo_trainer`` / ``amp_ppo_trainer``)
          * ``family`` — sniffed from the Robot node's ``asset_id``
            → RobotAsset.family

        Selecting a scene writes ``scene_id`` + auto-syncs the hidden
        ``scene_type`` param and applies the registry entry's
        ``defaults`` dict onto the node parameters so conditional rows
        (rough terrain, height-scan) reveal the right values the moment
        the picker closes.
        """
        _self_ref = self

        def _detect_backend() -> Optional[str]:
            """Walk the scene for a trainer node; return 'sb3' or 'isaac_lab'."""
            scene = _self_ref.scene()
            if scene is None:
                return None
            for item in scene.items():
                try:
                    nt = item.data(12)
                    if nt in ("il_ppo_trainer", "amp_ppo_trainer"):
                        return "isaac_lab"
                    if nt == "train":
                        return "sb3"
                except Exception:
                    continue
            return None

        def _detect_family() -> Optional[str]:
            """Walk the scene for the Robot node; resolve family via registry."""
            scene = _self_ref.scene()
            if scene is None:
                return None
            for item in scene.items():
                try:
                    if item.data(12) != "robot":
                        continue
                    nl = item.data(13)
                    if nl is None:
                        continue
                    aid = str((getattr(nl, "parameters", {}) or {}).get("asset_id", "")).strip()
                    if not aid:
                        continue
                    from src.system.training.robot_assets import get_asset, rescan
                    rescan()
                    return str(getattr(get_asset(aid), "family", "") or "") or None
                except Exception:
                    continue
            return None

        def _scene_choices_provider(
            _cur_val_fn: Callable[[], str] = get_val,
        ) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
            backend = _detect_backend()
            family = _detect_family()
            try:
                from src.system.training.scene_registry import list_scenes, rescan
                rescan()
                scenes = list_scenes(backend=backend, family=family)
            except Exception:
                scenes = []
            if not scenes:
                return (
                    ["(no scenes)"],
                    {"(no scenes)": {"title": "(no scenes)",
                                     "desc": "Scene registry empty for the current backend/family"}},
                )
            ids = [s.scene_id for s in scenes]
            meta: Dict[str, Dict[str, str]] = {}
            for s in scenes:
                backends = ", ".join(sorted(s.supported_backends)) or "any"
                fams = ", ".join(sorted(s.supported_families)) or "any family"
                meta[s.scene_id] = {
                    "title": s.name,
                    "desc": f"{s.description}    ·  {backends}  ·  {fams}",
                }
            return ids, meta

        initial_choices, initial_meta = _scene_choices_provider()
        current_val = get_val()
        if current_val not in initial_choices:
            current_val = initial_choices[0] if initial_choices else ""

        w = _RichChoicePicker(
            initial_choices,
            current=current_val,
            meta_map=initial_meta,
            leading_mode="checkbox",
            multi_select=False,
            choices_provider=_scene_choices_provider,
        )

        def _on_scene_pick(value: str) -> None:
            if not value or value.startswith("(no "):
                return
            updates: Dict[str, str] = {"scene_id": value}
            try:
                from src.system.training.scene_registry import get_scene, rescan
                rescan()
                s = get_scene(value)
                updates["scene_type"] = s.scene_type
                for k, v in (s.defaults or {}).items():
                    updates[k] = str(v)
            except Exception:
                pass
            write_params(updates)

        w.currentTextChanged.connect(_on_scene_pick)
        return w

    # ------------------------------------------------------------------
    # Joint init editor (JointInitNode.angles)
    # ------------------------------------------------------------------

    def _make_joint_init_editor(self, key, get_val, write_val, logic):
        """Popup per-joint angle editor.

        Button shows ``N angles set`` / ``none``. Click opens a
        :class:`QDialog` with a scrollable ``QFormLayout`` of
        :class:`QDoubleSpinBox` rows, one per joint. Joints are read
        from the scene's Robot node (``robot.asset_id`` → registry →
        ``joint_names``). Cancel discards; OK serializes the non-zero
        (or explicitly-set) entries back into ``key`` as a JSON dict.
        """
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import (
            QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
            QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget,
        )
        import math
        import json as _json

        _self_ref = self

        border = get_color("training_widget_border", "#3d3d3d")
        input_bg = get_color("training_widget_input_bg", "#1A1A1A")
        text_c = get_color("training_widget_text", "#cccccc")

        btn = QPushButton()
        btn.setFixedSize(WIDGET_W, PARAM_ROW_H - 2)
        btn.setStyleSheet(
            f"QPushButton {{ color: {text_c}; background: {input_bg}; "
            f"border: 1px solid {border}; border-radius: 3px; "
            f"font-size: 9px; padding: 0 6px; text-align: left; }}"
            f"QPushButton:hover {{ border: 1px solid #888; }}"
        )

        def _parse_current() -> Dict[str, float]:
            raw = str(get_val() or "{}").strip() or "{}"
            try:
                parsed = _json.loads(raw)
                if isinstance(parsed, dict):
                    return {str(k): float(v) for k, v in parsed.items()}
            except Exception:
                pass
            return {}

        def _resolve_joints() -> Tuple[List[str], str]:
            """Walk the scene for the Robot node and return its joint_names."""
            scene = _self_ref.scene()
            if scene is None:
                return [], ""
            for item in scene.items():
                if item is _self_ref:
                    continue
                try:
                    if item.data(12) != "robot":
                        continue
                    nl = item.data(13)
                    if nl is None:
                        continue
                    aid = str((getattr(nl, "parameters", {}) or {}).get("asset_id", "")).strip()
                    if not aid:
                        continue
                    from src.system.training.robot_assets import get_asset, rescan
                    rescan()
                    asset = get_asset(aid)
                    return list(asset.joint_names or []), aid
                except Exception:
                    continue
            return [], ""

        def _refresh_label():
            n = len(_parse_current())
            btn.setText(f"{n} angles set" if n else "none")

        def _open_dialog(_checked: bool = False):
            joints, asset_id = _resolve_joints()
            if not joints:
                btn.setText("— connect Robot")
                QTimer.singleShot(1200, _refresh_label)
                return

            dlg = QDialog()
            dlg.setWindowTitle(f"Joint Init — {asset_id or 'robot'}")
            dlg.setModal(True)
            dlg.resize(360, min(520, 80 + 30 * len(joints)))

            root = QVBoxLayout(dlg)
            root.setContentsMargins(8, 8, 8, 8)
            root.setSpacing(6)

            hint = QLabel(
                f"{len(joints)} joints — set any in radians (empty cells skipped)"
            )
            hint.setStyleSheet("color: #888; font-size: 10px;")
            root.addWidget(hint)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            form_wrap = QWidget()
            form = QFormLayout(form_wrap)
            form.setContentsMargins(4, 4, 4, 4)
            form.setSpacing(4)

            current = _parse_current()
            spinboxes: Dict[str, QDoubleSpinBox] = {}
            PI = math.pi
            for jn in joints:
                sb = QDoubleSpinBox()
                sb.setRange(-PI, PI)
                sb.setDecimals(3)
                sb.setSingleStep(0.01)
                sb.setSuffix(" rad")
                sb.setValue(float(current.get(jn, 0.0)))
                sb.setMinimumWidth(120)
                form.addRow(QLabel(jn), sb)
                spinboxes[jn] = sb

            scroll.setWidget(form_wrap)
            root.addWidget(scroll, 1)

            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok
                | QDialogButtonBox.StandardButton.Cancel
                | QDialogButtonBox.StandardButton.Reset
            )
            root.addWidget(buttons)

            def _on_reset():
                for sb in spinboxes.values():
                    sb.setValue(0.0)

            buttons.button(QDialogButtonBox.StandardButton.Reset).clicked.connect(_on_reset)
            buttons.accepted.connect(dlg.accept)
            buttons.rejected.connect(dlg.reject)

            if dlg.exec() == QDialog.DialogCode.Accepted:
                # Only persist entries that are explicitly non-zero so
                # an "unset" spinbox doesn't clobber the asset default.
                result = {
                    jn: round(sb.value(), 6)
                    for jn, sb in spinboxes.items()
                    if abs(sb.value()) > 1e-9
                }
                write_val(_json.dumps(result, ensure_ascii=False))
                _refresh_label()

        btn.clicked.connect(_open_dialog)
        _refresh_label()
        return btn

    # ------------------------------------------------------------------
    # Contact body picker widget (ActorSetting.contact_body_names)
    # ------------------------------------------------------------------

    def _make_contact_body_picker(self, key, get_val, write_val, logic):
        """Multi-select picker for contact-sensor body names.

        Button labelled ``Contact Bodies (N)`` that opens the standard
        ``_RichChoicePicker(multi_select=True)`` popup. Choices come from
        the connected Robot node's resolved asset:

          * ``base_link``
          * all bodies derived from ``joint_names`` via
            :func:`joint_name_to_link_name`
          * ``foot_link_names``

        Stores the selection as a JSON list in ``key``.
        """
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QPushButton
        import json as _json

        _self_ref = self

        def _resolve_bodies() -> list:
            """Find bodies via the connected Robot node or scene scan."""
            scene = _self_ref.scene()
            if scene is None:
                return []
            from src.system.training.robot_assets import get_asset, rescan
            from src.system.training.body_ir import joint_name_to_link_name

            def _collect(asset) -> list:
                bodies: list = []
                if asset.base_link:
                    bodies.append(asset.base_link)
                for jn in (asset.joint_names or []):
                    ln = joint_name_to_link_name(jn)
                    if ln and ln not in bodies:
                        bodies.append(ln)
                for fn in (asset.foot_link_names or []):
                    if fn not in bodies:
                        bodies.append(fn)
                return bodies

            for item in scene.items():
                if item is _self_ref:
                    continue
                try:
                    if item.data(12) != "robot":
                        continue
                    nl = item.data(13)
                    if nl is None:
                        continue
                    aid = str((getattr(nl, "parameters", {}) or {}).get("asset_id", "")).strip()
                    if not aid:
                        continue
                    rescan()
                    return _collect(get_asset(aid))
                except Exception:
                    continue
            return []

        def _parse_current() -> list:
            raw = str(get_val() or "").strip()
            if not raw:
                return []
            try:
                v = _json.loads(raw)
                if isinstance(v, list):
                    return [str(x) for x in v]
            except Exception:
                pass
            return []

        btn = QPushButton()
        btn.setFixedSize(WIDGET_W, PARAM_ROW_H - 2)
        border = get_color("training_widget_border", "#3d3d3d")
        input_bg = get_color("training_widget_input_bg", "#1A1A1A")
        text_c = get_color("training_widget_text", "#cccccc")
        btn.setStyleSheet(
            f"QPushButton {{ color: {text_c}; background: {input_bg}; "
            f"border: 1px solid {border}; border-radius: 3px; "
            f"font-size: 9px; padding: 0 6px; text-align: left; }}"
            f"QPushButton:hover {{ border: 1px solid #888; }}"
        )

        def _refresh_label():
            current = _parse_current()
            n = len(current)
            btn.setText(f"{n} selected" if n else "none")

        def _on_click(_checked=False):
            bodies = _resolve_bodies()
            if not bodies:
                btn.setText("— connect Robot")
                QTimer.singleShot(1200, _refresh_label)
                return
            current = _parse_current()
            picker = _RichChoicePicker(
                bodies,
                current_values=current,
                multi_select=True,
                # 'checkbox' leading mode uses a native QCheckBox; 'icon'
                # (the default) loads icon_*.svg per entry which is wrong
                # for free-form body names.
                leading_mode="checkbox",
                parent=btn,
            )
            # _RichChoicePicker is itself a QPushButton that opens on click —
            # trigger its menu programmatically, then listen for the result.
            def _on_selected(selected):
                sel = selected if isinstance(selected, list) else [selected]
                write_val(_json.dumps(sel, ensure_ascii=False))
                _refresh_label()
                try:
                    picker.deleteLater()
                except Exception:
                    pass

            picker.selectionChanged.connect(_on_selected)
            picker.click()

        btn.clicked.connect(_on_click)
        _refresh_label()
        return btn

    # ------------------------------------------------------------------
    # Action joint picker (ActorSetting.action_joint_names_expr)
    # ------------------------------------------------------------------

    def _make_action_joint_picker(self, key, get_val, write_val, logic):
        """Multi-select picker for action joint names.

        Button labelled ``N selected`` / ``all`` that opens the standard
        ``_RichChoicePicker(multi_select=True)`` popup. Choices come from
        the connected Robot node's resolved asset ``joint_names``.

        Stores the selection as a JSON list in ``key``. An empty list
        means "all joints" (equivalent to the old ``.*`` regex default).
        """
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QPushButton
        import json as _json

        _self_ref = self

        def _resolve_joints() -> list:
            """Find joint names via the connected Robot node."""
            scene = _self_ref.scene()
            if scene is None:
                return []
            from src.system.training.robot_assets import get_asset, rescan

            for item in scene.items():
                if item is _self_ref:
                    continue
                try:
                    if item.data(12) != "robot":
                        continue
                    nl = item.data(13)
                    if nl is None:
                        continue
                    aid = str((getattr(nl, "parameters", {}) or {}).get("asset_id", "")).strip()
                    if not aid:
                        continue
                    rescan()
                    asset = get_asset(aid)
                    return list(asset.joint_names or [])
                except Exception:
                    continue
            return []

        def _parse_current() -> list:
            raw = str(get_val() or "").strip()
            if not raw or raw == ".*":
                return []
            try:
                v = _json.loads(raw)
                if isinstance(v, list):
                    return [str(x) for x in v]
            except Exception:
                pass
            return []

        btn = QPushButton()
        btn.setFixedSize(WIDGET_W, PARAM_ROW_H - 2)
        border = get_color("training_widget_border", "#3d3d3d")
        input_bg = get_color("training_widget_input_bg", "#1A1A1A")
        text_c = get_color("training_widget_text", "#cccccc")
        btn.setStyleSheet(
            f"QPushButton {{ color: {text_c}; background: {input_bg}; "
            f"border: 1px solid {border}; border-radius: 3px; "
            f"font-size: 9px; padding: 0 6px; text-align: left; }}"
            f"QPushButton:hover {{ border: 1px solid #888; }}"
        )

        def _refresh_label():
            current = _parse_current()
            n = len(current)
            btn.setText(f"{n} selected" if n else "all")

        def _on_click(_checked=False):
            joints = _resolve_joints()
            if not joints:
                btn.setText("— connect Robot")
                QTimer.singleShot(1200, _refresh_label)
                return
            current = _parse_current()
            picker = _RichChoicePicker(
                joints,
                current_values=current,
                multi_select=True,
                leading_mode="checkbox",
                parent=btn,
            )

            def _on_selected(selected):
                sel = selected if isinstance(selected, list) else [selected]
                write_val(_json.dumps(sel, ensure_ascii=False))
                _refresh_label()
                try:
                    picker.deleteLater()
                except Exception:
                    pass

            picker.selectionChanged.connect(_on_selected)
            picker.click()

        btn.clicked.connect(_on_click)
        _refresh_label()
        return btn

    # ------------------------------------------------------------------
    # Gait preset table editor (TrainingCommands.gait_presets)
    # ------------------------------------------------------------------

    def _make_gait_preset_table(self, key, get_val, write_val, logic):
        """Gait preset table editor — Walk These Ways §3.

        Compact button showing ``N presets``. Click opens a
        :class:`QDialog` with a :class:`QTableWidget` carrying one row
        per preset. Columns:

          Name | Freq (Hz) | FL | FR | RL | RR | H_body (m) | H_step (m)

        Rows are editable via :class:`QDoubleSpinBox` / line-edit
        cells. "Add" appends a new row seeded from the currently-
        selected row's values; "Remove" deletes the selection; "Reset
        defaults" re-seeds from :data:`DEFAULT_PRESETS`. On OK the
        table serialises back into the node param as a JSON list of
        dicts matching :meth:`GaitPreset.to_dict`.
        """
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import (
            QDialog, QDialogButtonBox, QDoubleSpinBox, QHBoxLayout,
            QHeaderView, QLineEdit, QPushButton, QTableWidget,
            QTableWidgetItem, QVBoxLayout,
        )
        import json as _json

        border = get_color("training_widget_border", "#3d3d3d")
        input_bg = get_color("training_widget_input_bg", "#1A1A1A")
        text_c = get_color("training_widget_text", "#cccccc")

        COL_SPECS = [
            ("Name",    "str",   None),
            ("Freq",    "float", (0.5, 6.0, 2, 0.05)),
            ("FL",      "float", (0.0, 1.0, 3, 0.01)),
            ("FR",      "float", (0.0, 1.0, 3, 0.01)),
            ("RL",      "float", (0.0, 1.0, 3, 0.01)),
            ("RR",      "float", (0.0, 1.0, 3, 0.01)),
            ("H_body",  "float", (0.10, 0.60, 3, 0.005)),
            ("H_step",  "float", (0.01, 0.30, 3, 0.005)),
        ]

        def _parse_current() -> List[Dict[str, Any]]:
            raw = str(get_val() or "").strip()
            if not raw:
                from src.system.training.gait_presets import default_presets_json
                return default_presets_json()
            try:
                parsed = _json.loads(raw)
                if isinstance(parsed, list):
                    return [p for p in parsed if isinstance(p, dict)]
            except Exception:
                pass
            from src.system.training.gait_presets import default_presets_json
            return default_presets_json()

        btn = QPushButton()
        btn.setFixedSize(WIDGET_W, PARAM_ROW_H - 2)
        btn.setStyleSheet(
            f"QPushButton {{ color: {text_c}; background: {input_bg}; "
            f"border: 1px solid {border}; border-radius: 3px; "
            f"font-size: 9px; padding: 0 6px; text-align: left; }}"
            f"QPushButton:hover {{ border: 1px solid #888; }}"
        )

        def _refresh_label():
            n = len(_parse_current())
            btn.setText(f"{n} presets")

        def _populate_row(table: QTableWidget, row_idx: int, preset: Dict[str, Any]):
            phase = preset.get("phase") or [0.0, 0.5, 0.5, 0.0]
            if not isinstance(phase, (list, tuple)) or len(phase) != 4:
                phase = [0.0, 0.5, 0.5, 0.0]
            values = [
                str(preset.get("name", "unnamed")),
                float(preset.get("frequency", 2.5) or 2.5),
                float(phase[0]),
                float(phase[1]),
                float(phase[2]),
                float(phase[3]),
                float(preset.get("body_height", 0.35) or 0.35),
                float(preset.get("step_height", 0.08) or 0.08),
            ]
            for col_idx, (label, kind, spin_cfg) in enumerate(COL_SPECS):
                if kind == "str":
                    w = QLineEdit()
                    w.setText(str(values[col_idx]))
                    table.setCellWidget(row_idx, col_idx, w)
                else:
                    lo, hi, dec, step = spin_cfg
                    sb = QDoubleSpinBox()
                    sb.setRange(lo, hi)
                    sb.setDecimals(dec)
                    sb.setSingleStep(step)
                    sb.setValue(float(values[col_idx]))
                    table.setCellWidget(row_idx, col_idx, sb)

        def _row_to_dict(table: QTableWidget, row_idx: int) -> Dict[str, Any]:
            def _cell(col: int):
                return table.cellWidget(row_idx, col)
            name_w = _cell(0)
            name = name_w.text().strip() if name_w is not None else "unnamed"
            return {
                "name": name or "unnamed",
                "frequency": round(float(_cell(1).value()), 4),
                "phase": [
                    round(float(_cell(2).value()), 4),
                    round(float(_cell(3).value()), 4),
                    round(float(_cell(4).value()), 4),
                    round(float(_cell(5).value()), 4),
                ],
                "body_height": round(float(_cell(6).value()), 4),
                "step_height": round(float(_cell(7).value()), 4),
            }

        def _open_dialog(_checked: bool = False):
            current = _parse_current()

            dlg = QDialog()
            dlg.setWindowTitle("Gait Presets — Walk These Ways")
            dlg.setModal(True)
            dlg.resize(620, 360)

            root = QVBoxLayout(dlg)
            root.setContentsMargins(8, 8, 8, 8)
            root.setSpacing(6)

            table = QTableWidget(len(current), len(COL_SPECS))
            table.setHorizontalHeaderLabels([c[0] for c in COL_SPECS])
            table.verticalHeader().setDefaultSectionSize(26)
            header = table.horizontalHeader()
            header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            for i, p in enumerate(current):
                _populate_row(table, i, p)
            root.addWidget(table, 1)

            # Toolbar
            bar = QHBoxLayout()
            bar.setSpacing(4)
            add_btn = QPushButton("Add")
            rm_btn = QPushButton("Remove")
            reset_btn = QPushButton("Reset defaults")
            bar.addWidget(add_btn)
            bar.addWidget(rm_btn)
            bar.addStretch(1)
            bar.addWidget(reset_btn)
            root.addLayout(bar)

            def _on_add():
                cur_row = table.currentRow()
                seed: Dict[str, Any]
                if 0 <= cur_row < table.rowCount():
                    seed = _row_to_dict(table, cur_row)
                    seed["name"] = seed["name"] + "_copy"
                else:
                    seed = {
                        "name": "new", "frequency": 2.0,
                        "phase": [0.0, 0.5, 0.5, 0.0],
                        "body_height": 0.33, "step_height": 0.08,
                    }
                r = table.rowCount()
                table.insertRow(r)
                _populate_row(table, r, seed)
                table.selectRow(r)

            def _on_remove():
                r = table.currentRow()
                if 0 <= r < table.rowCount():
                    table.removeRow(r)

            def _on_reset():
                from src.system.training.gait_presets import default_presets_json
                defaults = default_presets_json()
                table.setRowCount(0)
                for i, p in enumerate(defaults):
                    table.insertRow(i)
                    _populate_row(table, i, p)

            add_btn.clicked.connect(_on_add)
            rm_btn.clicked.connect(_on_remove)
            reset_btn.clicked.connect(_on_reset)

            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok
                | QDialogButtonBox.StandardButton.Cancel
            )
            buttons.accepted.connect(dlg.accept)
            buttons.rejected.connect(dlg.reject)
            root.addWidget(buttons)

            if dlg.exec() == QDialog.DialogCode.Accepted:
                out = []
                for r in range(table.rowCount()):
                    try:
                        out.append(_row_to_dict(table, r))
                    except Exception:
                        continue
                write_val(_json.dumps(out, ensure_ascii=False))
                _refresh_label()

        btn.clicked.connect(_open_dialog)
        _refresh_label()
        return btn

    # ------------------------------------------------------------------
    # Output file list (Export.output_files)
    # ------------------------------------------------------------------

    def _make_output_file_list(self, key, get_val, write_val, logic):
        """Read-only bundle output manifest for the Export node.

        Reads sibling params (``bundle_name`` + ``version`` + the three
        ``include_*`` toggles) from this node's logic instance to
        compute:

          * the target bundle directory path
          * the list of files that will be produced
          * a ⚠ "Will Overwrite" tag on each row whose file already
            exists on disk

        Every row gets an Open button that hands the path to
        :class:`QDesktopServices` — it pops the OS file manager on
        existing files, or silently falls back to the parent directory
        for files that haven't been generated yet.

        The panel auto-refreshes whenever a sibling param changes by
        subscribing to ``scene.node_param_changed`` (same pattern as
        ``controller_status_panel`` / ``active_file_table``).
        """
        from PySide6.QtCore import QTimer, QUrl
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtWidgets import (
            QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
        )
        from pathlib import Path

        _self_ref = self

        border = get_color("training_widget_border", "#3d3d3d")
        input_bg = get_color("training_widget_input_bg", "#1A1A1A")
        text_c = get_color("training_widget_text", "#cccccc")
        muted = get_color("training_widget_label_text", "#888888")
        ok_c = "#4CAF50"
        warn_c = "#F59E0B"

        MAX_W = NODE_W - H_PAD * 2
        ROW_H = 20
        HDR_H = 14

        container = QWidget()
        container._full_row_widget = True
        container._preferred_height = PARAM_ROW_H

        root = QVBoxLayout(container)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(1)

        # No outer border — the Header label + separator line below
        # already provide enough visual structure; a frame here made
        # the Export node feel double-bordered against its parent.
        table_wrap = QFrame(container)
        table_wrap.setStyleSheet(
            f"background: transparent; border: none;"
        )
        tw_layout = QVBoxLayout(table_wrap)
        tw_layout.setContentsMargins(2, 2, 2, 2)
        tw_layout.setSpacing(0)
        root.addWidget(table_wrap)

        _row_widgets: list = []
        _height_cbs: list = []

        def _emit_h(h: int):
            container._preferred_height = h
            container.setMinimumHeight(h)
            container.setMaximumHeight(h)
            container.updateGeometry()
            for fn in _height_cbs:
                try:
                    fn(h)
                except Exception:
                    pass

        class _HSig:
            def connect(self_, fn):
                _height_cbs.append(fn)
            def emit(self_, h):
                _emit_h(h)

        container.height_changed = _HSig()
        container.preferred_row_height = lambda: max(PARAM_ROW_H, int(container._preferred_height))

        # ── helpers ───────────────────────────────────────────────────

        def _read_params() -> Dict[str, Any]:
            return (logic.parameters or {}) if logic is not None else {}

        def _resolve_workspace_root() -> Optional[Path]:
            """Walk up scene → view → parent chain to find the workspace
            name and compose the project root path. Fallback: cwd."""
            try:
                scene = _self_ref.scene()
            except RuntimeError:
                return None
            if scene is None:
                return None
            for view in scene.views():
                w = view
                while w is not None:
                    workspace = (
                        getattr(w, "_workspace_name", None)
                        or getattr(w, "_current_slug", None)
                    )
                    if workspace:
                        return Path.cwd() / "projects" / str(workspace)
                    try:
                        w = w.parentWidget()
                    except Exception:
                        break
            return None

        def _resolve_bundle_dir() -> Path:
            p = _read_params()
            bundle = str(p.get("bundle_name", "") or "trained_policy")
            version = str(p.get("version", "v1") or "v1")
            overwrite = str(p.get("overwrite", "true") or "true").strip().lower() == "true"
            if overwrite:
                full = bundle
            else:
                full = f"{bundle}_{version}" if version else bundle
            root = _resolve_workspace_root()
            if root is not None:
                return root / "training" / "exported" / full
            # Fallback: relative path under cwd so Open still opens
            # something sensible.
            return Path.cwd() / "training" / "exported" / full

        def _planned_files(bundle_dir: Path) -> List[Tuple[str, Path, bool]]:
            """Return [(display_name, abs_path, always_emitted)] for
            every file the next run will produce. ``always_emitted``
            is True for files that ship regardless of toggle state."""
            p = _read_params()
            def _b(k: str) -> bool:
                return str(p.get(k, "true") or "true").strip().lower() == "true"
            out: List[Tuple[str, Path, bool]] = []
            out.append(("manifest.yaml", bundle_dir / "manifest.yaml", True))
            out.append(("source.json", bundle_dir / "source.json", True))
            if _b("include_onnx"):
                out.append(("policy.onnx", bundle_dir / "policy.onnx", False))
            if _b("include_torchscript"):
                out.append(("policy.pt", bundle_dir / "policy.pt", False))
            if _b("include_normalization"):
                out.append(("normalization.pkl", bundle_dir / "normalization.pkl", False))
            return out

        # ── row rendering ─────────────────────────────────────────────

        def _lbl(text: str, color: str, width: Optional[int] = None,
                 bold: bool = False) -> QLabel:
            lb = QLabel(text)
            weight = "700" if bold else "400"
            lb.setStyleSheet(
                f"color: {color}; font-size: 9px; font-weight: {weight}; "
                f"background: transparent; border: none;"
            )
            if width is not None:
                lb.setFixedWidth(width)
            return lb

        def _open_path(path: Path) -> None:
            """Open *path* in the OS file manager."""
            try:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
            except Exception:
                pass

        def _make_open_button(path: Path) -> Optional[QPushButton]:
            """Return an Open button only when *path* actually exists
            on disk. Non-existent files return ``None`` so the caller
            can skip adding the button — this keeps the visual clean
            until post-training refresh makes the file appear."""
            if not path.exists():
                return None
            btn = QPushButton("Open")
            btn.setFixedHeight(ROW_H - 4)
            btn.setFixedWidth(42)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet(
                f"QPushButton {{ color: {text_c}; background: {input_bg}; "
                f"border: 1px solid {border}; border-radius: 2px; "
                f"font-size: 8px; padding: 0 4px; }}"
                f"QPushButton:hover {{ border: 1px solid #888; }}"
            )
            btn.setToolTip(str(path).replace("\\", "/"))
            btn.clicked.connect(lambda _=False, p=path: _open_path(p))
            return btn

        def _rebuild():
            # Startup pre-warm / canvas tear-down guard: if the owning
            # TrainingNodeItem has been destroyed between QTimer.singleShot
            # scheduling and firing, every Qt-child access below raises
            # RuntimeError. Check ``tw_layout.count()`` as a cheap liveness
            # probe and bail out silently — nothing to rebuild on a dead
            # widget.
            try:
                tw_layout.count()
            except RuntimeError:
                return
            for w in list(_row_widgets):
                try:
                    tw_layout.removeWidget(w)
                    w.deleteLater()
                except RuntimeError:
                    pass
            _row_widgets.clear()

            bundle_dir = _resolve_bundle_dir()

            # Header — bundle directory label + optional warn + Open.
            # No border on this row per user direction: the separator
            # line below already provides the visual break.
            dir_exists = bundle_dir.exists()
            hdr = QWidget(table_wrap)
            hdr.setFixedHeight(HDR_H + 4)
            hdr.setStyleSheet("QWidget { background: transparent; border: none; }")
            hl = QHBoxLayout(hdr)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.setSpacing(4)
            dir_label = _lbl(
                f"{bundle_dir.name}/", text_c if dir_exists else muted,
                bold=True,
            )
            dir_label.setToolTip(str(bundle_dir).replace("\\", "/"))
            hl.addWidget(dir_label, 1)
            if dir_exists:
                warn_lbl = _lbl("⚠", warn_c, bold=True)
                warn_lbl.setToolTip("Directory already exists — files below tagged ⚠ will be overwritten.")
                hl.addWidget(warn_lbl, 0)
                dir_btn = _make_open_button(bundle_dir)
                if dir_btn is not None:
                    hl.addWidget(dir_btn, 0)
            _row_widgets.append(hdr)
            tw_layout.addWidget(hdr)

            # Separator
            sep = QFrame(table_wrap)
            sep.setFixedHeight(1)
            sep.setStyleSheet(f"background: {border}; border: none;")
            _row_widgets.append(sep)
            tw_layout.addWidget(sep)

            # File rows
            files = _planned_files(bundle_dir)
            for idx, (display, path, always) in enumerate(files):
                row = QWidget(table_wrap)
                row.setFixedHeight(ROW_H)
                rl = QHBoxLayout(row)
                rl.setContentsMargins(0, 0, 0, 0)
                rl.setSpacing(4)

                exists = path.exists()
                # Uniform row background — zebra striping was leaving every
                # other row transparent, which read as "missing entry" at a
                # glance. Every planned file now paints the input background.
                row.setStyleSheet(f"QWidget {{ background: {input_bg}; border: none; }}")

                # Name
                name_lbl = _lbl(display, text_c if exists else muted)
                name_lbl.setToolTip(str(path).replace("\\", "/"))
                rl.addWidget(name_lbl, 1)

                # Overwrite warning tag
                if exists:
                    tag = _lbl("⚠", warn_c, bold=True, width=14)
                    tag.setToolTip("Will Overwrite")
                    rl.addWidget(tag, 0)

                # Status dot
                status = _lbl(
                    "Exists" if exists else "Planned",
                    ok_c if exists else muted,
                    width=44,
                )
                rl.addWidget(status, 0)

                # Open button: shown only when the file actually exists.
                # Post-training refresh (wired later by the user) will
                # re-render these rows and populate the buttons.
                open_btn = _make_open_button(path)
                if open_btn is not None:
                    rl.addWidget(open_btn, 0)

                _row_widgets.append(row)
                tw_layout.addWidget(row)

            total_h = HDR_H + 4 + 1 + ROW_H * len(files) + 6
            table_wrap.setFixedHeight(total_h)
            _emit_h(total_h + 2)

        # ── param-change subscription ────────────────────────────────

        _watch_keys = {
            "bundle_name", "version", "overwrite",
            "include_onnx", "include_torchscript", "include_normalization",
        }

        def _on_canvas_param_changed(node_type: str, param_key: str, _v: str):
            if node_type == "export" and param_key in _watch_keys:
                QTimer.singleShot(0, _rebuild)

        def _connect_scene_signal():
            # Guard against shiboken-deleted C++ items: if the owning
            # TrainingNodeItem was torn down between QTimer.singleShot
            # scheduling and firing (common during startup pre-warm),
            # ``_self_ref.scene()`` raises RuntimeError. Treat it as
            # "nothing to connect" — the widget is gone anyway.
            try:
                scene = _self_ref.scene()
            except RuntimeError:
                return
            if scene is not None and hasattr(scene, "node_param_changed"):
                try:
                    scene.node_param_changed.connect(_on_canvas_param_changed)
                except Exception:
                    pass

        QTimer.singleShot(0, _rebuild)
        QTimer.singleShot(50, _connect_scene_signal)
        return container

    # ------------------------------------------------------------------
    # Start Point compatibility panel (Export §2)
    # ------------------------------------------------------------------

    def _make_start_point_compat_panel(self, key, get_val, write_val, logic):
        """Read-only compatibility diff panel for the Export node.

        Renders :class:`StartPointCompatReport` as a compact table:

          * header row — Start Point label, load_mode, overall status dot
          * separator line
          * one row per :class:`CompatField` (Framework / Action dim /
            Decimation / Joint order) with old vs new values and a
            per-row status icon
          * footer summary line

        Auto-rebuilds when the canvas changes (subscribes to
        ``scene.node_param_changed`` which fires on both param edits
        and edge mutations via the graph_dirty hook). No direct file
        IO in here — that happens inside
        :meth:`TrainingContext.compat_report`.
        """
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout

        _self_ref = self

        border = get_color("training_widget_border", "#3d3d3d")
        text_c = get_color("training_widget_text", "#cccccc")
        muted = get_color("training_widget_label_text", "#888888")
        ok_c = "#4CAF50"
        warn_c = "#F59E0B"
        err_c = "#EF4444"
        dim_c = "#6B7280"

        MAX_W = NODE_W - H_PAD * 2
        HDR_H = 14
        ROW_H = 18
        FOOTER_H = 26

        container = QWidget()
        container._full_row_widget = True
        container._preferred_height = PARAM_ROW_H

        root = QVBoxLayout(container)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(1)

        panel = QFrame(container)
        # No outer border — matches the §1 output_file_list convention
        # (user wanted borderless panels inside the Export node body).
        panel.setStyleSheet("background: transparent; border: none;")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(2, 2, 2, 2)
        panel_layout.setSpacing(0)
        root.addWidget(panel)

        _row_widgets: list = []
        _height_cbs: list = []

        def _emit_h(h: int):
            container._preferred_height = h
            container.setMinimumHeight(h)
            container.setMaximumHeight(h)
            container.updateGeometry()
            for fn in _height_cbs:
                try:
                    fn(h)
                except Exception:
                    pass

        class _HSig:
            def connect(self_, fn):
                _height_cbs.append(fn)
            def emit(self_, h):
                _emit_h(h)

        container.height_changed = _HSig()
        container.preferred_row_height = lambda: max(PARAM_ROW_H, int(container._preferred_height))

        def _lbl(text: str, color: str, width: Optional[int] = None,
                 bold: bool = False, size_px: int = 9) -> QLabel:
            lb = QLabel(text)
            weight = "700" if bold else "400"
            lb.setStyleSheet(
                f"color: {color}; font-size: {size_px}px; font-weight: {weight}; "
                f"background: transparent; border: none;"
            )
            if width is not None:
                lb.setFixedWidth(width)
            return lb

        def _status_dot(status: str) -> QLabel:
            mapping = {
                "ok": ("●", ok_c),
                "warning": ("●", warn_c),
                "error": ("●", err_c),
                "none": ("○", dim_c),
            }
            glyph, col = mapping.get(status, ("○", dim_c))
            return _lbl(glyph, col, width=12, bold=True, size_px=12)

        def _status_icon(status: str) -> QLabel:
            mapping = {
                "ok": ("✓", ok_c),
                "warning": ("⚠", warn_c),
                "error": ("✗", err_c),
            }
            glyph, col = mapping.get(status, ("—", dim_c))
            return _lbl(glyph, col, width=14, bold=True)

        def _row(widgets, bg: str = "transparent") -> QWidget:
            rw = QWidget(panel)
            rw.setFixedHeight(ROW_H)
            rw.setStyleSheet(f"QWidget {{ background: {bg}; border: none; }}")
            rl = QHBoxLayout(rw)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(4)
            for w in widgets:
                rl.addWidget(w)
            return rw

        def _separator() -> QFrame:
            sep = QFrame(panel)
            sep.setFixedHeight(1)
            sep.setStyleSheet(f"background: {border}; border: none;")
            return sep

        def _build_report():
            """Call TrainingContext.compat_report with the current graph."""
            try:
                scene = _self_ref.scene()
            except RuntimeError:
                return None
            if scene is None:
                return None
            try:
                graph = scene.serialize_training_graph(for_compiler=False)
            except Exception:
                return None
            # Walk up to the workspace window to borrow its
            # TrainingContext instance — that's the one SafetyReview
            # uses, so the widget displays the same spec the compiler
            # would see.
            ctx = None
            for view in scene.views():
                w = view
                while w is not None:
                    getter = getattr(w, "_get_training_context", None)
                    if callable(getter):
                        try:
                            ctx = getter()
                            break
                        except Exception:
                            pass
                    try:
                        w = w.parentWidget()
                    except Exception:
                        break
                if ctx is not None:
                    break
            if ctx is None:
                return None
            try:
                return ctx.compat_report(graph)
            except Exception:
                return None

        def _rebuild():
            # Liveness probe — bail if the owning item has been torn
            # down between timer schedule and fire.
            try:
                panel_layout.count()
            except RuntimeError:
                return
            for w in list(_row_widgets):
                try:
                    panel_layout.removeWidget(w)
                    w.deleteLater()
                except RuntimeError:
                    pass
            _row_widgets.clear()

            report = _build_report()

            if report is None:
                msg = _lbl("(not ready)", muted)
                row_w = _row([msg])
                _row_widgets.append(row_w)
                panel_layout.addWidget(row_w)
                panel.setFixedHeight(ROW_H + 4)
                _emit_h(ROW_H + 6)
                return

            # Header row — label + mode + dot
            label_text = report.start_point_label or "(none)"
            hdr = _row([
                _lbl("Start", muted, width=34, bold=True),
                _lbl(label_text, text_c, bold=True),
                _lbl("·", muted),
                _lbl(report.load_mode, text_c),
                _status_dot(report.overall_status),
            ])
            _row_widgets.append(hdr)
            panel_layout.addWidget(hdr)

            if report.fields:
                sep = _separator()
                _row_widgets.append(sep)
                panel_layout.addWidget(sep)

                for idx, f in enumerate(report.fields):
                    row_w = _row([
                        _lbl(f.label, muted, width=64),
                        _lbl(str(f.old_value), text_c),
                        _lbl("→", muted, width=10),
                        _lbl(str(f.new_value), text_c),
                        _status_icon(f.status),
                    ])
                    if f.detail:
                        row_w.setToolTip(f.detail)
                    _row_widgets.append(row_w)
                    panel_layout.addWidget(row_w)

            if report.manifest_loaded is False and report.manifest_error:
                sep = _separator()
                _row_widgets.append(sep)
                panel_layout.addWidget(sep)
                err_row = _row([
                    _lbl("⚠", err_c, width=14, bold=True),
                    _lbl(report.manifest_error, err_c),
                ])
                _row_widgets.append(err_row)
                panel_layout.addWidget(err_row)

            # Footer summary
            footer = QLabel(report.summary or "")
            footer.setWordWrap(True)
            footer.setStyleSheet(
                f"color: {muted}; font-size: 8px; background: transparent; "
                f"border: none; padding: 2px 4px 0 4px;"
            )
            footer.setMinimumHeight(FOOTER_H)
            _row_widgets.append(footer)
            panel_layout.addWidget(footer)

            # Compute total height — 1 header + optional sep + N fields
            # + optional manifest err sep+row + footer. Conservative.
            n_field_rows = len(report.fields)
            total_h = ROW_H          # header
            if n_field_rows:
                total_h += 1 + ROW_H * n_field_rows
            if report.manifest_error:
                total_h += 1 + ROW_H
            total_h += FOOTER_H + 4
            panel.setFixedHeight(total_h)
            _emit_h(total_h + 2)

        def _on_canvas_param_changed(_node_type: str, _key: str, _value: str):
            QTimer.singleShot(0, _rebuild)

        def _connect_scene_signal():
            # Guard against shiboken-deleted C++ items: if the owning
            # TrainingNodeItem was torn down between QTimer.singleShot
            # scheduling and firing (common during startup pre-warm),
            # ``_self_ref.scene()`` raises RuntimeError. Treat it as
            # "nothing to connect" — the widget is gone anyway.
            try:
                scene = _self_ref.scene()
            except RuntimeError:
                return
            if scene is not None and hasattr(scene, "node_param_changed"):
                try:
                    scene.node_param_changed.connect(_on_canvas_param_changed)
                except Exception:
                    pass

        QTimer.singleShot(0, _rebuild)
        QTimer.singleShot(50, _connect_scene_signal)
        return container

    # ------------------------------------------------------------------
    # Review backend picker (Export §3)
    # ------------------------------------------------------------------

    def _make_review_backend_picker(self, key, get_val, write_val, logic):
        """Node-dedicated picker button for the Review backend.

        Single dropdown button (``_RichChoicePicker``) matching the
        scene_dropdown / contact_body_picker family. Choices come from
        :mod:`review_backends`, re-scanned every time the popup opens
        so that future plugin installs (Newton) surface without a
        canvas reload. Unavailable backends show in the popup with a
        disabled marker in their description and cannot be picked.
        """
        from src.system.training.review_backends import (
            list_backends, default_backend_id,
        )

        def _backend_choices_provider() -> Tuple[List[str], Dict[str, Dict[str, str]]]:
            try:
                backends = list_backends(available_only=False)
            except Exception:
                backends = []
            if not backends:
                return (
                    ["(none)"],
                    {"(none)": {"title": "(none)",
                                "desc": "No review backends registered"}},
                )
            ids: List[str] = []
            meta: Dict[str, Dict[str, str]] = {}
            for b in backends:
                ids.append(b.backend_id)
                desc = b.description
                if not b.available:
                    desc += "    ·  (not yet available)"
                meta[b.backend_id] = {
                    "title": b.display_name,
                    "desc": desc,
                }
            return ids, meta

        choices, meta = _backend_choices_provider()
        current = str(get_val() or "").strip()
        if current not in choices:
            current = default_backend_id() if default_backend_id() in choices else (choices[0] if choices else "")

        w = _RichChoicePicker(
            choices,
            current=current,
            meta_map=meta,
            leading_mode="checkbox",
            multi_select=False,
            choices_provider=_backend_choices_provider,
        )

        def _on_pick(value: str) -> None:
            if not value or value.startswith("(none"):
                return
            # Guard against the popup letting through a disabled backend
            # (the registry says which ones are unavailable).
            try:
                backends = {b.backend_id: b for b in list_backends(available_only=False)}
                entry = backends.get(value)
                if entry is not None and not entry.available:
                    return
            except Exception:
                pass
            write_val(value)

        w.currentTextChanged.connect(_on_pick)
        return w

    # ------------------------------------------------------------------
    # Review scene picker (Export §3)
    # ------------------------------------------------------------------

    def _make_review_scene_picker(self, key, get_val, write_val, write_params, logic):
        """Scene picker filtered by the current ``review_backend``.

        Same ``_RichChoicePicker`` widget the training-side scene
        picker uses, but points at :func:`scene_registry.list_review_scenes`
        so Isaac-only scenes disappear when MuJoCo is selected (and
        vice-versa). Re-scans on every popup open so toggling the
        backend button row immediately refreshes the available set.
        """
        _self_ref = self

        def _detect_family() -> Optional[str]:
            """Same robot-family probe the training scene picker uses."""
            scene = _self_ref.scene()
            if scene is None:
                return None
            for item in scene.items():
                try:
                    if item.data(12) != "robot":
                        continue
                    nl = item.data(13)
                    if nl is None:
                        continue
                    aid = str((getattr(nl, "parameters", {}) or {}).get("asset_id", "")).strip()
                    if not aid:
                        continue
                    from src.system.training.robot_assets import get_asset, rescan
                    rescan()
                    return str(getattr(get_asset(aid), "family", "") or "") or None
                except Exception:
                    continue
            return None

        def _current_review_backend() -> str:
            p = (logic.parameters or {}) if logic is not None else {}
            val = str(p.get("review_backend", "") or "").strip()
            if val:
                return val
            from src.system.training.review_backends import default_backend_id
            return default_backend_id()

        def _review_scene_choices(
            _cur_val_fn: Callable[[], str] = get_val,
        ) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
            backend = _current_review_backend()
            family = _detect_family()
            try:
                from src.system.training.scene_registry import (
                    list_review_scenes, rescan,
                )
                rescan()
                scenes = list_review_scenes(review_backend=backend, family=family)
            except Exception:
                scenes = []
            if not scenes:
                return (
                    ["(no scenes)"],
                    {"(no scenes)": {
                        "title": "(no scenes)",
                        "desc": f"No registered scenes support review backend {backend!r}",
                    }},
                )
            ids = [s.scene_id for s in scenes]
            meta: Dict[str, Dict[str, str]] = {}
            for s in scenes:
                fams = ", ".join(sorted(s.supported_families)) or "any family"
                meta[s.scene_id] = {
                    "title": s.name,
                    "desc": f"{s.description}    ·  {fams}",
                }
            return ids, meta

        initial_choices, initial_meta = _review_scene_choices()
        current_val = get_val()
        if current_val not in initial_choices:
            current_val = initial_choices[0] if initial_choices else ""

        w = _RichChoicePicker(
            initial_choices,
            current=current_val,
            meta_map=initial_meta,
            leading_mode="checkbox",
            multi_select=False,
            choices_provider=_review_scene_choices,
        )

        def _on_scene_pick(value: str) -> None:
            if not value or value.startswith("(no "):
                return
            write_val(value)

        w.currentTextChanged.connect(_on_scene_pick)
        return w

    # ------------------------------------------------------------------
    # Review launch button (Export §3)
    # ------------------------------------------------------------------

    def _make_review_launch_button(self, key, get_val, write_val, logic):
        """SafetyReview-gated "Launch Review" action button.

        Click pipeline:

          1. Walk up to the TrainingWorkspaceWindow (via scene →
             view → window).
          2. Call :meth:`_run_safety_review(blocking=True)` — any
             error-severity issue on the canvas stops here, the
             button reverts to idle, and the SafetyReview highlight
             is already painting red borders on the offending nodes.
          3. If clean, emit the workspace-level
             ``review_launch_requested`` signal with a payload
             ``{backend, scene_id, bundle_path}``. The workspace slot
             dispatches to the matching review session subprocess
             (isaac_review_session / il_review_session / ...).

        The subprocess lifetime is owned by the workspace — this
        widget never holds a QProcess / subprocess reference, so a
        canvas tear-down during review cannot orphan the viewer.
        """
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QPushButton

        _self_ref = self

        border = get_color("training_widget_border", "#3d3d3d")
        input_bg = get_color("training_widget_input_bg", "#1A1A1A")
        text_c = get_color("training_widget_text", "#cccccc")
        muted = get_color("training_widget_label_text", "#888888")
        accent_bg = "#00695C"

        btn = QPushButton("▶  Launch Review")
        btn.setFixedHeight(PARAM_ROW_H)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setToolTip(
            "Run SafetyReview, then spawn a review session using the "
            "selected backend + scene. The trained brain is loaded "
            "into a separate viewer process owned by the workspace — "
            "the canvas is safe to tear down mid-review."
        )

        def _idle_style():
            btn.setStyleSheet(
                f"QPushButton {{ color: #fff; background: {accent_bg}; "
                f"border: 1px solid {accent_bg}; border-radius: 4px; "
                f"font-size: 10px; font-weight: 600; padding: 4px 10px; }}"
                f"QPushButton:hover {{ background: #00897B; }}"
                f"QPushButton:disabled {{ color: {muted}; background: {input_bg}; "
                f"border: 1px solid {border}; }}"
            )
        _idle_style()

        def _walk_to_workspace():
            scene = _self_ref.scene()
            if scene is None:
                return None
            for view in scene.views():
                # Walk the parent widget chain from the view upward —
                # TrainingWorkspaceWindow is an intermediate QWidget,
                # NOT the top-level QMainWindow that view.window()
                # returns. The old code only checked the top window
                # and silently returned None.
                w = view
                while w is not None:
                    if hasattr(w, "_run_safety_review"):
                        return w
                    try:
                        w = w.parentWidget()
                    except Exception:
                        break
            return None

        def _on_click(_checked: bool = False):
            workspace = _walk_to_workspace()
            if workspace is None:
                return

            # SafetyReview gate — blocking, SB3 path only.
            # The SB3 TrainingSpecCompiler validates against SB3-only
            # node types (algo_config, env_assembler, …) which don't
            # exist on Isaac Lab canvases. Running it on an IL graph
            # always fails with a spurious "missing required node types"
            # error.  IL has its own validation in the config compiler.
            _backend = str(
                getattr(workspace, "_active_backend", "sb3") or "sb3"
            )
            if _backend != "isaac_lab":
                try:
                    ok = workspace._run_safety_review(blocking=True)
                except Exception:
                    ok = False
                if not ok:
                    return

            # Pull the current param snapshot from the node.
            params = (logic.parameters or {}) if logic is not None else {}
            payload = {
                "backend": str(params.get("review_backend", "mujoco") or "mujoco"),
                "scene_id": str(params.get("review_scene_id", "") or ""),
                "bundle_name": str(params.get("bundle_name", "") or ""),
                "version": str(params.get("version", "v1") or "v1"),
                "overwrite": str(params.get("overwrite", "true") or "true").lower() == "true",
            }

            scene = _self_ref.scene()
            if scene is None:
                return
            emit = getattr(scene, "review_launch_requested", None)
            if emit is None:
                # Signal not wired yet — surface a clear notice via
                # log path so users know the workspace hasn't been
                # upgraded to the new handler.
                try:
                    from src.system.core.logger import log_warning
                    log_warning(
                        "[Export] review_launch_requested signal missing — "
                        "workspace has not been upgraded to the Review Env §3 handler."
                    )
                except Exception:
                    pass
                return
            try:
                emit.emit(payload)
            except Exception:
                pass

        btn.clicked.connect(_on_click)
        return btn

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
        scene = self.scene()
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
                if scene is not None:
                    scene.removeItem(proxy)
            except Exception:
                pass
        self._param_proxies.clear()
        self._param_row_y_cache.clear()

    def _rebuild_geometry(self, rebuild_ports: bool = True) -> None:
        """Rebuild node geometry; keep existing ports when only params changed."""
        if rebuild_ports:
            scene = self.scene()
            for child in list(self.childItems()):
                child.setParentItem(None)
                if scene is not None:
                    scene.removeItem(child)
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
            # Preview clip list tracks motion_source / motion_file —
            # push a refresh into the live widget so pack changes are
            # reflected in the button text and dropdown immediately.
            if (self._ref_preview_button_widget is not None
                    and ("motion_source" in updates or "motion_file" in updates)):
                try:
                    self._ref_preview_button_widget._refresh_candidates()
                except Exception:
                    pass

        # ── motion_library_picker ─────────────────────────────────────
        # Unified Reference Motion picker. Drives BOTH motion_file (for
        # single-clip picks) and motion_source (for symbolic
        # generate:/loco:/pack: picks) so we can route everything through
        # one dropdown — see _MotionLibraryPicker docstring for layout.
        if w_type == "motion_library_picker":
            params = (logic.parameters or {}) if logic is not None else {}
            w = _MotionLibraryPicker(
                current_motion_file=str(params.get("motion_file", "")),
                current_motion_source=str(params.get("motion_source", "")),
                robot_type=self._robot_type_hint,
                write_params=write_params,
            )
            self._ref_motion_picker_widget = w
            return w

        # ── asset_dropdown ────────────────────────────────────────────
        # Phase_1 of AMP_design.yaml §3.custom_robot_assets: a closed
        # dropdown populated from the phase_1 registry via a live scan
        # of custom_mods/archives/ + menagerie/. Uses the project's
        # standard ``_RichChoicePicker`` (same look as IL Terrain Config
        # Terrain Type) with a ``choices_provider`` so the list
        # re-scans every time the popup opens — dropping new files
        # into archives is visible on the next click with no restart.
        if w_type == "asset_dropdown":
            family_filter = str(row.get("family_filter", "") or "").strip() or None

            def _has_usd(asset) -> bool:
                """True iff this asset can be spawned by Isaac Lab.

                Either an on-disk .usd file (rare for menagerie/archives)
                or a Nucleus URL marker (set by discovery for known
                menagerie robots). Archive assets that only ship URDF/
                MJCF return False — they need URDF→USD conversion before
                they can drive Isaac Lab training.
                """
                return asset.usd_path is not None or bool(asset.usd_url)

            def _rel_model_path(asset) -> str:
                """Short repo-relative description shown under the title.

                Format: ``<usd-source>  ·  <mjcf/urdf-source if any>``
                When no USD is available, prefix with ⚠ so the user can
                see at a glance that the asset is NOT Isaac Lab ready.
                """
                try:
                    repo_root = pathlib.Path(__file__).resolve().parents[2]
                except Exception:
                    repo_root = pathlib.Path.cwd()

                def _rel(p) -> str:
                    if p is None:
                        return ""
                    try:
                        return str(pathlib.Path(p).resolve().relative_to(repo_root))
                    except Exception:
                        return str(p)

                # ── Primary line: USD source (or warning) ──
                if asset.usd_path is not None:
                    primary = _rel(asset.usd_path)
                elif asset.usd_url:
                    primary = asset.usd_url
                else:
                    primary = "⚠ no USD — not Isaac Lab compatible"

                # ── Secondary line: mjcf/urdf for context ──
                secondary_bits = []
                if asset.mjcf_path is not None:
                    secondary_bits.append("mjcf=" + _rel(asset.mjcf_path))
                if asset.urdf_path is not None:
                    secondary_bits.append("urdf=" + _rel(asset.urdf_path))
                if secondary_bits:
                    return primary + "    " + " · ".join(secondary_bits)
                return primary

            def _asset_choices_provider(
                filt: Optional[str] = family_filter,
                current_val_fn: Callable[[], str] = get_val,
            ) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
                # Live rescan on every popup open — idempotent + cheap.
                try:
                    from src.system.training.robot_assets import (
                        rescan,
                        list_assets,
                    )
                    rescan()
                    assets = list_assets(family=filt)
                except Exception:
                    assets = []

                # Sort: USD-capable first, then USD-less. Within each
                # group keep alphabetical so users see Isaac-Lab-ready
                # robots at the top of the list.
                assets_sorted = sorted(
                    assets, key=lambda a: (0 if _has_usd(a) else 1, a.asset_id)
                )

                choices: List[str] = [a.asset_id for a in assets_sorted]
                meta_map: Dict[str, Dict[str, str]] = {}
                for a in assets_sorted:
                    title = a.asset_id if _has_usd(a) else f"⚠ {a.asset_id}"
                    meta_map[a.asset_id] = {
                        "title": title,
                        "desc": _rel_model_path(a),
                    }

                # Preserve unknown current value (e.g. from an old canvas)
                # so the node doesn't silently lose data on first render.
                current = current_val_fn()
                if current and current not in choices:
                    choices.append(current)
                    meta_map[current] = {
                        "title": current,
                        "desc": "(unknown — not in registry)",
                    }

                return choices, meta_map

            # Prime the initial list so the button shows something sensible
            # before the popup is opened for the first time.
            initial_choices, initial_meta = _asset_choices_provider()
            if not initial_choices:
                # Graceful empty state — single placeholder entry that
                # tells the user what to do. Picking it writes "" back.
                placeholder = "(no assets registered)"
                initial_choices = [placeholder]
                initial_meta = {
                    placeholder: {
                        "title": placeholder,
                        "desc": "Drop files into custom_mods/archives/<pkg>/resources/robots/",
                    }
                }

            current_val = get_val()
            if current_val not in initial_choices:
                current_val = initial_choices[0] if initial_choices else ""

            w = _RichChoicePicker(
                initial_choices,
                current=current_val,
                meta_map=initial_meta,
                leading_mode="checkbox",
                multi_select=False,
                choices_provider=_asset_choices_provider,
            )

            def _on_asset_pick(value: str, fn=write_val) -> None:
                # Never persist the placeholder sentinel
                if value.startswith("(no assets") or value.startswith("(unknown"):
                    return
                fn(value)

            w.currentTextChanged.connect(_on_asset_pick)
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
            def _get_candidates() -> List[tuple]:
                """Return list of (label, absolute_path) clips to preview.

                Phase_2 (AMP_design.yaml §3) — the unified picker writes
                both motion_file and motion_source. We apply the same
                precedence the backend uses (motion_source > motion_file)
                and expand pack: identifiers into their full member list
                so the user can scrub through individual clips in the UI.
                """
                if self._logic_node is None:
                    return []
                params = self._logic_node.parameters or {}
                ms = str(params.get("motion_source", "") or "").strip()
                if ms:
                    try:
                        from src.system.nodes.sys_nodes.training_nodes import (
                            ReferenceMotionNode,
                        )
                        resolved_files = ReferenceMotionNode.resolve_motion_source_files(ms)
                    except Exception:
                        return []
                    return [
                        (pathlib.Path(p).stem, p)
                        for p in resolved_files
                        if pathlib.Path(p).is_file()
                    ]
                # Fall through to motion_file when no motion_source set
                mf = str(params.get("motion_file", "") or "").strip()
                if mf and pathlib.Path(mf).is_file():
                    return [(pathlib.Path(mf).stem, mf)]
                return []

            def _get_fps() -> float:
                s = self._ref_fps_slider_widget
                if s is None:
                    return 30.0
                try:
                    return float(s._slider.value()) or 30.0
                except Exception:
                    return 30.0

            def _get_scene_xml_override() -> str:
                """Resolve the preview scene XML from the current canvas.

                Prefers the IL Robot Asset node's ``asset_id`` (what the
                training run will actually spawn) → ``resolve_for_mujoco``.
                Returns an empty string when no usable asset is on canvas,
                in which case the widget falls back to the generic
                robot_type → registered-scene lookup for SB3 canvases.
                """
                scene = self.scene()
                if scene is None:
                    return ""
                for it in scene.items():
                    try:
                        if it is self:
                            continue
                        if it.data(12) != "il_robot_asset":
                            continue
                        nl = it.data(13)
                        if nl is None:
                            continue
                        p = getattr(nl, "parameters", {}) or {}
                        aid = str(p.get("asset_id", "") or "").strip()
                        if not aid:
                            continue
                        from src.system.training.robot_assets import resolve_for_mujoco
                        mjcf = resolve_for_mujoco(aid)
                        return str(mjcf) if mjcf else ""
                    except Exception:
                        continue
                return ""

            def _get_mode() -> str:
                """Read the sibling preview_physics toggle: ON → physics, OFF → kinematic."""
                if self._logic_node is None:
                    return "physics"
                raw = str((self._logic_node.parameters or {}).get("preview_physics", "true") or "true")
                return "physics" if raw.lower() in ("true", "1", "yes", "on") else "kinematic"

            w = _PreviewButtonWidget(_get_candidates, _get_fps,
                                     lambda: self._robot_type_hint,
                                     get_scene_xml_override=_get_scene_xml_override,
                                     get_mode=_get_mode)
            # Keep a reference so write_params can re-trigger the clip
            # refresh whenever the sibling motion picker mutates
            # motion_file / motion_source (the widget instance stays
            # alive across param writes — write_params only refreshes
            # row conditions, it does not rebuild proxies).
            self._ref_preview_button_widget = w
            return w

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
                    load_mode = current_load_mode if current_load_mode != "scratch" else "resume"
                    write_params({
                        "start_point": choice,
                        "asset_id": "",
                        "checkpoint_file": "",
                        "load_mode": load_mode,
                    })
                    return

                if choice.startswith("run:"):
                    # Run-dir checkpoint. Default to warm_start_actor:
                    # the common use case here is seeding a new
                    # algorithm / reward layout from an existing
                    # policy, and warm_start_actor is correct in both
                    # that case and the pure-resume case (it just
                    # means "trust the user's reward changes, reset
                    # the optimizer"). Users who truly want a bitwise
                    # resume can switch the load_mode dropdown to
                    # 'resume' afterwards.
                    load_mode = (
                        current_load_mode
                        if current_load_mode in ("resume", "warm_start_actor")
                        else "warm_start_actor"
                    )
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
                        load_mode = "resume" if getattr(entry, "framework", "") == "sb3" else "warm_start_actor"
                except Exception:
                    if load_mode == "scratch":
                        load_mode = "resume"

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
            w = NodeInputer(
                current_text=get_val(),
                placeholder=str(row.get("placeholder", "")),
                mode="text",
            )
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
            w = NodeInputer(
                current_text=get_val(),
                mode="float",
                min_value=float(row.get("min", -1e9)),
                max_value=float(row.get("max", 1e9)),
                decimals=int(row.get("decimals", 6)),
            )
            w.textChanged.connect(write_val)
            return w

        # ── scientific_input ──────────────────────────────────────────
        if w_type == "scientific_input":
            w = NodeInputer(
                current_text=get_val(),
                placeholder="e.g. 3e-4",
                mode="scientific",
            )
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

            choices = [str(choice) for choice in row.get("choices", [])]
            choice_meta_raw = row.get("choice_meta") or {}
            if choice_meta_raw:
                meta_map: Dict[str, Dict[str, str]] = {}
                for choice in choices:
                    entry = choice_meta_raw.get(choice) or {}
                    default_meta = _default_choice_meta(choice)
                    meta_map[choice] = {
                        "title": str(entry.get("title", default_meta["title"])),
                        "desc":  str(entry.get("desc",  default_meta["desc"])),
                    }
                w = _RichChoicePicker(
                    choices,
                    current=get_val(),
                    meta_map=meta_map,
                    leading_mode="checkbox",
                    multi_select=False,
                )
            else:
                w = _DropdownChoicePicker(choices, get_val())
            w.currentTextChanged.connect(write_val)
            return w

        # ── toggle ────────────────────────────────────────────────────
        if w_type == "toggle":
            border = get_color("training_widget_border", "#3d3d3d")
            input_bg = get_color("training_widget_input_bg", "#1A1A1A")
            accent_bg = "#00695C"
            w = QPushButton()
            w.setCheckable(True)
            is_on = get_val().lower() in ("true", "1", "yes")
            w.setChecked(is_on)
            w.setText("ON" if is_on else "OFF")

            def _apply_toggle_style(checked: bool, btn: QPushButton = w) -> None:
                if checked:
                    btn.setStyleSheet(
                        f"QPushButton {{ color: #fff; background: {accent_bg}; "
                        f"border: 1px solid {accent_bg}; border-radius: 3px; "
                        f"font-size: 9px; font-weight: 600; padding: 0 6px; }}"
                        f"QPushButton:hover {{ background: #00897B; }}"
                    )
                else:
                    btn.setStyleSheet(
                        f"QPushButton {{ color: #888; background: {input_bg}; "
                        f"border: 1px solid {border}; border-radius: 3px; "
                        f"font-size: 9px; padding: 0 6px; }}"
                        f"QPushButton:hover {{ border: 1px solid #888; }}"
                    )
            _apply_toggle_style(is_on)

            def _on_toggle(checked: bool, btn: QPushButton = w, fn=write_val) -> None:
                btn.setText("ON" if checked else "OFF")
                _apply_toggle_style(checked, btn)
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

            # Accept both JSON list "[lo, hi]" and Python tuple "(lo, hi)"
            # syntax — older presets used tuple repr which json.loads
            # rejects, causing a silent fallback to the slider's full
            # range and hiding the real stored value from the user.
            def _parse_range_pair(raw: str) -> Optional[Tuple[float, float]]:
                s = str(raw or "").strip()
                if not s:
                    return None
                try:
                    parts = json.loads(s)
                    return float(parts[0]), float(parts[1])
                except Exception:
                    pass
                try:
                    import ast as _ast
                    parts = _ast.literal_eval(s)
                    return float(parts[0]), float(parts[1])
                except Exception:
                    return None

            parsed = _parse_range_pair(get_val())
            if parsed is not None:
                lo_v, hi_v = parsed
            else:
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

        # ── int_tuple ─────────────────────────────────────────────────
        # N inline integer spinboxes sharing a single WIDGET_W-width row.
        # Stores the combined value as a JSON list "[a, b, c]".
        if w_type == "int_tuple":
            from PySide6.QtWidgets import QAbstractSpinBox
            count = max(1, int(row.get("count", 3)))
            min_v = int(row.get("min", 1))
            max_v = int(row.get("max", 99_999))
            step  = int(row.get("step", 1))

            try:
                parts = json.loads(str(get_val()))
                if not isinstance(parts, list):
                    parts = []
            except Exception:
                parts = []
            while len(parts) < count:
                parts.append(min_v)
            parts = [int(p) for p in parts[:count]]

            container = QWidget()
            container.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
            container.setMaximumWidth(WIDGET_W)
            hl = QHBoxLayout(container)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.setSpacing(2)

            spins: List[QSpinBox] = []
            for i in range(count):
                sb = QSpinBox()
                sb.setRange(min_v, max_v)
                sb.setSingleStep(step)
                sb.setValue(parts[i])
                sb.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
                sb.setAlignment(Qt.AlignmentFlag.AlignCenter)
                hl.addWidget(sb, 1)
                spins.append(sb)

            def _on_tuple_changed(_v: int = 0, _spins: List[QSpinBox] = spins,
                                  fn=write_val) -> None:
                fn("[" + ", ".join(str(s.value()) for s in _spins) + "]")

            for sb in spins:
                sb.valueChanged.connect(_on_tuple_changed)
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
            storage_mode = str(row.get("storage", "space")).lower()
            option_meta_raw = row.get("option_meta") or row.get("choice_meta") or {}

            def _parse_current(raw: str, mode: str = storage_mode) -> List[str]:
                raw = str(raw or "").strip()
                if not raw:
                    return []
                if mode == "json_list":
                    try:
                        parsed = json.loads(raw)
                        if isinstance(parsed, list):
                            return [str(item) for item in parsed]
                    except Exception:
                        pass
                    return []
                if mode == "csv":
                    return [f.strip() for f in raw.split(",") if f.strip()]
                return raw.split()

            def _encode_selection(values: List[str], mode: str = storage_mode) -> str:
                if mode == "json_list":
                    return json.dumps(values)
                if mode == "csv":
                    return ",".join(values)
                return " ".join(values)

            meta_map: Dict[str, Dict[str, str]] = {}
            for opt in options:
                entry = option_meta_raw.get(opt) or {}
                default_meta = _default_choice_meta(opt)
                meta_map[opt] = {
                    "title": str(entry.get("title", default_meta["title"])),
                    "desc":  str(entry.get("desc",  default_meta["desc"])),
                }

            current_values = [v for v in _parse_current(get_val()) if v in options]
            picker = _RichChoicePicker(
                options,
                current_values=current_values,
                meta_map=meta_map,
                leading_mode="checkbox",
                multi_select=True,
            )

            def _on_picker_changed(values: List[str], fn=write_val,
                                   _encode=_encode_selection) -> None:
                fn(_encode(list(values)))

            picker.selectionChanged.connect(_on_picker_changed)
            return picker

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
            _reg_dispatch = {
                "rewards": reward_registry,
                "terminations": termination_registry,
                "il_rewards": il_reward_registry,
                "il_observations": il_obs_registry,
                "il_terminations": il_termination_registry,
            }
            # Backend-aware registry: "rewards_auto" / "terminations_auto" pick
            # SB3 vs IL registry based on the node's backend parameter so the
            # frontend stays a single items-list widget across both backends.
            if registry_kind == "rewards_auto":
                _be = (logic.get_parameter("backend", "sb3") if logic else "sb3")
                registry = il_reward_registry() if _be == "isaac_lab" else reward_registry()
            elif registry_kind == "terminations_auto":
                _be = (logic.get_parameter("backend", "sb3") if logic else "sb3")
                registry = il_termination_registry() if _be == "isaac_lab" else termination_registry()
            else:
                registry = _reg_dispatch.get(registry_kind, termination_registry)()
            editor = _RegistryModuleEditor(
                registry,
                get_val(),
                write_val,
                selector_title=str(row.get("selector_title", "Title")),
                sort_mode=str(row.get("sort_mode", "title_asc")),
                family_provider=self._resolve_canvas_robot_family,
                algorithm_provider=self._resolve_canvas_algorithm,
            )
            if registry_kind in ("rewards", "rewards_auto"):
                self._reward_editor_widget = editor
            elif registry_kind == "terminations":
                self._termination_editor_widget = editor
            return editor

        # ── body_ir_validator ─────────────────────────────────────────
        if w_type == "body_ir_validator":
            return self._make_body_ir_validator(key, get_val, write_val, logic)

        # ── active_file_table ─────────────────────────────────────────
        # Flat 3-row file-type table for the Robot node. Shows MJCF /
        # USD / URDF with Loaded status and a ✔ marker on the currently-
        # active file. Click a Loaded row to override; click the active
        # row again to revert to 'auto'.
        if w_type == "active_file_table":
            return self._make_active_file_table(key, get_val, write_val, logic)

        # ── controller_status_panel ───────────────────────────────────
        # Read-only live panel for TrainingCommandsNode. Mirrors the
        # sidebar Controller panel (mode + invert_vy) and shows the
        # fixed gamepad binding convention + the effective velocity
        # ranges derived from the gain sliders on the same node.
        if w_type == "controller_status_panel":
            return self._make_controller_status_panel(key, get_val, write_val, logic)

        # ── gait_preset_table ─────────────────────────────────────────
        # Compact button for TrainingCommandsNode.gait_presets. Click
        # opens a QDialog with a table editor (Name / Freq / FL / FR
        # / RL / RR / H_body / H_step). Stores the table as a JSON
        # list in the param. Seeded from DEFAULT_PRESETS on empty.
        if w_type == "gait_preset_table":
            return self._make_gait_preset_table(key, get_val, write_val, logic)

        # ── output_file_list ──────────────────────────────────────────
        # Read-only table for the unified Export node. Shows every
        # file the next training run will produce under the bundle
        # directory (computed from bundle_name + version + include_*),
        # with an Open button per row and a ⚠ "Will Overwrite" tag
        # on files that already exist on disk.
        if w_type == "output_file_list":
            return self._make_output_file_list(key, get_val, write_val, logic)

        # ── start_point_compat_panel ──────────────────────────────────
        # Read-only diff panel on the Export node showing whether the
        # connected Start Point bundle is compatible with the current
        # canvas. Consumes TrainingContext.compat_report(graph) — the
        # same source-of-truth SafetyReview's S2–S5 checks use.
        if w_type == "start_point_compat_panel":
            return self._make_start_point_compat_panel(key, get_val, write_val, logic)

        # ── review_backend_picker ─────────────────────────────────────
        # Horizontal button row sourced from review_backends registry.
        # MuJoCo default, Isaac Sim enabled, Newton placeholder greyed
        # out. Keeps the list extensible without hardcoding engines.
        if w_type == "review_backend_picker":
            return self._make_review_backend_picker(key, get_val, write_val, logic)

        # ── review_scene_picker ───────────────────────────────────────
        # Scene dropdown filtered by the Export node's current
        # review_backend selection. Decoupled from training PlayGround
        # — reviewers frequently test a policy in a fresh scene.
        if w_type == "review_scene_picker":
            return self._make_review_scene_picker(key, get_val, write_val, write_params, logic)

        # ── review_launch_button ──────────────────────────────────────
        # SafetyReview-gated Launch button that emits a workspace-level
        # signal. Process lifetime is owned by the workspace, not the
        # node, so a canvas tear-down can't orphan the subprocess.
        if w_type == "review_launch_button":
            return self._make_review_launch_button(key, get_val, write_val, logic)

        # ── scene_dropdown ────────────────────────────────────────────
        # Live-filtered scene registry picker for PlayGroundSettingNode.
        # Choices are pulled from src.system.training.scene_registry and
        # filtered by the current canvas backend (detected from the
        # Train node) and robot family (from the Robot node's asset).
        # Selecting a scene writes scene_id, auto-syncs scene_type, and
        # applies the registry entry's ``defaults`` dict.
        if w_type == "scene_dropdown":
            return self._make_scene_dropdown(key, get_val, write_val, write_params, logic)

        # ── joint_init_editor ─────────────────────────────────────────
        # Visual per-joint init angle editor for JointInitNode. Button
        # label shows "(N angles set)"; click opens a dialog with one
        # spinbox per joint (joint_names read from the scene's Robot
        # node). Stores the result as a JSON dict on the ``angles`` param.
        if w_type == "joint_init_editor":
            return self._make_joint_init_editor(key, get_val, write_val, logic)

        # ── contact_body_picker ───────────────────────────────────────
        # Multi-select picker for ActorSetting.contact_body_names.
        # Choices are sourced from the Robot node's resolved asset
        # (base_link + joint-derived links + foot_link_names). Stores
        # selection as a JSON list in the param. Label updates to show
        # "Contact Bodies (N)" reflecting the current selection count.
        if w_type == "contact_body_picker":
            return self._make_contact_body_picker(key, get_val, write_val, logic)

        # ── action_joint_picker ──────────────────────────────────────
        # Multi-select picker for ActorSetting.action_joint_names_expr.
        # Choices are sourced from the Robot node's resolved asset
        # (joint_names). Stores selection as a JSON list in the param.
        if w_type == "action_joint_picker":
            return self._make_action_joint_picker(key, get_val, write_val, logic)

        # ── stage_editor ──────────────────────────────────────────────
        # Dynamic stage list for ILPPOTrainerNode staged training.
        # Stores/reads a JSON list of stage dicts in the node param.
        if w_type == "stage_editor":
            w = _StageEditorWidget(
                initial_value=get_val(),
                write_fn=write_val,
            )
            return w

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
                if action == "review_il_export" and hasattr(scene, "il_export_review_requested"):
                    scene.il_export_review_requested.emit(self.get_parameters())
                if action == "scene_preview" and hasattr(scene, "scene_preview_requested"):
                    scene.scene_preview_requested.emit(self.get_parameters())
                if action == "init_pose_preview" and hasattr(scene, "init_pose_preview_requested"):
                    scene.init_pose_preview_requested.emit(self.get_parameters())

            btn.clicked.connect(_on_action)
            return btn

        # Unknown widget type — fall back to NodeInputer
        w = NodeInputer(current_text=get_val(), mode="text")
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

        # ── SafetyReview overlay border ───────────────────────────────
        # Painted last so it sits on top of the node body AND the
        # selection highlight — users need this signal to be the loudest
        # thing on the canvas when a pre-flight check fails.
        if self._safety_status in ("error", "warning"):
            if self._safety_status == "error":
                safety_color = QColor("#EF4444")   # bright red
                glow_color = QColor(239, 68, 68, 90)
            else:
                safety_color = QColor("#F59E0B")   # amber
                glow_color = QColor(245, 158, 11, 80)
            # Soft halo — 2.5 px outer stroke first so the inner crisp
            # stroke reads clearly against any background.
            painter.setPen(QPen(glow_color, 3.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(r.adjusted(-1.5, -1.5, 1.5, 1.5), 7.5, 7.5)
            # Crisp inner stroke
            painter.setPen(QPen(safety_color, 2.0))
            painter.drawRoundedRect(r, 6.0, 6.0)

    # ------------------------------------------------------------------
    # SafetyReview integration
    # ------------------------------------------------------------------

    def set_safety_status(self, severity: Optional[str]) -> None:
        """Paint (or clear) a SafetyReview overlay border on this node.

        Called by TrainingWorkspaceWindow after each :class:`TrainingContext`
        ``safety_review`` run. ``severity`` must be one of
        ``None`` (clear) / ``"warning"`` / ``"error"``.
        """
        if severity not in (None, "warning", "error"):
            severity = None
        if self._safety_status == severity:
            return
        self._safety_status = severity
        try:
            self.update()
        except Exception:
            pass
