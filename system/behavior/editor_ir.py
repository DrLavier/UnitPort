"""Behavior editor IR mapping.

Pure-Python backend model that mirrors the frontend editor semantics:

- Action packages on the main Action axis
- Body-part / limb lanes beneath the action axis
- Run modules placed on limb lanes only when behavior exists

This module intentionally sits between the raw ``BehaviorTimeline`` storage
shape and the frontend widgets so the product does not have to interpret raw
``motor_overlays`` directly everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from system.behavior.motor_topology import resolve_topology
from system.behavior.topology_fallback import heuristic_limb_for_track


@dataclass
class ActionPackageIR:
    action_id: str
    name: str
    kind: str
    start_time: float
    duration: float
    intent_id: str = ""
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "name": self.name,
            "kind": self.kind,
            "start_time": self.start_time,
            "duration": self.duration,
            "intent_id": self.intent_id,
            "params": dict(self.params),
        }


@dataclass
class RunSegmentIR:
    motor_id: str
    track_name: str
    start_time: float
    duration: float
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "motor_id": self.motor_id,
            "track_name": self.track_name,
            "start_time": self.start_time,
            "duration": self.duration,
            "params": dict(self.params),
        }


@dataclass
class RunModuleIR:
    module_id: str
    action_id: str
    action_name: str
    limb_id: str
    limb_label: str
    start_time: float
    duration: float
    track_names: List[str] = field(default_factory=list)
    motor_ids: List[str] = field(default_factory=list)
    segments: List[RunSegmentIR] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module_id": self.module_id,
            "action_id": self.action_id,
            "action_name": self.action_name,
            "limb_id": self.limb_id,
            "limb_label": self.limb_label,
            "start_time": self.start_time,
            "duration": self.duration,
            "track_names": list(self.track_names),
            "motor_ids": list(self.motor_ids),
            "segments": [segment.to_dict() for segment in self.segments],
        }


@dataclass
class LimbLaneIR:
    limb_id: str
    label: str
    modules: List[RunModuleIR] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "limb_id": self.limb_id,
            "label": self.label,
            "modules": [module.to_dict() for module in self.modules],
        }


@dataclass
class BehaviorEditorIR:
    actions: List[ActionPackageIR] = field(default_factory=list)
    limbs: List[LimbLaneIR] = field(default_factory=list)
    is_degraded: bool = False
    degraded_reason: str = ""

    def modules_for_action(self, action_id: str) -> List[RunModuleIR]:
        result: List[RunModuleIR] = []
        for limb in self.limbs:
            for module in limb.modules:
                if module.action_id == action_id:
                    result.append(module)
        result.sort(key=lambda item: (item.start_time, item.duration, item.limb_label))
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "actions": [action.to_dict() for action in self.actions],
            "limbs": [limb.to_dict() for limb in self.limbs],
            "is_degraded": self.is_degraded,
            "degraded_reason": self.degraded_reason,
        }


def _module_key(action_id: str, limb_id: str, start_time: float, duration: float) -> Tuple[str, str, int, int]:
    return (
        action_id,
        limb_id,
        int(round(start_time * 1000.0)),
        int(round(duration * 1000.0)),
    )


def build_behavior_editor_ir(
    timeline: Any,
    brand: str,
    robot_type: str,
) -> BehaviorEditorIR:
    """Map a BehaviorTimeline into frontend-facing Action + limb run modules."""
    if timeline is None or getattr(timeline, "is_empty", lambda: True)():
        return BehaviorEditorIR()

    topology = resolve_topology(brand, robot_type)
    action_segments = list(getattr(timeline, "action_segments", []) or [])
    actions = [
        ActionPackageIR(
            action_id=str(getattr(seg, "action_id", "") or ""),
            name=str(getattr(seg, "name", "") or ""),
            kind=str(getattr(seg, "kind", "movement") or "movement"),
            start_time=float(getattr(seg, "start_time", 0.0) or 0.0),
            duration=float(getattr(seg, "duration", 1.0) or 1.0),
            intent_id=str(getattr(seg, "intent_id", "") or ""),
            params=dict(getattr(seg, "params", {}) or {}),
        )
        for seg in action_segments
    ]
    action_index = {action.action_id: idx for idx, action in enumerate(actions)}
    action_name = {action.action_id: action.name for action in actions}

    lane_map: Dict[str, LimbLaneIR] = {}
    lane_order: List[str] = []
    if not topology.is_degraded:
        for limb in topology.all_limbs:
            lane_map[limb.limb_id] = LimbLaneIR(limb_id=limb.limb_id, label=limb.label)
            lane_order.append(limb.limb_id)

    grouped_modules: Dict[Tuple[str, str, int, int], RunModuleIR] = {}
    for overlay in list(getattr(timeline, "motor_overlays", []) or []):
        action_id = str(getattr(overlay, "action_id", "") or "")
        for seg in list(getattr(overlay, "motor_segments", []) or []):
            track_name = str(getattr(seg, "track_name", "") or "")
            if not track_name:
                continue

            if not topology.is_degraded:
                limb = topology.limb_for_track(track_name)
                if limb is not None:
                    limb_id = limb.limb_id
                    limb_label = limb.label
                else:
                    limb_id = f"stale:{track_name}"
                    limb_label = f"Unknown ({track_name})"
            else:
                limb_id, limb_label = heuristic_limb_for_track(track_name)

            if limb_id not in lane_map:
                lane_map[limb_id] = LimbLaneIR(limb_id=limb_id, label=limb_label)
                lane_order.append(limb_id)

            start_time = float(getattr(seg, "start_time", 0.0) or 0.0)
            duration = float(getattr(seg, "duration", 1.0) or 1.0)
            key = _module_key(action_id, limb_id, start_time, duration)
            module = grouped_modules.get(key)
            if module is None:
                module = RunModuleIR(
                    module_id=f"{action_id}:{limb_id}:{int(round(start_time * 1000.0))}:{int(round(duration * 1000.0))}",
                    action_id=action_id,
                    action_name=action_name.get(action_id, ""),
                    limb_id=limb_id,
                    limb_label=limb_label,
                    start_time=start_time,
                    duration=duration,
                )
                grouped_modules[key] = module
                lane_map[limb_id].modules.append(module)

            motor_id = str(getattr(seg, "motor_id", "") or "")
            if track_name not in module.track_names:
                module.track_names.append(track_name)
            if motor_id and motor_id not in module.motor_ids:
                module.motor_ids.append(motor_id)
            module.segments.append(
                RunSegmentIR(
                    motor_id=motor_id,
                    track_name=track_name,
                    start_time=start_time,
                    duration=duration,
                    params=dict(getattr(seg, "params", {}) or {}),
                )
            )

    limbs: List[LimbLaneIR] = []
    for limb_id in lane_order:
        lane = lane_map[limb_id]
        lane.modules.sort(
            key=lambda module: (
                module.start_time,
                module.duration,
                action_index.get(module.action_id, 10**6),
            )
        )
        limbs.append(lane)

    return BehaviorEditorIR(
        actions=actions,
        limbs=limbs,
        is_degraded=bool(topology.is_degraded),
        degraded_reason=str(topology.degraded_reason or ""),
    )
