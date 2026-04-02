#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ephemeral in-memory cache for Training Ground runtime metrics."""

from __future__ import annotations

import threading
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TrainingRunSample:
    step: int
    total: int
    reward_mean: float
    best_reward: float
    ep_len_mean: float
    status: str
    timestamp: float
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingRunEvent:
    event_type: str
    timestamp: float
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingRunTable:
    run_id: str
    policy_id: str
    policy_id_out: str
    algorithm: str
    total_timesteps: int
    created_at: float
    status: str = "created"
    finished_at: float = 0.0
    samples: List[TrainingRunSample] = field(default_factory=list)
    events: List[TrainingRunEvent] = field(default_factory=list)
    _milestone_cursor: int = -1


class TrainingRunCache:
    """Process-local storage for per-run metric series and milestone events."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._tables: Dict[str, TrainingRunTable] = {}

    def create_run(
        self,
        run_id: str,
        *,
        policy_id: str = "",
        policy_id_out: str = "",
        algorithm: str = "",
        total_timesteps: int = 0,
        status: str = "created",
    ) -> Dict[str, Any]:
        with self._lock:
            table = TrainingRunTable(
                run_id=str(run_id or "").strip(),
                policy_id=str(policy_id or "").strip(),
                policy_id_out=str(policy_id_out or "").strip(),
                algorithm=str(algorithm or "").strip(),
                total_timesteps=max(0, int(total_timesteps or 0)),
                created_at=time.time(),
                status=str(status or "created").strip() or "created",
            )
            self._tables[table.run_id] = table
            return self._serialize_table(table)

    def ensure_run(self, run_id: str, **kwargs) -> Dict[str, Any]:
        with self._lock:
            if str(run_id or "").strip() not in self._tables:
                return self.create_run(run_id, **kwargs)
            table = self._tables[str(run_id).strip()]
            self._merge_table_fields(table, kwargs)
            return self._serialize_table(table)

    def record_progress(
        self,
        run_id: str,
        *,
        step: int,
        total: int,
        reward_mean: float,
        best_reward: float,
        ep_len_mean: float = 0.0,
        status: str = "",
        timestamp: Optional[float] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            table = self._require_table(run_id)
            sample = TrainingRunSample(
                step=max(0, int(step or 0)),
                total=max(0, int(total or 0)),
                reward_mean=float(reward_mean or 0.0),
                best_reward=float(best_reward or 0.0),
                ep_len_mean=float(ep_len_mean or 0.0),
                status=str(status or "").strip(),
                timestamp=float(timestamp or time.time()),
                metrics=dict(metrics) if metrics else {},
            )
            table.samples.append(sample)
            if sample.total > 0:
                table.total_timesteps = sample.total
            if sample.status:
                table.status = sample.status
            return asdict(sample)

    def record_event(
        self,
        run_id: str,
        event_type: str,
        **payload,
    ) -> Dict[str, Any]:
        with self._lock:
            table = self._require_table(run_id)
            event = TrainingRunEvent(
                event_type=str(event_type or "event").strip() or "event",
                timestamp=time.time(),
                payload=dict(payload or {}),
            )
            table.events.append(event)
            if event.payload.get("status"):
                table.status = str(event.payload.get("status") or table.status)
            if event.payload.get("finished"):
                table.finished_at = event.timestamp
            return asdict(event)

    def update_status(self, run_id: str, status: str, *, finished: bool = False) -> Dict[str, Any]:
        with self._lock:
            table = self._require_table(run_id)
            table.status = str(status or table.status).strip() or table.status
            if finished:
                table.finished_at = time.time()
            return self._serialize_table(table)

    def summarize_progress_delta(self, run_id: str) -> Dict[str, Any]:
        with self._lock:
            table = self._require_table(run_id)
            if not table.samples:
                return {
                    "from_step": 0,
                    "to_step": 0,
                    "step_delta": 0,
                    "reward_mean_delta": 0.0,
                    "best_reward_delta": 0.0,
                    "sample_count": 0,
                    "latest_status": "",
                    "latest_reward_mean": 0.0,
                    "latest_best_reward": 0.0,
                }

            start_idx = min(max(table._milestone_cursor, 0), len(table.samples) - 1)
            end_idx = len(table.samples) - 1
            start_sample = table.samples[start_idx]
            end_sample = table.samples[end_idx]
            if table._milestone_cursor < 0:
                from_step = 0
                reward_base = 0.0
                best_base = 0.0
            else:
                from_step = start_sample.step
                reward_base = start_sample.reward_mean
                best_base = start_sample.best_reward

            summary = {
                "from_step": from_step,
                "to_step": end_sample.step,
                "step_delta": max(0, end_sample.step - from_step),
                "reward_mean_delta": end_sample.reward_mean - reward_base,
                "best_reward_delta": end_sample.best_reward - best_base,
                "sample_count": max(0, end_idx - table._milestone_cursor + 1),
                "latest_status": end_sample.status,
                "latest_reward_mean": end_sample.reward_mean,
                "latest_best_reward": end_sample.best_reward,
            }
            table._milestone_cursor = end_idx
            return summary

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            table = self._tables.get(str(run_id or "").strip())
            if table is None:
                return None
            return self._serialize_table(table)

    def list_runs(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [self._serialize_table(table) for table in self._tables.values()]

    def clear_run(self, run_id: str) -> None:
        with self._lock:
            self._tables.pop(str(run_id or "").strip(), None)

    def clear_all(self) -> None:
        with self._lock:
            self._tables.clear()

    def _require_table(self, run_id: str) -> TrainingRunTable:
        key = str(run_id or "").strip()
        if not key:
            raise KeyError("run_id is required")
        table = self._tables.get(key)
        if table is None:
            raise KeyError(f"Training run cache not found: {key}")
        return table

    @staticmethod
    def _merge_table_fields(table: TrainingRunTable, fields: Dict[str, Any]) -> None:
        if fields.get("policy_id"):
            table.policy_id = str(fields["policy_id"]).strip()
        if fields.get("policy_id_out"):
            table.policy_id_out = str(fields["policy_id_out"]).strip()
        if fields.get("algorithm"):
            table.algorithm = str(fields["algorithm"]).strip()
        if fields.get("total_timesteps") is not None:
            table.total_timesteps = max(0, int(fields["total_timesteps"] or 0))
        if fields.get("status"):
            table.status = str(fields["status"]).strip()

    @staticmethod
    def _serialize_table(table: TrainingRunTable) -> Dict[str, Any]:
        data = deepcopy(asdict(table))
        data.pop("_milestone_cursor", None)
        return data


_RUN_CACHE = TrainingRunCache()


def get_training_run_cache() -> TrainingRunCache:
    """Return the process-wide training run cache singleton."""
    return _RUN_CACHE
