"""CheckpointWatcher — polls Isaac Lab log_dir for new .pt checkpoint files.

RSL-RL saves checkpoints as: log_dir/<run_name>/model_<iteration>.pt
This QThread polls every 5 seconds, detects new files, and emits
``checkpoint_found(iteration, reward, path)`` for each.
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Set, Union

from PySide6.QtCore import QThread, Signal

log = logging.getLogger(__name__)

_MODEL_RE = re.compile(r"model_(\d+)\.pt$")


class CheckpointWatcher(QThread):
    """Background thread that watches for new Isaac Lab checkpoint files.

    Signals
    -------
    checkpoint_found(int, float, str)
        Emitted for each newly detected checkpoint:
        (iteration, reward_estimate, absolute_path).
    """

    checkpoint_found = Signal(int, float, str)

    def __init__(self, log_dir: Union[str, Path], poll_interval: float = 5.0) -> None:
        super().__init__()
        self._log_dir = Path(log_dir)
        self._poll_interval = poll_interval
        self._seen: Set[str] = set()
        self._stop = False

    def stop(self) -> None:
        """Request the watcher to stop on the next poll cycle."""
        self._stop = True

    def run(self) -> None:
        """Thread entry — poll until stopped."""
        log.info("CheckpointWatcher started: %s", self._log_dir)
        while not self._stop:
            try:
                self._scan()
            except Exception:
                log.exception("CheckpointWatcher scan error")
            # Sleep in small increments so stop requests are responsive
            for _ in range(int(self._poll_interval * 10)):
                if self._stop:
                    break
                time.sleep(0.1)
        log.info("CheckpointWatcher stopped")

    def _scan(self) -> None:
        """Walk log_dir for model_*.pt files not yet seen."""
        if not self._log_dir.exists():
            return
        for pt_file in sorted(self._log_dir.rglob("model_*.pt")):
            key = str(pt_file)
            if key in self._seen:
                continue
            self._seen.add(key)

            m = _MODEL_RE.search(pt_file.name)
            if not m:
                continue
            iteration = int(m.group(1))
            reward = self._estimate_reward(pt_file, iteration)
            log.info("Checkpoint detected: iter=%d reward=%.2f path=%s",
                     iteration, reward, pt_file)
            self.checkpoint_found.emit(iteration, reward, str(pt_file))

    def _estimate_reward(self, pt_path: Path, iteration: int) -> float:
        """Try to read the reward from a nearby log file or return 0.0."""
        # RSL-RL writes a progress.csv or logs to stdout.
        # Try to find a progress.csv next to the checkpoint.
        for candidate in [pt_path.parent / "progress.csv",
                          pt_path.parent.parent / "progress.csv"]:
            if candidate.exists():
                return self._read_reward_from_csv(candidate, iteration)
        return 0.0

    @staticmethod
    def _read_reward_from_csv(csv_path: Path, iteration: int) -> float:
        """Read reward for *iteration* from RSL-RL progress.csv.

        The CSV typically has columns: iteration, reward_mean, ...
        We scan for the row closest to the target iteration.
        """
        try:
            best_reward = 0.0
            best_dist = float("inf")
            with open(csv_path, "r", encoding="utf-8") as f:
                header = f.readline()
                # Find reward column index
                cols = [c.strip().lower() for c in header.split(",")]
                iter_col = -1
                reward_col = -1
                for i, c in enumerate(cols):
                    if c in ("iteration", "iter", "step"):
                        iter_col = i
                    if c in ("mean_reward", "reward_mean", "mean reward"):
                        reward_col = i
                if iter_col < 0 or reward_col < 0:
                    return 0.0
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) <= max(iter_col, reward_col):
                        continue
                    try:
                        it = int(float(parts[iter_col].strip()))
                        rew = float(parts[reward_col].strip())
                        dist = abs(it - iteration)
                        if dist < best_dist:
                            best_dist = dist
                            best_reward = rew
                    except (ValueError, IndexError):
                        continue
            return best_reward
        except Exception:
            return 0.0
