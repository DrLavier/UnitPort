"""Auto command schedule for the Isaac Review Session.

Per ``knowledge_base/IsaacSim_design.yaml`` §6 (auto_command_schedule):

When the user reviews a trained policy in AUTO mode, the Isaac Review
Session needs to drive the env's velocity command through a sequence
that systematically covers every fundamental locomotion behaviour
without leaving demonstration gaps.

Pure random sampling within the velocity range would work but has two
problems:

  1. Short review windows might miss the corners of the cube
     (e.g. forward-fast, backward, in-place spin) by chance.
  2. Random output is not reproducible — hard to compare two
     checkpoints when each gets a different command sequence.

The fix is a **curated 16-step deterministic schedule** derived from
the canvas's ``il_velocity_cmd`` ranges.  Each stage holds for
``hold_s`` seconds (default 4.0) and the cycle repeats indefinitely.
Total cycle time at default = 64 s.

This module is **pure Python** — it does NOT depend on torch, gym,
isaaclab, or anything from the Kit venv.  That keeps it importable
from both:

  * ``il_review_session.py`` running inside the Isaac Sim subprocess
  * ``isaac_review_session.py`` running in UnitPort's main venv (for
    UI preview / unit tests)

The dataclasses are equality-comparable so unit tests can lock the
schedule shape against regression.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommandStage:
    """One stage in the auto demo schedule.

    A stage is a (vx, vy, wz) command paired with a human-readable
    label and a hold time in seconds.  The Review Session sets the
    env's command tensor to (vx, vy, wz) when entering this stage and
    holds it for ``hold_s`` seconds before advancing.
    """

    name: str
    vx: float
    vy: float
    wz: float
    hold_s: float

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.vx, self.vy, self.wz)


@dataclass
class AutoCommandSchedule:
    """A complete cyclic demo schedule.

    The schedule advances every ``hold_s`` seconds (per stage) and
    loops back to stage 0 when it reaches the end.  Total cycle time
    is the sum of all per-stage hold times.

    Example
    -------
    >>> sched = build_auto_command_schedule(
    ...     vx_range=(-0.5, 3.0), vy_range=(-0.5, 0.5), wz_range=(-1.5, 1.5),
    ... )
    >>> len(sched.stages)
    16
    >>> sched.stages[0].name
    'Stand still'
    >>> round(sched.cycle_time_s, 1)
    64.0
    """

    stages: List[CommandStage]

    @property
    def cycle_time_s(self) -> float:
        return sum(s.hold_s for s in self.stages)

    def stage_at_time(self, t: float) -> Tuple[int, CommandStage]:
        """Return ``(stage_index, stage)`` for the given elapsed time.

        Time wraps modulo ``cycle_time_s`` so the schedule repeats
        forever.  ``t < 0`` is treated as 0.
        """
        if not self.stages:
            raise ValueError("AutoCommandSchedule has no stages")
        if t < 0:
            t = 0.0
        cycle = self.cycle_time_s
        if cycle <= 0:
            return 0, self.stages[0]
        local = t % cycle
        elapsed = 0.0
        for i, stage in enumerate(self.stages):
            elapsed += stage.hold_s
            if local < elapsed:
                return i, stage
        # Floating-point edge case at exactly cycle boundary
        return len(self.stages) - 1, self.stages[-1]


# ---------------------------------------------------------------------------
# Schedule builder
# ---------------------------------------------------------------------------


def build_auto_command_schedule(
    vx_range: Tuple[float, float],
    vy_range: Tuple[float, float],
    wz_range: Tuple[float, float],
    *,
    hold_s: float = 4.0,
) -> AutoCommandSchedule:
    """Build the curated 16-step demo schedule from a velocity command cube.

    Implements ``IsaacSim_design.yaml §6.algorithm`` exactly.  Stage
    order is fixed: stand → walk slow/medium/fast → walk+turn → in-place
    spin → strafe → walk+strafe → diagonal → backward → slow+spin →
    stand.  This order is intentional — it shows the most "interesting"
    motions first so a user only watching the first 30 s still sees a
    representative sample of the policy's capabilities.

    Parameters
    ----------
    vx_range, vy_range, wz_range
        Tuples of ``(low, high)`` from the canvas's ``il_velocity_cmd``
        node.  Negative ``vx_range[0]`` enables the "Backward" stage;
        otherwise that stage is replaced with a stand-still placeholder.
    hold_s
        Per-stage hold time in seconds.  Default 4.0 → total cycle 64 s.

    Returns
    -------
    AutoCommandSchedule
        16-stage deterministic demo schedule.
    """
    if hold_s <= 0:
        raise ValueError(f"hold_s must be > 0, got {hold_s}")

    vx_lo, vx_hi = float(vx_range[0]), float(vx_range[1])
    vy_lo, vy_hi = float(vy_range[0]), float(vy_range[1])
    wz_lo, wz_hi = float(wz_range[0]), float(wz_range[1])

    if vx_hi < vx_lo or vy_hi < vy_lo or wz_hi < wz_lo:
        raise ValueError(
            f"Velocity ranges must be (low, high) with low <= high. "
            f"Got vx={vx_range}, vy={vy_range}, wz={wz_range}"
        )

    vx_mid = 0.5 * (vx_lo + vx_hi)
    vx_slow = vx_lo + 0.3 * (vx_hi - vx_lo)
    vx_back = vx_lo if vx_lo < 0 else 0.0

    schedule_spec: List[Tuple[str, float, float, float]] = [
        ("Stand still",            0.0,     0.0,         0.0       ),
        ("Walk slow",              vx_slow, 0.0,         0.0       ),
        ("Walk medium",            vx_mid,  0.0,         0.0       ),
        ("Walk fast",              vx_hi,   0.0,         0.0       ),
        ("Walk + turn left",       vx_mid,  0.0,         wz_hi*0.5 ),
        ("Walk + turn right",      vx_mid,  0.0,         wz_lo*0.5 ),
        ("In-place spin left",     0.0,     0.0,         wz_hi     ),
        ("In-place spin right",    0.0,     0.0,         wz_lo     ),
        ("Strafe left",            0.0,     vy_hi,       0.0       ),
        ("Strafe right",           0.0,     vy_lo,       0.0       ),
        ("Walk + strafe left",     vx_mid,  vy_hi*0.5,   0.0       ),
        ("Walk + strafe right",    vx_mid,  vy_lo*0.5,   0.0       ),
        ("Diagonal motion",        vx_mid,  vy_hi*0.5,   wz_hi*0.5 ),
        ("Backward",               vx_back, 0.0,         0.0       ),
        ("Slow + spin",            vx_slow, 0.0,         wz_hi     ),
        ("Stand still",            0.0,     0.0,         0.0       ),
    ]

    stages = [
        CommandStage(name=name, vx=vx, vy=vy, wz=wz, hold_s=hold_s)
        for name, vx, vy, wz in schedule_spec
    ]
    return AutoCommandSchedule(stages=stages)


# ---------------------------------------------------------------------------
# Convenience helpers used by il_review_session.py
# ---------------------------------------------------------------------------


def format_stage_for_hud(stage: CommandStage, index: int, total: int) -> str:
    """Render a stage as a one-line HUD overlay string.

    Format::

        AUTO  Stage 5/16  'Walk + turn left'  vx=1.50 vy=0.00 wz=0.75
    """
    return (
        f"AUTO  Stage {index + 1}/{total}  '{stage.name}'  "
        f"vx={stage.vx:+.2f} vy={stage.vy:+.2f} wz={stage.wz:+.2f}"
    )


def parse_range_param(raw: str, fallback: Tuple[float, float]) -> Tuple[float, float]:
    """Parse a canvas range string like ``"[-0.5, 3.0]"`` or ``"(-1, 1)"``.

    Used by ``il_review_session.py`` to read the canvas's
    ``il_velocity_cmd`` ranges from the compiled config dict at startup.
    The compiler emits these as Python tuple literals; some legacy
    canvases use JSON list syntax.  Accept both via ``ast.literal_eval``.
    """
    s = str(raw or "").strip()
    if not s:
        return fallback
    try:
        import ast
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
            return float(parsed[0]), float(parsed[1])
    except Exception:
        pass
    return fallback


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick verification: print a schedule for the typical Go2 ranges.
    sched = build_auto_command_schedule(
        vx_range=(-0.5, 3.0),
        vy_range=(-0.5, 0.5),
        wz_range=(-1.5, 1.5),
    )
    print(f"Generated {len(sched.stages)}-stage schedule, "
          f"cycle = {sched.cycle_time_s:.1f} s")
    for i, stage in enumerate(sched.stages):
        print(format_stage_for_hud(stage, i, len(sched.stages)))

    # Time-lookup smoke test
    for t in (0.0, 4.5, 33.0, 64.0, 65.0):
        idx, stage = sched.stage_at_time(t)
        print(f"  t={t:5.1f}s → stage {idx}: {stage.name}")
