# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Agent roles for AI Build (blueprint §5).

A role is a bounded responsibility + a scoped C2 projection + a scoped tool set
+ write authority. v1 instantiates three roles (Design / Orchestrator / Critic);
the param-expert and integrator roles are reserved (their work is folded into
the Orchestrator and the deterministic C3 integrator respectively). The
``RoleSpec`` carries the system prompt; vocabulary (enums + reward bounds) and
the §9 guardrails are injected from the projection so nothing is hand-copied.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import FrozenSet, Optional

from ..blocks import CanvasBlock
from ..contracts import AgentRole
from ..registry_projection import RegistryProjection

__all__ = [
    "RoleSpec",
    "V1_ACTIVE_ROLES",
    "design_role",
    "orchestrator_role",
    "critic_role",
    "report_role",
    "chat_role",
]

# The roster active in v1; routing collapses every other producer onto these.
V1_ACTIVE_ROLES: FrozenSet[AgentRole] = frozenset(
    {AgentRole.DESIGN, AgentRole.ORCHESTRATOR, AgentRole.CRITIC}
)


@dataclass(frozen=True)
class RoleSpec:
    """One agent role: prompt + projection scope + write authority."""

    role: AgentRole
    system_prompt: str
    can_write: bool


# Shared domain guardrails (training_config.md §9) injected into producer prompts.
_GUARDRAILS = (
    "Hard rules you must never break:\n"
    "1. Reward weights MUST stay within each function's documented bounds; an "
    "out-of-bounds penalty freezes the robot's joints. Anchor weights to the "
    "same-family baseline; track_ang_vel_z should be ~half of track_lin_vel_xy; "
    "if you scale tracking up N×, scale penalties similarly.\n"
    "2. base_lin_vel goes ONLY in critic_privileged_terms, NEVER in obs_terms "
    "(it is unobservable on real hardware).\n"
    "3. Every enabled command item needs a reward edge; a zero-velocity item "
    "(stand) needs at least one negative term.\n"
    "4. velocity_command obs scale must be per-channel (e.g. [2,2,0.25]), not a "
    "flat scalar.\n"
    "5. Do not add hard terminations (e.g. command_follow_timeout) that fire "
    "before the skill is learned; use continuous reward pressure instead.\n"
    "6. Joints/bodies are referenced by IR role, never physical names.\n"
    "7. Stay on the locked backend; never mix isaac_lab and sb3_mujoco features.\n"
    "8. A training item's `clip` is a motion reference (an absolute path or a "
    "`pack:` ref). To train on only a SUB-RANGE of a clip — a 'segment' the user "
    "cut in the Clip Motion Editor — set that item's `clip` to "
    "`<clip_ref>#seg=<start>-<end>` (INCLUSIVE frame indices). The segments that "
    "exist are enumerated in the `motion_segments` vocabulary block; reference "
    "one by its EXACT `ref` string. NEVER invent frame numbers or fabricate a "
    "`#seg=` range that isn't listed — if the user asks for a segment not in "
    "`motion_segments`, say it doesn't exist rather than guessing.\n"
    "9. Reward configuration is normally COMPOSITE, not one flat global bundle. "
    "Put shared terms on the global page; when there are multiple command items "
    "(e.g. walk / turn / stand) prefer differentiating emphasis per item — a "
    "separate reward bundle per item via assign_reward_bundle — over one "
    "uniform global bundle, and use a pd-group page when a reward should apply "
    "only to a joint partition. This is a strong default, not a hard rule: a "
    "single-reward config is still valid when the task truly warrants it.\n"
    "10. Design rewards DIALECTICALLY (辩证面): a reward for doing X invites the "
    "question of a penalty for NOT doing X. The bounded command-tracking rewards "
    "SATURATE (a still robot under a command feels no gradient), so a following "
    "task usually also wants the UNBOUNDED command penalty (track_fail_penalty) "
    "and base-motion penalties for motion the command never asked for.\n"
    "11. Learn command-tracking AND posture together (姿态): include posture / "
    "joint-regularization terms (flat_orientation, joint_deviation_l1, "
    "dof_pos_limits, base_height) so the policy holds a stable posture, not only "
    "a velocity.\n"
    "12. After configuring, REVIEW for degenerate optima (局部解审视): abstract "
    "each reward into a boundary constraint and ask what actually maximises the "
    "sum — can the policy score by standing (missing track_fail_penalty), "
    "lifting one leg (feet_air_time without a phasing partner like gait), or just "
    "surviving (alive/termination dominating command pressure)? Add the "
    "counter-term where the task warrants it.\n"
    "These three are REASONING principles, not a fixed checklist — use judgment "
    "for the task (a task may legitimately omit gait; a non-locomotion task "
    "needs different terms). Each reward's 'category'/'hint' and the "
    "'reward_doctrine' block in the vocabulary tell you what a term pairs with "
    "and which degenerate optimum it guards against."
)


def _vocab_block(projection: RegistryProjection) -> str:
    return json.dumps(projection.vocabulary_summary(), ensure_ascii=False, indent=0)


def design_role(projection: RegistryProjection) -> RoleSpec:
    prompt = (
        "You are the Design agent for a legged-robot RL training canvas. The "
        "backend and robot are already locked (see vocabulary). Read the user's "
        "natural-language request and produce a concise architecture intent: "
        "which command items (walk/turn/stand/...), the locomotion emphasis, and "
        "any special features (gait, motion clips, or a SKILL — a discrete "
        "triggered behavior such as jump, authored via skill_items + a gated "
        "package; IsaacLab-only, see the 'skill_system' vocab block) — "
        "capabilities, not numbers. "
        "Do not call tools; reply with a short plain-text brief the Orchestrator "
        "will execute.\n\nLocked vocabulary:\n" + _vocab_block(projection)
    )
    return RoleSpec(role=AgentRole.DESIGN, system_prompt=prompt, can_write=False)


def orchestrator_role(
    projection: RegistryProjection,
    *,
    baseline_available: bool = True,
    block: Optional[CanvasBlock] = None,
    locked: str = "",
) -> RoleSpec:
    degradation = (
        "A same-family reference canvas is provided as a worked example; mirror "
        "its wiring and adapt parameters to the request."
        if baseline_available
        else "NO same-family reference exists for this task; you are building "
        "from the registry projection and port semantics alone. State this "
        "degradation in your reasoning."
    )
    # Block-by-block scope (AI Build v2): one block per pass, fewer tokens.
    scope = ""
    if block is not None:
        # Backend-scoped node types (drops this block's other-backend nodes, e.g.
        # SB3-only env_assembler on an isaac_lab run).
        scope = (
            f"\n\nYou are working on ONE canvas block right now: "
            f"**{block.default_title}**. Only place or modify nodes of these "
            f"types: {projection.block_schema_ids(block.schema_ids)}. Other blocks "
            "are handled in separate passes — do NOT touch them, and do not "
            "re-create nodes that already exist."
        )
    # Locked, user-supplied decisions (applied deterministically before you run).
    lock = ""
    if locked.strip():
        lock = (
            "\n\nThese decisions are already LOCKED by the user and applied to the "
            "canvas; do NOT change them, only build around them:\n" + locked.strip()
        )
    prompt = (
        "You are the Orchestrator agent. You build/modify a training canvas by "
        "calling the provided tools (place_node, connect, set_param, "
        "assign_reward_bundle, ...). Every write is validated immediately; if a "
        "tool returns an error, read it and correct your next call — the error "
        "names the real node_ids to use. Reference a node ONLY by a node_id "
        "returned by place_node or shown as a key in the canvas snapshot; NEVER "
        "by its schema/type name (e.g. use 'actor_setting_1', not "
        "'actor_setting'). Do NOT worry about node coordinates or overlap — the "
        "system arranges the canvas layout deterministically after each block; "
        "you only place and wire nodes. " + degradation
        + scope + lock
        + "\n\nWork in small, verifiable steps. Use assign_reward_bundle to wire "
        "rewards to a command item. To build a SKILL (a discrete triggered "
        "behavior such as jump), follow the 'skill_system' block in the "
        "vocabulary: set skill_items + a gated package on the training_motion node "
        "(IsaacLab-only). When the block is complete, stop calling "
        "tools.\n\n" + _GUARDRAILS + "\n\nLocked vocabulary:\n"
        + _vocab_block(projection)
    )
    return RoleSpec(role=AgentRole.ORCHESTRATOR, system_prompt=prompt, can_write=True)


def chat_role() -> RoleSpec:
    """The gate-declined direct-answer role (no knowledge base, no tools).

    Used when the silent intent gate judged the message NOT to be a canvas
    build/modify request needing the project knowledge base: the model answers
    from its own knowledge. Deliberately takes NO projection — attaching the
    knowledge base here would defeat the gate. Reuses ``DESIGN`` (a read-only
    role) for attribution, like the reporter.
    """
    prompt = (
        "You are UnitPort Studio's AI assistant, chatting inside the AI Build "
        "panel of a robot-training canvas tool. The user's message was judged "
        "not to be a canvas build/modify request, so no project knowledge base "
        "is attached. Answer helpfully and concisely from your own general "
        "knowledge, in the language the user wrote in. Do not call tools and "
        "do not invent project-specific facts (node names, robot SKUs, file "
        "paths); if the question needs project specifics you do not have, say "
        "so briefly and suggest rephrasing it as a canvas-building request."
    )
    return RoleSpec(role=AgentRole.DESIGN, system_prompt=prompt, can_write=False)


def critic_role(projection: RegistryProjection) -> RoleSpec:
    # The critic's prompt proper lives in validation.critic; this RoleSpec exists
    # for roster completeness / future routing. Read-only.
    return RoleSpec(
        role=AgentRole.CRITIC,
        system_prompt="Read-only baseline-relative pathology reviewer.",
        can_write=False,
    )


def report_role(projection: RegistryProjection) -> RoleSpec:
    """The Phase-3 reporter — a short natural-language summary of the build.

    Read-only (no tools): given the applied mutations + final canvas, it writes
    2-4 plain sentences for the user. Reuses ``DESIGN`` (a read-only role) for
    attribution; failure to produce it is non-fatal (the build already succeeded).
    """
    prompt = (
        "You are the Report agent. The training canvas has just been built and "
        "validated by the orchestration team. In 2-4 short sentences of plain, "
        "friendly natural language, tell the user what was configured — the "
        "robot, the command items / task, the locomotion or reward emphasis, and "
        "any notable settings. Be concrete but concise. Do not call tools, do not "
        "list raw node ids, do not invent anything not in the provided summary."
    )
    return RoleSpec(role=AgentRole.DESIGN, system_prompt=prompt, can_write=False)
