# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""Term-payload (reward / termination / observation) shape helpers.

Canvas nodes store enabled task-module terms as a dict keyed by the
preset / variant key. Each *value* has two legal shapes:

1. **Legacy scalar** — the raw weight (float) ::

        {"action_rate_penalty": -0.04}

2. **Structured dict** — adds optional ``variant`` and ``applies_to``
   metadata ::

        {
            "action_rate_penalty": {
                "weight":  -0.04,
                "variant": "aggressive",
                "applies_to": ["legged"]
            }
        }

   For ``termination_conditions`` the structured form additionally carries
   a per-condition ``grace_period_s`` (seconds) — a time-gated grace window
   during which the condition cannot fire (spawn/settle transient). The
   numeric ``weight`` is the condition's threshold for terminations. See
   :func:`parse_grace_period`.

Both forms are accepted on read; on write the structured form is only
emitted when at least one structured field deviates from default. This
keeps canvas JSON diffs minimal for users who don't yet use variants.

The same shape is used by ``reward_terms``, ``termination_conditions``,
``obs_terms``, and the discriminator slot map — hence one shared parser.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Tuple, Union

PayloadValue = Union[float, int, str, dict]
ParsedPayload = Tuple[float, Optional[str], List[str]]


def _coerce_weight(raw, fallback: float = 0.0) -> float:
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    if isinstance(raw, str):
        try:
            return float(raw.strip())
        except ValueError:
            return fallback
    return fallback


def _coerce_str_list(raw) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw] if raw else []
    if isinstance(raw, Iterable):
        return [str(x) for x in raw if x is not None]
    return []


def parse_term_payload(value: PayloadValue) -> ParsedPayload:
    """Decode either legacy or structured payload into a uniform triple.

    Returns ``(weight, variant, applies_to)``:

    * ``weight``     — coerced to ``float``; non-numeric → ``0.0``
    * ``variant``    — ``None`` means "use the preset" (legacy form
                       always yields ``None``); empty/``"preset"``
                       strings are normalised to ``None`` too so the
                       resolver gets a consistent contract.
    * ``applies_to`` — list of item-id / scope strings; missing → ``[]``
    """
    if isinstance(value, dict):
        weight = _coerce_weight(value.get("weight"), fallback=0.0)
        variant_raw = value.get("variant")
        if isinstance(variant_raw, str) and variant_raw not in ("", "preset"):
            variant: Optional[str] = variant_raw
        else:
            variant = None
        applies_to = _coerce_str_list(value.get("applies_to"))
        return weight, variant, applies_to
    return _coerce_weight(value, fallback=0.0), None, []


PAGE_GLOBAL = "__global__"
"""The always-present default reward page id (holds global / non-joint rewards).

缺口③ — the Rewards node's ``reward_terms`` is a **paged** dict
``{page_id: {reward_key: payload}}`` where ``page_id`` is either ``PAGE_GLOBAL``
(global rewards, no joint filter) or a pd_group id (a joint partition; only
joint-subset-capable rewards live here, emitted with that partition's joints).
"""


def is_paged_reward_terms(value) -> bool:
    """True iff ``value`` is the paged reward_terms shape (has the global page).

    New saves always include ``PAGE_GLOBAL``; legacy flat ``{reward: weight}``
    dicts never do — so this is an unambiguous discriminator (reward-registry
    keys and page-ids are disjoint namespaces).
    """
    return isinstance(value, dict) and PAGE_GLOBAL in value


def migrate_reward_terms_to_paged(value) -> dict:
    """Return ``reward_terms`` in paged shape, migrating legacy flat dicts.

    Already-paged → returned as-is. Legacy flat ``{reward: payload}`` → wrapped
    as ``{PAGE_GLOBAL: <flat>}`` (§8c on-disk-legacy compat: the caller WARNs).
    Empty / non-dict → ``{PAGE_GLOBAL: {}}``.
    """
    if is_paged_reward_terms(value):
        return dict(value)
    if isinstance(value, dict) and value:
        return {PAGE_GLOBAL: dict(value)}
    return {PAGE_GLOBAL: {}}


def iter_reward_pages(value):
    """Yield ``(page_id, terms_dict)`` over a (possibly legacy) reward_terms.

    Always yields the global page first. Migrates legacy flat dicts on the fly.
    """
    paged = migrate_reward_terms_to_paged(value)
    glob = paged.get(PAGE_GLOBAL)
    yield PAGE_GLOBAL, dict(glob) if isinstance(glob, dict) else {}
    for pid, terms in paged.items():
        if pid == PAGE_GLOBAL:
            continue
        yield pid, dict(terms) if isinstance(terms, dict) else {}


def parse_partitions(value: PayloadValue) -> dict:
    """Decode a reward's per-joint-subset ``partitions`` map from a payload.

    A partitionable reward (``TaskModuleItem.il_partition_source`` set, e.g.
    ``joint_deviation_l1``) may carry a ``partitions`` dict instead of a
    single scalar weight::

        {"joint_deviation_l1": {"partitions": {"hip_y": -0.1, "knee": -0.05}}}

    Keys are pd-group ids (the family's joint partitions); values are the
    per-partition reward weight. The compiler fans this one canvas term out
    into one ``RewTerm`` per partition, each restricted to that partition's
    physical joints (legged_gym's ``hip_dof_deviation`` / ``arm_dof_deviation``
    / ``waist_dof_deviation`` split). Absent / empty → ``{}`` (term runs on
    all joints, the legacy scalar-weight path). Non-numeric weights are
    dropped here (UI-side read); the compiler re-validates strictly at emit.
    """
    if not isinstance(value, dict):
        return {}
    raw = value.get("partitions")
    if not isinstance(raw, dict):
        return {}
    out: dict = {}
    for gid, w in raw.items():
        if not isinstance(gid, str) or not gid:
            continue
        if isinstance(w, (int, float)) and not isinstance(w, bool):
            out[gid] = float(w)
        elif isinstance(w, str):
            try:
                out[gid] = float(w.strip())
            except ValueError:
                continue
    return out


def parse_grace_period(value: PayloadValue) -> float:
    """Decode the per-condition ``grace_period_s`` (seconds) from a payload.

    Only the structured dict form carries it; the legacy scalar form (and a
    dict without the key) yields ``0.0`` — i.e. "no grace, current behavior".
    Non-numeric / negative values are clamped to ``0.0`` here (UI-side read);
    the compiler re-parses strictly (fail-loud, CLAUDE.md §8) at emit time.
    """
    if isinstance(value, dict):
        g = _coerce_weight(value.get("grace_period_s"), fallback=0.0)
        return g if g > 0.0 else 0.0
    return 0.0


def parse_item_value(value: PayloadValue) -> Optional[float]:
    """Decode a reward's *function-internal* custom param from a payload.

    The Rewards node "Value" column stores one tunable scalar a reward
    function consumes internally (e.g. ``base_height(target_height=…)``)
    under the structured form's ``"value"`` key. The legacy scalar form
    (and a dict without the key) yields ``None`` — meaning "unset, use the
    reward's own default" (for ``base_height`` the default 0.0 sentinel
    auto-resolves to the asset's nominal standing height). Returning
    ``None`` (not ``0.0``) lets callers distinguish "explicitly 0" from
    "never set"; the compiler defaults ``None`` → ``0.0`` at emit.

    ``std`` / ``threshold`` reward-shaping knobs are deliberately NOT this
    field — they get dedicated handling elsewhere.
    """
    if isinstance(value, dict) and "value" in value:
        raw = value.get("value")
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return float(raw)
        if isinstance(raw, str):
            try:
                return float(raw.strip())
            except ValueError:
                return None
    return None


def serialize_term_payload(
    weight: float,
    variant: Optional[str] = None,
    applies_to: Optional[List[str]] = None,
    grace_period_s: float = 0.0,
    value: Optional[float] = None,
    partitions: Optional[dict] = None,
) -> PayloadValue:
    """Encode the parts back into the smallest payload shape.

    Emits a bare scalar (legacy form) when ``variant`` is missing /
    ``"preset"`` AND ``applies_to`` is empty AND ``grace_period_s`` is 0
    AND ``value`` is None AND ``partitions`` is empty. Otherwise emits the
    structured dict. This minimises canvas-JSON diff churn for users who
    don't yet use variants / grace / per-item values / partitions.

    ``value`` is the reward's function-internal param (see
    :func:`parse_item_value`); pass ``None`` to omit it (the caller decides
    "unset" by comparing against the reward's registry default).

    ``partitions`` is the per-joint-subset weight map (see
    :func:`parse_partitions`). When present (non-empty) the structured form
    carries it and the top-level ``weight`` becomes a placeholder (the
    compiler fans the term out per partition and never reads it) — but it is
    still written for shape uniformity.
    """
    has_variant = bool(variant) and variant != "preset"
    has_scope = bool(applies_to)
    has_grace = float(grace_period_s) > 0.0
    has_value = value is not None
    has_parts = bool(partitions)
    if (not has_variant and not has_scope and not has_grace
            and not has_value and not has_parts):
        return float(weight)
    out: dict = {"weight": float(weight)}
    if has_variant:
        out["variant"] = variant
    if has_scope:
        out["applies_to"] = list(applies_to or [])
    if has_grace:
        out["grace_period_s"] = float(grace_period_s)
    if has_value:
        out["value"] = float(value)
    if has_parts:
        out["partitions"] = {str(k): float(v) for k, v in partitions.items()}
    return out


__all__ = [
    "PayloadValue",
    "ParsedPayload",
    "parse_term_payload",
    "parse_grace_period",
    "parse_item_value",
    "parse_partitions",
    "serialize_term_payload",
    "PAGE_GLOBAL",
    "is_paged_reward_terms",
    "migrate_reward_terms_to_paged",
    "iter_reward_pages",
]
