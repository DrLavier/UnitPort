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


def serialize_term_payload(
    weight: float,
    variant: Optional[str] = None,
    applies_to: Optional[List[str]] = None,
) -> PayloadValue:
    """Encode the triple back into the smallest payload shape.

    Emits a bare scalar (legacy form) when ``variant`` is missing /
    ``"preset"`` AND ``applies_to`` is empty. Otherwise emits the
    structured dict. This minimises canvas-JSON diff churn for users
    not yet using variants.
    """
    has_variant = bool(variant) and variant != "preset"
    has_scope = bool(applies_to)
    if not has_variant and not has_scope:
        return float(weight)
    out: dict = {"weight": float(weight)}
    if has_variant:
        out["variant"] = variant
    if has_scope:
        out["applies_to"] = list(applies_to or [])
    return out


__all__ = [
    "PayloadValue",
    "ParsedPayload",
    "parse_term_payload",
    "serialize_term_payload",
]
