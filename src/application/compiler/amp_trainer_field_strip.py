# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""RC-4 migration: strip deprecated AMP hyperparams off the trainer node.

The 9 AMP core hyperparams (``amp_reward_coef`` … ``lerp_schedule_json``)
have a single home — the ``discriminator`` node. The ``il_ppo_trainer`` /
``amp_trainer`` nodes historically carried duplicate copies (now
``deprecated = true`` in their manifests, UI-hidden). Existing canvases
still serialize the stale values, which:

  * confuse anyone reading the canvas JSON, and
  * make ``spec_compiler`` emit a PARAM_AUTHORITY_CONFLICT warning when
    they drift from the discriminator node.

This migrator removes those dead fields from the trainer node's ``params``
so the discriminator node is the unambiguous single source. Idempotent
via ``metadata.amp_trainer_fields_stripped_v1``. Shared between the canvas
load-time hook (``ui/canvas/page.py``) and the bootstrap sweep
(``bootstrap/migrate_canvas_strip_amp_trainer_fields.py``).

Pure dict-in/dict-out — no torch / no registers / no PyQt. Mutates the
canvas in place and returns ``True`` iff anything changed.
"""
from __future__ import annotations

from typing import Any, Dict

#: Idempotency stamp written into ``canvas["metadata"]``.
AMP_TRAINER_STRIP_FLAG = "amp_trainer_fields_stripped_v1"

#: The deprecated AMP hyperparam keys that live authoritatively on the
#: discriminator node. Keep in sync with DiscriminatorConfig / the
#: discriminator node manifest.
_DEPRECATED_TRAINER_AMP_KEYS = (
    "amp_reward_coef",
    "task_reward_lerp",
    "disc_grad_penalty",
    "disc_label_smoothing",
    "amp_replay_buffer_size",
    "num_preload_transitions",
    "disc_lr",
    "lerp_schedule",
    "lerp_schedule_json",
)

_TRAINER_SCHEMAS = ("il_ppo_trainer", "amp_trainer")


def strip_deprecated_amp_trainer_fields(canvas: Dict[str, Any]) -> bool:
    """Remove deprecated AMP hyperparams from trainer-node ``params``.

    Returns ``True`` iff at least one field was removed (or the stamp was
    newly written). No-op (returns ``False``) on already-migrated canvases
    and on canvases with no trainer node carrying the dead fields.
    """
    if not isinstance(canvas, dict):
        return False
    meta = canvas.get("metadata")
    if isinstance(meta, dict) and meta.get(AMP_TRAINER_STRIP_FLAG) is True:
        return False

    removed_any = False
    for node in (canvas.get("nodes") or []):
        if not isinstance(node, dict):
            continue
        if node.get("schema_id") not in _TRAINER_SCHEMAS:
            continue
        params = node.get("params")
        if not isinstance(params, dict):
            continue
        for key in _DEPRECATED_TRAINER_AMP_KEYS:
            if key in params:
                del params[key]
                removed_any = True

    if removed_any:
        if not isinstance(meta, dict):
            meta = {}
            canvas["metadata"] = meta
        meta[AMP_TRAINER_STRIP_FLAG] = True
    return removed_any


__all__ = [
    "strip_deprecated_amp_trainer_fields",
    "AMP_TRAINER_STRIP_FLAG",
]
