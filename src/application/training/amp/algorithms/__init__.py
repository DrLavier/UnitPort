"""amp.algorithms — discriminator + AMP-PPO orchestrator.

Lazy-attr contract: only :class:`AMPDiscriminator` is imported eagerly,
because it is pure-torch and main-venv-safe (no UI dependencies). The
``amp_ppo`` symbols (``AmpPpoTrainer``, ``AmpStyleRewardWrapper``,
``AmpRolloutCollector``, ``train_amp_ppo``) live inside ``amp_ppo.py``
which transitively imports ``unitport_sdk`` (and therefore ``PyQt6``).
That import is **only** valid in the main UnitPort process; Isaac Lab's
training subprocess runs inside Isaac Sim's own Python venv where PyQt6
isn't installed, so eagerly importing them here would break the moment
the Isaac Sim launcher's compiled env_cfg touches ``application.training.amp``
(which it does through ``mdp_events``).

PEP 562 ``__getattr__`` defers the ``amp_ppo`` import until something
actually asks for one of those names — the SB3 chain runs in the main
venv where PyQt6 is present, so the trigger is safe. The subprocess
path never touches any of the four AMP-PPO names, so the lazy import
never fires there.
"""
from __future__ import annotations

from application.training.amp.algorithms.discriminator import AMPDiscriminator

_AMP_PPO_NAMES = frozenset({
    "AmpPpoTrainer",
    "AmpStyleRewardWrapper",
    "AmpRolloutCollector",
    "train_amp_ppo",
})


def __getattr__(name: str):
    if name in _AMP_PPO_NAMES:
        from application.training.amp.algorithms import amp_ppo as _ap
        return getattr(_ap, name)
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


__all__ = [
    "AMPDiscriminator",
    "AmpPpoTrainer",
    "AmpStyleRewardWrapper",
    "AmpRolloutCollector",
    "train_amp_ppo",
]
