"""Gradient Penalty — Zero-centered R1 gradient penalty on expert transitions."""

from __future__ import annotations

from scripts.task_module import (
    discriminator_item,
)


INLINE_SOURCE = '''\
def compute_grad_pen(self, expert_state, expert_next_state, lambda_=10.0):
    """Zero-centered R1 gradient penalty on expert transitions.

    Default body: matches Peng 2021 / amp-rsl-rl. Override to swap in
    standard WGAN-GP (target = 1.0) or to change the norm.
    """
    import torch
    from torch import autograd
    expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
    expert_data.requires_grad = True
    disc = self.amp_linear(self.trunk(expert_data))
    ones = torch.ones(disc.size(), device=disc.device)
    grad = autograd.grad(
        outputs=disc, inputs=expert_data, grad_outputs=ones,
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    return float(lambda_) * (grad.norm(2, dim=1) - 0.0).pow(2).mean()
'''


ENTRY = discriminator_item(
    key='disc_compute_grad_pen',
    title='Gradient Penalty',
    desc='Zero-centered R1 gradient penalty on expert transitions. Default targets |grad|=0 (Peng 2021); override for standard WGAN-GP target=1 or different norm.',
    il_inline=INLINE_SOURCE,
)
