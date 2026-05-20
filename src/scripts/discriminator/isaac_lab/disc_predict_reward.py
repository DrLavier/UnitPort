"""Predict AMP Reward — Disc score → style reward."""

from __future__ import annotations

from scripts.task_module import (
    discriminator_item,
)


INLINE_SOURCE = '''\
def predict_amp_reward(self, state, next_state, normalizer=None):
    """Compute the AMP style reward for a batch of policy transitions.

    Default body: ``r = coef * softplus(d.clamp(max=logit_clamp_max))``
    -- Peng 2021's BCE-paired formulation. Override to switch to
    LSGAN-style ``-(d-1)^2``, sigmoid-based, etc.

    Returns (style_reward, disc_score) -- first is the per-step reward,
    second is the raw logit for logging.
    """
    import torch
    with torch.no_grad():
        self.eval()
        if normalizer is not None:
            state = normalizer.normalize_torch(state, self.device)
            next_state = normalizer.normalize_torch(next_state, self.device)
        d = self.amp_linear(self.trunk(torch.cat([state, next_state], dim=-1)))
        style_reward = self.amp_reward_coef * torch.nn.functional.softplus(
            d.clamp(max=self.logit_clamp_max)
        )
        self.train()
    return style_reward.squeeze(-1), d
'''


ENTRY = discriminator_item(
    key='disc_predict_reward',
    title='Predict AMP Reward',
    desc='Disc score → style reward. Default: amp_reward_coef × softplus(d.clamp(max=logit_clamp_max)). Override to switch to LSGAN ``-(d-1)^2``, sigmoid-based, etc.',
    il_inline=INLINE_SOURCE,
)
