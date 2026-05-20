"""Forward — MLP forward pass — trunk + linear head."""

from __future__ import annotations

from scripts.task_module import (
    discriminator_item,
)


INLINE_SOURCE = '''\
def forward(self, x):
    """Score a batch of concatenated (s_t, s_{t+1}) pairs.

    Default body: trunk MLP -> linear head. Override to add residual
    connections, attention, etc. Inputs:
      - self : the AMPDiscriminator instance (has self.trunk, self.amp_linear)
      - x    : torch.Tensor of shape (batch, 2 * amp_obs_dim)
    Returns: torch.Tensor of shape (batch, 1) -- raw logits.
    """
    h = self.trunk(x)
    return self.amp_linear(h)
'''


ENTRY = discriminator_item(
    key='disc_forward',
    title='Forward',
    desc='MLP forward pass — trunk + linear head. Override to add residuals, attention, dropout, etc.',
    il_inline=INLINE_SOURCE,
)
