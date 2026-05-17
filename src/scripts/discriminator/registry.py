"""AMP discriminator preset registry.

Three editable function bodies that drive ``AMPDiscriminator`` behaviour.
Each entry's ``il_inline`` carries the default Python source — the
RewardEditorPanel (parameterised on kind) lets the user inspect and
override these bodies. ``AMPDiscriminator.__init__`` execs any
user-edited source into a callable and binds it as a method-replacement
guard via the ``_OVERRIDE_SLOTS`` table.

Why a registry instead of editing ``discriminator.py`` directly:

* Edits survive UnitPort upgrades that re-vendor ``discriminator.py``.
* Edits don't pollute the vendored AMP_for_hardware code mirror.
* Three slots × isolated callables make it obvious which behaviour
  is being changed.

The ``kind="discriminator"`` registry is **AMP-specific** — vanilla PPO
has no discriminator, so a single sub-registry covers every backend
that supports AMP today (Isaac Lab only).

Slot contract — DO NOT add new keys without first extending
``_OVERRIDE_SLOTS`` in
``application/training/amp/algorithms/discriminator.py``
(once that module migrates from DEMO). The three slot keys in this
registry MUST match ``_OVERRIDE_SLOTS`` exactly:

    ("disc_forward",          "_user_fn_forward",         "forward")
    ("disc_compute_grad_pen", "_user_fn_grad_pen",        "compute_grad_pen")
    ("disc_predict_reward",   "_user_fn_predict_reward",  "predict_amp_reward")

These three are NOT per-robot presets — the GAN architecture and
loss shape are robot-agnostic by design. Users edit them to switch
between research variants (LSGAN vs softplus, Peng 2021 vs WGAN-GP,
ReLU vs LeakyReLU trunk, etc.) — those edits affect every robot the
discriminator runs against.

Runtime wiring (for context — not part of this preset migration):
the main process calls ``emit_inline_overrides("discriminator", path)``
to dump the current sub-registry to a JSON sidecar; the training
subprocess passes that path to ``AMPDiscriminator(user_overrides_file=...)``
which reads + execs the bodies on construction. The IR runtime that
glues these two halves together has not yet landed in RELEASE.
"""

from __future__ import annotations

from typing import Dict

from scripts.task_module import TaskModuleItem, discriminator_item


_DISC_DEFAULT_FORWARD = '''\
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

_DISC_DEFAULT_GRAD_PEN = '''\
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

_DISC_DEFAULT_PREDICT_REWARD = '''\
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


IL_DISC_REGISTRY: Dict[str, TaskModuleItem] = {
    "disc_forward": discriminator_item(
        key="disc_forward",
        title="Forward",
        desc="MLP forward pass — trunk + linear head. Override to add "
             "residuals, attention, dropout, etc.",
        il_inline=_DISC_DEFAULT_FORWARD,
    ),
    "disc_compute_grad_pen": discriminator_item(
        key="disc_compute_grad_pen",
        title="Gradient Penalty",
        desc="Zero-centered R1 gradient penalty on expert transitions. "
             "Default targets |grad|=0 (Peng 2021); override for standard "
             "WGAN-GP target=1 or different norm.",
        il_inline=_DISC_DEFAULT_GRAD_PEN,
    ),
    "disc_predict_reward": discriminator_item(
        key="disc_predict_reward",
        title="Predict AMP Reward",
        desc="Disc score → style reward. Default: amp_reward_coef × "
             "softplus(d.clamp(max=logit_clamp_max)). Override to switch "
             "to LSGAN ``-(d-1)^2``, sigmoid-based, etc.",
        il_inline=_DISC_DEFAULT_PREDICT_REWARD,
    ),
}


def il_disc_registry() -> Dict[str, TaskModuleItem]:
    return dict(IL_DISC_REGISTRY)


__all__ = [
    "IL_DISC_REGISTRY",
    "il_disc_registry",
]
