"""AMP discriminator network — small MLP scoring (s_t, s_{t+1}) pairs.

REWRITE-WITH-DEMO-REF from DEMO ``src/system/training/amp/algorithms/discriminator.py``
(itself BSD-3 NVIDIA + ETH Zurich). Stage 8 strips the canvas-side user-
override loader (DEMO Phase 0.3 feature) — that lands as a follow-up
once the SB3 AMP path stabilises.

Loss (paired with WGAN-GP-style zero-centered gradient penalty on expert):
    style_r = amp_reward_coef * softplus(clip(disc_logit, max=logit_clamp))
    grad_pen = lambda * mean(||∂d/∂(s,s')||²)  on expert pairs only

The disc is trained with BCEWithLogits in :class:`AmpPpoTrainer`; the
``softplus(logit)`` reward is the natural matching reward for that loss
(equivalent to ``-log(1 - sigmoid(logit))``).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


class AMPDiscriminator:
    """Transition-pair discriminator for AMP.

    Constructed lazily — the underlying ``nn.Module`` only materialises
    when torch is actually imported. This keeps the parent package
    importable without torch installed (Stage 8 unit tests can mock the
    disc; only the real AMP-PPO trainer requires torch).
    """

    def __init__(
        self,
        input_dim: int,
        amp_reward_coef: float,
        hidden_layer_sizes: Iterable[int],
        device: Optional[str] = None,
        task_reward_lerp: float = 0.5,
        logit_clamp_max: float = 4.0,
        user_overrides_file: str = "",
    ) -> None:
        import torch
        import torch.nn as nn

        self._torch = torch
        self.input_dim = int(input_dim)
        self.amp_reward_coef = float(amp_reward_coef)
        self.task_reward_lerp = float(task_reward_lerp)
        self.logit_clamp_max = float(logit_clamp_max)

        # P0.3 — User-edited function bodies. Loaded from a JSON sidecar
        # the main process writes via emit_inline_overrides("discriminator").
        # Each entry's il_inline is a full ``def name(self, ...): ...``
        # block; we exec it into a private dict and dispatch at call
        # time. Empty / missing file = pure default behaviour.
        self._user_fn_forward = None
        self._user_fn_grad_pen = None
        self._user_fn_predict_reward = None
        self._load_user_overrides(user_overrides_file)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # LeakyReLU(0.2) prevents the dying-ReLU collapse common to GAN
        # discriminators — see the DEMO discriminator docstring for the
        # full rationale (PatchGAN / WGAN convention).
        layers = []
        curr_in = self.input_dim
        for h in hidden_layer_sizes:
            layers.append(nn.Linear(curr_in, int(h)))
            layers.append(nn.LeakyReLU(negative_slope=0.2))
            curr_in = int(h)
        self.trunk = nn.Sequential(*layers).to(self.device)
        self.amp_linear = nn.Linear(curr_in, 1).to(self.device)
        self.trunk.train()
        self.amp_linear.train()

    # ------------------------------------------------------------------
    # P0.3 — User-override loader (canvas-supplied function bodies)
    # ------------------------------------------------------------------

    #: Map disc-registry key → (attribute, expected function name). The
    #: function name inside the user's source must match the second
    #: tuple element; if not, we fall back to taking the only callable
    #: in the namespace.
    _OVERRIDE_SLOTS = (
        ("disc_forward", "_user_fn_forward", "forward"),
        ("disc_compute_grad_pen", "_user_fn_grad_pen", "compute_grad_pen"),
        ("disc_predict_reward", "_user_fn_predict_reward", "predict_amp_reward"),
    )

    def _load_user_overrides(self, path: str) -> None:
        """Parse the disc-override JSON sidecar and bind user fns.

        The sidecar is written by the main process via
        ``emit_inline_overrides("discriminator")`` whenever the user
        edits a discriminator function body in the canvas's registry
        module editor. Empty path / missing file / parse error = pure
        default behaviour. Per-entry failures degrade gracefully
        (warn + keep default) so a single broken user fn cannot stop
        training launch — but the warning is loud, no silent drop.
        """
        if not path:
            return
        try:
            import json
            from pathlib import Path
            p = Path(path)
            if not p.is_file():
                return
            payload = json.loads(p.read_text(encoding="utf-8"))
            entries = (payload or {}).get("entries", {})
            if not isinstance(entries, dict) or not entries:
                return
        except Exception as exc:
            print(
                f"[UnitPort][AMP][WARN] disc overrides file unreadable "
                f"({exc}) — using defaults.",
                flush=True,
            )
            return

        for reg_key, attr, fn_name in self._OVERRIDE_SLOTS:
            entry = entries.get(reg_key)
            if not entry:
                continue
            src = (entry.get("il_inline") or "").strip()
            if not src:
                continue
            ns: dict = {}
            try:
                exec(src, ns, ns)  # noqa: S102 — intentional: user code
            except Exception as exc:
                print(
                    f"[UnitPort][AMP][WARN] disc override {reg_key} "
                    f"failed to compile ({exc}) — using default.",
                    flush=True,
                )
                continue
            fn = ns.get(fn_name)
            if not callable(fn):
                # Tolerate user-renamed function: take the single callable
                # in the namespace if exactly one is defined.
                callables = [v for v in ns.values() if callable(v)]
                if len(callables) == 1:
                    fn = callables[0]
                else:
                    print(
                        f"[UnitPort][AMP][WARN] disc override {reg_key}: "
                        f"could not resolve function (expected name "
                        f"{fn_name!r} or a single def) — using default.",
                        flush=True,
                    )
                    continue
            setattr(self, attr, fn)
            print(
                f"[UnitPort][AMP] applied user override: {reg_key} "
                f"→ {fn.__name__}()",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Forward + losses
    # ------------------------------------------------------------------

    def forward(self, x):
        """Score a batch of concatenated ``(s_t, s_{t+1})`` pairs."""
        if self._user_fn_forward is not None:
            return self._user_fn_forward(self, x)
        return self.amp_linear(self.trunk(x))

    __call__ = forward

    def parameters(self):
        return list(self.trunk.parameters()) + list(self.amp_linear.parameters())

    def to(self, device):
        """Move trunk + amp_linear to *device* and return ``self``.

        Mirrors ``nn.Module.to``'s chainable semantics so the runner
        construction ``AMPDiscriminator(...).to(self.device)`` keeps
        working. RELEASE's disc is a plain Python class (not an
        nn.Module subclass — keeps the module main-venv importable
        without a hard torch dep at module load), so the standard
        chainable ``.to`` has to be reimplemented by hand.
        """
        torch = self._torch
        self.device = torch.device(device)
        self.trunk.to(self.device)
        self.amp_linear.to(self.device)
        return self

    def train(self):
        self.trunk.train()
        self.amp_linear.train()

    def eval(self):
        self.trunk.eval()
        self.amp_linear.eval()

    def compute_grad_pen(
        self,
        expert_state,
        expert_next_state,
        lambda_: float = 10.0,
    ):
        """Zero-centered WGAN-GP gradient penalty on expert pairs only.

        Penalises non-zero ``||∂d/∂(s,s')||`` rather than the standard
        WGAN-GP target of 1. Stabilises AMP training in practice.
        """
        if self._user_fn_grad_pen is not None:
            return self._user_fn_grad_pen(
                self, expert_state, expert_next_state, lambda_
            )
        torch = self._torch
        from torch import autograd

        expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
        expert_data.requires_grad = True

        disc = self.amp_linear(self.trunk(expert_data))
        ones = torch.ones(disc.size(), device=disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=expert_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return float(lambda_) * grad.norm(2, dim=1).pow(2).mean()

    # ------------------------------------------------------------------
    # Persistence — atomic save/load of trunk + amp_linear weights
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        """Return a torch state dict bundling trunk + amp_linear + meta."""
        return {
            "trunk": self.trunk.state_dict(),
            "amp_linear": self.amp_linear.state_dict(),
            "input_dim": int(self.input_dim),
            "amp_reward_coef": float(self.amp_reward_coef),
            "task_reward_lerp": float(self.task_reward_lerp),
            "logit_clamp_max": float(self.logit_clamp_max),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load weights produced by :meth:`state_dict`."""
        self.trunk.load_state_dict(state["trunk"])
        self.amp_linear.load_state_dict(state["amp_linear"])

    def save(self, path) -> None:
        """Atomic write of :meth:`state_dict` to disk via temp + rename."""
        torch = self._torch
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        torch.save(self.state_dict(), str(tmp))
        os.replace(str(tmp), str(target))

    @classmethod
    def load(
        cls,
        path,
        *,
        hidden_layer_sizes: Iterable[int],
        device: Optional[str] = None,
    ) -> "AMPDiscriminator":
        """Reconstruct an :class:`AMPDiscriminator` from a saved file.

        ``hidden_layer_sizes`` is required because the trunk topology is
        not part of the state dict; pass the same value used at training
        time (``spec.il.amp.disc.hidden_dims``).
        """
        import torch

        state = torch.load(str(path), map_location=device or "cpu")
        inst = cls(
            input_dim=int(state["input_dim"]),
            amp_reward_coef=float(state["amp_reward_coef"]),
            hidden_layer_sizes=hidden_layer_sizes,
            device=device,
            task_reward_lerp=float(state.get("task_reward_lerp", 0.5)),
            logit_clamp_max=float(state.get("logit_clamp_max", 4.0)),
        )
        inst.load_state_dict(state)
        return inst

    def predict_amp_reward(
        self,
        state,
        next_state,
        normalizer=None,
    ) -> Tuple[object, object]:
        """Compute style reward + raw disc logit (no grad).

        Returns ``(style_reward, disc_logit)`` — caller blends with task
        reward via :attr:`task_reward_lerp`.

        Reward formula (Peng 2021 / amp-rsl-rl):
            ``style_r = coef * softplus(clip(logit, max=logit_clamp_max))``

        The clip protects the critic from runaway logits (disc drift):
        ``softplus(4) ≈ 4.018``, so at the BCE-with-label-smoothing
        equilibrium (logits ~±2.2) the clip is inactive but provides a
        circuit-breaker if disc destabilises.
        """
        if self._user_fn_predict_reward is not None:
            return self._user_fn_predict_reward(
                self, state, next_state, normalizer
            )
        torch = self._torch
        with torch.no_grad():
            self.eval()
            if normalizer is not None:
                state = normalizer.normalize_torch(state, self.device)
                next_state = normalizer.normalize_torch(next_state, self.device)
            x = torch.cat([state, next_state], dim=-1)
            d = self.amp_linear(self.trunk(x))
            style_reward = self.amp_reward_coef * torch.nn.functional.softplus(
                d.clamp(max=self.logit_clamp_max)
            )
            self.train()
        return style_reward.squeeze(-1), d


__all__ = ["AMPDiscriminator"]
