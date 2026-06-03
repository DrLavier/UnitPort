# SPDX-FileCopyrightText: 2026 SU CHANG
# SPDX-License-Identifier: Apache-2.0

"""ONNX / TorchScript inference engines — verbatim port from DEMO.

PolicyRunner picks one of these via ``policy_format`` in the bundle
manifest. Both engines:

* accept an unbatched ``(obs_dim,)`` numpy float vector
* internally add a batch dim, run inference, strip the batch dim
* return ``(action_dim,)`` float32

ONNXEngine uses ``onnxruntime`` (CPU); JITEngine uses ``torch.jit``.
``torch`` is optional — the import is deferred inside the methods so
JITEngine costs nothing if you only deploy ONNX bundles.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import numpy as np

# torch is an optional dependency; imported lazily inside JITEngine methods.


class InferenceEngine(ABC):
    """Abstract base for policy inference engines."""

    @abstractmethod
    def load(self, model_path: Path) -> None:
        """Load the model from *model_path*."""

    @abstractmethod
    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Run one inference step and return the action vector.

        For a recurrent policy (缺口①) the hidden state is threaded INSIDE
        the engine (kept as engine state across calls) so this signature and
        every PolicyRunner invariant around it stay unchanged. Episode
        boundaries call :meth:`reset_state`.
        """

    # ── recurrent-policy hooks (no-ops for feed-forward engines) ──────
    def is_recurrent(self) -> bool:
        """True iff the loaded policy carries a recurrent hidden state."""
        return False

    def reset_state(self, mask: Optional[np.ndarray] = None) -> None:
        """Reset the hidden state at an episode boundary.

        ``mask`` (optional, shape ``(batch,)``) selects which envs reset
        (non-zero = reset); ``None`` resets the whole state. No-op for
        feed-forward engines.
        """

    def set_recurrent(self, rnn_type: str, hidden_size: int, num_layers: int) -> None:
        """Declare the recurrent contract (rnn_type / hidden_size / num_layers).

        ONNXEngine self-detects from the graph's named inputs and only
        cross-checks here; JITEngine REQUIRES this (TorchScript inputs are
        unnamed) to allocate state and thread it. No-op for feed-forward.
        """


class ONNXEngine(InferenceEngine):
    """ONNX Runtime inference engine (CPU-only in Phase 1).

    Batch convention:
      - predict() accepts an unbatched observation vector of shape (obs_dim,).
      - The session receives input of shape (1, obs_dim) — batch size 1.
      - The first output tensor is used (Phase 1 assumption).
      - The batch dimension is stripped; the returned action has shape
        (action_dim,).

    Multiple outputs:
      If the ONNX model exposes more than one output, only index [0] is used.
      This is a documented Phase 1 simplification; Circle 8 will add output
      selection.
    """

    _DEFAULT_PROVIDERS = ["CPUExecutionProvider"]

    def __init__(self, providers: Optional[List[str]] = None):
        self._providers: List[str] = (
            providers if providers is not None else list(self._DEFAULT_PROVIDERS)
        )
        self._session = None          # onnxruntime.InferenceSession, created in load()
        self._input_name: Optional[str] = None
        # Recurrent state (缺口①) — populated in load() when the graph carries
        # ``h_in`` (and ``c_in`` for LSTM). Maps state-input name → current
        # ndarray; updated from the matching output each predict().
        self._recurrent: bool = False
        self._state: dict = {}                    # {"h_in": ndarray, ["c_in": ndarray]}
        self._state_out: dict = {}                # {"h_in": "h_out", "c_in": "c_out"}
        self._out_names: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, model_path: Path) -> None:
        """Load ONNX model from *model_path*.

        Raises FileNotFoundError if the file does not exist.
        """
        import onnxruntime as ort  # deferred import — not at module level

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"ONNXEngine.load(): model file not found: {model_path}"
            )

        self._session = ort.InferenceSession(
            str(model_path),
            providers=self._providers,
        )
        in_names = [i.name for i in self._session.get_inputs()]
        # ``obs`` input: the recurrent graph names it "obs"; the MLP graph
        # uses the first (only) input.
        self._input_name = "obs" if "obs" in in_names else in_names[0]

        # Recurrent self-detection (缺口①): graph carries h_in (+ c_in).
        self._recurrent = "h_in" in in_names
        self._state = {}
        self._state_out = {}
        if self._recurrent:
            self._out_names = [o.name for o in self._session.get_outputs()]
            pairs = [("h_in", "h_out")]
            if "c_in" in in_names:
                pairs.append(("c_in", "c_out"))
            for in_name, out_name in pairs:
                meta = next(i for i in self._session.get_inputs() if i.name == in_name)
                # shape [num_layers, batch, hidden]; batch is dynamic → 1.
                shape = [int(d) if isinstance(d, int) else 1 for d in meta.shape]
                self._state[in_name] = np.zeros(shape, dtype=np.float32)
                self._state_out[in_name] = out_name

    def is_recurrent(self) -> bool:
        return self._recurrent

    def reset_state(self, mask: Optional[np.ndarray] = None) -> None:
        if not self._recurrent:
            return
        for name, arr in self._state.items():
            if mask is None:
                arr.fill(0.0)
            else:
                # arr shape [L, batch, H]; zero columns where mask is non-zero.
                m = np.asarray(mask).reshape(-1).astype(bool)
                arr[:, m, :] = 0.0

    def set_recurrent(self, rnn_type: str, hidden_size: int, num_layers: int) -> None:
        # Self-detected from the graph; cross-check the contract and fail loud
        # (§8) on disagreement rather than silently replaying a mismatched policy.
        if not self._recurrent:
            raise RuntimeError(
                "ONNXEngine.set_recurrent(): deploy_contract declares a recurrent "
                "policy but the ONNX graph has no 'h_in' input — the exported "
                "policy.onnx is feed-forward. Re-export the bundle."
            )
        for name, arr in self._state.items():
            if int(arr.shape[0]) != int(num_layers) or int(arr.shape[2]) != int(hidden_size):
                raise RuntimeError(
                    f"ONNXEngine.set_recurrent(): contract rnn(layers={num_layers}, "
                    f"hidden={hidden_size}) disagrees with the ONNX '{name}' shape "
                    f"{tuple(arr.shape)} (layers={arr.shape[0]}, hidden={arr.shape[2]}). "
                    f"Bundle is inconsistent — re-export."
                )

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Run inference on a single (unbatched) observation vector.

        Raises RuntimeError if load() has not been called yet.
        Input is coerced to float32 and given a batch dimension before
        passing to the ONNX session.
        """
        if self._session is None:
            raise RuntimeError(
                "ONNXEngine.predict() called before load(). "
                "Call load(model_path) first."
            )

        obs_f32 = obs.astype(np.float32).flatten()
        # Add batch dimension: (obs_dim,) -> (1, obs_dim)
        batched = obs_f32[np.newaxis, :]

        if self._recurrent:
            feed = {self._input_name: batched}
            feed.update(self._state)
            outputs = self._session.run(None, feed)
            out_map = dict(zip(self._out_names, outputs))
            # Thread hidden state forward for the next step.
            for in_name, out_name in self._state_out.items():
                self._state[in_name] = np.asarray(out_map[out_name], dtype=np.float32)
            action_batched = out_map.get("action", outputs[0])
            return np.asarray(action_batched, dtype=np.float32).flatten()

        outputs = self._session.run(None, {self._input_name: batched})
        # Use first output only (Phase 1 assumption)
        action_batched = outputs[0]
        # Strip batch dimension: (1, action_dim) -> (action_dim,)
        action = np.asarray(action_batched, dtype=np.float32).flatten()
        return action


class JITEngine(InferenceEngine):
    """TorchScript (torch.jit) inference engine for Phase 2 policy execution.

    Only TorchScript-serialized models (saved with ``torch.jit.save`` /
    ``torch.jit.script`` / ``torch.jit.trace``) are supported.  Plain
    ``state_dict`` checkpoints or eager-mode ``.pt`` files are explicitly
    rejected — they require model-class reconstruction which is not safe here.

    Batch convention mirrors ONNXEngine:
      - predict() accepts an unbatched observation vector of shape (obs_dim,).
      - A batch dimension is added: (obs_dim,) -> (1, obs_dim).
      - First output tensor is used; batch dimension is stripped on return.
    """

    def __init__(self) -> None:
        self._model = None  # torch.jit.ScriptModule, set in load()
        # Recurrent state (缺口①). TorchScript inputs are unnamed, so the
        # recurrent contract MUST be declared via set_recurrent() (called by
        # PolicyRunner from deploy_contract). _h / _c are torch tensors.
        self._recurrent: bool = False
        self._rnn_type: str = ""
        self._hidden_size: int = 0
        self._num_layers: int = 0
        self._h = None
        self._c = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, model_path: Path) -> None:
        """Load a TorchScript model from *model_path*."""
        import torch  # deferred — optional dependency

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"JITEngine.load(): model file not found: {model_path}"
            )

        try:
            model = torch.jit.load(str(model_path), map_location="cpu")
            model.eval()
        except Exception as exc:
            raise RuntimeError(
                f"JITEngine.load(): failed to load TorchScript model from "
                f"'{model_path}': {exc}"
            ) from exc

        self._model = model

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Run inference on a single (unbatched) observation vector."""
        if self._model is None:
            raise RuntimeError(
                "JITEngine.predict() called before load(). "
                "Call load(model_path) first."
            )

        import torch  # deferred — optional dependency

        obs_f32 = obs.astype(np.float32).flatten()
        # Add batch dimension: (obs_dim,) -> (1, obs_dim)
        tensor = torch.tensor(obs_f32, dtype=torch.float32).unsqueeze(0)

        if self._recurrent:
            with torch.no_grad():
                if self._rnn_type == "lstm":
                    out = self._model(tensor, self._h, self._c)
                    action_t, self._h, self._c = out[0], out[1], out[2]
                else:
                    out = self._model(tensor, self._h)
                    action_t, self._h = out[0], out[1]
            return action_t.squeeze(0).detach().cpu().numpy().astype(np.float32)

        with torch.no_grad():
            output = self._model(tensor)

        if not isinstance(output, torch.Tensor):
            raise RuntimeError(
                f"JITEngine.predict(): model returned unexpected output type "
                f"{type(output).__name__}; expected a Tensor."
            )

        # Strip batch dimension and return float32 NumPy array
        action = output.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return action

    def is_recurrent(self) -> bool:
        return self._recurrent

    def set_recurrent(self, rnn_type: str, hidden_size: int, num_layers: int) -> None:
        import torch  # deferred — optional dependency
        self._recurrent = True
        self._rnn_type = (rnn_type or "gru").strip().lower()
        self._hidden_size = int(hidden_size)
        self._num_layers = int(num_layers)
        self._h = torch.zeros(self._num_layers, 1, self._hidden_size, dtype=torch.float32)
        self._c = (
            torch.zeros(self._num_layers, 1, self._hidden_size, dtype=torch.float32)
            if self._rnn_type == "lstm" else None
        )

    def reset_state(self, mask: Optional[np.ndarray] = None) -> None:
        if not self._recurrent:
            return
        import torch  # deferred — optional dependency
        if mask is None:
            if self._h is not None:
                self._h = torch.zeros_like(self._h)
            if self._c is not None:
                self._c = torch.zeros_like(self._c)
            return
        m = torch.as_tensor(np.asarray(mask).reshape(-1) != 0)
        if self._h is not None:
            self._h[:, m, :] = 0.0
        if self._c is not None:
            self._c[:, m, :] = 0.0
