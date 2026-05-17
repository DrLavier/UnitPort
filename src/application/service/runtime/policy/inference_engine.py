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
        """Run one inference step and return the action vector."""


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
        self._input_name = self._session.get_inputs()[0].name

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
