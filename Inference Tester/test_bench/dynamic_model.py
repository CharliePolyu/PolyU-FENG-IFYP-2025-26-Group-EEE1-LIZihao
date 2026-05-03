from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from inference_tester.interfaces import TargetModelWrapper


class DynamicModelWrapper(TargetModelWrapper):
    """Adapter for uploaded PyTorch models used in audit runs.

    The wrapped model is expected to output raw logits with shape
    ``(batch_size, num_classes)``.
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        input_shape: Tuple[int, ...],
        device: str = "cpu",
        use_top_k_masking: bool = False,
        top_k: int = 3,
    ) -> None:
        """Initialize wrapper state for uploaded-model inference."""
        self._model = model
        self._num_classes = num_classes
        self._input_shape = input_shape
        self._device = torch.device(device)
        self._use_top_k_masking = use_top_k_masking
        self._top_k = top_k
        self._model.to(self._device)
        self._model.eval()

    def get_expected_input_shape(self) -> Tuple[int, ...]:
        """Return expected input shape for this model."""
        return self._input_shape

    def get_num_classes(self) -> int:
        """Return number of classes this model predicts."""
        return self._num_classes

    def predict_proba(self, inputs: torch.Tensor | np.ndarray) -> np.ndarray:
        """Run forward pass and return class probabilities."""
        if isinstance(inputs, torch.Tensor):
            x = inputs.to(self._device)
        else:
            x = torch.as_tensor(inputs, dtype=torch.float32, device=self._device)

        with torch.no_grad():
            logits = self._model(x)
            if not isinstance(logits, torch.Tensor):
                raise TypeError(
                    f"Model must return torch.Tensor, got {type(logits)}. "
                    "Ensure your model outputs raw logits."
                )
            if logits.dim() != 2:
                raise ValueError(
                    f"Model output must be 2D (batch, classes), got shape {logits.shape}. "
                    "Ensure your model returns (batch_size, num_classes) logits."
                )
            
            if self._use_top_k_masking:
                logits = self._apply_top_k_mask(logits, k=self._top_k)
            
            probs_tensor = torch.softmax(logits, dim=1)
        return probs_tensor.cpu().numpy()
    
    def _apply_top_k_mask(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """Keep only top-k logits and mask others to -inf."""
        if k >= logits.size(1):
            return logits
        topk_values, topk_indices = torch.topk(logits, k=k, dim=1)
        masked_logits = torch.full_like(logits, float("-inf"))
        masked_logits.scatter_(1, topk_indices, topk_values)
        return masked_logits

    def get_embeddings(self, inputs: torch.Tensor | np.ndarray) -> np.ndarray:
        """Extract features for t-SNE, falling back to logits when needed."""
        if isinstance(inputs, torch.Tensor):
            x = inputs.to(self._device)
        else:
            x = torch.as_tensor(inputs, dtype=torch.float32, device=self._device)

        captured: list[torch.Tensor] = []

        def _hook(_module: nn.Module, _inp: tuple, output: torch.Tensor) -> None:
            captured.append(output.detach())

        try:
            last_linear = None
            for name, module in self._model.named_modules():
                if isinstance(module, nn.Linear):
                    last_linear = module

            if last_linear is None:
                with torch.no_grad():
                    logits = self._model(x)
                return logits.cpu().numpy()

            handle = last_linear.register_forward_hook(_hook)
            try:
                with torch.no_grad():
                    _ = self._model(x)
            finally:
                handle.remove()

            if captured:
                emb = captured[0]
                if emb.dim() > 2:
                    emb = emb.view(emb.size(0), -1)
                return emb.cpu().numpy()
            else:
                with torch.no_grad():
                    logits = self._model(x)
                return logits.cpu().numpy()
        except Exception:
            with torch.no_grad():
                logits = self._model(x)
            return logits.cpu().numpy()
