from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import ResNet50_Weights, resnet50

from inference_tester.interfaces import TargetModelWrapper


class MockModelWrapper(TargetModelWrapper):
    """Target model wrapper backed by a real ResNet-50 architecture.

    Despite the legacy class name (kept for compatibility with existing
    app imports), this implementation is no longer synthetic. It uses a
    mature CNN (ResNet-50), adapts the classifier head to CIFAR-10
    classes, and performs lightweight training in the test bench.

    Notes
    -----
    This class is intended for demo/testing flows in ``test_bench`` and
    is not part of the core attack API contract.
    """

    def __init__(
        self,
        num_classes: int = 10,
        data_root: str = "./data",
        train_epochs: int = 1,
        train_subset_size: int = 2000,
        train_batch_size: int = 64,
        learning_rate: float = 1e-4,
        device: Optional[str] = None,
        use_dp: bool = False,
        use_label_smoothing: bool = False,
        use_top_k_masking: bool = False,
        top_k: int = 3,
    ) -> None:
        self._num_classes = num_classes
        self._data_root = data_root
        self._train_epochs = train_epochs
        self._train_subset_size = train_subset_size
        self._train_batch_size = train_batch_size
        self._learning_rate = learning_rate
        self._device = torch.device(device) if device else torch.device("cpu")
        self._use_dp = use_dp
        self._use_label_smoothing = use_label_smoothing
        self._use_top_k_masking = use_top_k_masking
        self._top_k = top_k

        self._model = self._build_model()
        self._is_trained = False

    def _build_model(self) -> nn.Module:
        """Build a ResNet-50 model and adapt it to the task classes."""
        try:
            weights = ResNet50_Weights.IMAGENET1K_V2
            model = resnet50(weights=weights)
        except Exception:
            # Fallback when pretrained weights cannot be downloaded.
            model = resnet50(weights=None)

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, self._num_classes)
        model.to(self._device)
        return model

    @staticmethod
    def _preprocess_for_resnet(inputs: torch.Tensor) -> torch.Tensor:
        """Resize and normalize inputs for ResNet-style inference."""
        x = inputs.float()
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std

    def fit_if_needed(self) -> None:
        """Train the target model once before inference-time auditing."""
        if self._is_trained:
            return

        transform = transforms.ToTensor()
        dataset = datasets.CIFAR10(
            root=self._data_root,
            train=True,
            download=True,
            transform=transform,
        )

        subset_size = min(self._train_subset_size, len(dataset))
        rng = torch.Generator().manual_seed(2026)
        perm = torch.randperm(len(dataset), generator=rng).tolist()
        subset_indices = perm[:subset_size]
        subset = Subset(dataset, subset_indices)

        loader = DataLoader(
            subset,
            batch_size=self._train_batch_size,
            shuffle=True,
            num_workers=2,
        )

        label_smoothing_val = 0.1 if self._use_label_smoothing else 0.0
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing_val)
        if self._use_label_smoothing:
            print(f"[Label Smoothing] Applied with alpha=0.1 to reduce overconfidence.", flush=True)
        optimizer = Adam(self._model.parameters(), lr=self._learning_rate)

        if self._use_dp:
            try:
                from opacus import PrivacyEngine
                privacy_engine = PrivacyEngine()
                self._model, optimizer, loader = privacy_engine.make_private(
                    module=self._model,
                    optimizer=optimizer,
                    data_loader=loader,
                    noise_multiplier=1.0,
                    max_grad_norm=1.0,
                )
                print("[DP-SGD] PrivacyEngine attached. Training with differential privacy.", flush=True)
            except ImportError:
                print("[Warning] opacus not installed. Falling back to standard training.", flush=True)
            except Exception as e:
                print(f"[Warning] DP-SGD setup failed: {e}. Falling back to standard training.", flush=True)

        self._model.train()
        for _ in range(self._train_epochs):
            for inputs, labels in loader:
                inputs = inputs.to(self._device)
                labels = labels.to(self._device)

                optimizer.zero_grad()
                logits = self._model(self._preprocess_for_resnet(inputs))
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

        self._model.eval()
        self._is_trained = True

    def get_expected_input_shape(self) -> Tuple[int, ...]:
        """Return CIFAR-10 like input shape (3, 32, 32)."""
        return (3, 32, 32)

    def get_num_classes(self) -> int:
        return self._num_classes

    def predict_proba(self, inputs: torch.Tensor | np.ndarray) -> np.ndarray:
        """Return class probabilities from the trained ResNet-50 model."""
        self.fit_if_needed()

        if isinstance(inputs, torch.Tensor):
            x = inputs.to(self._device)
        else:
            x = torch.as_tensor(inputs, dtype=torch.float32, device=self._device)

        x = self._preprocess_for_resnet(x)
        with torch.no_grad():
            logits = self._model(x)
            if self._use_top_k_masking:
                logits = self._apply_top_k_mask(logits, k=self._top_k)
            probs_tensor = torch.softmax(logits, dim=1)
        return probs_tensor.cpu().numpy()

    def _apply_top_k_mask(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """Apply Top-K confidence masking to reduce information leakage.

        Keeps only the top-k logits; replaces others with -inf so they
        become ~0 after softmax. This prevents attackers from exploiting
        low-confidence predictions as membership signals.

        Parameters
        ----------
        logits : torch.Tensor
            Raw logits from model, shape (batch_size, num_classes).
        k : int
            Number of top predictions to keep (e.g., 3).

        Returns
        -------
        torch.Tensor
            Masked logits with same shape and device.
        """
        if k >= logits.size(1):
            return logits
        
        topk_values, topk_indices = torch.topk(logits, k=k, dim=1)
        masked_logits = torch.full_like(logits, float("-inf"))
        masked_logits.scatter_(1, topk_indices, topk_values)
        return masked_logits

    def get_embeddings(self, inputs: torch.Tensor | np.ndarray) -> np.ndarray:
        """Extract feature embeddings from the second-to-last layer (before logits).

        Uses a forward hook on the final classifier to capture pre-logits
        representations and returns an array shaped
        ``(batch_size, embedding_dim)``.
        """
        self.fit_if_needed()

        if isinstance(inputs, torch.Tensor):
            x = inputs.to(self._device)
        else:
            x = torch.as_tensor(inputs, dtype=torch.float32, device=self._device)

        x = self._preprocess_for_resnet(x)
        captured: list[torch.Tensor] = []

        def _hook(_module: nn.Module, inp: tuple) -> None:
            emb = inp[0]
            captured.append(emb.detach())

        handle = self._model.fc.register_forward_pre_hook(_hook)
        try:
            with torch.no_grad():
                _ = self._model(x)
        finally:
            handle.remove()

        if not captured:
            return np.empty((0, 2048), dtype=np.float32)
        emb_tensor = captured[0]
        return emb_tensor.cpu().numpy().astype(np.float32)

