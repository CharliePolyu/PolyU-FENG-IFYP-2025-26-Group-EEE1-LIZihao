from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

from inference_tester.exceptions import AuditInterrupted
from inference_tester.interfaces import AttackCallback, InterruptSignal


@dataclass
class ShadowTrainingConfig:
    """Configuration for training shadow models.

    Attributes
    ----------
    epochs:
        Number of training epochs per shadow model.

    learning_rate:
        Learning rate for the Adam optimizer.

    weight_decay:
        L2 regularization coefficient for the optimizer.

    device:
        Torch device string such as ``"cuda"`` or ``"cpu"``. If ``None``,
        the implementation will automatically choose CUDA if available,
        otherwise CPU.
    """

    epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    device: Optional[str] = None


class SimpleCNN(nn.Module):
    """A lightweight convolutional neural network for shadow models.

    The architecture is intentionally simple but expressive enough to
    capture non-trivial patterns for vision datasets such as CIFAR-10
    and MNIST. This keeps training time reasonable while still
    producing confidence distributions suitable for membership
    inference.
    """

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # The spatial resolution after two 2x2 poolings is 1/4 in each
        # dimension; we keep the exact flatten size flexible and compute
        # it at runtime using a dummy forward pass.
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # default for 32x32 inputs
        self.fc2 = nn.Linear(128, num_classes)

    def _infer_flat_features(self, input_shape: Tuple[int, ...]) -> int:
        """Infer the number of flattened features for arbitrary input size."""
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            x = self.pool(F.relu(self.bn1(self.conv1(dummy))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            return int(x.numel())

    def adapt_to_input_shape(self, input_shape: Tuple[int, ...]) -> None:
        """Adapt the first fully connected layer to a given input shape.

        This makes the network usable across different image resolutions
        without hard-coding the flatten size. The method should be
        called once after instantiation, before training.
        """
        flat_features = self._infer_flat_features(input_shape)
        if self.fc1.in_features != flat_features:
            self.fc1 = nn.Linear(flat_features, self.fc1.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def _resolve_device(config: ShadowTrainingConfig) -> torch.device:
    """Resolve the torch.device to use based on config and availability."""
    if config.device is not None:
        return torch.device(config.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_single_shadow_model(
    train_loader: DataLoader,
    input_shape: Tuple[int, ...],
    num_classes: int,
    config: Optional[ShadowTrainingConfig] = None,
    callback: Optional[AttackCallback] = None,
    interrupt_signal: Optional[InterruptSignal] = None,
    shadow_index: Optional[int] = None,
    interrupt_check_interval_batches: int = 16,
) -> SimpleCNN:
    """Train a single CNN shadow model on the provided data.

    Algorithm
    ---------
    1. Construct a ``SimpleCNN`` whose input channels and classifier
       output dimension match ``input_shape`` and ``num_classes``.
    2. Move the model to the selected device (GPU if available).
    3. For each epoch:
       a. Iterate over ``train_loader``.
       b. For each batch, perform a forward pass, compute
          ``CrossEntropyLoss``, backpropagate, and update parameters
          using Adam.
    4. Return the trained model.

    Parameters
    ----------
    train_loader:
        Data loader yielding ``(inputs, labels)`` for training this
        shadow model.

    input_shape:
        The expected single-sample input shape, e.g. ``(3, 32, 32)``.

    num_classes:
        Number of classes for the classification task.

    config:
        Optional ``ShadowTrainingConfig`` controlling training
        hyperparameters. Reasonable defaults are used if omitted.

    Returns
    -------
    SimpleCNN
        The trained shadow model in evaluation mode.
    """
    if config is None:
        config = ShadowTrainingConfig()

    device = _resolve_device(config)

    model = SimpleCNN(in_channels=input_shape[0], num_classes=num_classes)
    model.adapt_to_input_shape(input_shape)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    model.train()
    for epoch in range(config.epochs):
        if callback is not None:
            prefix = (
                f"Training shadow model {shadow_index}" if shadow_index is not None else "Training shadow model"
            )
            callback.on_progress(8.0, f"{prefix}: epoch {epoch + 1}/{config.epochs}")
        for batch_idx, (inputs, targets) in enumerate(train_loader, start=1):
            should_check = batch_idx % max(1, interrupt_check_interval_batches) == 0
            if should_check and interrupt_signal is not None and interrupt_signal.is_interrupted():
                raise AuditInterrupted(
                    stage="shadow_training_batch",
                    partial_results={
                        "mia": {
                            "status": "interrupted",
                            "phase": "shadow_training",
                            "shadow_model_index": shadow_index,
                            "completed_epochs": epoch,
                        }
                    },
                )
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def train_shadow_models(
    member_loader: DataLoader,
    k: int,
    input_shape: Tuple[int, ...],
    num_classes: int,
    config: Optional[ShadowTrainingConfig] = None,
    callback: Optional[AttackCallback] = None,
    interrupt_signal: Optional[InterruptSignal] = None,
    interrupt_check_interval_batches: int = 16,
) -> List[SimpleCNN]:
    """Train ``k`` shadow models on disjoint subsets of member candidates.

    The Membership Inference Attack relies on shadow models that mimic
    the behavior of the target model on both member and non-member
    examples. To approximate this setting, we partition the
    ``Member Candidates`` pool into ``k`` disjoint subsets and train one
    independent shadow model on each subset.

    Strategy
    --------
    1. Read the underlying dataset from ``member_loader.dataset``.
       This may already be a ``Subset`` produced by a previous split.
    2. Randomly partition the index set into ``k`` chunks with (almost)
       equal sizes.
    3. For each chunk:
       a. Construct a ``Subset`` backed by the original dataset.
       b. Wrap it in a ``DataLoader``.
       c. Call :func:`train_single_shadow_model`.

    Parameters
    ----------
    member_loader:
        Data loader over the full set of member candidates. Only its
        underlying ``dataset`` and length are used to build per-shadow
        subsets; the original loader is not modified.

    k:
        Number of shadow models to train. Must be >= 1.

    input_shape:
        The expected single-sample input shape.

    num_classes:
        Number of classes in the task.

    config:
        Optional shared training configuration for all shadow models.

    Returns
    -------
    List[SimpleCNN]
        A list of trained shadow models, each trained on a different
        subset of the member data.
    """
    if k <= 0:
        raise ValueError("Number of shadow models 'k' must be >= 1.")

    dataset = member_loader.dataset
    total_size = len(dataset)
    if total_size < k:
        raise ValueError(
            f"Not enough member samples ({total_size}) to train {k} shadow models."
        )

    # Build a deterministic index permutation so runs are reproducible.
    generator = torch.Generator().manual_seed(12345)
    indices = torch.randperm(total_size, generator=generator).tolist()

    # Split indices into k near-equal contiguous chunks.
    chunk_sizes: List[int] = []
    base_chunk = total_size // k
    remainder = total_size % k
    for i in range(k):
        size = base_chunk + (1 if i < remainder else 0)
        chunk_sizes.append(size)

    shadow_models: List[SimpleCNN] = []
    cursor = 0
    for i in range(k):
        if interrupt_signal is not None and interrupt_signal.is_interrupted():
            raise AuditInterrupted(
                stage="shadow_training_model_boundary",
                partial_results={
                    "mia": {
                        "status": "interrupted",
                        "phase": "shadow_training",
                        "shadow_models_trained": len(shadow_models),
                    }
                },
            )
        size = chunk_sizes[i]
        subset_indices = indices[cursor : cursor + size]
        cursor += size

        subset = Subset(dataset, subset_indices)
        subset_loader = DataLoader(
            subset,
            batch_size=member_loader.batch_size or 128,
            shuffle=True,
            num_workers=member_loader.num_workers,
        )

        model = train_single_shadow_model(
            train_loader=subset_loader,
            input_shape=input_shape,
            num_classes=num_classes,
            config=config,
            callback=callback,
            interrupt_signal=interrupt_signal,
            shadow_index=i + 1,
            interrupt_check_interval_batches=interrupt_check_interval_batches,
        )
        shadow_models.append(model)

    return shadow_models

