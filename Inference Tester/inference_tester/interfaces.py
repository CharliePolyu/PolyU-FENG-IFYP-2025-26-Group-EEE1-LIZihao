from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, Sequence, Tuple

import numpy as np
import torch


class TargetModelWrapper(ABC):
    """Abstract base class that standardizes how the inference tester
    interacts with a target model.

    Any model you want to audit (e.g., a PyTorch model, a sklearn model,
    or a remote service) should be wrapped by a concrete implementation
    of this class.

    The design goal is to keep the core inference-testing logic fully
    decoupled from the concrete model implementation and framework.
    """

    @abstractmethod
    def predict_proba(self, inputs: torch.Tensor | np.ndarray) -> np.ndarray:
        """Return class-probability predictions for a batch of inputs.

        Parameters
        ----------
        inputs:
            A batch of inputs. Implementations should accept either a
            PyTorch tensor (preferred for performance) or a NumPy array
            and internally handle conversion as needed.

        Returns
        -------
        np.ndarray
            A NumPy array of shape (batch_size, num_classes) where each
            row is a probability distribution over classes (sums to 1).
        """

    @abstractmethod
    def get_expected_input_shape(self) -> Tuple[int, ...]:
        """Return the expected single-sample input shape.

        Examples
        --------
        - For CIFAR-10 images: (3, 32, 32)
        - For MNIST images: (1, 28, 28)
        - For tabular features: (num_features,)
        """

    @abstractmethod
    def get_num_classes(self) -> int:
        """Return the number of classes predicted by the model."""


class AttackCallback(ABC):
    """Callback interface for reporting progress during attacks.

    The test bench or any other caller can implement this interface to
    receive progress updates from long-running operations such as
    training shadow models or running MIA/PIA pipelines.
    """

    @abstractmethod
    def on_progress(self, percentage: float, message: str) -> None:
        """Report progress of the attack pipeline.

        Parameters
        ----------
        percentage:
            Progress value in [0, 100]. Implementations should be
            tolerant to non-monotonic updates, but typical usage
            will be monotonically increasing.

        message:
            Human-readable status message describing the current step,
            e.g. "Training shadow model 1/5" or "Running MIA on target".
        """


class InterruptSignal(ABC):
    """Interface for cooperative, graceful interruption checks.

    Implementations should expose a cheap non-blocking check that returns
    `True` when the running audit should terminate at the next safe point.
    """

    @abstractmethod
    def is_interrupted(self) -> bool:
        """Return True when an interrupt has been requested."""


class SupportsAttackCallback(Protocol):
    """Structural protocol for objects that expose an AttackCallback.

    This is mainly provided for type-checking convenience in places
    where an object may *optionally* hold a callback instance.
    """

    callback: AttackCallback | None

