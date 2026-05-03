from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from inference_tester.exceptions import AuditInterrupted, ModelArchitectureMismatchError
from inference_tester.interfaces import AttackCallback, TargetModelWrapper
from inference_tester.interfaces import InterruptSignal


def _is_shape_mismatch_runtime_error(err: RuntimeError) -> bool:
    text = str(err).lower()
    indicators = (
        "size mismatch",
        "shape",
        "mat1 and mat2",
        "expected input",
        "invalid for input of size",
        "must match the size of tensor",
        "the size of tensor a",
        "the size of tensor b",
        "at non-singleton dimension",
        "dimension 1",
    )
    return any(flag in text for flag in indicators)


def _compute_true_class_distribution(
    loaders: Tuple[DataLoader, DataLoader],
    num_classes: int,
    callback: Optional[AttackCallback] = None,
    interrupt_signal: Optional[InterruptSignal] = None,
    interrupt_check_interval_batches: int = 16,
    min_interrupt_batches: int = 64,
    processed_batches_counter: Optional[List[int]] = None,
    max_batches: Optional[int] = None,
) -> np.ndarray:
    """Compute the empirical class distribution from ground-truth labels.

    Parameters
    ----------
    loaders:
        Tuple of data loaders (e.g. member and non-member loaders).

    num_classes:
        Total number of classes.

    callback:
        Optional progress callback.
    """
    counts = np.zeros(num_classes, dtype=np.float64)
    total = 0

    processed_batches = 0
    for i, loader in enumerate(loaders):
        if callback is not None:
            callback.on_progress(
                70.0 + 5.0 * i, "Accumulating true class distribution"
            )
        for batch_idx, (_inputs, targets) in enumerate(loader, start=1):
            if max_batches is not None and processed_batches >= max_batches:
                break
            processed_batches += 1
            if processed_batches_counter is not None:
                processed_batches_counter[0] += 1
            global_batches = (
                processed_batches_counter[0] if processed_batches_counter is not None else batch_idx
            )
            should_check = batch_idx % max(1, interrupt_check_interval_batches) == 0
            can_interrupt = global_batches >= max(0, min_interrupt_batches)
            if (
                should_check
                and can_interrupt
                and interrupt_signal is not None
                and interrupt_signal.is_interrupted()
            ):
                if total == 0:
                    raise AuditInterrupted(
                        stage="pia_true_distribution",
                        partial_results={
                            "pia": {
                                "status": "interrupted",
                                "phase": "true_distribution",
                                "processed_batches": int(global_batches),
                                "processed_samples": 0,
                            }
                        },
                    )
                partial_true = counts / float(total)
                raise AuditInterrupted(
                    stage="pia_true_distribution",
                    partial_results={
                        "pia": {
                            "status": "interrupted",
                            "phase": "true_distribution",
                            "processed_batches": int(global_batches),
                            "processed_samples": int(total),
                            "true_distribution": partial_true.astype(float).tolist(),
                        }
                    },
                )
            if isinstance(targets, torch.Tensor):
                y = targets.cpu().numpy()
            else:
                y = np.asarray(targets)
            for label in y:
                if 0 <= int(label) < num_classes:
                    counts[int(label)] += 1.0
                    total += 1
        if max_batches is not None and processed_batches >= max_batches:
            break

    if total == 0:
        raise RuntimeError("No samples available to compute true distribution.")

    return counts / float(total)


def _compute_inferred_distribution_from_model(
    target_model: TargetModelWrapper,
    loaders: Tuple[DataLoader, DataLoader],
    num_classes: int,
    callback: Optional[AttackCallback] = None,
    interrupt_signal: Optional[InterruptSignal] = None,
    interrupt_check_interval_batches: int = 16,
    min_interrupt_batches: int = 64,
    processed_batches_counter: Optional[List[int]] = None,
    max_batches: Optional[int] = None,
) -> np.ndarray:
    """Infer class distribution only from target-model predictions.

    The attacker is assumed not to know true labels. Instead, they
    query the black-box model and aggregate predicted probabilities
    across all queried samples, normalizing the sum to obtain an
    estimate of underlying class proportions.
    """
    probs_sum = np.zeros(num_classes, dtype=np.float64)

    def _adapt_probability_dim(probs_np: np.ndarray, expected_dim: int) -> np.ndarray:
        current_dim = int(probs_np.shape[1])
        if current_dim > expected_dim:
            probs_np = probs_np[:, :expected_dim]
        elif current_dim < expected_dim:
            pad = expected_dim - current_dim
            probs_np = np.pad(probs_np, ((0, 0), (0, pad)), mode="constant", constant_values=0.0)
        row_sum = probs_np.sum(axis=1, keepdims=True)
        row_sum[row_sum <= 0.0] = 1.0
        return probs_np / row_sum

    processed_batches = 0
    for i, loader in enumerate(loaders):
        if callback is not None:
            callback.on_progress(
                80.0 + 5.0 * i, "Querying target model for PIA"
            )
        for batch_idx, (inputs, _targets) in enumerate(loader, start=1):
            if max_batches is not None and processed_batches >= max_batches:
                break
            processed_batches += 1
            if processed_batches_counter is not None:
                processed_batches_counter[0] += 1
            global_batches = (
                processed_batches_counter[0] if processed_batches_counter is not None else batch_idx
            )
            should_check = batch_idx % max(1, interrupt_check_interval_batches) == 0
            can_interrupt = global_batches >= max(0, min_interrupt_batches)
            if (
                should_check
                and can_interrupt
                and interrupt_signal is not None
                and interrupt_signal.is_interrupted()
            ):
                total_mass = probs_sum.sum()
                if total_mass <= 0.0:
                    raise AuditInterrupted(
                        stage="pia_inferred_distribution",
                        partial_results={
                            "pia": {
                                "status": "interrupted",
                                "phase": "inferred_distribution",
                                "processed_batches": int(global_batches),
                                "processed_probability_mass": 0.0,
                            }
                        },
                    )
                partial_inferred = probs_sum / total_mass
                raise AuditInterrupted(
                    stage="pia_inferred_distribution",
                    partial_results={
                        "pia": {
                            "status": "interrupted",
                            "phase": "inferred_distribution",
                            "processed_batches": int(global_batches),
                            "processed_probability_mass": float(total_mass),
                            "inferred_distribution": partial_inferred.astype(float).tolist(),
                        }
                    },
                )
            if isinstance(inputs, torch.Tensor):
                batch_inputs = inputs
            else:
                batch_inputs = torch.as_tensor(inputs)
            try:
                probs = target_model.predict_proba(batch_inputs)
            except RuntimeError as e:
                if _is_shape_mismatch_runtime_error(e):
                    raise ModelArchitectureMismatchError(str(e)) from e
                raise
            probs_np = np.asarray(probs, dtype=np.float64)
            if probs_np.ndim != 2:
                raise ValueError(
                    f"Target model predict_proba must return 2D matrix, got shape={probs_np.shape}."
                )
            probs_np = _adapt_probability_dim(probs_np, num_classes)
            probs_sum += probs_np.sum(axis=0)
        if max_batches is not None and processed_batches >= max_batches:
            break

    total_mass = probs_sum.sum()
    if total_mass <= 0.0:
        raise RuntimeError("Inferred probability mass is zero; check model outputs.")

    return probs_sum / total_mass


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Compute KL divergence D_KL(p || q) with numerical stability."""
    p_safe = p + eps
    q_safe = q + eps
    return float(np.sum(p_safe * np.log(p_safe / q_safe)))


def _mae(p: np.ndarray, q: np.ndarray) -> float:
    """Mean Absolute Error between two distributions."""
    return float(np.mean(np.abs(p - q)))


def run_property_inference_attack(
    target_model: TargetModelWrapper,
    member_loader: DataLoader,
    non_member_loader: DataLoader,
    num_classes: int,
    callback: Optional[AttackCallback] = None,
    interrupt_signal: Optional[InterruptSignal] = None,
    interrupt_check_interval_batches: int = 16,
    min_interrupt_batches: int = 64,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """Run a simple Property Inference Attack (PIA) on class distribution.

    High-Level Idea
    ---------------
    We treat the (member + non-member) union as a proxy for the private
    training distribution of the target model. The attacker:

    1. Queries the model on all available samples and aggregates its
       predicted probabilities across classes.
    2. Normalizes this aggregate to obtain an **inferred** class
       distribution.
    3. Compares this inferred distribution with the empirical
       distribution computed from ground-truth labels (available only
       for evaluation, not to the attacker) using:
       - Kullback-Leibler (KL) divergence.
       - Mean Absolute Error (MAE).

    Parameters
    ----------
    target_model:
        The model under audit.

    member_loader:
        Data loader of member samples.

    non_member_loader:
        Data loader of non-member samples.

    num_classes:
        Number of classes in the dataset and model output.

    callback:
        Optional progress callback.

    Returns
    -------
    Dict[str, Any]
        JSON-serializable dictionary containing:

        - ``"kl_divergence"``: float
        - ``"mae"``: float
        - ``"true_distribution"``: List[float]
        - ``"inferred_distribution"``: List[float]
    """
    loaders = (member_loader, non_member_loader)
    processed_batches_counter = [0]

    if callback is not None:
        callback.on_progress(68.0, "Computing true class distribution for PIA")

    true_dist = _compute_true_class_distribution(
        loaders=loaders,
        num_classes=num_classes,
        callback=callback,
        interrupt_signal=interrupt_signal,
        interrupt_check_interval_batches=interrupt_check_interval_batches,
        min_interrupt_batches=min_interrupt_batches,
        processed_batches_counter=processed_batches_counter,
        max_batches=max_batches,
    )

    if callback is not None:
        callback.on_progress(78.0, "Inferring class distribution from target model")

    inferred_dist = _compute_inferred_distribution_from_model(
        target_model=target_model,
        loaders=loaders,
        num_classes=num_classes,
        callback=callback,
        interrupt_signal=interrupt_signal,
        interrupt_check_interval_batches=interrupt_check_interval_batches,
        min_interrupt_batches=min_interrupt_batches,
        processed_batches_counter=processed_batches_counter,
        max_batches=max_batches,
    )

    kl = _kl_divergence(true_dist, inferred_dist)
    mae = _mae(true_dist, inferred_dist)

    if callback is not None:
        callback.on_progress(90.0, "Finished PIA; preparing results")

    return {
        "kl_divergence": kl,
        "mae": mae,
        "true_distribution": true_dist.astype(float).tolist(),
        "inferred_distribution": inferred_dist.astype(float).tolist(),
    }

