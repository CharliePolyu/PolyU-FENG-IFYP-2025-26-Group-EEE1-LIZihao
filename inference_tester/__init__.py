from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data_manager import prepare_shadow_data
from .exceptions import AuditInterrupted, ModelArchitectureMismatchError
from .interfaces import AttackCallback, InterruptSignal, TargetModelWrapper
from .core.mia import run_membership_inference_attack
from .core.pia import run_property_inference_attack
from .core.shadow_trainer import ShadowTrainingConfig, train_shadow_models


class _AdaptedTargetModelWrapper(TargetModelWrapper):
    """Adapter that force-aligns target output dim to dataset class space."""

    def __init__(self, base: TargetModelWrapper, expected_num_classes: int) -> None:
        self._base = base
        self._expected_num_classes = int(expected_num_classes)

    def predict_proba(self, inputs: torch.Tensor | np.ndarray) -> np.ndarray:
        probs = np.asarray(self._base.predict_proba(inputs), dtype=np.float32)
        if probs.ndim != 2:
            raise ValueError(
                f"Target predict_proba must return 2D array, got shape={probs.shape}."
            )
        current_dim = int(probs.shape[1])
        if current_dim > self._expected_num_classes:
            probs = probs[:, : self._expected_num_classes]
        elif current_dim < self._expected_num_classes:
            pad_width = self._expected_num_classes - current_dim
            probs = np.pad(probs, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)

        # Keep row-wise probabilities normalized for downstream metrics.
        row_sum = probs.sum(axis=1, keepdims=True)
        row_sum[row_sum <= 0.0] = 1.0
        return probs / row_sum

    def get_expected_input_shape(self) -> tuple[int, ...]:
        return self._base.get_expected_input_shape()

    def get_num_classes(self) -> int:
        return self._expected_num_classes

    def __getattr__(self, item: str) -> Any:
        # Delegate optional capabilities like get_embeddings.
        return getattr(self._base, item)


def _calculate_risk_score(
    mia_auc: Optional[float],
    generalization_gap: Optional[float],
    kl_divergence: Optional[float],
) -> int:
    """Calculate overall privacy risk score (0-100) based on MIA, gap, and PIA.
    
    Score formula:
    - MIA component: (AUC - 0.5) * 200, normalized to [0, 100]
    - Gap component: (gap / 0.20) * 100, normalized to [0, 100]
    - PIA component: 100 - (KL / 0.5) * 100, normalized to [0, 100]
    - Overall: 0.5 * MIA + 0.25 * Gap + 0.25 * PIA
    
    Returns
    -------
    int
        Risk score from 0 (low) to 100 (high).
    """
    score_mia = 0.0
    if mia_auc is not None:
        score_mia = max(0.0, min(100.0, (mia_auc - 0.5) * 200.0))
    
    score_gap = 0.0
    if generalization_gap is not None:
        score_gap = max(0.0, min(100.0, (generalization_gap / 0.20) * 100.0))
    
    score_pia = 0.0
    if kl_divergence is not None:
        score_pia = max(0.0, min(100.0, 100.0 - (kl_divergence / 0.5) * 100.0))
    
    overall = 0.5 * score_mia + 0.25 * score_gap + 0.25 * score_pia
    return round(overall)


def _evaluate_target_model_health(
    target_model: TargetModelWrapper,
    member_loader: DataLoader,
    non_member_loader: DataLoader,
    member_max_batches: Optional[int],
    non_member_max_batches: Optional[int],
) -> Dict[str, float]:
    """Evaluate target model accuracy on member (train) and non-member (test) sets."""
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

    correct, total = 0, 0
    for batch_idx, (inputs, targets) in enumerate(member_loader, start=1):
        if member_max_batches is not None and batch_idx > member_max_batches:
            break
        try:
            probs = target_model.predict_proba(inputs)
        except RuntimeError as e:
            if _is_shape_mismatch_runtime_error(e):
                raise ModelArchitectureMismatchError(str(e)) from e
            raise
        preds = np.argmax(probs, axis=1)
        if hasattr(targets, "numpy"):
            y = targets.numpy()
        else:
            y = np.asarray(targets)
        correct += int(np.sum(preds == y))
        total += len(y)
    train_acc = float(correct / total) if total > 0 else 0.0

    correct, total = 0, 0
    for batch_idx, (inputs, targets) in enumerate(non_member_loader, start=1):
        if non_member_max_batches is not None and batch_idx > non_member_max_batches:
            break
        try:
            probs = target_model.predict_proba(inputs)
        except RuntimeError as e:
            if _is_shape_mismatch_runtime_error(e):
                raise ModelArchitectureMismatchError(str(e)) from e
            raise
        preds = np.argmax(probs, axis=1)
        if hasattr(targets, "numpy"):
            y = targets.numpy()
        else:
            y = np.asarray(targets)
        correct += int(np.sum(preds == y))
        total += len(y)
    test_acc = float(correct / total) if total > 0 else 0.0

    gap = train_acc - test_acc
    return {"train_acc": train_acc, "test_acc": test_acc, "gap": gap}


def _normalize_config(user_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge user-provided config with sensible defaults.

    The configuration is intentionally lightweight and JSON-friendly so
    that external callers (e.g. a UI or orchestrator service) can pass
    it directly over an API boundary.
    """
    default: Dict[str, Any] = {
        "data": {
            "batch_size": 128,
            "num_workers": 2,
            "member_ratio": 0.8,
        },
        "interrupt": {
            "check_interval_batches": 16,
            "pia_min_interrupt_batches": 64,
        },
        "shadow": {
            "k": 5,
            "epochs": 5,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "device": None,
        },
        "mia": {
            "target_member_max_batches": None,
            "target_non_member_max_batches": None,
        },
        "pia": {
            "max_batches": None,
        },
    }

    if user_config is None:
        return default

    # Perform a section-level merge for top-level config groups.
    merged = default.copy()
    for section in ("data", "shadow", "interrupt", "mia", "pia"):
        if section in user_config and isinstance(user_config[section], dict):
            merged_section = merged[section].copy()
            merged_section.update(user_config[section])
            merged[section] = merged_section
    return merged


def run_inference_audit(
    target_model: TargetModelWrapper,
    metadata: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
    callback: Optional[AttackCallback] = None,
    interrupt_signal: Optional[InterruptSignal] = None,
) -> Dict[str, Any]:
    """Run a full inference-privacy audit (MIA + PIA) on a target model.

    This is the main entry point exposed by the ``inference_tester``
    package. It orchestrates dataset preparation, shadow-model
    training, membership inference, and property inference in a
    decoupled, backend-only fashion.

    Pipeline
    --------
    1. **Data preparation**:
       Use :func:`prepare_shadow_data` to download and split the
       specified dataset into member and non-member candidate sets.
    2. **Shadow model training**:
       Train ``k`` convolutional shadow models on disjoint subsets of
       the member-candidate pool.
    3. **Membership Inference Attack (MIA)**:
       Use the shadow models to train a logistic regression
       meta-classifier that predicts membership from confidence vectors,
       and apply it to the target model's outputs.
    4. **Property Inference Attack (PIA)**:
       Query the target model to infer the overall class distribution
       and compare it against the empirical distribution.

    Parameters
    ----------
    target_model:
        The model to be audited, wrapped as a ``TargetModelWrapper`` so
        that this package remains agnostic to the underlying framework
        (PyTorch, TensorFlow, sklearn, remote API, etc.).

    metadata:
        Dictionary describing the dataset domain and related parameters.
        Must include at least ``"dataset_domain"``. See
        :func:`prepare_shadow_data` for details.

    config:
        Optional configuration dictionary. Structure:

        .. code-block:: python

            {
                "data": {
                    "batch_size": int,
                    "num_workers": int,
                },
                "shadow": {
                    "k": int,
                    "epochs": int,
                    "learning_rate": float,
                    "weight_decay": float,
                    "device": Optional[str],
                },
            }

        Missing values are filled with sensible defaults.

    callback:
        Optional ``AttackCallback`` implementation used to receive
        progress updates through the pipeline.

    Returns
    -------
    Dict[str, Any]
        JSON-serializable dictionary, suitable for direct return from a
        web API, with the structure:

        .. code-block:: python

            {
                "metadata": {...},   # echo of input
                "config_used": {...},
                "results": {
                    "mia": {...},    # metrics + visualization arrays
                    "pia": {...},
                },
            }
    """
    cfg = _normalize_config(config)
    payload: Dict[str, Any] = {
        "status": "completed",
        "stopped_at": None,
        "metadata": metadata,
        "config_used": cfg,
        "results": {
            "mia": None,
            "pia": None,
            "target_model_health": None,
        },
    }

    def _has_meaningful_results() -> bool:
        # Require at least one complete section (MIA/PIA) before honoring stop.
        return payload["results"]["mia"] is not None or payload["results"]["pia"] is not None

    def _check_interrupt(stage: str) -> None:
        if (
            interrupt_signal is not None
            and interrupt_signal.is_interrupted()
            and _has_meaningful_results()
        ):
            raise AuditInterrupted(stage=stage, partial_results={})

    try:
        # Step 1: Prepare datasets.
        if callback is not None:
            callback.on_progress(5.0, "Preparing datasets for shadow models")

        data_cfg = cfg["data"]
        member_loader, non_member_loader, num_classes, input_shape = prepare_shadow_data(
            metadata=metadata,
            batch_size=int(data_cfg.get("batch_size", 128)),
            num_workers=int(data_cfg.get("num_workers", 2)),
            member_ratio=float(data_cfg.get("member_ratio", 0.8)),
        )
        adapted_target_model = _AdaptedTargetModelWrapper(
            base=target_model,
            expected_num_classes=int(num_classes),
        )
        _check_interrupt("after_data_preparation")

        # Step 2: Evaluate target-model health before attacks.
        if callback is not None:
            callback.on_progress(6.0, "Evaluating target model health")
        payload["results"]["target_model_health"] = _evaluate_target_model_health(
            target_model=adapted_target_model,
            member_loader=member_loader,
            non_member_loader=non_member_loader,
            member_max_batches=cfg.get("mia", {}).get("target_member_max_batches"),
            non_member_max_batches=cfg.get("mia", {}).get("target_non_member_max_batches"),
        )
        _check_interrupt("after_target_health")

        # Step 3: Train shadow models.
        shadow_cfg = cfg["shadow"]
        interrupt_cfg = cfg["interrupt"]
        interrupt_check_interval_batches = max(
            1, int(interrupt_cfg.get("check_interval_batches", 16))
        )
        pia_min_interrupt_batches = max(
            0, int(interrupt_cfg.get("pia_min_interrupt_batches", 64))
        )
        shadow_config = ShadowTrainingConfig(
            epochs=int(shadow_cfg.get("epochs", 5)),
            learning_rate=float(shadow_cfg.get("learning_rate", 1e-3)),
            weight_decay=float(shadow_cfg.get("weight_decay", 0.0)),
            device=shadow_cfg.get("device"),
        )

        if callback is not None:
            callback.on_progress(8.0, "Training shadow models")

        k = int(shadow_cfg.get("k", 5))
        shadow_models = train_shadow_models(
            member_loader=member_loader,
            k=k,
            input_shape=input_shape,
            num_classes=num_classes,
            config=shadow_config,
            callback=callback,
            # Defer interruption until meaningful results are available.
            interrupt_signal=None,
            interrupt_check_interval_batches=interrupt_check_interval_batches,
        )
        _check_interrupt("after_shadow_training")

        # Step 4: Run Membership Inference Attack (MIA).
        if callback is not None:
            callback.on_progress(20.0, "Running Membership Inference Attack (MIA)")

        mia_results = run_membership_inference_attack(
            target_model=adapted_target_model,
            shadow_models=shadow_models,
            member_loader=member_loader,
            non_member_loader=non_member_loader,
            device=torch.device(shadow_config.device)
            if shadow_config.device is not None
            else None,
            callback=callback,
            # Keep MIA atomic so early-stop payloads remain meaningful.
            interrupt_signal=None,
            interrupt_check_interval_batches=interrupt_check_interval_batches,
            target_member_max_batches=cfg.get("mia", {}).get("target_member_max_batches"),
            target_non_member_max_batches=cfg.get("mia", {}).get("target_non_member_max_batches"),
            expected_num_classes=int(num_classes),
        )
        tsne_charts = mia_results.pop("tsne_charts", None)
        roc_curve_data = mia_results.pop("roc_curve", None)
        payload["results"]["mia"] = mia_results
        if tsne_charts is not None:
            payload["results"]["tsne_charts"] = tsne_charts
        if roc_curve_data is not None:
            payload["results"]["mia_charts"] = {"roc_curve": roc_curve_data}
        _check_interrupt("after_mia")

        # Step 5: Run Property Inference Attack (PIA).
        if callback is not None:
            callback.on_progress(60.0, "Running Property Inference Attack (PIA)")

        pia_results = run_property_inference_attack(
            target_model=adapted_target_model,
            member_loader=member_loader,
            non_member_loader=non_member_loader,
            num_classes=num_classes,
            callback=callback,
            interrupt_signal=interrupt_signal,
            interrupt_check_interval_batches=interrupt_check_interval_batches,
            min_interrupt_batches=pia_min_interrupt_batches,
            max_batches=cfg.get("pia", {}).get("max_batches"),
        )
        payload["results"]["pia"] = pia_results

        # Step 6: Compute the overall risk score.
        if callback is not None:
            callback.on_progress(95.0, "Calculating overall risk score")
        payload["results"]["risk_score"] = _calculate_risk_score(
            mia_auc=roc_curve_data.get("auc") if roc_curve_data else None,
            generalization_gap=payload["results"]["target_model_health"].get("gap") if payload["results"]["target_model_health"] else None,
            kl_divergence=pia_results.get("kl_divergence"),
        )

        if callback is not None:
            callback.on_progress(100.0, "Inference audit complete")
        return payload
    except AuditInterrupted as exc:
        if callback is not None:
            callback.on_progress(100.0, f"Audit interrupted at {exc.stage}")
        payload["status"] = "interrupted"
        payload["stopped_at"] = exc.stage
        partial = exc.partial_results or {}
        if "mia" in partial:
            payload["results"]["mia"] = partial["mia"]
        if "pia" in partial:
            payload["results"]["pia"] = partial["pia"]
        return payload


__all__ = ["run_inference_audit", "TargetModelWrapper", "AttackCallback", "InterruptSignal"]

