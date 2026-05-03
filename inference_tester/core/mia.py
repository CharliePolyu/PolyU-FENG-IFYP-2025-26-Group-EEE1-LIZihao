from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from inference_tester.exceptions import AuditInterrupted, ModelArchitectureMismatchError
from inference_tester.interfaces import AttackCallback, TargetModelWrapper
from inference_tester.interfaces import InterruptSignal

TSNE_MAX_SAMPLES_PER_CLASS = 200


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


def _to_numpy_probs(logits: torch.Tensor) -> np.ndarray:
    """Convert logits tensor to NumPy probability array via softmax."""
    with torch.no_grad():
        probs = torch.softmax(logits, dim=1)
    return probs.cpu().numpy()


def _adapt_logits_dim(logits: torch.Tensor, expected_dim: int) -> torch.Tensor:
    """Adapt logits width to expected class dimension."""
    current_dim = int(logits.shape[1])
    if current_dim == expected_dim:
        return logits
    if current_dim > expected_dim:
        return logits[:, :expected_dim]
    pad_right = expected_dim - current_dim
    return F.pad(logits, (0, pad_right), mode="constant", value=-100.0)


def _adapt_probability_dim(probs: np.ndarray, expected_dim: int) -> np.ndarray:
    """Gracefully adapt probability matrix width for meta-classifier."""
    current_dim = int(probs.shape[1])
    if current_dim > expected_dim:
        probs = probs[:, :expected_dim]
    elif current_dim < expected_dim:
        pad = expected_dim - current_dim
        probs = np.pad(probs, ((0, 0), (0, pad)), mode="constant", constant_values=0.0)

    row_sum = probs.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 0.0] = 1.0
    return probs / row_sum


def _collect_shadow_confidences(
    shadow_models: Sequence[nn.Module],
    member_loader: DataLoader,
    non_member_loader: DataLoader,
    device: torch.device,
    callback: Optional[AttackCallback] = None,
    interrupt_signal: Optional[InterruptSignal] = None,
    interrupt_check_interval_batches: int = 16,
    expected_num_classes: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect confidence vectors from shadow models for meta-classifier.

    For each shadow model, we:
    - Run it on all member-candidate batches and label outputs as
      membership ``1``.
    - Run it on all non-member-candidate batches and label outputs as
      membership ``0``.

    The resulting feature matrix is a stack of per-sample probability
    vectors across all shadow models, and the label vector encodes
    membership status.
    """
    features: List[np.ndarray] = []
    labels: List[int] = []

    total_models = len(shadow_models)
    for idx, model in enumerate(shadow_models, start=1):
        if interrupt_signal is not None and interrupt_signal.is_interrupted():
            raise AuditInterrupted(
                stage="mia_shadow_confidence_collection",
                partial_results={
                    "mia": {
                        "status": "interrupted",
                        "phase": "shadow_confidence_collection",
                        "shadow_models_processed": idx - 1,
                    }
                },
            )
        if callback is not None:
            callback.on_progress(
                float(10 + 10 * idx / max(total_models, 1)),
                f"Collecting shadow confidences ({idx}/{total_models})",
            )

        model.eval()
        model.to(device)

        # Member-candidate samples map to label 1.
        for member_batch_idx, (inputs, _targets) in enumerate(member_loader, start=1):
            should_check = member_batch_idx % max(1, interrupt_check_interval_batches) == 0
            if should_check and interrupt_signal is not None and interrupt_signal.is_interrupted():
                raise AuditInterrupted(
                    stage="mia_shadow_confidence_collection_member",
                    partial_results={
                        "mia": {
                            "status": "interrupted",
                            "phase": "shadow_confidence_collection",
                            "shadow_models_processed": idx - 1,
                        }
                    },
                )
            inputs = inputs.to(device)
            logits = model(inputs)
            if expected_num_classes is not None and logits.dim() == 2:
                logits = _adapt_logits_dim(logits, expected_num_classes)
            probs_np = _to_numpy_probs(logits)
            features.append(probs_np)
            labels.extend([1] * probs_np.shape[0])

        # Non-member-candidate samples map to label 0.
        for non_member_batch_idx, (inputs, _targets) in enumerate(non_member_loader, start=1):
            should_check = non_member_batch_idx % max(1, interrupt_check_interval_batches) == 0
            if should_check and interrupt_signal is not None and interrupt_signal.is_interrupted():
                raise AuditInterrupted(
                    stage="mia_shadow_confidence_collection_non_member",
                    partial_results={
                        "mia": {
                            "status": "interrupted",
                            "phase": "shadow_confidence_collection",
                            "shadow_models_processed": idx - 1,
                        }
                    },
                )
            inputs = inputs.to(device)
            logits = model(inputs)
            if expected_num_classes is not None and logits.dim() == 2:
                logits = _adapt_logits_dim(logits, expected_num_classes)
            probs_np = _to_numpy_probs(logits)
            features.append(probs_np)
            labels.extend([0] * probs_np.shape[0])

    if not features:
        raise RuntimeError("No shadow features collected; check data loaders.")

    x = np.concatenate(features, axis=0)
    y = np.asarray(labels, dtype=np.int32)
    return x, y


def _collect_target_confidences_and_labels(
    target_model: TargetModelWrapper,
    member_loader: DataLoader,
    non_member_loader: DataLoader,
    callback: Optional[AttackCallback] = None,
    interrupt_signal: Optional[InterruptSignal] = None,
    interrupt_check_interval_batches: int = 16,
    target_member_max_batches: Optional[int] = None,
    target_non_member_max_batches: Optional[int] = None,
    expected_num_classes: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect target-model confidences on member/non-member sets.

    Returns
    -------
    member_probs:
        Probability matrix for member samples.

    non_member_probs:
        Probability matrix for non-member samples.

    member_labels:
        Ground-truth membership labels for member samples (all ones).

    non_member_labels:
        Ground-truth membership labels for non-member samples (all
        zeros).
    """
    member_probs_list: List[np.ndarray] = []
    non_member_probs_list: List[np.ndarray] = []

    # Collect probabilities on the member set.
    for member_batch_idx, (inputs, _targets) in enumerate(member_loader, start=1):
        if target_member_max_batches is not None and member_batch_idx > target_member_max_batches:
            break
        should_check = member_batch_idx % max(1, interrupt_check_interval_batches) == 0
        if should_check and interrupt_signal is not None and interrupt_signal.is_interrupted():
            break
        if callback is not None:
            callback.on_progress(40.0, "Querying target model on member set")
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
        probs_np = np.asarray(probs, dtype=np.float32)
        if expected_num_classes is not None and probs_np.ndim == 2:
            probs_np = _adapt_probability_dim(probs_np, expected_num_classes)
        member_probs_list.append(probs_np)

    # Collect probabilities on the non-member set.
    for non_member_batch_idx, (inputs, _targets) in enumerate(non_member_loader, start=1):
        if (
            target_non_member_max_batches is not None
            and non_member_batch_idx > target_non_member_max_batches
        ):
            break
        should_check = non_member_batch_idx % max(1, interrupt_check_interval_batches) == 0
        if should_check and interrupt_signal is not None and interrupt_signal.is_interrupted():
            break
        if callback is not None:
            callback.on_progress(45.0, "Querying target model on non-member set")
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
        probs_np = np.asarray(probs, dtype=np.float32)
        if expected_num_classes is not None and probs_np.ndim == 2:
            probs_np = _adapt_probability_dim(probs_np, expected_num_classes)
        non_member_probs_list.append(probs_np)

    fallback_dim = expected_num_classes or target_model.get_num_classes()
    member_probs = (
        np.concatenate(member_probs_list, axis=0)
        if member_probs_list
        else np.empty((0, fallback_dim), dtype=np.float32)
    )
    non_member_probs = (
        np.concatenate(non_member_probs_list, axis=0)
        if non_member_probs_list
        else np.empty((0, fallback_dim), dtype=np.float32)
    )

    member_labels = np.ones(member_probs.shape[0], dtype=np.int32)
    non_member_labels = np.zeros(non_member_probs.shape[0], dtype=np.int32)

    return member_probs, non_member_probs, member_labels, non_member_labels


def _collect_target_embeddings_for_tsne(
    target_model: TargetModelWrapper,
    member_loader: DataLoader,
    non_member_loader: DataLoader,
    max_samples: int = TSNE_MAX_SAMPLES_PER_CLASS,
    callback: Optional[AttackCallback] = None,
    interrupt_signal: Optional[InterruptSignal] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Collect embeddings from target model for t-SNE visualization.

    Returns (member_embeddings, non_member_embeddings) or None if the model
    does not support embedding extraction.
    """
    if not hasattr(target_model, "get_embeddings"):
        return None

    member_embs: List[np.ndarray] = []
    non_member_embs: List[np.ndarray] = []
    member_count = 0
    non_member_count = 0

    for inputs, _ in member_loader:
        if member_count >= max_samples:
            break
        if interrupt_signal is not None and interrupt_signal.is_interrupted():
            return None
        if callback is not None:
            callback.on_progress(38.0, "Collecting embeddings for t-SNE (members)")
        batch_inputs = inputs if isinstance(inputs, torch.Tensor) else torch.as_tensor(inputs)
        try:
            embs = target_model.get_embeddings(batch_inputs)
        except RuntimeError as e:
            if _is_shape_mismatch_runtime_error(e):
                raise ModelArchitectureMismatchError(str(e)) from e
            raise
        take = min(embs.shape[0], max_samples - member_count)
        member_embs.append(embs[:take])
        member_count += take

    for inputs, _ in non_member_loader:
        if non_member_count >= max_samples:
            break
        if interrupt_signal is not None and interrupt_signal.is_interrupted():
            return None
        if callback is not None:
            callback.on_progress(39.0, "Collecting embeddings for t-SNE (non-members)")
        batch_inputs = inputs if isinstance(inputs, torch.Tensor) else torch.as_tensor(inputs)
        try:
            embs = target_model.get_embeddings(batch_inputs)
        except RuntimeError as e:
            if _is_shape_mismatch_runtime_error(e):
                raise ModelArchitectureMismatchError(str(e)) from e
            raise
        take = min(embs.shape[0], max_samples - non_member_count)
        non_member_embs.append(embs[:take])
        non_member_count += take

    if not member_embs and not non_member_embs:
        return None

    emb_dim = (
        member_embs[0].shape[1] if member_embs
        else non_member_embs[0].shape[1] if non_member_embs
        else 0
    )
    member_arr = np.concatenate(member_embs, axis=0) if member_embs else np.empty((0, emb_dim))
    non_member_arr = np.concatenate(non_member_embs, axis=0) if non_member_embs else np.empty((0, emb_dim))
    return member_arr, non_member_arr


def _run_tsne_and_format(
    member_embs: np.ndarray,
    non_member_embs: np.ndarray,
    callback: Optional[AttackCallback] = None,
    interrupt_signal: Optional[InterruptSignal] = None,
) -> Optional[Dict[str, Any]]:
    """Run t-SNE on embeddings and return coordinates for charting."""
    if interrupt_signal is not None and interrupt_signal.is_interrupted():
        return None

    total = member_embs.shape[0] + non_member_embs.shape[0]
    if total < 4:
        return None

    all_embs = np.concatenate([member_embs, non_member_embs], axis=0).astype(np.float64)
    n_member = member_embs.shape[0]

    if callback is not None:
        callback.on_progress(64.0, "Running t-SNE dimension reduction")

    tsne = TSNE(n_components=2, init="pca", random_state=42, perplexity=min(30, total - 1))
    coords = tsne.fit_transform(all_embs)

    member_coords = coords[:n_member].tolist()
    non_member_coords = coords[n_member:].tolist()

    return {"members": member_coords, "non_members": non_member_coords}


def run_membership_inference_attack(
    target_model: TargetModelWrapper,
    shadow_models: Sequence[nn.Module],
    member_loader: DataLoader,
    non_member_loader: DataLoader,
    device: Optional[torch.device] = None,
    callback: Optional[AttackCallback] = None,
    interrupt_signal: Optional[InterruptSignal] = None,
    interrupt_check_interval_batches: int = 16,
    target_member_max_batches: Optional[int] = None,
    target_non_member_max_batches: Optional[int] = None,
    expected_num_classes: Optional[int] = None,
) -> Dict[str, Any]:
    """Run a black-box Membership Inference Attack (MIA).

    High-Level Procedure
    --------------------
    1. **Shadow feature collection**:
       Use each shadow model to generate confidence vectors on both
       member and non-member candidate sets. These serve as training
       data for the attack model (meta-classifier).
    2. **Meta-classifier training**:
       Train a logistic regression classifier on the collected features
       to distinguish members (label=1) from non-members (label=0).
    3. **Target model probing**:
       Query the target model with the same member/non-member splits,
       obtaining confidence vectors for each sample.
    4. **Membership prediction & evaluation**:
       Feed the target confidences through the meta-classifier to obtain
       membership predictions, then compute Accuracy, Precision, Recall
       and F1-score.
    5. **Visualization data**:
       Extract the maximum predicted confidence per sample for members
       and non-members separately so that the UI can draw an overlapping
       histogram of confidence gaps.

    Parameters
    ----------
    target_model:
        The model under audit, wrapped as a ``TargetModelWrapper``.

    shadow_models:
        Sequence of trained shadow models (typically CNNs) that mimic
        the target model's behavior.

    member_loader:
        Data loader over member-candidate samples (ground truth label=1
        for MIA evaluation).

    non_member_loader:
        Data loader over non-member-candidate samples (ground truth
        label=0).

    device:
        Torch device used for running shadow models. If ``None``,
        defaults to CUDA if available, otherwise CPU.

    callback:
        Optional progress callback.

    Returns
    -------
    Dict[str, Any]
        A JSON-serializable dictionary with keys:

        - ``"accuracy"``: float
        - ``"precision"``: float
        - ``"recall"``: float
        - ``"f1"``: float
        - ``"confidence_gap"``: {
              ``"member_max"``: List[float],
              ``"non_member_max"``: List[float]
          }
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if callback is not None:
        callback.on_progress(10.0, "Starting shadow feature extraction for MIA")

    # 1) Train-time data for the meta-classifier using shadow models.
    x_shadow, y_shadow = _collect_shadow_confidences(
        shadow_models=shadow_models,
        member_loader=member_loader,
        non_member_loader=non_member_loader,
        device=device,
        callback=callback,
        interrupt_signal=interrupt_signal,
        interrupt_check_interval_batches=interrupt_check_interval_batches,
        expected_num_classes=expected_num_classes,
    )

    if interrupt_signal is not None and interrupt_signal.is_interrupted():
        raise AuditInterrupted(
            stage="mia_before_meta_training",
            partial_results={
                "mia": {
                    "status": "interrupted",
                    "phase": "before_meta_training",
                }
            },
        )

    if callback is not None:
        callback.on_progress(30.0, "Training logistic regression meta-classifier")

    # 2) Train logistic regression attack model.
    attack_model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
    )
    attack_model.fit(x_shadow, y_shadow)

    # 3) Collect target model confidences for evaluation.
    if interrupt_signal is not None and interrupt_signal.is_interrupted():
        raise AuditInterrupted(
            stage="mia_after_meta_training",
            partial_results={
                "mia": {
                    "status": "interrupted",
                    "phase": "after_meta_training",
                }
            },
        )

    if callback is not None:
        callback.on_progress(35.0, "Collecting target model confidences for MIA")

    (
        member_probs,
        non_member_probs,
        member_labels,
        non_member_labels,
    ) = _collect_target_confidences_and_labels(
        target_model=target_model,
        member_loader=member_loader,
        non_member_loader=non_member_loader,
        callback=callback,
        interrupt_signal=interrupt_signal,
        interrupt_check_interval_batches=interrupt_check_interval_batches,
        target_member_max_batches=target_member_max_batches,
        target_non_member_max_batches=target_non_member_max_batches,
        expected_num_classes=expected_num_classes or int(x_shadow.shape[1]),
    )

    if interrupt_signal is not None and interrupt_signal.is_interrupted():
        partial_size = int(member_probs.shape[0] + non_member_probs.shape[0])
        if partial_size > 0:
            x_partial = np.concatenate([member_probs, non_member_probs], axis=0)
            y_partial = np.concatenate([member_labels, non_member_labels], axis=0)
            y_partial_pred = attack_model.predict(x_partial)
            p_acc = float(accuracy_score(y_partial, y_partial_pred))
            p_precision, p_recall, p_f1, _ = precision_recall_fscore_support(
                y_partial,
                y_partial_pred,
                average="binary",
                pos_label=1,
                zero_division=0,
            )
            raise AuditInterrupted(
                stage="mia_target_query_interrupted",
                partial_results={
                    "mia": {
                        "status": "interrupted",
                        "phase": "target_query",
                        "partial_samples": partial_size,
                        "accuracy": p_acc,
                        "precision": float(p_precision),
                        "recall": float(p_recall),
                        "f1": float(p_f1),
                        "confidence_gap": {
                            "member_max": member_probs.max(axis=1).astype(float).tolist()
                            if member_probs.shape[0] > 0
                            else [],
                            "non_member_max": non_member_probs.max(axis=1).astype(float).tolist()
                            if non_member_probs.shape[0] > 0
                            else [],
                        },
                    }
                },
            )
        raise AuditInterrupted(
            stage="mia_target_query_interrupted",
            partial_results={
                "mia": {
                    "status": "interrupted",
                    "phase": "target_query",
                    "partial_samples": 0,
                }
            },
        )

    x_target = np.concatenate([member_probs, non_member_probs], axis=0)
    y_true = np.concatenate([member_labels, non_member_labels], axis=0)

    if callback is not None:
        callback.on_progress(55.0, "Predicting membership for target samples")

    try:
        y_pred = attack_model.predict(x_target)
        y_score = attack_model.predict_proba(x_target)[:, 1]  # P(member=1)
    except ValueError as e:
        err_text = str(e).lower()
        if "features" in err_text or "expecting" in err_text:
            raise ModelArchitectureMismatchError(
                f"Meta-classifier feature mismatch between shadow and target confidences: {e}"
            ) from e
        raise

    # Compute ROC curve and AUC.
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    auc_val = float(roc_auc_score(y_true, y_score))

    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    # Prepare max-confidence arrays for histogram-style visualization.
    member_max_conf = member_probs.max(axis=1)
    non_member_max_conf = non_member_probs.max(axis=1)

    # Build t-SNE chart data from extracted embeddings when available.
    tsne_charts: Optional[Dict[str, Any]] = None
    if interrupt_signal is None or not interrupt_signal.is_interrupted():
        emb_result = _collect_target_embeddings_for_tsne(
            target_model=target_model,
            member_loader=member_loader,
            non_member_loader=non_member_loader,
            max_samples=TSNE_MAX_SAMPLES_PER_CLASS,
            callback=callback,
            interrupt_signal=interrupt_signal,
        )
        if emb_result is not None and interrupt_signal is not None and interrupt_signal.is_interrupted():
            emb_result = None
        if emb_result is not None:
            member_embs, non_member_embs = emb_result
            tsne_charts = _run_tsne_and_format(
                member_embs=member_embs,
                non_member_embs=non_member_embs,
                callback=callback,
                interrupt_signal=interrupt_signal,
            )

    if callback is not None:
        callback.on_progress(65.0, "Finished MIA; preparing results")

    result: Dict[str, Any] = {
        "accuracy": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confidence_gap": {
            "member_max": member_max_conf.astype(float).tolist(),
            "non_member_max": non_member_max_conf.astype(float).tolist(),
        },
        "roc_curve": {
            "fpr": fpr.astype(float).tolist(),
            "tpr": tpr.astype(float).tolist(),
            "auc": auc_val,
        },
    }
    if tsne_charts is not None:
        result["tsne_charts"] = tsne_charts
    return result

