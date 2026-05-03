"""Universal LLM-based expert diagnosis for security audit metrics.

This module provides a scalable analyzer that can process any JSON metrics
and generate professional diagnostic reports via the DeepSeek API.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import traceback
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)
MAX_METRICS_CHARS = 24000


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _summarize_numeric_list(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    sorted_vals = sorted(values)
    n = len(sorted_vals)

    def _pct(p: float) -> float:
        idx = int(round((n - 1) * p))
        idx = max(0, min(n - 1, idx))
        return float(sorted_vals[idx])

    return {
        "count": n,
        "min": float(sorted_vals[0]),
        "max": float(sorted_vals[-1]),
        "mean": float(sum(sorted_vals) / n),
        "p50": _pct(0.50),
        "p90": _pct(0.90),
        "head": [float(v) for v in sorted_vals[:3]],
        "tail": [float(v) for v in sorted_vals[-3:]],
    }


def _sanitize_for_llm(obj: Any, list_limit: int = 64, depth: int = 0, max_depth: int = 8) -> Any:
    """Compress potentially huge metrics payloads for LLM context safety."""
    if depth > max_depth:
        return "<omitted: max depth reached>"

    if isinstance(obj, dict):
        # Keep stable key order for deterministic prompts.
        return {str(k): _sanitize_for_llm(v, list_limit, depth + 1, max_depth) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        seq = list(obj)
        if not seq:
            return []

        # Large numeric sequences -> summarize statistically.
        if len(seq) > list_limit and all(_is_number(v) for v in seq):
            vals = [float(v) for v in seq]
            return {
                "_summary": "numeric_list_truncated",
                **_summarize_numeric_list(vals),
            }

        # Large mixed/nested sequences -> keep head/tail only.
        if len(seq) > list_limit:
            head_n = max(1, list_limit // 2)
            tail_n = max(1, list_limit - head_n)
            return {
                "_summary": "list_truncated",
                "count": len(seq),
                "head": [_sanitize_for_llm(v, list_limit, depth + 1, max_depth) for v in seq[:head_n]],
                "tail": [_sanitize_for_llm(v, list_limit, depth + 1, max_depth) for v in seq[-tail_n:]],
            }

        return [_sanitize_for_llm(v, list_limit, depth + 1, max_depth) for v in seq]

    return obj


def _extract_core_metrics(metrics: dict) -> dict:
    """Final fallback summary when payload is still too large."""
    mia = metrics.get("mia", {}) if isinstance(metrics.get("mia"), dict) else {}
    pia = metrics.get("pia", {}) if isinstance(metrics.get("pia"), dict) else {}
    health = (
        metrics.get("target_model_health", {})
        if isinstance(metrics.get("target_model_health"), dict)
        else {}
    )
    roc = (
        metrics.get("mia_roc_curve", {})
        if isinstance(metrics.get("mia_roc_curve"), dict)
        else {}
    )

    return {
        "risk_score": metrics.get("risk_score"),
        "mia": {
            "accuracy": mia.get("accuracy"),
            "precision": mia.get("precision"),
            "recall": mia.get("recall"),
            "f1": mia.get("f1"),
            "auc": roc.get("auc") or mia.get("auc"),
        },
        "target_model_health": {
            "train_acc": health.get("train_acc"),
            "test_acc": health.get("test_acc"),
            "gap": health.get("gap") or health.get("generalization_gap"),
        },
        "pia": {
            "kl_divergence": pia.get("kl_divergence"),
            "mae": pia.get("mae"),
        },
        "_note": "Large arrays were truncated for context safety.",
    }


def _make_json_serializable(obj: object) -> object:
    """Convert numpy/types to Python builtins for JSON serialization."""
    try:
        import numpy as np  # noqa: PLC0415
    except ImportError:
        np = None  # type: ignore[assignment]

    if isinstance(obj, dict):
        return {str(k): _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(x) for x in obj]
    if np is not None and isinstance(obj, np.ndarray):
        return obj.astype(float).tolist()
    if np is not None and isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    return obj


def generate_expert_report(module_name: str, metrics_data: dict, use_dp: bool = False, risk_score: int = 0) -> str:
    """Generate a professional expert diagnostic report for given audit metrics.

    Uses the DeepSeek API (OpenAI-compatible) to analyze any JSON-structured
    metrics and produce a concise security/privacy assessment with actionable
    mitigation strategies. Designed to be universal: teammates can pass their
    own test results (various JSON structures) for different security audits.

    Parameters
    ----------
    module_name : str
        Human-readable name of the audit module (e.g. "Privacy Leakage (MIA & PIA)").
    metrics_data : dict
        Any JSON-serializable dictionary of metrics to analyze.
    use_dp : bool
        If True, this is a DP-SGD defense comparison run.
    risk_score : int
        Overall privacy risk score (0-100).

    Returns
    -------
    str
        Expert diagnostic report in English, or a fail-safe message if the
        API is unavailable or any error occurs.
    """
    try:
        if OpenAI is None:
            return "AI Expert analysis is currently unavailable. Please check the API configuration."

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key or not api_key.strip():
            return "AI Expert analysis is currently unavailable. Please check the API configuration."

        client = OpenAI(
            api_key=api_key.strip(),
            base_url="https://api.deepseek.com",
        )

        serializable_metrics = _make_json_serializable(metrics_data)
        compact_metrics = _sanitize_for_llm(serializable_metrics, list_limit=64)
        metrics_str = json.dumps(compact_metrics, ensure_ascii=False, separators=(",", ":"))

        # Multi-stage payload reduction to avoid context overflow.
        if len(metrics_str) > MAX_METRICS_CHARS:
            compact_metrics = _sanitize_for_llm(serializable_metrics, list_limit=16)
            metrics_str = json.dumps(compact_metrics, ensure_ascii=False, separators=(",", ":"))
        if len(metrics_str) > MAX_METRICS_CHARS:
            compact_metrics = _extract_core_metrics(
                serializable_metrics if isinstance(serializable_metrics, dict) else {}
            )
            metrics_str = json.dumps(compact_metrics, ensure_ascii=False, separators=(",", ":"))

        logger.info("LLM metrics payload chars=%d", len(metrics_str))

        if use_dp:
            user_prompt = (
                f"This is a FOLLOW-UP audit after applying Differential Privacy (DP-SGD). "
                f"The metrics are: {metrics_str}. "
                f"Overall Privacy Risk Score: {risk_score}/100. "
                "Please provide a professional diagnostic report in ENGLISH structured in 3 sections:\n\n"
                "1. DIAGNOSTIC ANALYSIS: Compare these DP metrics to baseline. Evaluate if DP successfully mitigated privacy risks.\n"
                "2. MITIGATION STRATEGIES: Provide 2-3 technical recommendations for further improvement.\n"
                "3. COMPLIANCE CHECK: Based on the risk score, assess compliance with GDPR Article 25 (Privacy by Design) "
                "and NIST AI RMF. If score > 70, explicitly flag as '[!] NON-COMPLIANT'.\n\n"
                "Keep the report concise (200-250 words). Output plain text only: no markdown, no asterisks, no hashtags."
            )
        else:
            user_prompt = (
                f"Please analyze the following test results for the [{module_name}] audit. "
                f"The metrics are: {metrics_str}. "
                f"Overall Privacy Risk Score: {risk_score}/100. "
                "Provide a professional diagnostic report in ENGLISH structured in 3 sections:\n\n"
                "1. DIAGNOSTIC ANALYSIS: Evaluate the privacy risks based on the metrics.\n"
                "2. MITIGATION STRATEGIES: Provide 2-3 actionable technical recommendations.\n"
                "3. COMPLIANCE CHECK: Based on the risk score, assess compliance with GDPR Article 25 (Privacy by Design) "
                "and NIST AI RMF. If score > 70, explicitly flag as '[!] NON-COMPLIANT'. "
                "If score <= 30, state '[✓] COMPLIANT'. Otherwise, state '[~] PARTIAL COMPLIANCE'.\n\n"
                "Keep the report concise (200-250 words). Output plain text only: no markdown, no asterisks, no hashtags."
            )

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": "You are a Top-tier AI Security and Privacy Expert with deep knowledge of GDPR and NIST AI Risk Management Framework. Always respond in plain text only - no markdown, no asterisks for bold, no special formatting symbols.",
                },
                {"role": "user", "content": user_prompt},
            ],
        )

        choice = response.choices[0]
        if choice.message and choice.message.content:
            return choice.message.content.strip()
        return "AI Expert analysis is currently unavailable. Please check the API configuration."
    except Exception as e:
        logger.exception("LLM analyzer failed: %s", e)
        traceback.print_exc(file=sys.stderr)  # visible in terminal when running Flask
        return "AI Expert analysis is currently unavailable. Please check the API configuration."
