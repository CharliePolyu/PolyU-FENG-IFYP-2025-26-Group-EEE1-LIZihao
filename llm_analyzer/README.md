# llm_analyzer Guide

`llm_analyzer` is a reusable module that converts structured audit metrics
(any JSON-serializable payload) into a human-readable expert diagnosis.

## Features

- Accepts flexible metric schemas (MIA/PIA/health/risk-score or custom fields)
- Compresses oversized payloads before LLM requests
- Calls DeepSeek via an OpenAI-compatible API
- Returns a stable fallback message when API access is unavailable

## Installation

Install project dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variable

```text
DEEPSEEK_API_KEY=your_api_key_here
```

## Minimal Example

```python
from llm_analyzer import generate_expert_report

report = generate_expert_report(
    module_name="Privacy Leakage (MIA & PIA)",
    metrics_data={
        "mia": {"accuracy": 0.71, "precision": 0.69, "recall": 0.74, "f1": 0.71},
        "pia": {"kl_divergence": 0.08, "mae": 0.03},
        "target_model_health": {"train_acc": 0.95, "test_acc": 0.86, "gap": 0.09},
        "risk_score": 62,
    },
    use_dp=False,
    risk_score=62,
)

print(report)
```

## Parameters

- `module_name: str`  
  Human-readable name for the audit module.

- `metrics_data: dict`  
  Any JSON-serializable metrics dictionary.

- `use_dp: bool = False`  
  Whether to frame analysis as a post-DP follow-up.

- `risk_score: int = 0`  
  Overall risk score (0-100).

## Return Value

- On success: plain-text expert report (English)
- On failure: stable fallback string for UI-safe display

