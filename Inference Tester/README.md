# Inference Tester User Guide

`Inference Tester` is a privacy-audit toolkit for ML inference systems.
It includes a backend audit engine and a demo web UI.

Main capabilities:

- Membership Inference Attack (MIA)
- Property Inference Attack (PIA)
- Target model health checks (train/test accuracy and generalization gap)
- Overall risk scoring (0-100)
- Optional LLM-based expert diagnosis

## 1. Project Structure

```text
Inference Tester/
├── inference_tester/         # Core audit engine (MIA/PIA/shadow training)
├── test_bench/               # Flask API + demo web page
├── llm_analyzer/             # Reusable standalone LLM analyzer
├── data/                     # Dataset/cache storage
├── requirements.txt
└── README.md
```

## 2. Setup

Create and activate a Python virtual environment:

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate
pip install -r requirements.txt
```

If you want LLM expert analysis, configure:

```text
DEEPSEEK_API_KEY=your_api_key_here
```

If the key is not set, the audit still runs and returns a fallback LLM message.

## 3. Run the Service

From the `Inference Tester` directory:

```bash
python -m test_bench.app
```

Default URL:

- `http://127.0.0.1:5000`

## 4. Web Usage

You can run audits in two ways:

- Upload your own `.pth/.pt` model
- Use the built-in mock model

Common options:

- `shadow_models_count` (1-10)
- `shadow_epochs` (1/5/10)
- `probe_sample_size` (`small`/`medium`/`full`)
- `member_ratio` (0.1-0.9)
- `defenses.top_k_masking` (bool)
- `use_cache` (bool)

Note: For uploaded models, DP-SGD and Label Smoothing are training-time defenses and are intentionally ignored.

## 5. API Overview

### 5.1 Run Audit

- `POST /api/inference/run`

Supported request modes:

- `application/json` (mock path, no file upload)
- `multipart/form-data` (uploaded model path)
  - `model_file`: `.pth` or `.pt`
  - `config`: JSON string

Main response fields:

- `run_id`
- `logs`
- `results` (includes `mia`, `pia`, `target_model_health`, `risk_score`)
- `llm_analysis`
- `model_info`
- `resolved_data_options`

### 5.2 Poll Progress

- `GET /api/inference/progress?run_id=<id>`

### 5.3 Interrupt Run

- `POST /api/inference/interrupt`

Body:

```json
{"run_id":"your_run_id"}
```

### 5.4 Regenerate Expert Analysis

- `POST /api/inference/regenerate_expert_analysis`

## 6. Uploaded Model Requirements

Uploaded models should satisfy:

- Extension is `.pth` or `.pt`
- The file is an executable model object (`nn.Module` or TorchScript), not only `state_dict`
- Forward output is 2D logits: `(batch_size, num_classes)`

If you hit pickle class-resolution issues (`Can't get attribute ...`), exporting to TorchScript is recommended.

## 7. Cache and Performance

- Cache directory: `test_bench/cache/`
- Set `use_cache=true` to reuse completed runs with matching config
- First-time runtime is usually dominated by shadow-model training

## 8. FAQ

- **Why do I see `DP-SGD is ignored for uploaded models`?**  
  Uploaded-model audits are inference-only. DP-SGD is a training-time method, so ignoring it here is expected.

- **Why do I get HTTP 200 but no expected defense effect?**  
  HTTP 200 means request processing succeeded, not that every optional defense was applied. Check logs and `model_info`.

- **How do I fix input-size mismatch errors?**  
  Use `input_size=auto` first, or set `data_options.input_size` explicitly to match your model.

## 9. Notes for Integrators

- Core logic lives in `inference_tester/`
- API/UI adapter layer lives in `test_bench/`
- `llm_analyzer/` can be copied and reused as a standalone module

