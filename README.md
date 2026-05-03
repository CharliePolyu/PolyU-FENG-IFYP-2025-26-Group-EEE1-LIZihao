
# Inference Tester

Inference Tester is a privacy-audit toolkit for machine learning inference systems.  
It provides a backend audit engine and a demo web UI to evaluate privacy risks using:

- **Membership Inference Attack (MIA)**
- **Property Inference Attack (PIA)**
- **Target model health metrics** (train/test accuracy + generalization gap)
- **Overall privacy risk scoring** (0–100)
- **Optional LLM expert diagnosis** (DeepSeek API, OpenAI-compatible)

---

## Key Features

- Upload and audit custom PyTorch models (`.pth` / `.pt`)
- Built-in mock model path for quick smoke testing
- MIA metrics + ROC/AUC output
- PIA metrics (KL divergence + MAE)
- t-SNE feature-space visualization
- Defense comparison workflow:
  - Top-K Masking (inference-time)
  - Label Smoothing (training-time)
  - DP-SGD (training-time)
- Run progress polling and graceful interruption
- Result caching for repeated configurations
- Optional AI-generated expert report

---

## Project Structure

```text
Inference Tester/
├── inference_tester/         # Core audit engine (MIA/PIA/shadow training)
├── test_bench/               # Flask API + demo UI
│   └── templates/index.html
├── llm_analyzer/             # Standalone LLM analyzer module
├── test_models/              # Optional local test models (.pt)
├── requirements.txt
├── QUICK_START_V2.md
└── README.md
```

---

## Requirements

- Python 3.10+ recommended
- pip / virtualenv

Main dependencies (from `requirements.txt`):

- `torch`
- `torchvision`
- `scikit-learn`
- `Flask`
- `numpy`
- `python-dotenv`
- `openai`
- `opacus` (for DP training path)

---

## Quick Start

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
# source venv/bin/activate

pip install -r requirements.txt
python -m test_bench.app
```

Open:

- `http://127.0.0.1:5000`

---

## Environment Variables

Optional (for LLM expert analysis only):

```text
DEEPSEEK_API_KEY=your_api_key_here
```

If not configured, the audit still runs normally and returns a fallback message for the expert report section.

---

## Usage

### 1) Baseline Audit
- Upload a model (`.pth` / `.pt`) or use mock mode
- Configure preset/custom options
- Run baseline audit
- Review risk score, MIA/PIA metrics, health metrics, and charts

### 2) Defense Comparison
- Run baseline first
- Enable defense options and run defense test
- Compare updated metrics and ROC/risk score changes

---

## Important Notes on Defenses

For **uploaded models**, the pipeline is inference-only.  
Therefore:

- **Top-K Masking** is supported
- **DP-SGD** and **Label Smoothing** are training-time defenses and will be ignored in uploaded-model path

This behavior is expected.

---

## API Overview

### `POST /api/inference/run`
Run a full audit.

Supported request modes:
- `application/json` (mock path)
- `multipart/form-data` (uploaded model path)
  - `model_file`: `.pth` or `.pt`
  - `config`: JSON string

### `GET /api/inference/progress?run_id=<id>`
Poll run status and progress.

### `POST /api/inference/interrupt`
Request graceful stop.

### `POST /api/inference/regenerate_expert_analysis`
Regenerate expert diagnosis from existing metrics.

---

## Uploaded Model Requirements

Uploaded model files should:
- be `.pth` or `.pt`
- contain an executable model object (`nn.Module` or TorchScript), not only `state_dict`
- output logits with shape `(batch_size, num_classes)`

If you see pickle/class resolution issues (e.g. `Can't get attribute ...`), export to TorchScript.

---

## Caching

- Cache directory: `test_bench/cache/`
- Enable via `use_cache=true`
- Reuses completed runs when effective config matches

---

## Troubleshooting

- **`Failed to fetch` in UI**  
  Usually backend restart/disconnect; check Flask terminal logs.

- **Model input/shape mismatch**  
  Try auto input-size detection first, then set explicit input size if needed.

- **No expert analysis**  
  Verify `DEEPSEEK_API_KEY` and network/API availability.

---

## Security & Repository Hygiene

Recommended before publishing:
- Do **not** commit `.env`
- Do **not** commit cache/output folders
- Do **not** commit virtual environments (`venv`, `.venv`)
- Use Git LFS for large model files (`.pt` / `.pth`) if needed

---
