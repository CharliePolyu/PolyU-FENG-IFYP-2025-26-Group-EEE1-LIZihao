# Quick Start Guide (v2)

This guide is for UI users and demo operators who want to run the system quickly.

Note: model-download scripts are not included in this repo. Prepare/export your model separately, then upload `.pth/.pt` in the UI.

## 1) Start the App

```bash
python -m venv venv
venv\Scripts\Activate.ps1          # Windows
# source venv/bin/activate         # macOS/Linux
pip install -r requirements.txt
python -m test_bench.app
```

Open `http://127.0.0.1:5000`.

Optional (for LLM analysis):

```text
DEEPSEEK_API_KEY=your_api_key_here
```

## 2) Baseline Audit Flow

1. (Optional) Upload a model file (`.pth` or `.pt`) in **Target Model File**
   - The UI calls model sniffer automatically and suggests dataset when possible.
2. Select preset mode (`Fast`, `Standard`, `Strict`, or `Custom`)
3. Click **Run Baseline Audit**
4. Watch progress in status + console
5. Review output:
   - Risk score
   - MIA metrics + ROC/AUC
   - PIA metrics
   - Target model health
   - t-SNE chart

## 3) Defense Comparison Flow

1. Run baseline first
2. Defense panel appears
3. Select any defense combination:
   - Label Smoothing (training-time)
   - Top-K Masking (inference-time)
   - DP-SGD (training-time)
4. Click **Run Defense Test**
5. Compare baseline vs defense:
   - Overlaid ROC curves
   - Updated risk score
   - Console logs

## 4) Upload Rules (Important)

- Supported files: `.pth`, `.pt`
- Expected model output: raw logits `(batch_size, num_classes)`
- Avoid saving only state_dict when you want direct upload
- For uploaded models:
  - Top-K masking is supported
  - DP-SGD / Label Smoothing are ignored (training-time methods)

## 5) Risk Score Interpretation

- `0-30`: low risk
- `31-70`: medium risk
- `71-100`: high risk

The score is computed from MIA, generalization gap, and PIA signals.

## 6) Most Common Issues

- **Model load failed**  
  Usually means incompatible file format or architecture mismatch.

- **No expert analysis generated**  
  Check `DEEPSEEK_API_KEY` and network/API availability.

- **Defense panel not visible**  
  Baseline run must complete first.

- **Failed to fetch**  
  Usually backend restart/disconnect. Retry after checking Flask logs.

## 7) Recommended Test Plan

1. Fast baseline (smoke test)
2. Standard baseline (stable metrics)
3. Defense test: Top-K only
4. Defense test: Label + Top-K
5. Defense test: Label + Top-K + DP-SGD

Export the PDF report after final comparison.
