from __future__ import annotations

"""Flask API entry for the Inference Tester demo bench.

This module wires HTTP endpoints to the backend audit pipeline. It is
intended for local demo/testing usage and can also serve as a reference
implementation for frontend integration.
"""

import hashlib
import json
import logging
import os
import __main__
import sys
from pathlib import Path
from threading import Event
from dotenv import load_dotenv

load_dotenv()
# Ensure sibling `inference_tester` is importable when this script is run directly.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from threading import Lock
from typing import Any, Dict, List
from uuid import uuid4
from werkzeug.utils import secure_filename

from flask import Flask, jsonify, render_template, request

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn

from inference_tester import AttackCallback, InterruptSignal, run_inference_audit
from inference_tester.exceptions import ModelArchitectureMismatchError
from inference_tester.llm_analyzer import generate_expert_report
from .mock_model import MockModelWrapper
from .dynamic_model import DynamicModelWrapper


class ToyCNN(nn.Module):
    """Compat class for unpickling toy models saved from generate_toy_model.py.

    These files are often saved via `torch.save(model, "...pth")` under `__main__`.
    Registering the same class name in this backend allows torch.load to resolve
    `__main__.ToyCNN` safely for this known architecture.
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.features(x)
        return self.classifier(x)


def _register_known_pickle_classes() -> None:
    """Register known class symbols used by pickle-based .pth uploads."""
    # Most toy scripts save under __main__.ToyCNN; expose that symbol for torch.load().
    if not hasattr(__main__, "ToyCNN"):
        setattr(__main__, "ToyCNN", ToyCNN)


class FlaskAttackCallback(AttackCallback):
    """Simple callback that stores progress logs for the Flask API."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.logs: List[Dict[str, Any]] = []

    def on_progress(self, percentage: float, message: str) -> None:
        pct = float(percentage)
        msg = str(message)
        self.logs.append({"percentage": pct, "message": msg})
        phase = _infer_phase_from_message(msg)
        with _run_progress_lock:
            if self.run_id in _run_progress:
                _run_progress[self.run_id]["percentage"] = pct
                _run_progress[self.run_id]["message"] = msg
                _run_progress[self.run_id]["phase"] = phase


class EventInterruptSignal(InterruptSignal):
    """Interrupt signal backed by a threading.Event."""

    def __init__(self, event: Event) -> None:
        self._event = event

    def is_interrupted(self) -> bool:
        return self._event.is_set()


app = Flask(__name__, template_folder="templates")
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload size
app.config['UPLOAD_FOLDER'] = Path(__file__).resolve().parent / "uploads"
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)

_model_cache: Dict[str, MockModelWrapper] = {}
_model_cache_lock = Lock()
_interrupt_events: Dict[str, Event] = {}
_interrupt_events_lock = Lock()
_run_progress: Dict[str, Dict[str, Any]] = {}
_run_progress_lock = Lock()
_audit_cache_dir = Path(__file__).resolve().parent / "cache"
_audit_cache_lock = Lock()

ALLOWED_EXTENSIONS = {'.pth', '.pt'}

# Resolve runtime device once at startup so wrappers behave consistently.
# Set FORCE_CPU=1 to force CPU execution in unstable local CUDA setups.
_force_cpu_env = str(os.getenv("FORCE_CPU", "0")).strip().lower()
_force_cpu = _force_cpu_env in {"1", "true", "yes", "on"}
_DEFAULT_DEVICE: str = "cpu" if _force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Compute device selected: %s", _DEFAULT_DEVICE.upper())


def _get_runtime_device_info() -> Dict[str, Any]:
    """Return runtime device metadata for frontend and logs."""
    info: Dict[str, Any] = {
        "device": _DEFAULT_DEVICE,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if _DEFAULT_DEVICE != "cuda" or not torch.cuda.is_available():
        return info
    try:
        dev_idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(dev_idx)
        total_vram_gb = round(float(props.total_memory) / (1024 ** 3), 2)
        info.update(
            {
                "cuda_device_index": int(dev_idx),
                "gpu_name": str(props.name),
                "gpu_total_memory_gb": total_vram_gb,
            }
        )
    except Exception as e:
        logger.warning("Failed to query CUDA device properties: %s", e)
    return info


def _compute_audit_cache_key(
    metadata: Dict[str, Any], 
    config: Dict[str, Any],
    use_dp: bool = False,
    use_label_smoothing: bool = False,
    use_top_k_masking: bool = False,
    input_size: int = 32,
) -> str:
    """Compute a deterministic cache key from model + audit config + defenses."""
    canonical = json.dumps(
        {
            "dataset_domain": metadata.get("dataset_domain", "unknown"),
            "data_root": metadata.get("data_root", "./data"),
            "split_seed": metadata.get("split_seed", 42),
            "shadow_k": config.get("shadow", {}).get("k", 5),
            "shadow_epochs": config.get("shadow", {}).get("epochs", 5),
            "probe_mia": config.get("mia", {}).get("target_member_max_batches"),
            "probe_pia": config.get("pia", {}).get("max_batches"),
            "member_ratio": config.get("data", {}).get("member_ratio", 0.5),
            "use_dp": use_dp,
            "label_smoothing": use_label_smoothing,
            "top_k_masking": use_top_k_masking,
            "input_size": input_size,
        },
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:32]


def _error_response(
    code: str,
    message: str,
    status_code: int = 400,
    hints: List[str] | None = None,
) -> Any:
    payload: Dict[str, Any] = {
        "status": "error",
        "code": code,
        "message": message,
    }
    if hints:
        payload["hints"] = hints
    return jsonify(payload), status_code


def _resolve_input_size(raw: Any, default: int = 32) -> int:
    """Resolve expected input size from request payload."""
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    # Keep a bounded safe range; UI currently sends values like 32 or 224.
    return value if 16 <= value <= 1024 else default


def _resolve_requested_input_size(raw: Any) -> int | None:
    """Resolve user requested input size; returns None for auto mode."""
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip().lower() in {"auto", ""}:
        return None
    return _resolve_input_size(raw, default=32)


def _parse_data_options(body: Dict[str, Any], fallback_domain: str = "cifar10") -> Dict[str, Any]:
    """Parse data options from request while keeping backward compatibility."""
    raw = body.get("data_options", {})
    data_opts = raw if isinstance(raw, dict) else {}

    domain_raw = str(
        data_opts.get(
            "dataset_domain",
            body.get("dataset_domain", body.get("target_dataset", "auto")),
        )
    ).strip().lower()
    if domain_raw in {"", "auto"}:
        dataset_domain = fallback_domain
    else:
        dataset_domain = domain_raw

    data_root = str(data_opts.get("data_root", body.get("data_root", "./data")))
    split_seed_raw = data_opts.get("split_seed", body.get("split_seed", 42))
    try:
        split_seed = int(split_seed_raw)
    except (TypeError, ValueError):
        split_seed = 42

    requested_input_size = _resolve_requested_input_size(
        data_opts.get("input_size", body.get("input_size", "auto"))
    )
    strict_validation = bool(data_opts.get("strict_validation", False))
    channels = str(data_opts.get("channels", "auto")).strip().lower()
    num_classes_raw = data_opts.get("num_classes", "auto")
    num_classes_override: int | None = None
    if not (isinstance(num_classes_raw, str) and num_classes_raw.strip().lower() in {"", "auto"}):
        try:
            parsed_classes = int(num_classes_raw)
            if parsed_classes > 0:
                num_classes_override = parsed_classes
        except (TypeError, ValueError):
            num_classes_override = None
    normalization = data_opts.get("normalization", "auto")
    dataset_path = data_opts.get("dataset_path", None)

    return {
        "dataset_domain": dataset_domain,
        "data_root": data_root,
        "split_seed": split_seed,
        "requested_input_size": requested_input_size,
        "strict_validation": strict_validation,
        "channels": channels,
        "num_classes": num_classes_override,
        "num_classes_raw": num_classes_raw,
        "normalization": normalization,
        "dataset_path": dataset_path,
    }


def _is_shape_mismatch_message(message: str) -> bool:
    text = message.lower()
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


def _is_device_mismatch_message(message: str) -> bool:
    text = message.lower()
    indicators = (
        "input type (torch.floattensor) and weight type (torch.cuda.floattensor)",
        "input type (torch.cuda.floattensor) and weight type (torch.floattensor)",
        "expected all tensors to be on the same device",
        "found at least two devices, cpu and cuda",
    )
    return any(flag in text for flag in indicators)


def _infer_input_shape_for_model(
    model: nn.Module,
    device_obj: torch.device,
    preferred_input_size: int | None = None,
) -> tuple[int, int, int]:
    """Infer a compatible square input shape by probing common image sizes."""
    candidates: list[int] = []
    if preferred_input_size is not None:
        candidates.append(_resolve_input_size(preferred_input_size))
    candidates.extend([32, 64, 96, 128, 160, 192, 224, 256])

    # Deduplicate while preserving order.
    seen: set[int] = set()
    ordered_candidates = [x for x in candidates if not (x in seen or seen.add(x))]

    last_runtime_error: RuntimeError | None = None
    for size in ordered_candidates:
        input_shape: tuple[int, int, int] = (3, size, size)
        try:
            with torch.no_grad():
                output = model(torch.randn(1, *input_shape, device=device_obj))
            if isinstance(output, torch.Tensor) and output.dim() == 2:
                return input_shape
        except RuntimeError as e:
            last_runtime_error = e
            if _is_shape_mismatch_message(str(e)):
                continue
            raise

    if last_runtime_error is not None:
        raise ValueError(
            "Model architecture mismatch: "
            f"{last_runtime_error}. Please check your model's expected input dimensions "
            "and number of output classes."
        ) from last_runtime_error

    raise ValueError(
        "Model architecture mismatch: unable to infer a compatible input size "
        "from common candidates (32-256). Please provide a compatible model."
    )


def _load_audit_cache(cache_key: str) -> Dict[str, Any] | None:
    """Load cached audit results from disk. Returns None if miss or error."""
    cache_file = _audit_cache_dir / f"{cache_key}.json"
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _to_json_serializable(obj: object) -> object:
    """Convert numpy types to Python builtins for JSON serialization."""
    try:
        import numpy as np  # noqa: PLC0415
    except ImportError:
        np = None  # type: ignore[assignment]

    if isinstance(obj, dict):
        return {str(k): _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(x) for x in obj]
    if np is not None and isinstance(obj, np.ndarray):
        return obj.astype(float).tolist()
    if np is not None and isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    return obj


def _save_audit_cache(cache_key: str, data: Dict[str, Any]) -> None:
    """Save audit results to disk cache."""
    # Only cache completed runs with meaningful results
    if data.get("status") != "completed":
        logger.info("Cache skip: status=%s (not completed)", data.get("status"))
        print(f"[Cache] Skip: status={data.get('status')} (need completed)", flush=True)
        return
    if not data.get("results", {}).get("mia") and not data.get("results", {}).get("pia"):
        logger.info("Cache skip: no mia/pia results")
        print("[Cache] Skip: no mia/pia results", flush=True)
        return

    _audit_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = _audit_cache_dir / f"{cache_key}.json"
    try:
        serializable = _to_json_serializable(data)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False)
        logger.info("Cache saved: %s", cache_file.resolve())
        print(f"[Cache] Saved to {cache_file.resolve()}", flush=True)
    except (OSError, TypeError) as e:
        logger.warning("Cache save failed: %s", e)
        print(f"[Cache] Save failed: {e}", flush=True)


def _infer_phase_from_message(message: str) -> str:
    msg = message.lower()
    if "shadow model" in msg or "training shadow" in msg:
        return "shadow_training"
    if "membership inference" in msg or "mia" in msg:
        return "mia"
    if "property inference" in msg or "pia" in msg:
        return "pia"
    if "preparing dataset" in msg or "preparing" in msg:
        return "preparing_data"
    if "complete" in msg:
        return "completed"
    if "interrupted" in msg:
        return "interrupted"
    return "running"


@app.route("/", methods=["GET"])
def index() -> str:
    """Serve the main HTML page for running audits."""
    return render_template("index.html", default_device=_DEFAULT_DEVICE.upper())


@app.route("/api/sniff_model", methods=["POST"])
def sniff_model() -> Any:
    """Sniff uploaded model and suggest a default dataset."""
    model_file = request.files.get("model_file")
    if not model_file or model_file.filename == "":
        return _error_response(
            code="MISSING_MODEL_FILE",
            message="No model file provided for sniffing.",
            hints=["Upload a .pth or .pt file in model_file."],
        )

    ext = Path(model_file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return _error_response(
            code="INVALID_MODEL_EXTENSION",
            message="Invalid file type for sniffing. Allowed: .pth, .pt.",
        )

    sniff_id = str(uuid4())
    filename = secure_filename(f"sniff_{sniff_id}_{model_file.filename}")
    upload_path = app.config["UPLOAD_FOLDER"] / filename
    try:
        model_file.save(str(upload_path))
        model = _load_model_for_sniff(upload_path, torch.device("cpu"))
        num_classes = _sniff_model_num_classes(model, torch.device("cpu"))
    except Exception as e:
        logger.warning("Model sniffing failed: %s", e)
        return _error_response(
            code="MODEL_SNIFF_FAILED",
            message=f"Failed to sniff model: {e}",
            hints=[
                "Try exporting to TorchScript (.pt) for maximum compatibility.",
                "Ensure model forward returns logits with shape (batch, num_classes).",
            ],
            status_code=400,
        )
    finally:
        upload_path.unlink(missing_ok=True)

    if num_classes == 10:
        return jsonify(
            {
                "num_classes": 10,
                "suggested_dataset": "cifar10",
                "message": "🤖 Model Sniffed: 10 classes. Auto-selected CIFAR-10.",
            }
        )
    if num_classes == 100:
        return jsonify(
            {
                "num_classes": 100,
                "suggested_dataset": "cifar100",
                "message": "🤖 Model Sniffed: 100 classes. Auto-selected CIFAR-100.",
            }
        )
    return jsonify(
        {
            "num_classes": num_classes,
            "suggested_dataset": "unknown",
            "message": (
                f"⚠️ Sniffed {num_classes} classes. No exact built-in match. "
                "Please select a fallback dataset; the system will use OOD Dimension Adaptation."
            ),
        }
    )


def _load_pytorch_model_from_file(
    file_path: Path,
    device: str = "cpu",
    expected_input_size: int | None = None,
) -> tuple[nn.Module, int, tuple[int, int, int], str, str]:
    """Load user uploaded model (.pth or .pt) with robust fallbacks."""
    device_obj = torch.device(device)
    load_error: Exception | None = None

    # First try TorchScript for .pt files.
    if file_path.suffix.lower() == ".pt":
        try:
            scripted = torch.jit.load(str(file_path), map_location=device_obj)
            scripted.eval()
            model = scripted
            model_format = "torchscript"
        except Exception as e:  # fallback to torch.load
            load_error = e
            model = None
            model_format = "pickle"
    else:
        model = None
        model_format = "pickle"

    # Fallback: standard pickle/full-model load.
    if model is None:
        _register_known_pickle_classes()
        try:
            model = torch.load(file_path, map_location=device_obj, weights_only=False)
        except Exception as e:
            err_text = str(e)
            if "Can't get attribute" in err_text or "ModuleNotFoundError" in err_text:
                raise ValueError(
                    "模型加载失败：该 .pth 文件依赖原始类定义（pickle 限制）。"
                    "建议改用 TorchScript 导出：torch.jit.script(model).save('xxx.pt')，"
                    "或在后端提供对应模型类定义。"
                ) from e
            if load_error is not None:
                logger.warning("TorchScript load also failed: %s", load_error)
            raise ValueError(
                "模型加载失败：文件可能损坏或不是可执行的 PyTorch 模型。"
            ) from e

    if not isinstance(model, nn.Module):
        raise ValueError(
            f"上传文件不是 nn.Module（实际类型: {type(model)}）。"
            "请上传完整模型（torch.save(model, path)）或 TorchScript 模型（.pt）。"
        )

    model.to(device_obj)
    model.eval()
    resolved_device = str(device_obj)

    # Probe common image sizes to infer a compatible input shape.
    try:
        input_shape = _infer_input_shape_for_model(
            model=model,
            device_obj=device_obj,
            preferred_input_size=expected_input_size,
        )
    except RuntimeError as e:
        # Some TorchScript graphs are traced with hardcoded CPU moves.
        # If CUDA probing fails due to device mismatch, safely fallback to CPU.
        if device_obj.type == "cuda" and _is_device_mismatch_message(str(e)):
            logger.warning(
                "Detected device-mismatch TorchScript graph. Falling back to CPU for uploaded model. error=%s",
                e,
            )
            device_obj = torch.device("cpu")
            model.to(device_obj)
            model.eval()
            resolved_device = "cpu"
            input_shape = _infer_input_shape_for_model(
                model=model,
                device_obj=device_obj,
                preferred_input_size=expected_input_size,
            )
        else:
            raise
    num_classes: int | None = None

    # Try inferring class count from the last Linear layer.
    for module in model.modules():
        if isinstance(module, nn.Linear):
            num_classes = int(module.out_features)

    # Verify output shape with a dummy forward pass.
    with torch.no_grad():
        output = model(torch.randn(1, *input_shape, device=device_obj))

    if not isinstance(output, torch.Tensor):
        raise ValueError("模型 forward 输出不是 Tensor。")
    if output.dim() != 2:
        raise ValueError(
            f"模型输出维度应为 2D (batch, classes)，实际为 {tuple(output.shape)}。"
        )

    inferred_classes = int(output.shape[1])
    if num_classes is None or num_classes != inferred_classes:
        num_classes = inferred_classes

    logger.info(
        "Model loaded: classes=%d input_shape=%s device=%s",
        num_classes,
        input_shape,
        resolved_device,
    )
    return model, num_classes, input_shape, model_format, resolved_device


def _extract_logits_tensor(model_output: Any) -> torch.Tensor:
    """Extract a logits tensor from model forward output."""
    if isinstance(model_output, torch.Tensor):
        return model_output
    if isinstance(model_output, (tuple, list)) and model_output:
        first = model_output[0]
        if isinstance(first, torch.Tensor):
            return first
    raise ValueError("Model output is not a tensor/tuple-of-tensor logits.")


def _load_model_for_sniff(file_path: Path, device: torch.device) -> nn.Module:
    """Prefer TorchScript load, fallback to pickle load on CPU."""
    model: nn.Module | None = None
    jit_error: Exception | None = None
    try:
        model = torch.jit.load(str(file_path), map_location=device)
    except Exception as e:
        jit_error = e

    if model is None:
        _register_known_pickle_classes()
        try:
            model = torch.load(file_path, map_location=device, weights_only=False)
        except Exception as e:
            if jit_error is not None:
                logger.warning("Sniffer TorchScript load failed: %s", jit_error)
            raise ValueError(f"Unable to load model for sniffing: {e}") from e

    if not isinstance(model, nn.Module):
        raise ValueError(f"Uploaded object is not nn.Module: {type(model)}")
    model.to(device)
    model.eval()
    return model


def _sniff_model_num_classes(model: nn.Module, device: torch.device) -> int:
    """Probe common input tensors and infer class dimension."""
    probes = [
        torch.randn(1, 3, 224, 224, device=device),
        torch.randn(1, 1, 28, 28, device=device),
    ]
    last_error: Exception | None = None
    for dummy in probes:
        try:
            with torch.no_grad():
                output = model(dummy)
            logits = _extract_logits_tensor(output)
            if logits.dim() != 2:
                raise ValueError(f"Unexpected output shape: {tuple(logits.shape)}")
            return int(logits.shape[1])
        except Exception as e:
            last_error = e
            continue
    raise ValueError(f"Unable to sniff model output dimension: {last_error}")


@app.route("/api/inference/run", methods=["POST"])
def run_test() -> Any:
    """Run a full inference audit on uploaded model or mock model."""
    if 'model_file' in request.files:
        return _run_with_uploaded_model()
    else:
        body = request.get_json(silent=True) or {}
        return _run_with_mock_model(body)


def _run_with_uploaded_model() -> Any:
    """Handle audit with user-uploaded PyTorch model."""
    model_file = request.files.get('model_file')
    
    if not model_file or model_file.filename == '':
        logger.error("No model file in request")
        return _error_response(
            code="MISSING_MODEL_FILE",
            message="No model file provided in request.",
            hints=["Attach a .pth or .pt file in the model upload field."],
        )
    
    logger.info("Received file: %s (size: %d bytes)", model_file.filename, len(model_file.read()))
    model_file.seek(0)
    
    file_ext = Path(model_file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        logger.error("Invalid file extension: %s", file_ext)
        return _error_response(
            code="INVALID_MODEL_EXTENSION",
            message="Invalid file type. Allowed extensions: .pth, .pt.",
        )
    
    try:
        config_json = request.form.get('config', '{}')
        logger.info("Config JSON received: %s", config_json[:200])
        body = json.loads(config_json)
    except json.JSONDecodeError as e:
        logger.exception("JSON parse failed: %s", e)
        return _error_response(
            code="INVALID_CONFIG_JSON",
            message=f"Failed to parse config JSON: {e}",
        )
    data_options = _parse_data_options(body, fallback_domain="cifar10")
    requested_input_size = data_options["requested_input_size"]
    
    run_id = str(body.get("run_id") or uuid4())
    
    with _interrupt_events_lock:
        interrupt_event = _interrupt_events.get(run_id)
        if interrupt_event is None:
            interrupt_event = Event()
            _interrupt_events[run_id] = interrupt_event
        else:
            interrupt_event.clear()
    
    with _run_progress_lock:
        _run_progress[run_id] = {
            "status": "running",
            "phase": "uploading_model",
            "percentage": 0.0,
            "message": f"Uploading and loading model on {_DEFAULT_DEVICE.upper()}",
        }
    logger.info(
        "[Run Start] run_id=%s source=uploaded device=%s",
        run_id,
        _DEFAULT_DEVICE.upper(),
    )
    
    filename = secure_filename(f"{run_id}_{model_file.filename}")
    upload_path = app.config['UPLOAD_FOLDER'] / filename
    
    try:
        model_file.save(str(upload_path))
        logger.info("Model uploaded to %s", upload_path)
    except Exception as e:
        logger.exception("File save failed: %s", e)
        with _interrupt_events_lock:
            _interrupt_events.pop(run_id, None)
        return _error_response(
            code="UPLOAD_SAVE_FAILED",
            message=f"Failed to save uploaded file: {e}",
            status_code=500,
        )
    
    try:
        model_nn, num_classes, input_shape, model_format, resolved_model_device = _load_pytorch_model_from_file(
            upload_path,
            device=_DEFAULT_DEVICE,
            expected_input_size=requested_input_size,
        )
        resolved_input_size = int(input_shape[1])
    except ValueError as e:
        upload_path.unlink(missing_ok=True)
        with _interrupt_events_lock:
            _interrupt_events.pop(run_id, None)
        logger.error("Model load failed for %s: %s", upload_path, e)
        return _error_response(
            code="MODEL_LOAD_FAILED",
            message=str(e),
            hints=[
                "Try exporting TorchScript (.pt) for maximum compatibility.",
                "Ensure model output is logits with shape (batch_size, num_classes).",
            ],
        )

    if data_options["strict_validation"] and data_options["num_classes"] is not None:
        expected_classes = int(data_options["num_classes"])
        if expected_classes != num_classes:
            upload_path.unlink(missing_ok=True)
            with _interrupt_events_lock:
                _interrupt_events.pop(run_id, None)
            return _error_response(
                code="NUM_CLASSES_MISMATCH",
                message=(
                    f"Strict validation failed: model outputs {num_classes} classes, "
                    f"but expected {expected_classes}."
                ),
                hints=["Set num_classes to auto, or upload a matching model."],
                status_code=400,
            )

    # Uploaded-model audits are inference-only in this pipeline.
    use_dp = bool(body.get("use_dp", False))
    defenses = body.get("defenses", {})
    use_label_smoothing = bool(defenses.get("label_smoothing", False))
    use_top_k_masking = bool(defenses.get("top_k_masking", False))

    if use_dp:
        logger.warning("DP-SGD is ignored for uploaded models (training-time defense).")
    if use_label_smoothing:
        logger.warning("Label Smoothing is ignored for uploaded models (training-time defense).")

    target_model = DynamicModelWrapper(
        model=model_nn,
        num_classes=num_classes,
        input_shape=input_shape,
        device=resolved_model_device,
        use_top_k_masking=use_top_k_masking,
        top_k=3,
    )
    
    shadow_models_count = int(body.get("shadow_models_count", 5))
    shadow_models_count = max(1, min(10, shadow_models_count))

    shadow_epochs = int(body.get("shadow_epochs", 5))
    if shadow_epochs not in (1, 5, 10):
        shadow_epochs = 5

    probe_sample_size = str(body.get("probe_sample_size", "full")).lower()
    probe_to_max_batches = {
        "small": 10,
        "medium": 50,
        "full": None,
    }
    max_probe_batches = probe_to_max_batches.get(probe_sample_size, None)

    member_ratio = float(body.get("member_ratio", 0.5))
    member_ratio = max(0.1, min(0.9, member_ratio))

    metadata: Dict[str, Any] = {
        "dataset_domain": data_options["dataset_domain"],
        "data_root": data_options["data_root"],
        "split_seed": data_options["split_seed"],
        "input_size": resolved_input_size,
        "strict_validation": data_options["strict_validation"],
        "channels": data_options["channels"],
        "num_classes": data_options["num_classes"],
        "normalization": data_options["normalization"],
        "dataset_path": data_options["dataset_path"],
    }
    resolved_data_options: Dict[str, Any] = {
        "dataset_domain": metadata["dataset_domain"],
        "data_root": metadata["data_root"],
        "split_seed": metadata["split_seed"],
        "input_size": resolved_input_size,
        "strict_validation": metadata["strict_validation"],
        "channels": metadata["channels"],
        "num_classes": metadata["num_classes"],
        "normalization": metadata["normalization"],
        "dataset_path": metadata["dataset_path"],
    }

    config: Dict[str, Any] = {
        "data": {
            "batch_size": 128,
            "num_workers": 2,
            "member_ratio": member_ratio,
        },
        "interrupt": {
            "check_interval_batches": int(body.get("check_interval_batches", 16)),
            "pia_min_interrupt_batches": int(body.get("pia_min_interrupt_batches", 64)),
        },
        "shadow": {
            "k": shadow_models_count,
            "epochs": shadow_epochs,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "device": None,
        },
        "mia": {
            "target_member_max_batches": max_probe_batches,
            "target_non_member_max_batches": max_probe_batches,
        },
        "pia": {
            "max_batches": max_probe_batches,
        },
    }

    runtime_device_info = _get_runtime_device_info()
    model_info = {
        "source": "uploaded",
        "filename": model_file.filename,
        "format": model_format,
        "num_classes": num_classes,
        "input_shape": list(input_shape),
        "requested_input_size": requested_input_size if requested_input_size is not None else "auto",
        "resolved_input_size": resolved_input_size,
        "dataset_domain": metadata["dataset_domain"],
        "data_root": metadata["data_root"],
        "resolved_model_device": resolved_model_device,
        **runtime_device_info,
        "runtime_device": runtime_device_info.get("device"),
        "device": resolved_model_device,
    }
    use_cache = bool(body.get("use_cache", False))

    audit_cache_key = _compute_audit_cache_key(
        metadata, 
        config,
        use_dp=use_dp,
        use_label_smoothing=use_label_smoothing,
        use_top_k_masking=use_top_k_masking,
        input_size=resolved_input_size,
    )

    if use_cache:
        with _audit_cache_lock:
            cached = _load_audit_cache(audit_cache_key)
        if cached is not None:
            with _run_progress_lock:
                if run_id in _run_progress:
                    _run_progress[run_id]["status"] = "completed"
                    _run_progress[run_id]["phase"] = "completed"
                    _run_progress[run_id]["percentage"] = 100.0
                    _run_progress[run_id]["message"] = "Loaded from cache"
            with _interrupt_events_lock:
                _interrupt_events.pop(run_id, None)
            
            if upload_path:
                upload_path.unlink(missing_ok=True)
            
            cached_logs = [{"percentage": 100.0, "message": "Loaded from cache (same configuration)"}]
            return jsonify({
                "run_id": run_id,
                "logs": cached_logs,
                "model_info": model_info,
                "resolved_data_options": resolved_data_options,
                **cached,
            })

    return _execute_audit(
        target_model,
        metadata,
        config,
        use_dp,
        use_label_smoothing,
        use_top_k_masking,
        audit_cache_key,
        run_id,
        interrupt_event,
        cleanup_path=upload_path,
        model_info=model_info,
        resolved_data_options=resolved_data_options,
    )


def _run_with_mock_model(body: Dict[str, Any]) -> Any:
    """Handle audit with hardcoded mock ResNet-50 model."""
    run_id = str(body.get("run_id") or uuid4())
    with _interrupt_events_lock:
        interrupt_event = _interrupt_events.get(run_id)
        if interrupt_event is None:
            interrupt_event = Event()
            _interrupt_events[run_id] = interrupt_event
        else:
            interrupt_event.clear()
    with _run_progress_lock:
        _run_progress[run_id] = {
            "status": "running",
            "phase": "initializing",
            "percentage": 0.0,
            "message": f"Starting inference audit on {_DEFAULT_DEVICE.upper()}",
        }
    logger.info(
        "[Run Start] run_id=%s source=mock device=%s",
        run_id,
        _DEFAULT_DEVICE.upper(),
    )

    shadow_models_count = int(body.get("shadow_models_count", 5))
    shadow_models_count = max(1, min(10, shadow_models_count))

    shadow_epochs = int(body.get("shadow_epochs", 5))
    if shadow_epochs not in (1, 5, 10):
        shadow_epochs = 5

    probe_sample_size = str(body.get("probe_sample_size", "full")).lower()
    probe_to_max_batches = {
        "small": 10,
        "medium": 50,
        "full": None,
    }
    max_probe_batches = probe_to_max_batches.get(probe_sample_size, None)

    member_ratio = float(body.get("member_ratio", 0.5))
    member_ratio = max(0.1, min(0.9, member_ratio))
    data_options = _parse_data_options(body, fallback_domain="cifar10")
    requested_input_size = data_options["requested_input_size"]
    input_size = _resolve_input_size(requested_input_size, default=32)

    metadata: Dict[str, Any] = {
        "dataset_domain": data_options["dataset_domain"],
        "data_root": data_options["data_root"],
        "split_seed": data_options["split_seed"],
        "input_size": input_size,
        "strict_validation": data_options["strict_validation"],
        "channels": data_options["channels"],
        "num_classes": data_options["num_classes"],
        "normalization": data_options["normalization"],
        "dataset_path": data_options["dataset_path"],
    }
    resolved_data_options: Dict[str, Any] = {
        "dataset_domain": metadata["dataset_domain"],
        "data_root": metadata["data_root"],
        "split_seed": metadata["split_seed"],
        "input_size": input_size,
        "strict_validation": metadata["strict_validation"],
        "channels": metadata["channels"],
        "num_classes": metadata["num_classes"],
        "normalization": metadata["normalization"],
        "dataset_path": metadata["dataset_path"],
    }

    config: Dict[str, Any] = {
        "data": {
            "batch_size": 128,
            "num_workers": 2,
            "member_ratio": member_ratio,
        },
        "interrupt": {
            "check_interval_batches": int(body.get("check_interval_batches", 16)),
            "pia_min_interrupt_batches": int(body.get("pia_min_interrupt_batches", 64)),
        },
        "shadow": {
            "k": shadow_models_count,
            "epochs": shadow_epochs,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "device": None,
        },
        "mia": {
            "target_member_max_batches": max_probe_batches,
            "target_non_member_max_batches": max_probe_batches,
        },
        "pia": {
            "max_batches": max_probe_batches,
        },
    }

    use_cache = bool(body.get("use_cache", False))
    use_dp = bool(body.get("use_dp", False))
    defenses = body.get("defenses", {})
    use_label_smoothing = bool(defenses.get("label_smoothing", False))
    use_top_k_masking = bool(defenses.get("top_k_masking", False))
    audit_cache_key = _compute_audit_cache_key(
        metadata, 
        config,
        use_dp=use_dp,
        use_label_smoothing=use_label_smoothing,
        use_top_k_masking=use_top_k_masking,
        input_size=input_size,
    )

    model_info = {
        "source": "mock",
        "filename": "mock_resnet50",
        "format": "internal",
        "num_classes": 10,
        "input_shape": [3, input_size, input_size],
        "requested_input_size": requested_input_size if requested_input_size is not None else "auto",
        "resolved_input_size": input_size,
        "dataset_domain": metadata["dataset_domain"],
        "data_root": metadata["data_root"],
        **_get_runtime_device_info(),
    }
    if use_cache:
        with _audit_cache_lock:
            cached = _load_audit_cache(audit_cache_key)
        if cached is not None:
            with _run_progress_lock:
                if run_id in _run_progress:
                    _run_progress[run_id]["status"] = "completed"
                    _run_progress[run_id]["phase"] = "completed"
                    _run_progress[run_id]["percentage"] = 100.0
                    _run_progress[run_id]["message"] = "Loaded from cache"
            with _interrupt_events_lock:
                _interrupt_events.pop(run_id, None)
            cached_logs = [{"percentage": 100.0, "message": "Loaded from cache (same configuration)"}]
            return jsonify({
                "run_id": run_id,
                "logs": cached_logs,
                "model_info": model_info,
                "resolved_data_options": resolved_data_options,
                **cached,
            })

    cache_key = (
        f"{metadata.get('dataset_domain', 'unknown')}"
        f"|{metadata.get('data_root', './data')}"
        f"|size={input_size}"
        f"|dp={use_dp}|ls={use_label_smoothing}|topk={use_top_k_masking}"
    )
    with _model_cache_lock:
        model = _model_cache.get(cache_key)
        if model is None:
            model = MockModelWrapper(
                num_classes=10,
                data_root=str(metadata.get("data_root", "./data")),
                train_epochs=1,
                train_subset_size=2000,
                train_batch_size=64,
                learning_rate=1e-4,
                device=_DEFAULT_DEVICE,
                use_dp=use_dp,
                use_label_smoothing=use_label_smoothing,
                use_top_k_masking=use_top_k_masking,
                top_k=3,
            )
            model.fit_if_needed()
            _model_cache[cache_key] = model
    
    return _execute_audit(
        model,
        metadata,
        config,
        use_dp,
        use_label_smoothing,
        use_top_k_masking,
        audit_cache_key,
        run_id,
        interrupt_event,
        cleanup_path=None,
        model_info=model_info,
        resolved_data_options=resolved_data_options,
    )


def _execute_audit(
    target_model: Any,
    metadata: Dict[str, Any],
    config: Dict[str, Any],
    use_dp: bool,
    use_label_smoothing: bool,
    use_top_k_masking: bool,
    audit_cache_key: str,
    run_id: str,
    interrupt_event: Event,
    cleanup_path: Path | None = None,
    model_info: Dict[str, Any] | None = None,
    resolved_data_options: Dict[str, Any] | None = None,
) -> Any:
    """Execute the inference audit pipeline."""
    callback = FlaskAttackCallback(run_id=run_id)
    runtime_info = _get_runtime_device_info()
    device_label = str(runtime_info.get("device", "cpu")).upper()
    gpu_name = runtime_info.get("gpu_name")
    gpu_mem = runtime_info.get("gpu_total_memory_gb")
    if gpu_name and gpu_mem is not None:
        runtime_msg = (
            f"Runtime compute device: {device_label} | GPU: {gpu_name} | VRAM: {gpu_mem} GB"
        )
    else:
        runtime_msg = f"Runtime compute device: {device_label}"
    callback.logs.append(
        {
            "percentage": 0.0,
            "message": runtime_msg,
        }
    )

    try:
        results = run_inference_audit(
            target_model=target_model,
            metadata=metadata,
            config=config,
            callback=callback,
            interrupt_signal=EventInterruptSignal(interrupt_event),
        )
        with _run_progress_lock:
            if run_id in _run_progress:
                _run_progress[run_id]["status"] = (
                    "interrupted" if results.get("status") == "interrupted" else "completed"
                )
                _run_progress[run_id]["phase"] = (
                    "interrupted" if results.get("status") == "interrupted" else "completed"
                )
                _run_progress[run_id]["percentage"] = 100.0
                _run_progress[run_id]["message"] = (
                    f"Interrupted at {results.get('stopped_at', 'unknown')}"
                    if results.get("status") == "interrupted"
                    else "Inference audit complete"
                )
        with _interrupt_events_lock:
            _interrupt_events.pop(run_id, None)

        report_error = None
        try:
            result_metrics = results.get("results", {}) if isinstance(results, dict) else {}
            overall_risk = int(result_metrics.get("risk_score", 0) or 0)

            # Keep compatibility with both old and new payload layouts.
            mia_metrics = result_metrics.get("mia") or result_metrics.get("attacks", {}).get("mia")
            pia_metrics = result_metrics.get("pia") or result_metrics.get("attacks", {}).get("pia")
            health_metrics = result_metrics.get("target_model_health")
            roc_metrics = result_metrics.get("mia_charts", {}).get("roc_curve")

            expert_report = generate_expert_report(
                "Privacy Leakage (MIA & PIA)",
                {
                    "mia": mia_metrics,
                    "pia": pia_metrics,
                    "target_model_health": health_metrics,
                    "mia_roc_curve": roc_metrics,
                    "risk_score": overall_risk,
                },
                use_dp=use_dp,
                risk_score=overall_risk,
            )
            results["llm_analysis"] = {"expert_diagnosis": expert_report}
        except Exception as e:
            logger.exception("Failed to generate LLM report: %s", e)
            report_error = str(e)
            results["llm_analysis"] = {
                "expert_diagnosis": f"[LLM Error] Failed to generate expert analysis: {e}"
            }

        if not results.get("_interrupted") and results.get("status") != "interrupted":
            with _audit_cache_lock:
                _save_audit_cache(audit_cache_key, {
                    "status": "completed",
                    "logs": callback.logs,
                    "results": results["results"],
                    "llm_analysis": results.get("llm_analysis"),
                })
        
        if cleanup_path:
            cleanup_path.unlink(missing_ok=True)
            logger.info("Cleaned up uploaded model: %s", cleanup_path)

        return jsonify({
            "run_id": run_id,
            "logs": callback.logs,
            "results": results["results"],
            "llm_analysis": results.get("llm_analysis"),
            "report_error": report_error,
            "model_info": model_info,
            "resolved_data_options": resolved_data_options,
        })
    except ModelArchitectureMismatchError as e:
        with _run_progress_lock:
            if run_id in _run_progress:
                _run_progress[run_id]["status"] = "error"
                _run_progress[run_id]["phase"] = "error"
                _run_progress[run_id]["message"] = str(e)
        with _interrupt_events_lock:
            _interrupt_events.pop(run_id, None)
        if cleanup_path:
            cleanup_path.unlink(missing_ok=True)
        logger.warning("Model architecture mismatch during run %s: %s", run_id, e)
        return _error_response(
            code="MODEL_INPUT_MISMATCH",
            message=str(e),
            hints=[
                "Use Auto mode or update data_options.input_size.",
                "Verify model output classes match dataset labels.",
                "For ImageNet-class models (1000 classes), use a matching dataset/domain instead of CIFAR-10.",
            ],
            status_code=400,
        )
    except RuntimeError as e:
        if _is_shape_mismatch_message(str(e)):
            message = (
                "Model architecture mismatch: "
                f"{e}. Please check your model's expected input dimensions and number of output classes."
            )
            with _run_progress_lock:
                if run_id in _run_progress:
                    _run_progress[run_id]["status"] = "error"
                    _run_progress[run_id]["phase"] = "error"
                    _run_progress[run_id]["message"] = message
            with _interrupt_events_lock:
                _interrupt_events.pop(run_id, None)
            if cleanup_path:
                cleanup_path.unlink(missing_ok=True)
            logger.warning("Runtime shape mismatch during run %s: %s", run_id, e)
            return _error_response(
                code="MODEL_INPUT_MISMATCH",
                message=message,
                hints=[
                    "Try Auto input-size detection (recommended).",
                    "Check if uploaded model was trained for 224x224 or another fixed size.",
                ],
                status_code=400,
            )

        with _run_progress_lock:
            if run_id in _run_progress:
                _run_progress[run_id]["status"] = "error"
                _run_progress[run_id]["message"] = str(e)
        with _interrupt_events_lock:
            _interrupt_events.pop(run_id, None)
        if cleanup_path:
            cleanup_path.unlink(missing_ok=True)
        logger.exception("Runtime error during inference run: %s", e)
        return _error_response(
            code="RUNTIME_ERROR",
            message=str(e),
            status_code=500,
        )
    except Exception as e:
        with _run_progress_lock:
            if run_id in _run_progress:
                _run_progress[run_id]["status"] = "error"
                _run_progress[run_id]["message"] = str(e)
        with _interrupt_events_lock:
            _interrupt_events.pop(run_id, None)
        
        if cleanup_path:
            cleanup_path.unlink(missing_ok=True)
        
        logger.exception("Error during inference run: %s", e)
        return _error_response(
            code="UNHANDLED_ERROR",
            message=str(e),
            status_code=500,
        )


@app.route("/api/inference/regenerate_expert_analysis", methods=["POST"])
def regenerate_expert_analysis() -> Any:
    """Regenerate AI expert diagnosis from existing MIA/PIA metrics."""
    body = request.get_json(silent=True) or {}
    mia_metrics = body.get("mia_metrics")
    pia_metrics = body.get("pia_metrics")
    if mia_metrics is None and pia_metrics is None:
        return jsonify({"ok": False, "message": "Missing mia_metrics and pia_metrics"}), 400

    use_dp = bool(body.get("use_dp", False))
    risk_score = int(body.get("risk_score", 0))
    expert_text = generate_expert_report(
        "Privacy Leakage (MIA & PIA)",
        {"mia_metrics": mia_metrics, "pia_metrics": pia_metrics},
        use_dp=use_dp,
        risk_score=risk_score,
    )
    return jsonify({"ok": True, "expert_analysis": expert_text})


@app.route("/api/inference/interrupt", methods=["POST"])
def interrupt_run() -> Any:
    """Request graceful interruption for a running audit."""
    body = request.get_json(silent=True) or {}
    run_id = str(body.get("run_id", "")).strip()
    if not run_id:
        return jsonify({"ok": False, "message": "Missing run_id"}), 400

    with _interrupt_events_lock:
        event = _interrupt_events.get(run_id)
        if event is None:
            return jsonify({"ok": False, "message": "run_id not found"}), 404
        event.set()
    with _run_progress_lock:
        if run_id in _run_progress:
            _run_progress[run_id]["status"] = "stopping"
            _run_progress[run_id]["message"] = "Stop requested; waiting next checkpoint"
            _run_progress[run_id]["phase"] = "stopping"
    return jsonify({"ok": True, "message": "Interrupt requested", "run_id": run_id})


@app.route("/api/inference/progress", methods=["GET"])
def get_progress() -> Any:
    """Get current progress for a run_id."""
    run_id = str(request.args.get("run_id", "")).strip()
    if not run_id:
        return jsonify({"ok": False, "message": "Missing run_id"}), 400
    with _run_progress_lock:
        progress = _run_progress.get(run_id)
        if progress is None:
            # Frontend may poll before /run registers run_id, or after backend restart.
            # Return a soft "initializing" state instead of 404 to avoid noisy errors.
            return jsonify(
                {
                    "ok": True,
                    "run_id": run_id,
                    "status": "initializing",
                    "phase": "initializing",
                    "percentage": 0.0,
                    "message": "Run not registered yet or backend restarted.",
                }
            )
        return jsonify({"ok": True, "run_id": run_id, **progress})


if __name__ == "__main__":
    # Keep long-running audits stable by default; disable reloader unless enabled.
    _debug_env = str(os.getenv("FLASK_DEBUG", "0")).strip().lower()
    _reloader_env = str(os.getenv("FLASK_USE_RELOADER", "0")).strip().lower()
    debug_mode = _debug_env in {"1", "true", "yes", "on"}
    use_reloader = _reloader_env in {"1", "true", "yes", "on"}
    host = str(os.getenv("HOST", "127.0.0.1"))
    port = int(os.getenv("PORT", "5000"))
    app.run(debug=debug_mode, use_reloader=use_reloader, threaded=True, host=host, port=port)

