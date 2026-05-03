from __future__ import annotations

from typing import Any, Dict


class AuditInterrupted(Exception):
    """Raised when an inference audit is gracefully interrupted.

    The exception carries:
    - `stage`: where interruption happened.
    - `partial_results`: best-effort partial metrics/results computed so far.
    """

    def __init__(self, stage: str, partial_results: Dict[str, Any]) -> None:
        super().__init__(f"Inference audit interrupted at stage: {stage}")
        self.stage = stage
        self.partial_results = partial_results


class ModelArchitectureMismatchError(Exception):
    """Raised when target-model input/output shape mismatches audit data."""

    def __init__(self, original_error: str) -> None:
        message = (
            "Model architecture mismatch: "
            f"{original_error}. Please check your model's expected input dimensions "
            "and number of output classes."
        )
        super().__init__(message)
        self.original_error = original_error

