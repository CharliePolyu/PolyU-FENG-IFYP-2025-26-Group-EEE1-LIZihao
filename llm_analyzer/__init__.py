"""Standalone LLM Expert Analyzer package.

Provides reusable expert-diagnosis utilities for audit metrics via
DeepSeek's OpenAI-compatible API.
"""

from __future__ import annotations

from .llm_analyzer import generate_expert_report

__all__ = ["generate_expert_report"]
