"""Backwards-compatible re-export for the standalone LLM analyzer package.

The concrete implementation lives in the top-level ``llm_analyzer``
package so it can be reused independently of the test bench.
"""

from __future__ import annotations

from llm_analyzer import generate_expert_report

__all__ = ["generate_expert_report"]
