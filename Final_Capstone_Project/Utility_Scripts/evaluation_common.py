#################################################################################
# 
# Massachusetts Institute of Technology - xPRO
# RAG and Context Engineering: Designing and Building Production-Grade AI Systems
# 
# Student: Miguel Herrera
# Section: B
# Date and Time: 2026-09-22 19:31:56 -07:00
#
# Description: Shared, checkpoint-agnostic evaluation helpers for the capstone
#              RAG engine: timestamped console logging, the OPENROUTER_API_KEY
#              guard, a RAGAS judge factory (make_judge), a validated JSONL loader,
#              and the DiscreteMetric "correctness" pass/fail grader factory. Lifted
#              out of the Checkpoint 5.1 agent solution so callers share one
#              implementation instead of duplicating it per checkpoint.
#
#################################################################################

"""Reusable RAGAS evaluation helpers shared across capstone checkpoints."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from Final_Capstone_Project.Utility_Scripts.ragas_vertexai_shim import ensure_ragas_vertexai_shim

ensure_ragas_vertexai_shim()
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def console_log(message: str, level: str = "INFO") -> None:
    """Write a timestamped diagnostic message to the console."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}", flush=True)


def require_api_key() -> None:
    """Ensure OPENROUTER_API_KEY is available, loading a .env file if needed."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        load_dotenv()
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit(
            "\n[setup] OPENROUTER_API_KEY is not set.\n"
            "  Create a file named '.env' with: OPENROUTER_API_KEY=sk-or-your-key-here\n"
        )


def make_judge(judge_model: str, base_url: str = OPENROUTER_BASE_URL):
    """Build a RAGAS judge LLM bound to the OpenRouter-compatible client."""
    return llm_factory(
        judge_model,
        client=OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url=base_url),
    )


def build_correctness_metric() -> DiscreteMetric:
    """Return the DiscreteMetric pass/fail correctness grader used by the harness."""
    return DiscreteMetric(
        name="correctness",
        prompt=(
            "You are grading a retrieval-augmented answer against reference grading notes.\n"
            "Return 'pass' if the response is factually consistent with the grading notes and "
            "captures their main point(s) — even if it omits some minor details or is worded "
            "differently. Return 'fail' only if the response contradicts the notes, is "
            "unsupported by them, or misses the central point.\n"
            "Response: {response}\nGrading Notes: {grading_notes}"
        ),
        allowed_values=["pass", "fail"],
    )


def load_jsonl(path: Path) -> list[dict]:
    """Load and validate a JSONL question file (requires question + grading_notes)."""
    records = []
    required_fields = {"question", "grading_notes"}
    with open(path, "r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL record at {path}:{line_number} is not an object")
            missing_fields = required_fields - record.keys()
            if missing_fields:
                missing = ", ".join(sorted(missing_fields))
                raise ValueError(f"JSONL record at {path}:{line_number} is missing: {missing}")
            for field in required_fields:
                if not isinstance(record[field], str) or not record[field].strip():
                    raise ValueError(
                        f"JSONL record at {path}:{line_number} has an empty or non-text '{field}' field"
                    )
            records.append(record)
    return records
