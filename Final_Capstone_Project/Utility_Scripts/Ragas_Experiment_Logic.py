#################################################################################
# 
# Massachusetts Institute of Technology - xPRO
# RAG and Context Engineering: Designing and Building Production-Grade AI Systems
# 
# Student: Miguel Herrera
# Section: B
# Date and Time: 2026-09-13 18:18:45 -07:00
#
# Description: Reusable RAGAS experiment runner for Checkpoint 4.1. Executes each
#              selected question through the retriever and answer generator, scores
#              it with DiscreteMetric, reports ORIGINAL/PARAPHRASED PASS or FAIL,
#              saves experiment CSV artifacts, and writes summary log entries.
#
#################################################################################

"""Reusable RAGAS experiment orchestration for Checkpoint 4.1."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from ragas import Dataset, experiment


LogFunction = Callable[[str, str], None]


def _format_relative_path(path: Path | str) -> str:
    """Return path relative to the repository parent directory (e.g. MIT_CAPSTONE/...)."""
    p = Path(path).resolve()
    try:
        repo_root = Path(__file__).resolve().parents[2]
        return str(p.relative_to(repo_root.parent))
    except Exception:
        parts = p.parts
        if "MIT_CAPSTONE" in parts:
            idx = parts.index("MIT_CAPSTONE")
            return str(Path(*parts[idx:]))
        return str(p)


def build_experiment(
    retriever: Any,
    judge: Any,
    correctness_metric: Any,
    kind: str = "hybrid",
    label: str = "",
):
    """Create an async RAGAS experiment for one retriever and judge."""
    progress = {"done": 0, "total": 0, "label": label}
    progress["last_original"] = None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "main" if "ORIGINAL" in label.upper() else "paraphrased" if "PARAPHRASE" in label.upper() else label.lower()
    prefix = f"{timestamp}_{tag}" if tag else timestamp

    @experiment(name_prefix=prefix)
    async def run_experiment(row):
        response = retriever.query(row["question"])
        score = correctness_metric.score(
            llm=judge,
            response=response,
            grading_notes=row["grading_notes"],
        )
        progress["done"] += 1
        preview = row["question"][:60]
        if len(row["question"]) > 60:
            preview += "..."
        return {**row, "retriever": kind, "response": response, "score": score.value}

    run_experiment.progress = progress
    return run_experiment


async def evaluate_dataset(
    retriever: Any,
    judge: Any,
    correctness_metric: Any,
    dataset: Dataset,
    label: str,
    ragas_root: str | Path,
    log: LogFunction,
    kind: str = "hybrid",
) -> dict:
    """Run, save, and summarize one RAGAS dataset evaluation."""
    total = len(dataset)
        # print(f"\nEvaluating {label}: {total} questions with the {kind} retriever...")
    experiment_runner = build_experiment(retriever, judge, correctness_metric, kind=kind, label=label)
    experiment_runner.progress["done"] = 0
    experiment_runner.progress["total"] = total
    experiment_runner.progress["label"] = label

    results = await experiment_runner.arun(dataset)
    ordered_results = list(results)
    if "PARAPHRASE" in label.upper():
        ordered_results = sorted(
            ordered_results,
            key=lambda result: (
                result.get("original_order", float("inf")),
                result.get("paraphrase_order", float("inf")),
            ),
        )
    is_paraphrased = "PARAPHRASE" in label.upper()
    last_original = None
    for index, result in enumerate(ordered_results, 1):
        preview = result["question"][:60]
        if len(result["question"]) > 60:
            preview += "..."
        if is_paraphrased:
            original_question = result.get("original_question", "")
            if original_question != last_original and original_question:
                print(f"[ORIGINAL] {original_question}")
                print(f"[GRADING_NOTE] {result.get('original_grading_notes', '')}")
                last_original = original_question
            print(
                f"    [PARAPHRASED {index}] {index}/{total} tested "
                f"({str(result['score']).upper()})  {preview}"
            )
        else:
            print(
                f"[ORIGINAL] {index}/{total} tested "
                f"({str(result['score']).upper()})  {preview}"
            )
    passes = sum(1 for result in ordered_results if result["score"] == "pass")
    result_total = len(ordered_results)
    failures = [result["question"] for result in ordered_results if result["score"] != "pass"]
    results.save()
    csv_path = Path(ragas_root) / "experiments" / f"{results.name}.csv"
    rel_csv_path = _format_relative_path(csv_path)
        # print(f"  {label}: {passes}/{result_total} passed  ->  {csv_path.resolve()}")
    log(f"{label} RESULT", f"{passes}/{result_total} passed; csv={rel_csv_path}")
    return {
        "label": label,
        "passes": passes,
        "total": result_total,
        "failures": failures,
        "csv_path": rel_csv_path,
    }
