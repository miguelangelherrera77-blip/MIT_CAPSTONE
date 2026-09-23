#################################################################################
# 
# Massachusetts Institute of Technology - xPRO
# RAG and Context Engineering: Designing and Building Production-Grade AI Systems
# 
# Student: Miguel Herrera
# Section: B
# Date and Time: 2026-09-22 19:31:56 -07:00
#
# Description: Shared dataset assembly for the capstone RAGAS harness. Resolves the
#              originals/paraphrases JSONL sources (LLM- or manually-generated) and
#              builds two RAGAS datasets, returning the resolved paths and the
#              per-original paraphrase coverage alongside the datasets (instead of
#              mutating module globals). Lifted out of the Checkpoint 5.1 ReAct agent
#              solution so callers share one implementation.
#
#################################################################################

"""Reusable originals/paraphrases dataset loading for the RAGAS harness."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

from Final_Capstone_Project.Utility_Scripts.evaluation_common import console_log, load_jsonl
from Final_Capstone_Project.Utility_Scripts.ragas_vertexai_shim import ensure_ragas_vertexai_shim

ensure_ragas_vertexai_shim()
from ragas import Dataset


@dataclass
class SplitDatasets:
    """Result of load_split_datasets: the two datasets plus provenance metadata."""

    originals: Dataset
    paraphrases: Dataset
    originals_path: Path
    variants_path: Path
    paraphrase_counts: dict[str, int] = field(default_factory=dict)


def resolve_dataset_paths(
    test_variables_dir: Path,
    main_question_source: str = "llm",
) -> tuple[Path, Path]:
    """Resolve the originals/variants datasets for the requested main-question source."""
    source = str(main_question_source).lower()
    if source not in {"llm", "manual"}:
        raise ValueError("main_question_source must be 'llm' or 'manual'.")

    test_variables_dir = Path(test_variables_dir)
    originals_path = test_variables_dir / "Test_Questions_LLM_Generated.jsonl"
    manual_originals_path = test_variables_dir / "Test_Questions_Manually_Generated.jsonl"
    variants_path = test_variables_dir / "Test_Questions_LLM_Paraphrased_Evaluation.jsonl"
    manual_paraphrased_evaluation_path = (
        test_variables_dir / "Test_Questions_Manually_Paraphrased_Evaluation.jsonl"
    )

    if source == "manual":
        original_candidates = [
            manual_originals_path,
            test_variables_dir / "Backup" / "Test_Questions_Manually_Generated.jsonl",
        ]
    else:
        original_candidates = [
            originals_path,
            test_variables_dir / "Backup" / "Test_Main_Questions_LLM_Generated.jsonl",
        ]
    orig = next((path for path in original_candidates if path.is_file()), original_candidates[0])

    if source == "manual":
        variant_candidates = [
            manual_paraphrased_evaluation_path,
            test_variables_dir / "Test_Questions_Manually_Paraphrased.jsonl",
        ]
    else:
        variant_candidates = [
            variants_path,
            test_variables_dir / "Test_Questions_LLM_Paraphrased_Evaluation.jsonl",
        ]
    var = next((path for path in variant_candidates if path.is_file()), variant_candidates[0])

    return orig, var


def load_split_datasets(
    test_variables_dir: Path,
    ragas_root: str | Path,
    number_questions: int | None = None,
    random_mode: str = "N",
    main_question_source: str = "llm",
) -> SplitDatasets:
    """Build two RAGAS datasets from the resolved originals and variants files."""
    if number_questions is not None and (
        isinstance(number_questions, bool) or not isinstance(number_questions, int)
    ):
        raise ValueError("number_questions must be an integer or None.")

    random_mode = str(random_mode).upper()
    if random_mode not in {"Y", "N"}:
        raise ValueError("random_mode must be either 'Y' or 'N'.")

    orig_path, var_path = resolve_dataset_paths(test_variables_dir, main_question_source)
    if not orig_path.is_file():
        console_log(f"Originals file does not exist: {orig_path}", "ERROR")
        raise FileNotFoundError(f"Originals dataset not found at {orig_path}")
    if not var_path.is_file():
        console_log(f"Variants file does not exist: {var_path}", "ERROR")
        raise FileNotFoundError(f"Variants dataset not found at {var_path}")

    originals = load_jsonl(orig_path)
    original_questions = {o["question"].strip() for o in originals}
    combined = load_jsonl(var_path)
    paraphrases = [r for r in combined if r["question"].strip() not in original_questions]
    if not paraphrases:
        paraphrases = combined
    max_questions = len(originals)
    if number_questions is not None and number_questions <= 0:
        console_log(f"Invalid question limit: {number_questions}.", "ERROR")
        raise ValueError("--number_questions must be a positive integer.")
    if number_questions is not None and number_questions > max_questions:
        console_log(
            f"Requested {number_questions} original questions, but only {max_questions} are available.",
            "ERROR",
        )
        raise ValueError(
            f"--number_questions={number_questions} exceeds the available original questions "
            f"(max available: {max_questions})."
        )

    def _group_key(row: dict) -> tuple[str, tuple[str, ...]]:
        return row.get("grading_notes", ""), tuple(row.get("sources", []))

    paraphrases_by_original = {}
    for row in paraphrases:
        paraphrases_by_original.setdefault(_group_key(row), []).append(row)

    if random_mode == "Y":
        selected_originals = random.sample(originals, number_questions) if number_questions is not None else originals
    else:
        selected_originals = originals[:number_questions] if number_questions is not None else originals

    selected_paraphrases = []
    for original_order, original in enumerate(selected_originals):
        original_question = original["question"]
        for paraphrase_order, paraphrase in enumerate(
            paraphrases_by_original.get(_group_key(original), [])
        ):
            selected_paraphrases.append({
                **paraphrase,
                "original_question": original_question,
                "original_grading_notes": original["grading_notes"],
                "original_order": original_order,
                "paraphrase_order": paraphrase_order,
            })

    original_by_key = {
        (row.get("grading_notes", ""), tuple(row.get("sources", []))): row["question"]
        for row in selected_originals
    }
    counts_by_original: dict[str, int] = {question: 0 for question in original_by_key.values()}
    for row in selected_paraphrases:
        key = (row.get("grading_notes", ""), tuple(row.get("sources", [])))
        original_question = original_by_key.get(key)
        if original_question is not None:
            counts_by_original[original_question] += 1

    def _to_dataset(name: str, rows: list[dict]) -> Dataset:
        ds = Dataset(name=name, backend="local/csv", root_dir=str(ragas_root))
        for r in rows:
            dataset_row = {
                "question": r["question"],
                "grading_notes": r["grading_notes"],
            }
            if r.get("original_question"):
                dataset_row["original_question"] = r["original_question"]
            if r.get("original_grading_notes"):
                dataset_row["original_grading_notes"] = r["original_grading_notes"]
            if "original_order" in r:
                dataset_row["original_order"] = r["original_order"]
            if "paraphrase_order" in r:
                dataset_row["paraphrase_order"] = r["paraphrase_order"]
            if "evaluation_category" in r:
                dataset_row["evaluation_category"] = r["evaluation_category"]
            ds.append(dataset_row)
        ds.save()
        return ds

    return SplitDatasets(
        originals=_to_dataset("wiki_eval_originals", selected_originals),
        paraphrases=_to_dataset("wiki_eval_paraphrases", selected_paraphrases),
        originals_path=orig_path,
        variants_path=var_path,
        paraphrase_counts=counts_by_original,
    )
