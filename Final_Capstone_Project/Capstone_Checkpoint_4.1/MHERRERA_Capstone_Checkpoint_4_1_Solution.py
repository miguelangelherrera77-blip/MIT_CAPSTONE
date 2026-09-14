#################################################################################
# 
# Massachusetts Institute of Technology - xPRO
# RAG and Context Engineering: Designing and Building Production-Grade AI Systems
# 
# Student: Miguel Herrera
# Section: B
# Date and Time: 2026-09-13 18:18:45 -07:00
#
# Description: Interactive Checkpoint 4.1 evaluation harness for the Wikipedia RAG
#              engine. Selects manual or LLM-generated original/paraphrase datasets,
#              runs lexical BM25, semantic Chroma, graph, hybrid, or combined retrieval,
#              evaluates generated answers with RAGAS DiscreteMetric, and records results.
#
#################################################################################

r"""Capstone Checkpoint 4.1 — Evaluation Infrastructure and Baseline Diagnosis (SOLUTION).
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code / PyCharm / Jupytext.

This solution evaluates the Wikipedia Retrieval Engine capstone system with RAGAS
(an LLM-judge DiscreteMetric, exactly as in Lab 3.1), using configurable retrieval
methods (Lexical BM25, Semantic Chroma Vector, Graph-augmented, or Hybrid) over the
Wikipedia corpus. It supports evaluating:

    * ORIGINALS   -> Main grounded evaluation questions
    * PARAPHRASES -> Derived query variants reusing ground-truth notes
    * BOTH        -> Side-by-side comparison with Delta and robustness verdict
"""

# %% [markdown]
# # Capstone Checkpoint 4.1 — RAGAS Evaluation: Originals vs Paraphrases
# **MO-LLM Module 3 / Required Capstone Checkpoint**
#
# Uses RAGAS `DiscreteMetric` + `experiment()` (as in Lab 3.1) with the Checkpoint 2.1
# hybrid retriever. Evaluates two datasets separately and compares pass rates.

# %% [markdown]
# ## Setup
# 1. Python 3.11 or 3.12.
# 2. pip install -r requirements.txt (ragas, langchain-openai, langchain-chroma, rank-bm25, python-dotenv, openai).
# 3. Use the OpenRouter API key provided for this program.
# 4. Create a `.env` file with:  OPENROUTER_API_KEY=sk-or-v1-...

# %%
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import os
import configparser
import json
import asyncio
import argparse
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Final_Capstone_Project.Retrieval_Methods.bm25_retrieval import load_wikipedia_chunks
from Final_Capstone_Project.Retrieval_Methods.bm25_retrieval import BM25_CANDIDATES
from Final_Capstone_Project.Retrieval_Methods.hybrid_retrieval import HybridRetriever, get_embeddings
from Final_Capstone_Project.Retrieval_Methods.vector_retrieval import VECTOR_CANDIDATES
from Final_Capstone_Project.Ranking_Techniques.weighted_fusion_ranking import (
    FUSED_TOP_K,
    WEIGHT_BM25,
    WEIGHT_VECTOR,
)
from Final_Capstone_Project.Utility_Scripts.ragas_vertexai_shim import ensure_ragas_vertexai_shim

ensure_ragas_vertexai_shim()
from ragas import Dataset
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric

from Final_Capstone_Project.Utility_Scripts.Ragas_Experiment_Logic import evaluate_dataset

# %%
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_config = configparser.ConfigParser()
_config.read(Path(__file__).resolve().parents[1] / "retrieval.conf")
LLM_MODEL = _config.get("llm", "model", fallback="openai/gpt-5.4-mini")
EMBEDDING_MODEL = _config.get("embedding", "model", fallback="openai/text-embedding-3-small")
JUDGE_MODEL = _config.get("llm", "judge_model", fallback=LLM_MODEL)
TEMPERATURE = _config.getfloat("llm", "temperature", fallback=0.2)
MAX_TOKENS = _config.getint("llm", "max_tokens", fallback=1024)


# ── paths ────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = Path(__file__).resolve().parent
FINAL_CAPSTONE_DIR = CHECKPOINT_DIR.parent
CHROMA_DIR = str(FINAL_CAPSTONE_DIR / "Capstone_Database" / "Capstone_Chroma_DB")

TEST_VARIABLES_DIR = FINAL_CAPSTONE_DIR / "Test_Variables"
ORIGINALS_PATH = TEST_VARIABLES_DIR / "Test_Questions_LLM_Generated.jsonl"
MANUAL_ORIGINALS_PATH = TEST_VARIABLES_DIR / "Test_Questions_Manually_Generated.jsonl"
VARIANTS_PATH = TEST_VARIABLES_DIR / "Test_Questions_LLM_Paraphrased_Evaluation.jsonl"
MANUAL_PARAPHRASED_EVALUATION_PATH = (
    TEST_VARIABLES_DIR / "Test_Questions_Manually_Paraphrased_Evaluation.jsonl"
)

RAGAS_ROOT = str(FINAL_CAPSTONE_DIR / "Ragas_Experiments")
LOG_PATH = CHECKPOINT_DIR / "checkpoint_4_1_evaluation.log"
TEST_RESULTS_LOG = Path(RAGAS_ROOT) / "detailed_test_results.log"
CURRENT_ORIGINALS_PATH: Path | None = None
CURRENT_VARIANTS_PATH: Path | None = None
CURRENT_PARAPHRASE_COUNTS: dict[str, int] = {}


def resolve_dataset_paths(main_question_source: str = "llm") -> tuple[Path, Path]:
    """Resolve datasets using the requested source for main questions."""
    source = str(main_question_source).lower()
    if source not in {"llm", "manual"}:
        raise ValueError("main_question_source must be 'llm' or 'manual'.")

    if source == "manual":
        original_candidates = [
            MANUAL_ORIGINALS_PATH,
            TEST_VARIABLES_DIR / "Backup" / "Test_Questions_Manually_Generated.jsonl",
        ]
    else:
        original_candidates = [
            ORIGINALS_PATH,
            TEST_VARIABLES_DIR / "Backup" / "Test_Main_Questions_LLM_Generated.jsonl",
        ]
    orig = next((path for path in original_candidates if path.is_file()), original_candidates[0])

    if source == "manual":
        variant_candidates = [
            MANUAL_PARAPHRASED_EVALUATION_PATH,
            TEST_VARIABLES_DIR / "Test_Questions_Manually_Paraphrased.jsonl",
        ]
    else:
        variant_candidates = [
            VARIANTS_PATH,
            TEST_VARIABLES_DIR / "Test_Questions_LLM_Paraphrased_Evaluation.jsonl",
        ]
    var = next((path for path in variant_candidates if path.is_file()), variant_candidates[0])

    return orig, var

# %%
def console_log(message: str, level: str = "INFO") -> None:
    """Write a timestamped diagnostic message to the console."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}", flush=True)


def ensure_workspace_setup() -> None:
    """Run the full shared Setup.py bootstrap before any menu or evaluation work begins."""
    setup_script = Path(__file__).resolve().parents[1] / "Utility_Scripts" / "Setup.py"
    if not setup_script.is_file():
        console_log(f"Setup script not found at {setup_script}; continuing without bootstrap.", "WARNING")
        return

    console_log(f"Running full workspace bootstrap: {setup_script.name} --build")
    try:
        subprocess.run(
            [sys.executable, str(setup_script), "--build"],
            check=True,
            cwd=str(setup_script.parent.parent),
            capture_output=False,
        )
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Workspace bootstrap failed with exit code {exc.returncode}: {setup_script} --build") from exc


def require_api_key() -> None:
    if not os.environ.get("OPENROUTER_API_KEY"):
        load_dotenv()
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit(
            "\n[setup] OPENROUTER_API_KEY is not set.\n"
            "  Create a file named '.env' with: OPENROUTER_API_KEY=sk-or-your-key-here\n"
        )


def log(label: str, text: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {label}\n{text}\n{'-' * 72}\n")


def prepend_test_results(entry: str) -> None:
    """Place the newest detailed test result before older sessions."""
    existing = TEST_RESULTS_LOG.read_text(encoding="utf-8") if TEST_RESULTS_LOG.is_file() else ""
    TEST_RESULTS_LOG.write_text(entry + existing, encoding="utf-8")


# %% [markdown]
# ## RAGAS evaluation harness (DiscreteMetric judge — same idea as Lab 3.1)

# %%
def make_judge():
    return llm_factory(
        JUDGE_MODEL,
        client=OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url=OPENROUTER_BASE_URL),
    )


judge = None  # built in main() after the API-key check

correctness_metric = DiscreteMetric(
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


# %% [markdown]
# ## Datasets — originals vs paraphrases (Step 3)

# %%
def _load_jsonl(path: Path) -> list[dict]:
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


def load_split_datasets(
    number_questions: int | None = None,
    random_mode: str = "N",
    main_question_source: str = "llm",
) -> tuple[Dataset, Dataset]:
    """Build two RAGAS datasets from the resolved originals and variants files."""
    if number_questions is not None and (
        isinstance(number_questions, bool) or not isinstance(number_questions, int)
    ):
        raise ValueError("number_questions must be an integer or None.")

    random_mode = str(random_mode).upper()
    if random_mode not in {"Y", "N"}:
        raise ValueError("random_mode must be either 'Y' or 'N'.")

    orig_path, var_path = resolve_dataset_paths(main_question_source)
    if not orig_path.is_file():
        console_log(f"Originals file does not exist: {orig_path}", "ERROR")
        raise FileNotFoundError(f"Originals dataset not found at {orig_path}")
    if not var_path.is_file():
        console_log(f"Variants file does not exist: {var_path}", "ERROR")
        raise FileNotFoundError(f"Variants dataset not found at {var_path}")

    originals = _load_jsonl(orig_path)
    original_questions = {o["question"].strip() for o in originals}
    combined = _load_jsonl(var_path)
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

    global CURRENT_ORIGINALS_PATH, CURRENT_VARIANTS_PATH, CURRENT_PARAPHRASE_COUNTS
    CURRENT_ORIGINALS_PATH = orig_path
    CURRENT_VARIANTS_PATH = var_path
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
    CURRENT_PARAPHRASE_COUNTS = counts_by_original

    def _to_dataset(name: str, rows: list[dict]) -> Dataset:
        ds = Dataset(name=name, backend="local/csv", root_dir=RAGAS_ROOT)
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
            ds.append(dataset_row)
        ds.save()
        return ds
    return (
        _to_dataset("wiki_eval_originals", selected_originals),
        _to_dataset("wiki_eval_paraphrases", selected_paraphrases),
    )


# %% [markdown]
# ## Experiment (Step 4)

def append_single_test_results(
    dataset_name: str,
    results: dict,
    search_method: str,
    main_question_source: str,
) -> None:
    """Append a structured log entry when evaluating a single dataset (main or paraphrased)."""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    def _rate(d: dict) -> str:
        return f"{d['passes']}/{d['total']} ({(d['passes'] / d['total'] * 100) if d['total'] else 0:.0f}%)"

    def _fail_block(d: dict) -> str:
        if not d["failures"]:
            return "  Failures    : none\n"
        lines = ["  Failures    : {} question(s) FAILED".format(len(d["failures"]))]
        for i, q in enumerate(d["failures"], 1):
            preview = q if len(q) <= 110 else q[:107] + "..."
            lines.append(f"    {i}. {preview}")
        return "\n".join(lines) + "\n"

    entry = []
    entry.append("=" * 80)
    entry.append(f"TEST SESSION  |  {date_str} {time_str}")
    entry.append("=" * 80)
    entry.append(
        f"Test type   : RAGAS correctness evaluation ({search_method.capitalize()} search; "
        f"{main_question_source}-generated source)"
    )
    entry.append(f"Date        : {date_str}")
    entry.append(f"Time        : {time_str}")
    entry.append(
        f"Config      : Search={search_method.capitalize()}, "
        f"candidates=BM25:{BM25_CANDIDATES}/Vector:{VECTOR_CANDIDATES}, Top-K={FUSED_TOP_K}, "
        f"answer={LLM_MODEL}, judge={JUDGE_MODEL}, metric=RAGAS DiscreteMetric")
    entry.append("")
    entry.append("-" * 80)
    entry.append(f"EVALUATION DATASET: {dataset_name.upper()}")
    entry.append("-" * 80)
    entry.append(f"  Result      : {_rate(results)} PASSED")
    entry.append(f"  Output CSV  : {results['csv_path']}")
    entry.append(f"  Dataset source: {main_question_source}-generated questions")
    entry.append(_fail_block(results).rstrip("\n"))
    if dataset_name.lower().startswith("paraphrased"):
        entry.append("")
        entry.append("PARAPHRASE COVERAGE PER ORIGINAL")
        entry.append("-" * 80)
        for original_question, count in CURRENT_PARAPHRASE_COUNTS.items():
            preview = original_question if len(original_question) <= 110 else original_question[:107] + "..."
            entry.append(f"  {count} paraphrased question(s) tested for original: {preview}")
    entry.append("=" * 80)
    entry.append("")

    prepend_test_results("\n".join(entry) + "\n")
    print(f"Test results appended to: {TEST_RESULTS_LOG}")


def append_test_results(
    originals: dict,
    paraphrases: dict,
    delta: float,
    verdict: str,
    search_method: str = "hybrid",
    main_question_source: str = "llm",
) -> None:
    """Append a structured, medium-detail entry to test_results.log for each run."""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    def _rate(d: dict) -> str:
        return f"{d['passes']}/{d['total']} ({(d['passes'] / d['total'] * 100) if d['total'] else 0:.0f}%)"

    def _fail_block(d: dict) -> str:
        if not d["failures"]:
            return "  Failures    : none\n"
        lines = ["  Failures    : {} question(s) FAILED".format(len(d["failures"]))]
        for i, q in enumerate(d["failures"], 1):
            preview = q if len(q) <= 110 else q[:107] + "..."
            lines.append(f"    {i}. {preview}")
        return "\n".join(lines) + "\n"

    orig_path = CURRENT_ORIGINALS_PATH or resolve_dataset_paths(main_question_source)[0]
    var_path = CURRENT_VARIANTS_PATH or resolve_dataset_paths(main_question_source)[1]
    entry = []
    entry.append("=" * 80)
    entry.append(f"TEST SESSION  |  {date_str} {time_str}")
    entry.append("=" * 80)
    entry.append(
        f"Test type   : Side-by-side RAGAS correctness evaluation ({search_method.capitalize()} search; "
        f"{main_question_source}-generated source)"
    )
    entry.append(f"Date        : {date_str}")
    entry.append(f"Time        : {time_str}")
    entry.append(
        f"Config      : Search={search_method.capitalize()}, "
        f"candidates=BM25:{BM25_CANDIDATES}/Vector:{VECTOR_CANDIDATES}, Top-K={FUSED_TOP_K}, "
        f"answer={LLM_MODEL}, judge={JUDGE_MODEL}, metric=RAGAS DiscreteMetric")
    entry.append("")
    entry.append("-" * 80)
    entry.append("ORIGINAL QUESTIONS")
    entry.append("-" * 80)
    entry.append(f"  Dataset     : {orig_path.name}")
    entry.append(f"  Dataset source: {main_question_source}-generated Main Questions")
    entry.append(f"  Result      : {_rate(originals)} PASSED")
    entry.append(f"  Output CSV  : {originals['csv_path']}")
    entry.append(_fail_block(originals).rstrip("\n"))
    entry.append("")
    entry.append("-" * 80)
    entry.append("PARAPHRASED QUESTIONS")
    entry.append("-" * 80)
    entry.append(f"  Dataset     : {var_path.name} (paraphrase rows)")
    entry.append(f"  Result      : {_rate(paraphrases)} PASSED")
    entry.append(f"  Output CSV  : {paraphrases['csv_path']}")
    entry.append(_fail_block(paraphrases).rstrip("\n"))
    entry.append("")
    entry.append("PARAPHRASE COVERAGE PER ORIGINAL")
    entry.append("-" * 80)
    if CURRENT_PARAPHRASE_COUNTS:
        for original_question, count in CURRENT_PARAPHRASE_COUNTS.items():
            preview = original_question if len(original_question) <= 110 else original_question[:107] + "..."
            entry.append(f"  {count} paraphrased question(s) tested for original: {preview}")
    else:
        entry.append("  No original-to-paraphrase coverage mapping was available.")
    entry.append("")
    entry.append("-" * 80)
    entry.append("SIDE-BY-SIDE COMPARISON (rephrasing robustness)")
    entry.append("-" * 80)
    o_rate = (originals["passes"] / originals["total"]) if originals["total"] else 0.0
    p_rate = (paraphrases["passes"] / paraphrases["total"]) if paraphrases["total"] else 0.0
    entry.append(f"  ORIGINALS    : {originals['passes']}/{originals['total']}  =  {o_rate:.0%}")
    entry.append(f"  PARAPHRASES  : {paraphrases['passes']}/{paraphrases['total']}  =  {p_rate:.0%}")
    entry.append(f"  DELTA        : {delta:+.0%}")
    entry.append(f"  Verdict      : {verdict}")
    entry.append("=" * 80)
    entry.append("")

    prepend_test_results("\n".join(entry) + "\n")
    print(f"Test results appended to: {TEST_RESULTS_LOG}")


# %% [markdown]
# ## Main — side-by-side comparison

# %%
async def main(
    dataset_type: str = "both",
    search_method: str = "hybrid",
    number_questions: int | None = None,
    random_mode: str = "N",
    main_question_source: str = "llm",
):
    valid_search_methods = {"hybrid", "semantic", "lexical", "graph", "graph_enabled", "all"}
    if search_method not in valid_search_methods:
        raise ValueError(
            f"search_method must be one of: {', '.join(sorted(valid_search_methods))}."
        )
    valid_datasets = {"main", "paraphrased", "both"}
    if dataset_type not in valid_datasets:
        raise ValueError(
            f"dataset_type must be one of: {', '.join(sorted(valid_datasets))}."
        )
    if number_questions is not None and (
        isinstance(number_questions, bool) or not isinstance(number_questions, int)
    ):
        raise ValueError("number_questions must be an integer or None.")
    if str(random_mode).upper() not in {"Y", "N"}:
        raise ValueError("random_mode must be either 'Y' or 'N'.")
    if main_question_source not in {"llm", "manual"}:
        raise ValueError("main_question_source must be 'llm' or 'manual'.")

    require_api_key()
    global judge
    judge = make_judge()

    docs = load_wikipedia_chunks()
    if not docs:
        console_log("No Wikipedia documents were loaded; cannot evaluate retrieval.", "ERROR")
        raise ValueError("The Wikipedia corpus is empty.")

    embedding_function = (
        get_embeddings(EMBEDDING_MODEL)
        if search_method in {"semantic", "graph", "graph_enabled", "hybrid", "all"}
        else None
    )
    retriever = HybridRetriever(
        docs,
        search_method=search_method,
        llm_model=LLM_MODEL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        embedding_function=embedding_function,
        chroma_dir=CHROMA_DIR,
    )
    originals_ds, paraphrases_ds = load_split_datasets(
        number_questions=number_questions,
        random_mode=random_mode,
        main_question_source=main_question_source,
    )

    if dataset_type == "main":
            # console_log("Beginning MAIN (Originals) dataset evaluation.")
        results = await evaluate_dataset(
            retriever, judge, correctness_metric, originals_ds, "ORIGINALS", RAGAS_ROOT, log,
            kind=search_method,
        )
        rate = (results["passes"] / results["total"]) if results["total"] else 0.0
        print("\n" + "=" * 72)
        print(f"MAIN QUESTIONS EVALUATION ({search_method.capitalize()} search)")
        print("-" * 72)
        print(f"  Result : {results['passes']}/{results['total']} = {rate:.0%} PASSED")
        print("=" * 72)
        append_single_test_results("Main Questions", results, search_method, main_question_source)

    elif dataset_type == "paraphrased":
            # console_log("[PARAPHRASED] Beginning paraphrased dataset evaluation.")
        results = await evaluate_dataset(
            retriever, judge, correctness_metric, paraphrases_ds, "PARAPHRASES", RAGAS_ROOT, log,
            kind=search_method,
        )
        rate = (results["passes"] / results["total"]) if results["total"] else 0.0
        print("\n" + "=" * 72)
        print(f"PARAPHRASED QUESTIONS EVALUATION ({search_method.capitalize()} search)")
        print("-" * 72)
        print(f"  Result : {results['passes']}/{results['total']} = {rate:.0%} PASSED")
        print("=" * 72)
        append_single_test_results("Paraphrased Questions", results, search_method, main_question_source)

    else:
            # console_log("Beginning side-by-side ORIGINALS evaluation.")
        originals = await evaluate_dataset(
            retriever, judge, correctness_metric, originals_ds, "ORIGINALS", RAGAS_ROOT, log,
            kind=search_method,
        )
        console_log(
            f"ORIGINALS evaluation complete: {originals['passes']}/{originals['total']} passed; "
            f"failures={len(originals['failures'])}."
        )
            # console_log("[PARAPHRASED] Beginning side-by-side paraphrase evaluation.")
        paraphrases = await evaluate_dataset(
            retriever, judge, correctness_metric, paraphrases_ds, "PARAPHRASES", RAGAS_ROOT, log,
            kind=search_method,
        )
        console_log(
            f"[PARAPHRASED] Evaluation complete: {paraphrases['passes']}/{paraphrases['total']} passed; "
            f"failures={len(paraphrases['failures'])}."
        )

        orig_rate = (originals["passes"] / originals["total"]) if originals["total"] else 0.0
        para_rate = (paraphrases["passes"] / paraphrases["total"]) if paraphrases["total"] else 0.0
        delta = para_rate - orig_rate

        if delta < -0.05:
            verdict = "BRITTLE to rephrasing: paraphrases score lower than originals."
        elif delta > 0.05:
            verdict = "Paraphrases scored higher -- check for lucky wording or judge leniency."
        else:
            verdict = "ROBUST to rephrasing: pass rates are comparable."

        console_log(
            f"Computed comparison: originals={orig_rate:.0%}, paraphrases={para_rate:.0%}, "
            f"delta={delta:+.0%}, verdict={verdict}"
        )

        print("\n" + "=" * 72)
        print(f"SIDE-BY-SIDE COMPARISON  (RAGAS correctness, {search_method.capitalize()} search)")
        print("-" * 72)
        print(f"  ORIGINALS    : {originals['passes']}/{originals['total']}  =  {orig_rate:.0%}")
        print(f"  PARAPHRASES  : {paraphrases['passes']}/{paraphrases['total']}  =  {para_rate:.0%}")
        print(f"  DELTA (para - orig) : {delta:+.0%}")
        print("-" * 72)
        print(f"  {verdict}")
        print("=" * 72)
        log("COMPARISON",
            f"originals={originals['passes']}/{originals['total']} ({orig_rate:.0%}); "
            f"paraphrases={paraphrases['passes']}/{paraphrases['total']} ({para_rate:.0%}); delta={delta:+.0%}")
        append_test_results(
            originals,
            paraphrases,
            delta,
            verdict,
            search_method,
            main_question_source,
        )

    console_log("Checkpoint 4.1 evaluation completed.")


DATASET_MAP = {
    "1": "main",
    "main": "main",
    "originals": "main",
    "2": "paraphrased",
    "paraphrased": "paraphrased",
    "variants": "paraphrased",
    "3": "both",
    "both": "both",
    "all": "both",
    "4": "quit",
    "quit": "quit",
}

RETRIEVAL_METHOD_MAP = {
    "1": "lexical",
    "lexical": "lexical",
    "bm25": "lexical",
    "2": "semantic",
    "semantic": "semantic",
    "vector": "semantic",
    "3": "graph",
    "graph": "graph",
    "graph_enabled": "graph",
    "4": "hybrid",
    "hybrid": "hybrid",
    "5": "all",
    "all": "all",
    "6": "quit",
    "quit": "quit",
}


def prompt_dataset() -> tuple[str, str] | None:
    """Prompt for the evaluation question source."""
    print("\nSelect Evaluation Dataset:")
    print("  1. Manually Generated Questions")
    print("  2. LLM Generated Questions")
    print("  3. Both")
    print("  4. Quit")
    choices = {"1": "manual", "2": "llm"}
    while True:
        try:
            choice = input("Enter dataset selection [1-4]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return None
        if choice in choices:
            source = choices[choice]
            question_set = prompt_question_set(source)
            if question_set is None:
                return None
            return question_set, source
        if choice == "3":
            print("Confirmed: testing both LLM-generated Main and paraphrased questions.")
            return "both", "llm"
        if choice == "4":
            print("Quit selected. Exiting without running.")
            return None
        print("Invalid choice. Enter 1 (manual), 2 (LLM), 3 (both), or 4 (quit).")


def prompt_question_set(source: str) -> str | None:
    """Prompt for original or paraphrased questions after selecting a source."""
    source_label = "manually generated" if source == "manual" else "LLM-generated"
    print(f"\nSelect Question Set ({source_label} source):")
    print("  1. Original Questions")
    print("  2. Paraphrase Questions")
    print("  3. Quit")
    while True:
        try:
            choice = input("Enter question-set selection [1-3]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return None
        if choice == "1":
            print(f"Confirmed: testing {source_label} Original Questions.")
            return "main"
        if choice == "2":
            print(f"Confirmed: testing {source_label} Paraphrase Questions.")
            return "paraphrased"
        if choice == "3":
            print("Quit selected. Exiting without running.")
            return None
        print("Invalid choice. Enter 1 (original), 2 (paraphrase), or 3 (quit).")


def prompt_retrieval_method() -> str | None:
    """Prompt the user to select a retrieval method from options 1-6."""
    print("\nSelect Search Method:")
    print("  1. Lexical Search: BM25 matches exact terms, names, dates, and phrases")
    print("  2. Semantic Search: Chroma dense vectors find conceptual meaning and paraphrases")
    print("  3. Graph-enabled Search: Graph DB follows explicit article and chunk connections")
    print("  4. Hybrid Search: BM25 + Chroma vectors combine exact-term and deep-meaning retrieval")
    print("  5. Lexical+Semantic+Graph: BM25 + vectors + Graph DB cover exact terms, deep meanings, and explicit entity connections")
    print("  6. Quit (Exit without running)")
    while True:
        try:
            choice = input("Enter search method selection [1-6]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return None
        if choice in RETRIEVAL_METHOD_MAP:
            mapped = RETRIEVAL_METHOD_MAP[choice]
            if mapped == "quit":
                print("Quit selected. Exiting without running.")
                return None
            return mapped
        print("Invalid choice. Please enter 1 (lexical), 2 (semantic), 3 (graph), 4 (hybrid), 5 (all), or 6 (quit).")


def prompt_question_limit() -> int | None:
    """Prompt for an optional question limit during interactive runs."""
    print("\nSelect Evaluation Size:")
    print("  Press Enter to evaluate every available question in the selected dataset(s).")
    print("  Or enter a positive integer to evaluate only that many questions per dataset.")
    while True:
        try:
            value = input("Enter question limit [all]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return None
        if not value:
            print("Evaluation size: all available questions.")
            return None
        try:
            limit = int(value)
        except ValueError:
            print("Invalid question limit. Enter a positive integer or press Enter for all questions.")
            continue
        if limit <= 0:
            print("Question limit must be greater than zero.")
            continue
        return limit


def prompt_random_mode(number_questions: int | None) -> str:
    """Prompt whether a limited evaluation should sample questions randomly."""
    if number_questions is None:
        return "N"
    print("\nSelect Question Sampling:")
    print("  Y. Randomly sample the requested number from each dataset")
    print("  N. Use the first questions from each dataset in file order")
    while True:
        try:
            choice = input("Random sampling [N]: ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return "N"
        if not choice:
            return "N"
        if choice in {"Y", "N"}:
            return choice
        print("Invalid sampling choice. Enter Y for random sampling or N for file order.")


if __name__ == "__main__":
    console_log("Parsing command-line arguments.")
    parser = argparse.ArgumentParser(description="Run the Checkpoint 4.1 RAGAS evaluation.")
    parser.add_argument(
        "--input_type",
        "--dataset",
        dest="dataset",
        choices=("1", "2", "3", "4", "main", "paraphrased", "both", "quit"),
        default=None,
        help="Evaluation dataset: 1/main, 2/paraphrased, 3/both, 4/quit.",
    )
    parser.add_argument(
        "--retrieval",
        "--search_method",
        dest="retrieval",
        choices=("1", "2", "3", "4", "5", "6", "lexical", "semantic", "graph", "hybrid", "all", "quit"),
        default=None,
        help="Retrieval search method: 1/lexical, 2/semantic, 3/graph, 4/hybrid, 5/all, 6/quit.",
    )
    parser.add_argument(
        "--main-source",
        choices=("llm", "manual"),
        default=None,
        help="Main-question JSONL source: llm or manual. Prompts interactively when omitted for main/both datasets.",
    )
    parser.add_argument(
        "--numbers",
        "--number_questions",
        "-n",
        dest="number_questions",
        type=int,
        default=None,
        help="Maximum number of questions to evaluate from each dataset. Must not exceed the smallest dataset size.",
    )
    parser.add_argument(
        "--random",
        choices=("Y", "y", "N", "n"),
        default=None,
        help="Y/y = sample number_questions at random; N/n = take the first number_questions rows from each dataset.",
    )
    args = parser.parse_args()
    interactive_mode = args.dataset is None or args.retrieval is None
    main_question_source = args.main_source

    # Ensure the shared local workspace scaffolding is ready before any interactive prompts or menu choices are displayed.
    ensure_workspace_setup()

    # 1. Resolve dataset selection
    dataset_choice = args.dataset
    if dataset_choice is not None:
        dataset_choice = DATASET_MAP.get(dataset_choice.lower(), dataset_choice)
        if dataset_choice == "quit":
            console_log("Quit selected via dataset argument. Exiting without running.")
            sys.exit(0)
    else:
        dataset_selection = prompt_dataset()
        if dataset_selection is None:
            sys.exit(0)
        dataset_choice, main_question_source = dataset_selection

    if args.main_source is not None:
        main_question_source = args.main_source
    if main_question_source is None:
        main_question_source = "llm"

    # 2. Resolve retrieval search method selection
    retrieval_method = args.retrieval
    if retrieval_method is not None:
        retrieval_method = RETRIEVAL_METHOD_MAP.get(retrieval_method.lower(), retrieval_method)
        if retrieval_method == "quit":
            console_log("Quit selected via retrieval argument. Exiting without running.")
            sys.exit(0)
    else:
        retrieval_method = prompt_retrieval_method()
        if retrieval_method is None:
            sys.exit(0)

    number_questions = args.number_questions
    if number_questions is None and interactive_mode:
        number_questions = prompt_question_limit()

    random_mode = args.random.upper() if args.random else None
    if random_mode is None:
        random_mode = prompt_random_mode(number_questions) if interactive_mode else "N"

    console_log(
        f"Execution configuration: dataset={dataset_choice}, "
        f"retrieval_method={retrieval_method}, number_questions={number_questions}, "
        f"random={random_mode}, main_question_source={main_question_source}"
    )
    asyncio.run(
        main(
            dataset_type=dataset_choice,
            search_method=retrieval_method,
            number_questions=number_questions,
            random_mode=random_mode,
            main_question_source=main_question_source,
        )
    )
