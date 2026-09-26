#################################################################################
# 
# Massachusetts Institute of Technology - xPRO
# RAG and Context Engineering: Designing and Building Production-Grade AI Systems
# 
# Student: Miguel Herrera
# Section: B
# Date and Time: 2026-09-22 19:31:56 -07:00
#
# Description: Interactive Checkpoint 5.1 solution for the Wikipedia RAG engine. Lets
#              the user pick a retrieval engine — Context-Aware (history-aware, no
#              planning) or Agentic Dynamic (LangGraph ReAct: plan / retrieve /
#              graph_expand / clarify / answer) — then run either a RAGAS DiscreteMetric
#              evaluation or an interactive chat (with history and per-turn token usage).
#              Evaluation also supports a "Compare Both" mode that scores both engines
#              on the same dataset and logs a side-by-side comparison. All evaluation
#              results are written to the agentic detailed-results log. The CLI wires
#              the shared menus, dataset loader, RAGAS harness, and report writers.
#
#################################################################################

r"""Capstone Checkpoint 6.1 — ReAct Tool-Using Agent (SOLUTION).
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code / PyCharm / Jupytext.

This solution offers two retrieval engines over the Wikipedia capstone corpus and a
common RAGAS harness. The Agentic Dynamic engine is a LangGraph ReAct agent whose
"plan" node observes the conversation, executed queries, and retrieved chunks, then
emits ONE JSON action each step (no upfront plan):

    plan ──▶ retrieve        (BM25 + vector search of the Wikipedia corpus)
         ──▶ graph_expand    (graph-augmented neighborhood expansion; graph methods only)
         ──▶ clarify         (ask the user a question, then re-plan)
         ──▶ answer          (respond to the user)  ──▶ done

The Context-Aware engine is a non-agentic, history-aware retriever (no planning). Either
engine can be scored through the RAGAS DiscreteMetric harness as a drop-in retriever, or
driven interactively in CHAT mode (conversation history + per-turn token usage).

    * ENGINE      -> Context-Aware, Agentic Dynamic, or Compare Both (eval only)
    * MAIN        -> Main grounded evaluation questions
    * PARAPHRASES -> Derived query variants reusing ground-truth notes
    * BOTH        -> Originals + paraphrases (each logged as its own entry)
    * CHAT        -> Interactive multi-turn chat with history
"""

# %% [markdown]
# # Capstone Checkpoint 6.1 — ReAct Tool-Using Agent
# **MO-LLM Module 5 / Required Capstone Checkpoint**
#
# A LangGraph plan/retrieve/graph_expand/clarify/answer agent (as in Lab 5.2) that
# reuses the Checkpoint 2.1 hybrid retriever and the capstone Wikipedia corpus. Supports
# the Checkpoint 6.1 RAGAS evaluation menus plus an interactive chat-with-history mode.

# %% [markdown]
# ## Setup
# 1. Python 3.11 or 3.12.
# 2. pip install -r requirements.txt (ragas, langgraph, langchain-openai, langchain-chroma, rank-bm25, python-dotenv, openai).
# 3. Use the OpenRouter API key provided for this program.
# 4. Create a `.env` file with:  OPENROUTER_API_KEY=sk-or-v1-...

# %%
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import json
import os
import configparser
import asyncio
import argparse
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Final_Capstone_Project.Retrieval_Methods.bm25_retrieval import load_wikipedia_chunks
from Final_Capstone_Project.Retrieval_Methods.bm25_retrieval import BM25_CANDIDATES
from Final_Capstone_Project.Retrieval_Methods.dynamic_agent_retrieval import ToolUsingAgent
from Final_Capstone_Project.Retrieval_Methods.context_aware_retrieval import ContextAwareRetriever
from Final_Capstone_Project.Retrieval_Methods.vector_retrieval import VECTOR_CANDIDATES
from Final_Capstone_Project.Ranking_Techniques.weighted_fusion_ranking import (
    FUSED_TOP_K,
    WEIGHT_BM25,
    WEIGHT_VECTOR,
)
from Final_Capstone_Project.Utility_Scripts.Ragas_Experiment_Logic import evaluate_dataset
from Final_Capstone_Project.Utility_Scripts.evaluation_common import (
    build_correctness_metric,
    console_log,
    make_judge,
    require_api_key,
)
from Final_Capstone_Project.Utility_Scripts.dataset_loader import load_split_datasets
from Final_Capstone_Project.Utility_Scripts.test_report import (
    EngineRun,
    ReportConfig,
    append_engine_comparison,
    append_single_engine_result,
)
from Final_Capstone_Project.Utility_Scripts.evaluation_menus import (
    AGENT_BASE_RETRIEVAL_MAP,
    DATASET_MAP,
    ENGINE_MAP,
    MODE_MAP,
    RETRIEVAL_METHOD_MAP,
    prompt_agent_base_retrieval_method,
    prompt_dataset,
    prompt_evaluation_type,
    prompt_engine,
    prompt_mode,
    prompt_question_limit,
    prompt_random_mode,
    prompt_retrieval_method,
)
from Final_Capstone_Project.Utility_Scripts.checkpoint_6_1_cost_experiment_prototype import (
    ExperimentResult,
    configuration_pairs,
    estimate_cost_usd,
)
from Final_Capstone_Project.Utility_Scripts.token_usage import TokenUsage, format_tokens

# %%
_config = configparser.ConfigParser()
_config.read(Path(__file__).resolve().parents[1] / "retrieval.conf")
LLM_MODEL = _config.get("llm", "model", fallback="openai/gpt-5.4-mini")
JUDGE_MODEL = _config.get("llm", "judge_model", fallback=LLM_MODEL)


# ── paths ────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = Path(__file__).resolve().parent
FINAL_CAPSTONE_DIR = CHECKPOINT_DIR.parent
CHROMA_DIR = str(FINAL_CAPSTONE_DIR / "Capstone_Database" / "Capstone_Chroma_DB")

# Cosmetic checkpoint label derived from the folder name (e.g. "Capstone_Checkpoint_5.1"
# -> "CAPSTONE 6.1"). Self-updating for any future Capstone_Checkpoint_x.x folder.
CAPSTONE_LABEL = "CAPSTONE " + CHECKPOINT_DIR.name.split("_")[-1]

TEST_VARIABLES_DIR = FINAL_CAPSTONE_DIR / "Test_Variables"
RAGAS_ROOT = str(FINAL_CAPSTONE_DIR / "Ragas_Experiments")
LOG_PATH = CHECKPOINT_DIR / "checkpoint_6_1_agent.log"
TEST_RESULTS_LOG = Path(RAGAS_ROOT) / "detailed_test_results.log"
CHECKPOINT_6_1_REPORT_LOG = Path(RAGAS_ROOT) / "checkpoint_6_1_security_cost.log"
# Engine-comparison runs write here, leaving the single-engine log above untouched.
TEST_RESULTS_AGENTIC_LOG = Path(RAGAS_ROOT) / "detailed_test_results_agentic.log"
TEST_RESULTS_FAILURES_LOG = CHECKPOINT_6_1_REPORT_LOG
COST_EXPERIMENT_LOG = CHECKPOINT_6_1_REPORT_LOG
FAILURE_DATASET_PATH = TEST_VARIABLES_DIR / "Test_Questions_Manually_Failure.json"

# Checkpoint-specific values threaded into every detailed-report entry.
REPORT_CONFIG = ReportConfig(
    capstone_label=CAPSTONE_LABEL,
    test_results_log=TEST_RESULTS_LOG,
    bm25_candidates=BM25_CANDIDATES,
    vector_candidates=VECTOR_CANDIDATES,
    fused_top_k=FUSED_TOP_K,
    llm_model=LLM_MODEL,
    judge_model=JUDGE_MODEL,
    weight_bm25=WEIGHT_BM25,
    weight_vector=WEIGHT_VECTOR,
)


# %%
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


def log(label: str, text: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {label}\n{text}\n{'-' * 72}\n")


def load_failure_dataset(number_questions: int | None = None, random_mode: str = "N"):
    """Load the manually authored failure questions into a RAGAS dataset."""
    if not FAILURE_DATASET_PATH.is_file():
        raise FileNotFoundError(f"Failure dataset not found at {FAILURE_DATASET_PATH}")
    random_mode = str(random_mode).upper()
    if random_mode not in {"Y", "N"}:
        raise ValueError("random_mode must be either 'Y' or 'N'.")
    if number_questions is not None and (
        isinstance(number_questions, bool) or not isinstance(number_questions, int)
    ):
        raise ValueError("number_questions must be an integer or None.")
    if number_questions is not None and number_questions <= 0:
        raise ValueError("--number_questions must be a positive integer.")

    with FAILURE_DATASET_PATH.open("r", encoding="utf-8") as fh:
        records = json.load(fh)
    if not isinstance(records, list):
        raise ValueError(f"Failure dataset must contain a JSON array: {FAILURE_DATASET_PATH}")
    for index, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise ValueError(f"Failure dataset record {index} is not an object")
        for field in ("question", "grading_notes"):
            if not isinstance(record.get(field), str) or not record[field].strip():
                raise ValueError(f"Failure dataset record {index} has an invalid '{field}' field")
    if number_questions is not None and number_questions > len(records):
        raise ValueError(
            f"--number_questions={number_questions} exceeds the available failure questions "
            f"(max available: {len(records)})."
        )
    if random_mode == "Y" and number_questions is not None:
        import random
        records = random.sample(records, number_questions)
    elif number_questions is not None:
        records = records[:number_questions]

    from ragas import Dataset
    dataset = Dataset(name="wiki_eval_failures", backend="local/csv", root_dir=RAGAS_ROOT)
    for record in records:
        dataset.append({
            "question": record["question"],
            "grading_notes": record["grading_notes"],
            **({"evaluation_category": record["evaluation_category"]}
               if record.get("evaluation_category") else {}),
        })
    dataset.save()
    return dataset


def attach_eval_metrics(results: dict, retriever) -> dict:
    """Add cumulative latency and workflow steps to an evaluation result."""
    latency_seconds, workflow_steps = retriever.get_eval_metrics()
    results["latency_seconds"] = latency_seconds
    results["workflow_steps"] = workflow_steps
    return results


@dataclass
class EngineCostComparison:
    configuration: str
    context: ExperimentResult
    agent: ExperimentResult


def append_cost_experiment_report(
    results: list[ExperimentResult],
    search_method: str,
    comparisons: list[EngineCostComparison],
) -> None:
    """Append one model-ladder cost comparison to the dedicated cost log."""
    lines = [
        "=" * 80,
        f"{CAPSTONE_LABEL} COST EXPERIMENT SESSION  |  {datetime.now():%Y-%m-%d %H:%M:%S}",
        "=" * 80,
        "Dataset    : Test_Questions_Manually_Failure.json",
        f"Search     : {search_method}",
        "Note       : RAGAS judge tokens and embedding costs are excluded from these totals.",
        "",
        "CONFIGURATION                                      RESULT   TOKENS   EST. COST",
        "-" * 80,
    ]
    for result in results:
        lines.append(
            f"{result.configuration:48s} {result.passes}/{result.questions} "
            f"({result.pass_rate:.0%})  {format_tokens(result.total_tokens):>7s}  "
            f"${result.estimated_cost_usd:.5f}  {result.latency_seconds:.2f}s  "
            f"steps={result.workflow_steps}"
        )
    lines.extend(["", "Planner/answer token detail", "-" * 80])
    for result in results:
        lines.append(
            f"  {result.configuration}: planner={format_tokens(result.planner_usage.total_tokens)}, "
            f"answer={format_tokens(result.answer_usage.total_tokens)}"
        )
    lines.extend(["", "CONTEXT-AWARE VS AGENTIC COST COMPARISON", "-" * 80])
    lines.append("CONFIGURATION                                      CONTEXT-AWARE          AGENTIC")
    for comparison in comparisons:
        context = comparison.context
        agent = comparison.agent
        lines.append(
            f"{comparison.configuration:48s} "
            f"{context.passes}/{context.questions} | {format_tokens(context.total_tokens):>7s} | "
            f"${context.estimated_cost_usd:.5f} | {context.latency_seconds:.2f}s | steps={context.workflow_steps}  "
            f"{agent.passes}/{agent.questions} | {format_tokens(agent.total_tokens):>7s} | "
            f"${agent.estimated_cost_usd:.5f} | {agent.latency_seconds:.2f}s | steps={agent.workflow_steps}"
        )
    lines.extend(["=" * 80, "",])
    COST_EXPERIMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with COST_EXPERIMENT_LOG.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


async def run_cost_experiment(
    docs: list[dict],
    *,
    search_method: str = "hybrid",
    number_questions: int = 4,
    random_mode: str = "N",
) -> None:
    """Run an opt-in Lab 6.2-style model ladder over the failure-question suite."""
    global judge
    judge = make_judge(JUDGE_MODEL)
    dataset = load_failure_dataset(number_questions, random_mode)
    experiment_results = []
    engine_comparisons = []
    for pair in configuration_pairs():
        console_log(
            f"[cost] Running {pair.name}: planner={pair.planner.name}, "
            f"answer={pair.answer.name}."
        )
        agent = ToolUsingAgent(
            docs,
            search_method=search_method,
            plan_model=pair.planner.name,
            answer_model=pair.answer.name,
        )
        agent.reset_eval_usage()
        scored = await evaluate_dataset(
            agent,
            judge,
            correctness_metric,
            dataset,
            f"COST {pair.name}",
            RAGAS_ROOT,
            log,
            kind=search_method,
        )
        plan_usage, answer_usage = agent.get_eval_usage()
        experiment_results.append(
            agent_result := ExperimentResult(
                configuration=pair.name,
                questions=scored["total"],
                passes=scored["passes"],
                planner_usage=plan_usage,
                answer_usage=answer_usage,
                estimated_cost_usd=estimate_cost_usd(
                    plan_usage,
                    answer_usage,
                    pair.planner,
                    pair.answer,
                ),
            )
        )
        agent_latency, agent_steps = agent.get_eval_metrics()
        agent_result.latency_seconds = agent_latency
        agent_result.workflow_steps = agent_steps

        context_engine = ContextAwareRetriever(
            docs,
            search_method=search_method,
            llm_model=pair.answer.name,
        )
        context_engine.reset_eval_usage()
        context_scored = await evaluate_dataset(
            context_engine,
            judge,
            correctness_metric,
            dataset,
            f"COST CONTEXT-AWARE {pair.name}",
            RAGAS_ROOT,
            log,
            kind=search_method,
        )
        context_plan, context_answer = context_engine.get_eval_usage()
        context_result = ExperimentResult(
            configuration=pair.name,
            questions=context_scored["total"],
            passes=context_scored["passes"],
            planner_usage=context_plan,
            answer_usage=context_answer,
            estimated_cost_usd=estimate_cost_usd(
                context_plan,
                context_answer,
                pair.planner,
                pair.answer,
            ),
        )
        context_latency, context_steps = context_engine.get_eval_metrics()
        context_result.latency_seconds = context_latency
        context_result.workflow_steps = context_steps
        engine_comparisons.append(
            EngineCostComparison(pair.name, context_result, agent_result)
        )

    append_cost_experiment_report(experiment_results, search_method, engine_comparisons)
    print("\n" + "=" * 80)
    print("CHECKPOINT 6.1 COST EXPERIMENT")
    print("-" * 80)
    for result in experiment_results:
        print(
            f"{result.configuration:48s} {result.passes}/{result.questions} "
            f"({result.pass_rate:.0%})  {format_tokens(result.total_tokens):>7s}  "
            f"${result.estimated_cost_usd:.5f}"
        )
    print("\nContext-Aware vs Agentic")
    for comparison in engine_comparisons:
        print(
            f"{comparison.configuration:48s} "
            f"context={comparison.context.passes}/{comparison.context.questions}, "
            f"agent={comparison.agent.passes}/{comparison.agent.questions}"
        )
    print(f"Report: {COST_EXPERIMENT_LOG}")
    print("=" * 80)


# %% [markdown]
# ## RAGAS evaluation harness (DiscreteMetric judge — same idea as Lab 3.1)
# The judge factory, correctness metric, JSONL loader, and originals/paraphrases
# dataset splitter now live in the shared Utility_Scripts modules
# (evaluation_common, dataset_loader). The judge itself is built in main() after
# the API-key check.

# %%
judge = None  # built in main() after the API-key check
correctness_metric = build_correctness_metric()


# %% [markdown]
# ## Engine comparison — Context-Aware vs Agentic Dynamic (writes the agentic log)

# %%
async def _evaluate_engine(retriever, dataset, label, search_method):
    """Reset the engine's eval usage, run one dataset, and return (results, EngineRun-fields)."""
    retriever.reset_eval_usage()
    results = await evaluate_dataset(
        retriever, judge, correctness_metric, dataset, label, RAGAS_ROOT, log,
        kind=search_method,
    )
    attach_eval_metrics(results, retriever)
    plan_usage, answer_usage = retriever.get_eval_usage()
    return results, plan_usage, answer_usage


async def run_engine_comparison(
    docs: list[dict],
    *,
    dataset_type: str = "both",
    search_method: str = "hybrid",
    number_questions: int | None = None,
    random_mode: str = "N",
    main_question_source: str = "llm",
) -> None:
    """Evaluate BOTH engines on the same dataset and write a side-by-side comparison.

    The comparison entry goes to the separate agentic results log; the single-engine
    detailed log is left untouched."""
    global judge
    judge = make_judge(JUDGE_MODEL)

    if dataset_type == "failure":
        dataset = load_failure_dataset(number_questions, random_mode)
        dataset_name, label, report_log = (
            "FAILURE/DIAGNOSTIC QUESTIONS",
            "FAILURE TESTS",
            TEST_RESULTS_FAILURES_LOG,
        )
    else:
        split = load_split_datasets(
            TEST_VARIABLES_DIR,
            RAGAS_ROOT,
            number_questions=number_questions,
            random_mode=random_mode,
            main_question_source=main_question_source,
        )
        # Compare on a single dataset: paraphrases when explicitly requested, else main.
        if dataset_type == "paraphrased":
            dataset, dataset_name, label = split.paraphrases, "PARAPHRASED QUESTIONS", "PARAPHRASES"
        else:
            dataset, dataset_name, label = split.originals, "MAIN QUESTIONS", "ORIGINALS"
        report_log = TEST_RESULTS_AGENTIC_LOG

    console_log(f"[compare] Evaluating Context-Aware retriever ({search_method} search)...")
    context_engine = ContextAwareRetriever(docs, search_method=search_method)
    context_results, context_plan, context_answer = await _evaluate_engine(
        context_engine, dataset, label, search_method
    )

    console_log(f"[compare] Evaluating Agentic Dynamic retriever ({search_method} search)...")
    agent_engine = ToolUsingAgent(docs, search_method=search_method)
    agent_results, agent_plan, agent_answer = await _evaluate_engine(
        agent_engine, dataset, label, search_method
    )

    append_engine_comparison(
        REPORT_CONFIG,
        report_log,
        dataset_name,
        EngineRun("Context-Aware retriever", context_results, context_plan, context_answer),
        EngineRun("Agentic Dynamic (ReAct)", agent_results, agent_plan, agent_answer),
        search_method,
        main_question_source,
    )

    context_rate = context_results["passes"] / context_results["total"] if context_results["total"] else 0.0
    agent_rate = agent_results["passes"] / agent_results["total"] if agent_results["total"] else 0.0
    print("\n" + "=" * 72)
    print(f"ENGINE COMPARISON ({search_method.capitalize()} search, {dataset_name.title()})")
    print("-" * 72)
    print(f"  Context-Aware : {context_results['passes']}/{context_results['total']} = {context_rate:.0%}")
    print(f"  Agentic       : {agent_results['passes']}/{agent_results['total']} = {agent_rate:.0%}")
    print(f"  Delta (agent - context) : {agent_rate - context_rate:+.0%}")
    print("=" * 72)
    console_log(f"{CAPSTONE_LABEL} engine comparison completed.")


def view_previous_results() -> None:
    """Display the latest available failure and cost summaries without API calls."""
    print("\nPrevious Checkpoint 6.1 Results")
    for label, path in (
        ("Unified security and cost report", CHECKPOINT_6_1_REPORT_LOG),
        ("Standard Agentic results", TEST_RESULTS_AGENTIC_LOG),
    ):
        print(f"\n--- {label}: {path} ---")
        if not path.is_file():
            print("No results have been recorded yet.")
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        visible_lines = lines[:80] if path != CHECKPOINT_6_1_REPORT_LOG else lines[-80:]
        print("\n".join(visible_lines))


# %% [markdown]
# ## Main — RAGAS evaluation (agent as retriever) or interactive chat

# %%
async def main(
    mode: str = "eval",
    engine: str = "agent",
    dataset_type: str = "both",
    search_method: str = "hybrid",
    number_questions: int | None = None,
    random_mode: str = "N",
    main_question_source: str = "llm",
):
    valid_modes = {"eval", "chat"}
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of: {', '.join(sorted(valid_modes))}.")
    valid_engines = {"context", "agent", "both"}
    if engine not in valid_engines:
        raise ValueError(f"engine must be one of: {', '.join(sorted(valid_engines))}.")
    if engine == "both" and mode != "eval":
        raise ValueError("engine='both' (comparison) is only supported in eval mode.")
    valid_search_methods = {"hybrid", "semantic", "lexical", "graph", "graph_enabled", "all"}
    if search_method not in valid_search_methods:
        raise ValueError(
            f"search_method must be one of: {', '.join(sorted(valid_search_methods))}."
        )
    valid_datasets = {"main", "paraphrased", "both", "failure"}
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
    if dataset_type == "failure":
        main_question_source = "manual"

    require_api_key()

    docs = load_wikipedia_chunks()
    if not docs:
        console_log("No Wikipedia documents were loaded; cannot run the agent.", "ERROR")
        raise ValueError("The Wikipedia corpus is empty.")

    # ── Engine comparison mode (Context-Aware vs Agentic Dynamic) ───────────
    if engine == "both":
        await run_engine_comparison(
            docs,
            dataset_type=dataset_type,
            search_method=search_method,
            number_questions=number_questions,
            random_mode=random_mode,
            main_question_source=main_question_source,
        )
        return

    # Build the selected retrieval engine. Both expose query() (drop-in for the RAGAS
    # harness) and chat() (interactive, with conversation history).
    if engine == "context":
        retriever = ContextAwareRetriever(docs, search_method=search_method)
        engine_label = "Context-Aware retriever"
    else:
        retriever = ToolUsingAgent(docs, search_method=search_method)
        engine_label = "ReAct agent"
    # Reflect the chosen engine in the detailed-report Config line.
    REPORT_CONFIG.engine_label = engine_label
    console_log(f"{engine_label} ready over {len(docs)} documents ({search_method} search).")

    # ── Interactive chat mode (with history) ────────────────────────────────
    if mode == "chat":
        console_log(f"{CAPSTONE_LABEL} interactive chat (with history) starting [{engine_label}].")
        retriever.chat()
        console_log(f"{CAPSTONE_LABEL} chat session completed.")
        return

    # ── RAGAS evaluation mode (the engine is a drop-in retriever) ───────────
    global judge
    judge = make_judge(JUDGE_MODEL)

    failure_ds = load_failure_dataset(number_questions, random_mode) if dataset_type == "failure" else None
    if failure_ds is None:
        split = load_split_datasets(
            TEST_VARIABLES_DIR,
            RAGAS_ROOT,
            number_questions=number_questions,
            random_mode=random_mode,
            main_question_source=main_question_source,
        )
        originals_ds, paraphrases_ds = split.originals, split.paraphrases

    # Evaluate one dataset, capturing this engine's cumulative token/call effort, and
    # write a clean per-engine entry to the agentic results log.
    async def evaluate_and_log(dataset, label, dataset_name, report_log):
        retriever.reset_eval_usage()
        results = await evaluate_dataset(
            retriever, judge, correctness_metric, dataset, label, RAGAS_ROOT, log,
            kind=search_method,
        )
        attach_eval_metrics(results, retriever)
        plan_usage, answer_usage = retriever.get_eval_usage()
        rate = (results["passes"] / results["total"]) if results["total"] else 0.0
        print("\n" + "=" * 72)
        print(f"{dataset_name.upper()} EVALUATION ({search_method.capitalize()} search, {engine_label})")
        print("-" * 72)
        print(f"  Result : {results['passes']}/{results['total']} = {rate:.0%} PASSED")
        print("=" * 72)
        append_single_engine_result(
            REPORT_CONFIG,
            report_log,
            dataset_name,
            EngineRun(engine_label, results, plan_usage, answer_usage),
            search_method,
            main_question_source,
        )

    if dataset_type == "main":
        await evaluate_and_log(originals_ds, "ORIGINALS", "Main Questions", TEST_RESULTS_AGENTIC_LOG)
    elif dataset_type == "paraphrased":
        await evaluate_and_log(paraphrases_ds, "PARAPHRASES", "Paraphrased Questions", TEST_RESULTS_AGENTIC_LOG)
    elif dataset_type == "failure":
        await evaluate_and_log(
            failure_ds,
            "FAILURE TESTS",
            "Failure/Diagnostic Questions",
            TEST_RESULTS_FAILURES_LOG,
        )
    else:
        # Log each dataset as its own clean per-engine entry.
        await evaluate_and_log(originals_ds, "ORIGINALS", "Main Questions", TEST_RESULTS_AGENTIC_LOG)
        await evaluate_and_log(paraphrases_ds, "PARAPHRASES", "Paraphrased Questions", TEST_RESULTS_AGENTIC_LOG)

    console_log(f"{CAPSTONE_LABEL} {engine_label} evaluation completed.")


# %% [markdown]
# ## CLI entrypoint
# The interactive menus (mode / dataset / question set / retrieval method / size /
# sampling) and their answer-normalization maps live in the shared
# Utility_Scripts.evaluation_menus module (imported above). This block only wires
# the argparse CLI to those prompts and dispatches to main().

# %%
if __name__ == "__main__":
    console_log("Parsing command-line arguments.")
    parser = argparse.ArgumentParser(description="Run the Checkpoint 6.1 ReAct agent (RAGAS evaluation or interactive chat).")
    parser.add_argument(
        "--mode",
        choices=("1", "2", "3", "eval", "chat", "quit"),
        default=None,
        help="Run mode: 1/chat (interactive chat with history), 2/eval (RAGAS evaluation), 3/quit.",
    )
    parser.add_argument(
        "--engine",
        choices=("1", "2", "3", "4", "context", "agent", "both", "quit"),
        default=None,
        help="Retrieval engine: 1/context, 2/agent, 3/both (eval-only comparison), 4/quit.",
    )
    parser.add_argument(
        "--input_type",
        "--dataset",
        dest="dataset",
        choices=("1", "2", "3", "4", "main", "paraphrased", "both", "failure", "quit"),
        default=None,
        help="Evaluation dataset: main, paraphrased, both, or failure diagnostic questions.",
    )
    parser.add_argument(
        "--retrieval",
        "--search_method",
        dest="retrieval",
        choices=("1", "2", "3", "4", "5", "lexical", "semantic", "hybrid", "graph", "quit"),
        default=None,
        help="Retrieval search method: 1/lexical, 2/semantic, 3/hybrid, 4/graph (hybrid with graph enabled), 5/quit.",
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
    parser.add_argument(
        "--cost-experiment",
        action="store_true",
        help="Run the opt-in Lab 6.2-style model-ladder and mixed-model cost experiment.",
    )
    args = parser.parse_args()

    # Ensure the shared local workspace scaffolding is ready before any prompts or evaluation.
    ensure_workspace_setup()

    if args.cost_experiment:
        retrieval_method = args.retrieval
        if retrieval_method is not None:
            retrieval_method = RETRIEVAL_METHOD_MAP.get(retrieval_method.lower(), retrieval_method)
        else:
            retrieval_method = "hybrid"
        number_questions = args.number_questions if args.number_questions is not None else 4
        random_mode = args.random.upper() if args.random else "N"
        console_log(
            f"Execution configuration: cost_experiment=True, retrieval_method={retrieval_method}, "
            f"number_questions={number_questions}, random={random_mode}"
        )
        asyncio.run(
            run_cost_experiment(
                load_wikipedia_chunks(),
                search_method=retrieval_method,
                number_questions=number_questions,
                random_mode=random_mode,
            )
        )
        sys.exit(0)

    # 0. Resolve top-level mode (evaluation vs interactive chat)
    run_mode = args.mode
    if run_mode is not None:
        run_mode = MODE_MAP.get(run_mode.lower(), run_mode)
        if run_mode == "quit":
            console_log("Quit selected via mode argument. Exiting without running.")
            sys.exit(0)
    else:
        run_mode = prompt_mode()
        if run_mode is None:
            sys.exit(0)

    if args.mode is None and run_mode == "eval":
        evaluation_type = prompt_evaluation_type()
        if evaluation_type is None:
            sys.exit(0)
        if evaluation_type == "view":
            view_previous_results()
            sys.exit(0)
        if evaluation_type == "cost":
            retrieval_method = prompt_retrieval_method()
            if retrieval_method is None:
                sys.exit(0)
            number_questions = prompt_question_limit()
            random_mode = prompt_random_mode(number_questions)
            print("\nWarning: the cost experiment makes multiple API calls and may incur costs.")
            try:
                confirmation = input("Run cost experiment? [Y/N]: ").strip().upper()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                sys.exit(0)
            if confirmation != "Y":
                print("Cost experiment cancelled.")
                sys.exit(0)
            asyncio.run(
                run_cost_experiment(
                    load_wikipedia_chunks(),
                    search_method=retrieval_method,
                    number_questions=number_questions or 5,
                    random_mode=random_mode,
                )
            )
            sys.exit(0)

    # 0b. Resolve the retrieval engine (Context-Aware vs Agentic Dynamic) for both modes.
    engine_choice = args.engine
    if engine_choice is not None:
        engine_choice = ENGINE_MAP.get(engine_choice.lower(), engine_choice)
        if engine_choice == "quit":
            console_log("Quit selected via engine argument. Exiting without running.")
            sys.exit(0)
    else:
        # "Compare Both" is eval-only, so it is hidden from the chat engine menu.
        engine_choice = prompt_engine(allow_compare=(run_mode == "eval"))
        if engine_choice is None:
            sys.exit(0)

    # Guard the CLI path (--engine both --mode chat): comparison is eval-only.
    if engine_choice == "both" and run_mode == "chat":
        console_log("Engine comparison ('both') is eval-only; switching to evaluation mode.")
        run_mode = "eval"

    # The Dynamic Agent always has graph expansion available, so it uses a base-retriever
    # menu (lexical/semantic/hybrid); the Context-Aware engine uses the full search menu.
    if engine_choice == "agent":
        retrieval_map = AGENT_BASE_RETRIEVAL_MAP
        prompt_search_method = prompt_agent_base_retrieval_method
    else:
        retrieval_map = RETRIEVAL_METHOD_MAP
        prompt_search_method = prompt_retrieval_method

    # Interactive chat needs only a retrieval method; the rest of the eval menus are skipped.
    interactive_mode = args.mode is None or (run_mode == "eval" and (args.dataset is None or args.retrieval is None))
    main_question_source = args.main_source

    if run_mode == "chat":
        retrieval_method = args.retrieval
        if retrieval_method is not None:
            retrieval_method = retrieval_map.get(retrieval_method.lower(), retrieval_method)
            if retrieval_method == "quit":
                console_log("Quit selected via retrieval argument. Exiting without running.")
                sys.exit(0)
        else:
            retrieval_method = prompt_search_method()
            if retrieval_method is None:
                sys.exit(0)
        console_log(f"Execution configuration: mode=chat, engine={engine_choice}, retrieval_method={retrieval_method}")
        asyncio.run(main(mode="chat", engine=engine_choice, search_method=retrieval_method))
        sys.exit(0)

    # 1. Resolve dataset selection (evaluation mode)
    dataset_choice = args.dataset
    if dataset_choice is not None:
        dataset_choice = DATASET_MAP.get(dataset_choice.lower(), dataset_choice)
        if dataset_choice == "quit":
            console_log("Quit selected via dataset argument. Exiting without running.")
            sys.exit(0)
    else:
        dataset_selection = prompt_dataset(include_failure=True)
        if dataset_selection is None:
            sys.exit(0)
        dataset_choice, main_question_source = dataset_selection

    if args.main_source is not None:
        main_question_source = args.main_source
    if main_question_source is None:
        main_question_source = "llm"

    # 2. Resolve retrieval search method selection (engine-appropriate menu)
    retrieval_method = args.retrieval
    if retrieval_method is not None:
        retrieval_method = retrieval_map.get(retrieval_method.lower(), retrieval_method)
        if retrieval_method == "quit":
            console_log("Quit selected via retrieval argument. Exiting without running.")
            sys.exit(0)
    else:
        retrieval_method = prompt_search_method()
        if retrieval_method is None:
            sys.exit(0)

    number_questions = args.number_questions
    if number_questions is None and interactive_mode:
        number_questions = prompt_question_limit()

    random_mode = args.random.upper() if args.random else None
    if random_mode is None:
        random_mode = prompt_random_mode(number_questions) if interactive_mode else "N"

    console_log(
        f"Execution configuration: mode=eval, engine={engine_choice}, dataset={dataset_choice}, "
        f"retrieval_method={retrieval_method}, number_questions={number_questions}, "
        f"random={random_mode}, main_question_source={main_question_source}"
    )
    asyncio.run(
        main(
            mode="eval",
            engine=engine_choice,
            dataset_type=dataset_choice,
            search_method=retrieval_method,
            number_questions=number_questions,
            random_mode=random_mode,
            main_question_source=main_question_source,
        )
    )
