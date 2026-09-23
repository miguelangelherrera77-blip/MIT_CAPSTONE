#################################################################################
# 
# Massachusetts Institute of Technology - xPRO
# RAG and Context Engineering: Designing and Building Production-Grade AI Systems
# 
# Student: Miguel Herrera
# Section: B
# Date and Time: 2026-09-22 19:31:56 -07:00
#
# Description: Shared detailed test-result report writers for the capstone RAGAS
#              harness. Provides both the originals-vs-paraphrases robustness writers
#              and the engine-oriented writers: a clean single-engine result entry
#              and a Context-Aware vs Agentic side-by-side comparison (correctness,
#              LLM call/token effort, passes-per-1K-tokens, and category deltas). Each
#              entry records the effective search method and fusion-ranking weights,
#              and is prepended newest-first to the log path passed in by the caller.
#              Checkpoint-specific values (label, config, models) come via ReportConfig.
#
#################################################################################

"""Reusable detailed test-result report writers for the RAGAS harness."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from Final_Capstone_Project.Utility_Scripts.token_usage import TokenUsage, format_tokens


@dataclass
class ReportConfig:
    """Checkpoint-specific values threaded into every report entry."""

    capstone_label: str
    test_results_log: Path
    bm25_candidates: int
    vector_candidates: int
    fused_top_k: int
    llm_model: str
    judge_model: str
    engine_label: str = "ReAct tool-using"
    weight_bm25: float = 0.5
    weight_vector: float = 0.5


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


def _category_block(d: dict) -> list[str]:
    stats = d.get("category_stats", {})
    if not stats:
        return []
    lines = ["CATEGORY BREAKDOWN", "-" * 80]
    for category, category_result in sorted(stats.items()):
        lines.append(
            f"  {category:30s}: {category_result['passes']}/{category_result['total']} "
            f"({category_result['rate']:.0%})"
        )
    return lines


def _search_method_label(search_method: str) -> str:
    """Friendly display name for a search method (maps 'graph' -> 'Hybrid + Graph')."""
    labels = {
        "lexical": "Lexical",
        "semantic": "Semantic",
        "hybrid": "Hybrid",
        "graph": "Hybrid + Graph",
        "graph_enabled": "Hybrid + Graph",
    }
    return labels.get(search_method.lower(), search_method.capitalize())


def _ranking_note(config: ReportConfig, search_method: str) -> str:
    """Describe the fusion ranking in effect (only fused methods actually rank/fuse)."""
    if search_method.lower() in {"hybrid", "graph", "graph_enabled", "all"}:
        return (
            f"weighted fusion (BM25 {config.weight_bm25:g} / Vector {config.weight_vector:g})"
        )
    if search_method.lower() == "lexical":
        return "BM25 native order (no fusion)"
    if search_method.lower() == "semantic":
        return "vector-distance order (no fusion)"
    return "n/a"


def _config_line(config: ReportConfig, search_method: str) -> str:
    return (
        f"Config      : Engine={config.engine_label}, Search={_search_method_label(search_method)}, "
        f"candidates=BM25:{config.bm25_candidates}/Vector:{config.vector_candidates}, "
        f"Top-K={config.fused_top_k}, Ranking={_ranking_note(config, search_method)}, "
        f"answer={config.llm_model}, judge={config.judge_model}, metric=RAGAS DiscreteMetric"
    )


def prepend_test_results(test_results_log: Path, entry: str) -> None:
    """Place the newest detailed test result before older sessions."""
    test_results_log = Path(test_results_log)
    existing = test_results_log.read_text(encoding="utf-8") if test_results_log.is_file() else ""
    test_results_log.write_text(entry + existing, encoding="utf-8")


def append_single_test_results(
    config: ReportConfig,
    dataset_name: str,
    results: dict,
    search_method: str,
    main_question_source: str,
    paraphrase_counts: dict[str, int],
) -> None:
    """Append a structured log entry when evaluating a single dataset (main or paraphrased)."""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    entry = []
    entry.append("=" * 80)
    entry.append(f"{config.capstone_label} AGENT TEST SESSION  |  {date_str} {time_str}")
    entry.append("=" * 80)
    entry.append(
        f"Test type   : ReAct agent RAGAS correctness evaluation ({search_method.capitalize()} search; "
        f"{main_question_source}-generated source)"
    )
    entry.append(f"Date        : {date_str}")
    entry.append(f"Time        : {time_str}")
    entry.append(_config_line(config, search_method))
    entry.append("")
    entry.append("-" * 80)
    entry.append(f"EVALUATION DATASET: {dataset_name.upper()}")
    entry.append("-" * 80)
    entry.append(f"  Result      : {_rate(results)} PASSED")
    entry.append(f"  Output CSV  : {results['csv_path']}")
    entry.append(f"  Dataset source: {main_question_source}-generated questions")
    entry.append(_fail_block(results).rstrip("\n"))
    category_lines = _category_block(results)
    if category_lines:
        entry.append("")
        entry.extend(category_lines)
    if dataset_name.lower().startswith("paraphrased"):
        entry.append("")
        entry.append("PARAPHRASE COVERAGE PER ORIGINAL")
        entry.append("-" * 80)
        for original_question, count in paraphrase_counts.items():
            preview = original_question if len(original_question) <= 110 else original_question[:107] + "..."
            entry.append(f"  {count} paraphrased question(s) tested for original: {preview}")
    entry.append("=" * 80)
    entry.append("")

    prepend_test_results(config.test_results_log, "\n".join(entry) + "\n")
    print(f"Test results appended to: {config.test_results_log}")


def append_test_results(
    config: ReportConfig,
    originals: dict,
    paraphrases: dict,
    delta: float,
    verdict: str,
    originals_path: Path,
    variants_path: Path,
    paraphrase_counts: dict[str, int],
    search_method: str = "hybrid",
    main_question_source: str = "llm",
) -> None:
    """Append a structured, medium-detail entry to the detailed results log for each run."""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    orig_path = Path(originals_path)
    var_path = Path(variants_path)
    entry = []
    entry.append("=" * 80)
    entry.append(f"{config.capstone_label} AGENT TEST SESSION  |  {date_str} {time_str}")
    entry.append("=" * 80)
    entry.append(
        f"Test type   : Side-by-side ReAct agent RAGAS evaluation ({search_method.capitalize()} search; "
        f"{main_question_source}-generated source)"
    )
    entry.append(f"Date        : {date_str}")
    entry.append(f"Time        : {time_str}")
    entry.append(_config_line(config, search_method))
    entry.append("")
    entry.append("-" * 80)
    entry.append("ORIGINAL QUESTIONS")
    entry.append("-" * 80)
    entry.append(f"  Dataset     : {orig_path.name}")
    entry.append(f"  Dataset source: {main_question_source}-generated Main Questions")
    entry.append(f"  Result      : {_rate(originals)} PASSED")
    entry.append(f"  Output CSV  : {originals['csv_path']}")
    entry.append(_fail_block(originals).rstrip("\n"))
    original_category_lines = _category_block(originals)
    if original_category_lines:
        entry.append("")
        entry.extend(original_category_lines)
    entry.append("")
    entry.append("-" * 80)
    entry.append("PARAPHRASED QUESTIONS")
    entry.append("-" * 80)
    entry.append(f"  Dataset     : {var_path.name} (paraphrase rows)")
    entry.append(f"  Result      : {_rate(paraphrases)} PASSED")
    entry.append(f"  Output CSV  : {paraphrases['csv_path']}")
    entry.append(_fail_block(paraphrases).rstrip("\n"))
    paraphrase_category_lines = _category_block(paraphrases)
    if paraphrase_category_lines:
        entry.append("")
        entry.extend(paraphrase_category_lines)
    entry.append("")
    entry.append("PARAPHRASE COVERAGE PER ORIGINAL")
    entry.append("-" * 80)
    if paraphrase_counts:
        for original_question, count in paraphrase_counts.items():
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
    original_categories = originals.get("category_stats", {})
    paraphrase_categories = paraphrases.get("category_stats", {})
    if original_categories or paraphrase_categories:
        entry.append("")
        entry.append("CATEGORY-LEVEL DELTA ANALYSIS")
        entry.append("-" * 80)
        for category in sorted(set(original_categories) | set(paraphrase_categories)):
            original_result = original_categories.get(category, {"passes": 0, "total": 0, "rate": 0.0})
            paraphrase_result = paraphrase_categories.get(category, {"passes": 0, "total": 0, "rate": 0.0})
            entry.append(
                f"  {category:30s}: originals={original_result['rate']:.0%} "
                f"({original_result['passes']}/{original_result['total']}), "
                f"paraphrases={paraphrase_result['rate']:.0%} "
                f"({paraphrase_result['passes']}/{paraphrase_result['total']}), "
                f"delta={paraphrase_result['rate'] - original_result['rate']:+.0%}"
            )
    entry.append("=" * 80)
    entry.append("")

    prepend_test_results(config.test_results_log, "\n".join(entry) + "\n")
    print(f"Test results appended to: {config.test_results_log}")


# ── engine-vs-engine comparison report (Context-Aware vs Agentic Dynamic) ─────
@dataclass
class EngineRun:
    """One engine's evaluation of a dataset plus its cumulative eval effort.

    ``results`` is the dict returned by evaluate_dataset (passes/total/failures/
    csv_path/category_stats). ``plan_usage``/``answer_usage`` are the cumulative
    TokenUsage tallied across the eval (plan is empty for the Context-Aware engine)."""

    engine_label: str
    results: dict
    plan_usage: TokenUsage
    answer_usage: TokenUsage

    @property
    def passes(self) -> int:
        return self.results["passes"]

    @property
    def total(self) -> int:
        return self.results["total"]

    @property
    def rate(self) -> float:
        return self.passes / self.total if self.total else 0.0

    @property
    def llm_calls(self) -> int:
        return self.plan_usage.calls + self.answer_usage.calls

    @property
    def total_tokens(self) -> int:
        return self.plan_usage.total_tokens + self.answer_usage.total_tokens


def _engine_section(run: EngineRun, search_method: str) -> list[str]:
    """Render one engine's identity, result, effort, failures, and categories."""
    r = run.results
    lines = ["-" * 80, f"ENGINE: {run.engine_label.upper()}", "-" * 80]
    # The agent (which plans) has graph expansion ONLY when the selected method is
    # graph-enabled; annotate accordingly so the logged method reads correctly.
    graph_enabled = search_method.lower() in {"graph", "graph_enabled", "all"}
    if run.plan_usage.calls and graph_enabled:
        method_label = f"{_search_method_label(search_method)} base (graph expansion enabled)"
    else:
        method_label = _search_method_label(search_method)
    lines.append(f"  Search method : {method_label}")
    lines.append(f"  Result        : {_rate(r)} PASSED")
    lines.append(f"  Output CSV    : {r['csv_path']}")
    if run.plan_usage.calls:
        plan_note = f"plan={run.plan_usage.calls}"
    else:
        plan_note = "plan=0 (planning is only in the Agentic Engine)"
    lines.append(f"  LLM calls     : {plan_note}, answer={run.answer_usage.calls}")
    lines.append(
        f"  Tokens        : {format_tokens(run.total_tokens)} total "
        f"({format_tokens(run.plan_usage.input_tokens + run.answer_usage.input_tokens)} prompt + "
        f"{format_tokens(run.plan_usage.output_tokens + run.answer_usage.output_tokens)} completion)"
    )
    lines.append(_fail_block(r).rstrip("\n"))
    category_lines = _category_block(r)
    if category_lines:
        lines.append("")
        lines.extend(category_lines)
    return lines


def _comparison_section(context: EngineRun, agent: EngineRun) -> list[str]:
    """Render the side-by-side deltas: correctness, cost, efficiency, categories."""
    lines = ["-" * 80, "SIDE-BY-SIDE COMPARISON  (Context-Aware vs Agentic Dynamic)", "-" * 80]
    lines.append(f"  Context-Aware : {_rate(context.results)}   |   Agentic : {_rate(agent.results)}")
    delta = agent.rate - context.rate
    lines.append(f"  Correctness delta (agent - context) : {delta:+.0%}")
    lines.append(
        f"  LLM calls     : context={context.llm_calls}  |  agent={agent.llm_calls} "
        f"(+{agent.llm_calls - context.llm_calls})"
    )
    lines.append(
        f"  Tokens        : context={format_tokens(context.total_tokens)}  |  "
        f"agent={format_tokens(agent.total_tokens)}"
    )

    def per_1k(run: EngineRun) -> float:
        return (run.passes / (run.total_tokens / 1000)) if run.total_tokens else 0.0

    lines.append(
        f"  Passes / 1K tokens : context={per_1k(context):.2f}  |  agent={per_1k(agent):.2f}"
    )
    if delta < -0.05:
        verdict = "Agentic UNDERPERFORMS on correctness for this config."
    elif delta > 0.05:
        verdict = "Agentic IMPROVES correctness — weigh against its higher token/call cost."
    else:
        verdict = "Correctness COMPARABLE — Context-Aware wins on cost."
    lines.append(f"  Verdict       : {verdict}")

    context_categories = context.results.get("category_stats", {})
    agent_categories = agent.results.get("category_stats", {})
    categories = sorted(set(context_categories) | set(agent_categories))
    if categories:
        lines.append("")
        lines.append("  CATEGORY-LEVEL DELTA (agent - context)")
        lines.append("  " + "-" * 78)
        for category in categories:
            context_rate = context_categories.get(category, {"rate": 0.0})["rate"]
            agent_rate = agent_categories.get(category, {"rate": 0.0})["rate"]
            lines.append(
                f"    {category:30s}: context={context_rate:.0%}, "
                f"agent={agent_rate:.0%}, delta={agent_rate - context_rate:+.0%}"
            )
    return lines


def append_engine_comparison(
    config: ReportConfig,
    agentic_log: Path,
    dataset_name: str,
    context: EngineRun,
    agent: EngineRun,
    search_method: str,
    main_question_source: str,
) -> None:
    """Prepend a Context-Aware vs Agentic comparison entry to the agentic results log.

    Writes to a SEPARATE file (``agentic_log``) so the single-engine detailed log is
    left untouched; uses the same newest-first prepend semantics."""
    now = datetime.now()
    run_id = f"{now:%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    entry = []
    entry.append("=" * 80)
    entry.append(f"{config.capstone_label} — ENGINE COMPARISON SESSION  |  {now:%Y-%m-%d %H:%M:%S}")
    entry.append("=" * 80)
    entry.append(f"Run ID     : {run_id}")
    entry.append(f"Dataset    : {dataset_name}  ({main_question_source}-generated source)")
    entry.append("Engines    : Context-Aware retriever  vs  Agentic Dynamic (ReAct)")
    entry.append(
        f"Config     : Search={_search_method_label(search_method)}, "
        f"candidates=BM25:{config.bm25_candidates}/Vector:{config.vector_candidates}, "
        f"Top-K={config.fused_top_k}, Ranking={_ranking_note(config, search_method)}, "
        f"answer={config.llm_model}, judge={config.judge_model}, metric=RAGAS DiscreteMetric"
    )
    entry.append("")
    entry.extend(_engine_section(context, search_method))
    entry.append("")
    entry.extend(_engine_section(agent, search_method))
    entry.append("")
    entry.extend(_comparison_section(context, agent))
    entry.append("=" * 80)
    entry.append("")

    prepend_test_results(agentic_log, "\n".join(entry) + "\n")
    print(f"Engine comparison appended to: {agentic_log}")


def append_single_engine_result(
    config: ReportConfig,
    agentic_log: Path,
    dataset_name: str,
    engine: EngineRun,
    search_method: str,
    main_question_source: str,
) -> None:
    """Prepend a clean single-engine result entry to the agentic results log.

    Used for standalone Context-Aware or Agentic runs (no cross-engine comparison).
    Reuses the same per-engine section as the comparison writer so a lone run is
    formatted identically, and writes newest-first to ``agentic_log``."""
    now = datetime.now()
    run_id = f"{now:%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    entry = []
    entry.append("=" * 80)
    entry.append(f"{config.capstone_label} — ENGINE TEST SESSION  |  {now:%Y-%m-%d %H:%M:%S}")
    entry.append("=" * 80)
    entry.append(f"Run ID     : {run_id}")
    entry.append(f"Dataset    : {dataset_name}  ({main_question_source}-generated source)")
    entry.append(f"Engine     : {engine.engine_label}")
    entry.append(
        f"Config     : Search={_search_method_label(search_method)}, "
        f"candidates=BM25:{config.bm25_candidates}/Vector:{config.vector_candidates}, "
        f"Top-K={config.fused_top_k}, Ranking={_ranking_note(config, search_method)}, "
        f"answer={config.llm_model}, judge={config.judge_model}, metric=RAGAS DiscreteMetric"
    )
    entry.append("")
    entry.extend(_engine_section(engine, search_method))
    entry.append("=" * 80)
    entry.append("")

    prepend_test_results(agentic_log, "\n".join(entry) + "\n")
    print(f"Test results appended to: {agentic_log}")
