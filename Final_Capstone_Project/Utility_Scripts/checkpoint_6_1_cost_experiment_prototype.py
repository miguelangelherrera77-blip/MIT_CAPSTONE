#################################################################################
# 
# Massachusetts Institute of Technology - xPRO
# RAG and Context Engineering: Designing and Building Production-Grade AI Systems
# 
# Student: Miguel Herrera
# Section: B
# Date and Time: 2026-09-26 00:00:00 -07:00
#
# Description: Prototype for the Checkpoint 6.1 model-ladder and mixed-model cost
#              experiment. Defines the experiment contract, per-role token/cost
#              accounting, and a reviewable synthetic comparison without calling an API.
#
#################################################################################

"""Prototype Lab 6.2-style cost experiment for the Checkpoint 6.1 agent."""
from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Callable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Final_Capstone_Project.Utility_Scripts.token_usage import TokenUsage, format_tokens


@dataclass(frozen=True)
class ModelProfile:
    name: str
    input_price_per_mtok: float
    output_price_per_mtok: float


@dataclass(frozen=True)
class ModelPair:
    name: str
    planner: ModelProfile
    answer: ModelProfile


@dataclass
class ExperimentResult:
    configuration: str
    questions: int
    passes: int
    planner_usage: TokenUsage
    answer_usage: TokenUsage
    estimated_cost_usd: float
    latency_seconds: float = 0.0
    workflow_steps: int = 0

    @property
    def total_tokens(self) -> int:
        return self.planner_usage.total_tokens + self.answer_usage.total_tokens

    @property
    def pass_rate(self) -> float:
        return self.passes / self.questions if self.questions else 0.0


MODEL_LADDER = (
    ModelProfile("google/gemma-4-31b-it:free", 0.0, 0.0),
    ModelProfile("openai/gpt-4o-mini", 0.15, 0.60),
    ModelProfile("openai/gpt-5.4-mini", 0.20, 0.80),
)
EXPENSIVE_MODEL = ModelProfile("openai/gpt-5.2-pro", 10.0, 30.0)
MIXED_CONFIGURATION = ModelPair(
    "mixed: strong planner + economical answer",
    planner=EXPENSIVE_MODEL,
    answer=MODEL_LADDER[1],
)


Runner = Callable[[Sequence[str], str, str], ExperimentResult]


def estimate_cost_usd(
    planner_usage: TokenUsage,
    answer_usage: TokenUsage,
    planner: ModelProfile,
    answer: ModelProfile,
) -> float:
    """Estimate cost from input/output token counts and model rates."""
    planner_cost = (
        planner_usage.input_tokens * planner.input_price_per_mtok
        + planner_usage.output_tokens * planner.output_price_per_mtok
    ) / 1_000_000
    answer_cost = (
        answer_usage.input_tokens * answer.input_price_per_mtok
        + answer_usage.output_tokens * answer.output_price_per_mtok
    ) / 1_000_000
    return planner_cost + answer_cost


def configuration_pairs() -> list[ModelPair]:
    """Return all-single-model ladder cases plus the mixed-model case."""
    pairs = [ModelPair(profile.name, profile, profile) for profile in MODEL_LADDER]
    pairs.append(MIXED_CONFIGURATION)
    return pairs


def run_cost_experiment(
    questions: Sequence[str],
    runner: Runner,
) -> list[ExperimentResult]:
    """Run a small question suite through each model configuration.

    ``runner`` is the future Checkpoint 6.1 adapter. It receives the questions,
    planner model name, and answer model name, and returns measured usage/results.
    """
    results = []
    for pair in configuration_pairs():
        result = runner(questions, pair.planner.name, pair.answer.name)
        result.configuration = pair.name
        result.estimated_cost_usd = estimate_cost_usd(
            result.planner_usage,
            result.answer_usage,
            pair.planner,
            pair.answer,
        )
        results.append(result)
    return results


def format_report(results: Sequence[ExperimentResult]) -> str:
    """Render the comparison that the future 6.1 report can log."""
    lines = [
        "CHECKPOINT 6.1 COST EXPERIMENT PROTOTYPE",
        "Configuration                                      Result   Tokens   Cost",
        "-" * 78,
    ]
    for result in results:
        lines.append(
            f"{result.configuration:48s} "
            f"{result.passes}/{result.questions} ({result.pass_rate:.0%})  "
            f"{format_tokens(result.total_tokens):>7s}  "
            f"${result.estimated_cost_usd:.5f}"
        )
    return "\n".join(lines)


def synthetic_runner(
    questions: Sequence[str],
    planner_model: str,
    answer_model: str,
) -> ExperimentResult:
    """Provide deterministic sample data so the prototype can be reviewed offline."""
    planner_tokens = 2_400 if "pro" in planner_model else 1_500
    answer_tokens = 1_600 if "mini" in answer_model else 1_800
    passes = len(questions) if "pro" in planner_model else max(0, len(questions) - 1)
    return ExperimentResult(
        configuration=f"{planner_model} -> {answer_model}",
        questions=len(questions),
        passes=passes,
        planner_usage=TokenUsage(input_tokens=planner_tokens, output_tokens=600, calls=len(questions)),
        answer_usage=TokenUsage(input_tokens=answer_tokens, output_tokens=500, calls=len(questions)),
        estimated_cost_usd=0.0,
    )


if __name__ == "__main__":
    prototype_questions = [
        "What information can be verified from the retrieved Wikipedia evidence?",
        "Quote the relevant sentence exactly if it is present.",
        "What information is unavailable from the corpus?",
        "Which retrieved documents appear to conflict?",
    ]
    print(format_report(run_cost_experiment(prototype_questions, synthetic_runner)))
    print("\nNo API calls were made; synthetic results demonstrate the proposed report shape.")
