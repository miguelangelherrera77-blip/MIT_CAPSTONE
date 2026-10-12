#################################################################################
# 
# Massachusetts Institute of Technology - xPRO
# RAG and Context Engineering: Designing and Building Production-Grade AI Systems
# 
# Student: Miguel Herrera
# Section: B
# Date and Time: 2026-09-22 19:31:56 -07:00
#
# Description: Token-usage accounting and console formatting for the retrieval
#              engines. Reads exact prompt/completion counts from a LangChain
#              AIMessage when the provider reports them (naming the OpenRouter
#              provider and model), and otherwise estimates locally with tiktoken
#              (offline, no API cost), clearly labeling estimated figures. Provides
#              the TokenUsage accumulator, a K/M/B token formatter, a startup banner,
#              per-turn and cumulative session panels (plan vs answer breakdown), so
#              both the Context-Aware and Agentic engines report usage consistently.
#
#################################################################################

"""Token-usage accounting and console formatting for the retrieval engines."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── local tokenizer fallback (best-effort, offline) ─────────────────────────
_LOCAL_ENCODER = None
_LOCAL_ENCODER_READY = False


def _get_local_encoder():
    """Return a cached tiktoken encoder, or None if unavailable (kept fully offline)."""
    global _LOCAL_ENCODER, _LOCAL_ENCODER_READY
    if _LOCAL_ENCODER_READY:
        return _LOCAL_ENCODER
    _LOCAL_ENCODER_READY = True
    try:
        import tiktoken

        _LOCAL_ENCODER = tiktoken.get_encoding("cl100k_base")
    except Exception:
        # No tiktoken, or its vocab could not be loaded offline: fall back to a
        # coarse word-based heuristic in estimate_tokens().
        _LOCAL_ENCODER = None
    return _LOCAL_ENCODER


def estimate_tokens(text: str) -> int:
    """Estimate the token count of a string locally (no API call).

    Uses tiktoken when available; otherwise a simple ~4-chars-per-token heuristic."""
    if not text:
        return 0
    encoder = _get_local_encoder()
    if encoder is not None:
        return len(encoder.encode(text))
    # Heuristic fallback: roughly 4 characters per token, at least one token per word.
    return max(len(text.split()), (len(text) + 3) // 4)


def format_tokens(count: int) -> str:
    """Abbreviate a token count with K/M/B/T suffixes for compact, readable output.

    Counts below 1,000 are shown exactly (e.g. 34). Larger values use one decimal,
    trimming a trailing '.0' (e.g. 3,280 -> 3.3K, 1,000,000 -> 1M, 4,610 -> 4.6K)."""
    number = float(count)
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(number) >= threshold:
            scaled = number / threshold
            text = f"{scaled:.1f}".rstrip("0").rstrip(".")
            return f"{text}{suffix}"
    return str(int(number))


def _message_text(message: Any) -> str:
    """Extract the text content of an AIMessage (or plain string) for estimation."""
    content = getattr(message, "content", message)
    return content if isinstance(content, str) else str(content)


def _model_name(message: Any) -> str:
    """Best-effort model name reported by the provider on a LangChain AIMessage."""
    metadata = getattr(message, "response_metadata", None) or {}
    return str(metadata.get("model_name") or metadata.get("model") or "").strip()


def usage_from_message(message: Any, prompt_text: str = "") -> "TokenUsage":
    """Build a TokenUsage from a LangChain AIMessage.

    Prefers the provider-reported usage_metadata (exact). When that is missing, falls
    back to a local tiktoken estimate of the prompt (input) and the response (output),
    and marks the result as estimated. The provider's model name is captured when the
    response reports it."""
    model = _model_name(message)
    usage_metadata = getattr(message, "usage_metadata", None)
    if usage_metadata:
        input_tokens = int(usage_metadata.get("input_tokens", 0) or 0)
        output_tokens = int(usage_metadata.get("output_tokens", 0) or 0)
        if input_tokens or output_tokens:
            return TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                calls=1,
                estimated=False,
                model=model,
            )

    # Fallback: estimate both sides locally.
    return TokenUsage(
        input_tokens=estimate_tokens(prompt_text),
        output_tokens=estimate_tokens(_message_text(message)),
        calls=1,
        estimated=True,
        model=model,
    )


@dataclass
class TokenUsage:
    """Accumulates input/output token counts across one or more LLM calls.

    ``estimated`` is True when ANY contributing figure came from the local tokenizer
    rather than provider-reported usage, so displays can label the total honestly."""

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    estimated: bool = False
    model: str = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, other: "TokenUsage") -> "TokenUsage":
        """Merge another usage record into this one, summing tokens and call counts."""
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.calls += other.calls
        self.estimated = self.estimated or other.estimated
        if other.model and not self.model:
            self.model = other.model
        return self

    def label(self) -> str:
        """Return the source label appended to displayed figures."""
        return " (Local Tokenizer estimated)" if self.estimated else ""

    def describe(self) -> str:
        """One-line 'total (prompt + completion)' description with source label.

        Uses the standard API usage vocabulary: prompt / completion / total tokens."""
        return (
            f"{format_tokens(self.total_tokens)} total "
            f"({format_tokens(self.input_tokens)} prompt + "
            f"{format_tokens(self.output_tokens)} completion){self.label()}"
        )


# ── read-friendly console panels ─────────────────────────────────────────────
_PANEL_WIDTH = 60

# All LLM calls in this project route through OpenRouter (see OPENROUTER_BASE_URL).
PROVIDER_NAME = "OpenRouter"


def _source_note(estimated: bool, model: str = "") -> str:
    """Describe where the token counts came from, naming the provider/model if known.

    Provider-reported counts name the OpenRouter provider (and the model when the
    response reported it). Estimated counts name the local tokenizer instead, since
    those figures were computed locally rather than returned by the provider."""
    if estimated:
        return "Local Tokenizer estimate"
    if model:
        return f"{PROVIDER_NAME} · {model} (provider-reported)"
    return f"{PROVIDER_NAME} (provider-reported)"


def _panel(title: str, rows: list[str]) -> str:
    """Render a titled box around the given rows for tidy console output."""
    top = f"┌─ {title} " + "─" * max(0, _PANEL_WIDTH - len(title) - 4) + "┐"
    bottom = "└" + "─" * (_PANEL_WIDTH - 2) + "┘"
    body = "\n".join(f"  {row}" for row in rows)
    return f"{top}\n{body}\n{bottom}"


def banner(label: str, width: int = _PANEL_WIDTH) -> None:
    """Print two blank lines then an asterisk-bordered box enclosing ``label``."""
    inner_width = max(width, len(label) + 4) - 2
    border = "*" * (inner_width + 2)
    print("\n\n" + border)
    print("*" + label.center(inner_width) + "*")
    print(border, flush=True)


def format_plan_answer_usage(
    plan: "TokenUsage",
    answer: "TokenUsage",
    title: str = "Token usage",
) -> str:
    """Render a plan-vs-answer token breakdown panel under the given title.

    Used per-turn ("Token usage") and for the end-of-session cumulative summary."""
    total = plan.total_tokens + answer.total_tokens
    estimated = plan.estimated or answer.estimated
    rows = []

    if plan.calls:
        plan_calls = f"{plan.calls} call" + ("s" if plan.calls != 1 else "")
        rows.append(
            f"{'Plan':<10} {plan_calls:<9} {format_tokens(plan.total_tokens):>6} total   "
            f"({format_tokens(plan.input_tokens)} prompt + {format_tokens(plan.output_tokens)} completion) "
            f"[Agentic Engine]"
        )
    else:
        rows.append(
            f"{'Plan':<10} {'0 calls':<9}      0 total   "
            f"(no plan tokens used — planning is only in the Agentic Engine)"
        )

    answer_calls = f"{answer.calls} call" + ("s" if answer.calls != 1 else "")
    rows.append(
        f"{'Answer':<10} {answer_calls:<9} {format_tokens(answer.total_tokens):>6} total   "
        f"({format_tokens(answer.input_tokens)} prompt + {format_tokens(answer.output_tokens)} completion)"
    )

    if total:
        plan_share = plan.total_tokens / total
        answer_share = answer.total_tokens / total
        rows.append(
            f"{'Total':<10} {'':<9} {format_tokens(total):>6} total   "
            f"(plan {plan_share:.0%} · answer {answer_share:.0%})"
        )
    model = answer.model or plan.model
    rows.append(f"{'Source':<10} {_source_note(estimated, model)}")
    return _panel(title, rows)


def format_session_total(plan: "TokenUsage", answer: "TokenUsage") -> str:
    """Render the cumulative plan/answer token totals for the whole chat session."""
    return format_plan_answer_usage(plan, answer, title="Session token usage (cumulative)")
