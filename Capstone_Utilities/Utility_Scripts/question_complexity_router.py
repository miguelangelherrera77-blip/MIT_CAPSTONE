"""Deterministic, local routing heuristic for Checkpoint 8.1 chat.

This classifier makes no model or network calls. It sends questions with explicit
multi-part, comparison, cross-source, or synthesis signals to the Agentic Dynamic
planner. Other questions use the history-aware Context-Aware path. The result is a
cost/complexity heuristic, not a correctness or safety judgment.
"""

# #################################################################################
#
# Massachusetts Institute of Technology - xPRO
# RAG and Context Engineering: Designing and Building Production-Grade AI Systems
#
# Student: Miguel Herrera
# Section: B
# Date and Time: 2026-10-12 06:48:53 -07:00
# Header reviewed: 2026-10-12 11:34:17 -07:00
#
# Description: Classifies Checkpoint 8.1 chat questions locally and selects the
#              Context-Aware or Agentic Dynamic retrieval path without an LLM call.
#
# #################################################################################

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

EngineName = Literal["context", "agent"]

_COMPLEXITY_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "comparison or relationship request",
        re.compile(
            r"\b(compare|contrast|differences?|similarities|relationship|"
            r"versus|vs\.?|pros and cons|trade-?offs?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "cross-source synthesis request",
        re.compile(
            r"\b(across (?:multiple )?(?:documents|articles|sources)|"
            r"multiple (?:documents|articles|sources)|synthesize|"
            r"combine (?:the )?(?:evidence|information|findings))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "multiple requested analysis steps",
        re.compile(
            r"\b(?:and then|then also|also)\s+"
            r"(?:explain|compare|identify|describe|summari[sz]e|analy[sz]e|"
            r"evaluate|determine|trace)\b",
            re.IGNORECASE,
        ),
    ),
)

_INTERROGATIVE_RE = re.compile(
    r"\b(?:what|who|when|where|why|how|which)\b",
    re.IGNORECASE,
)
_TASK_CONNECTOR_RE = re.compile(
    r"\b(?:and|also|then|as well as|in addition)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RoutingDecision:
    """The selected engine and the deterministic rule(s) that selected it."""

    engine: EngineName
    reasons: tuple[str, ...]


def classify_question(question: str) -> RoutingDecision:
    """Route an input using explicit, explainable text patterns only.

    A question is considered complex if it contains a comparison/cross-source
    signal, multiple interrogative intents connected by a task connector, or an
    explicit second analysis step. A lone "what/how" word is not sufficient.
    All other non-empty questions go to Context-Aware retrieval, which already
    supports conversation history. Ambiguous-looking cases are not guaranteed to
    be detected, so callers should keep manual engine selection available.
    """
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    normalized = question.strip()
    if not normalized:
        raise ValueError("question must not be empty")

    reasons: list[str] = []
    # These semantic task words are strong signals even in a single sentence.
    for reason, pattern in _COMPLEXITY_RULES[:2]:
        if pattern.search(normalized):
            reasons.append(reason)

    # Multiple question marks almost always indicate more than one user request.
    if normalized.count("?") > 1:
        reasons.append("multiple question marks")

    # Require both multiple interrogative intents and a connector, so an ordinary
    # question containing words such as "what" and "how" is not over-routed.
    if (
        len(_INTERROGATIVE_RE.findall(normalized)) > 1
        and _TASK_CONNECTOR_RE.search(normalized)
    ):
        reasons.append("multiple connected question intents")

    if _COMPLEXITY_RULES[2][1].search(normalized):
        reasons.append(_COMPLEXITY_RULES[2][0])

    if reasons:
        return RoutingDecision("agent", tuple(dict.fromkeys(reasons)))
    return RoutingDecision("context", ("single-intent/default path",))
