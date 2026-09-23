#################################################################################
# 
# Massachusetts Institute of Technology - xPRO
# RAG and Context Engineering: Designing and Building Production-Grade AI Systems
# 
# Student: Miguel Herrera
# Section: B
# Date and Time: 2026-09-22 19:31:56 -07:00
#
# Description: Conversation history-aware retriever for the Wikipedia RAG engine.
#              Wraps the shared HybridRetriever with rolling chat memory: the chat
#              loop folds recent turns into BOTH the retrieval query (so follow-ups
#              resolve) and the grounded answer prompt. It performs NO autonomous
#              planning (that is the Agentic Dynamic Retriever's job). Exposes query()
#              (stateless, drop-in for the RAGAS harness) and chat() (interactive,
#              with conversation history), and tracks cumulative evaluation token
#              usage (reset_eval_usage / get_eval_usage) for the results log.
#
#################################################################################

"""Conversation history-aware (non-agentic) retrieval over the Wikipedia corpus."""
from __future__ import annotations

import configparser
import os
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from Final_Capstone_Project.Retrieval_Methods.hybrid_retrieval import HybridRetriever, get_embeddings
from Final_Capstone_Project.Utility_Scripts.token_usage import (
    TokenUsage,
    banner,
    format_plan_answer_usage,
    format_session_total,
    usage_from_message,
)


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "retrieval.conf"
_config = configparser.ConfigParser()
_config.read(CONFIG_PATH)
LLM_MODEL = _config.get("llm", "model", fallback="openai/gpt-5.4-mini")
EMBEDDING_MODEL = _config.get("embedding", "model", fallback="openai/text-embedding-3-small")
TEMPERATURE = _config.getfloat("llm", "temperature", fallback=0.2)
MAX_TOKENS = _config.getint("llm", "max_tokens", fallback=1024)

# The Chroma directory is resolved relative to this module unless the caller overrides it.
CHROMA_DIR = str(CONFIG_PATH.parent / "Capstone_Database" / "Capstone_Chroma_DB")

# Number of prior conversation turns folded into retrieval + answer prompts (chat only).
HISTORY_WINDOW = _config.getint("context_aware", "history_window", fallback=6)

ANSWER_SYSTEM = (
    "You are a helpful assistant for a Wikipedia retrieval engine. Answer the question "
    "using ONLY the provided Wikipedia article excerpts, and quote from them where you "
    "can. Consider the prior conversation, which may add important context for resolving "
    "the current question. If the retrieved excerpts do not contain enough information, "
    "say so explicitly and do not speculate or use outside knowledge. If the user is "
    "simply asking to exit, answer exactly: Exiting"
)


def console_log(message: str, level: str = "INFO") -> None:
    """Write a timestamped diagnostic message to the console."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}", flush=True)


# ── conversation history-aware retriever (grounded RAG, no planning) ──────────
class ContextAwareRetriever:
    """A history-aware retriever that grounds answers in HybridRetriever context and
    folds recent conversation turns into each retrieval query and answer prompt. It does
    NOT plan or select tools. Exposes query() (stateless, drop-in for the RAGAS harness)
    and a chat() loop (with conversation history)."""

    def __init__(
        self,
        docs: list[dict],
        *,
        search_method: str = "hybrid",
        llm_model: str = LLM_MODEL,
        embedding_model: str = EMBEDDING_MODEL,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
        history_window: int = HISTORY_WINDOW,
        api_key: str | None = None,
        base_url: str = OPENROUTER_BASE_URL,
        chroma_dir: str | None = None,
    ):
        self._search_method = str(search_method).lower()
        resolved_api_key = api_key or os.environ["OPENROUTER_API_KEY"]
        resolved_chroma_dir = chroma_dir or CHROMA_DIR
        self._llm = ChatOpenAI(
            model=llm_model,
            api_key=resolved_api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        embedding_function = get_embeddings(embedding_model)
        # "all" is not a HybridRetriever mode on its own; map it to hybrid retrieval,
        # matching the agent's handling of the shared retrieval stack.
        retrieve_method = "hybrid" if self._search_method == "all" else self._search_method
        self._hybrid = HybridRetriever(
            docs,
            search_method=retrieve_method,
            llm_model=llm_model,
            api_key=resolved_api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            embedding_function=embedding_function,
            chroma_dir=resolved_chroma_dir,
        )
        self._history_window = max(0, int(history_window))
        # Cumulative token accounting across an evaluation run (every query() adds into
        # this). This engine never plans, so only the answer usage accrues. Reset via
        # reset_eval_usage() before an eval and read via get_eval_usage() after.
        self._eval_answer_usage = TokenUsage()

    def reset_eval_usage(self) -> None:
        """Clear the cumulative evaluation token accounting before an eval run."""
        self._eval_answer_usage = TokenUsage()

    def get_eval_usage(self) -> tuple[TokenUsage, TokenUsage]:
        """Return the cumulative (plan_usage, answer_usage); plan is always empty here."""
        return TokenUsage(), self._eval_answer_usage

    @staticmethod
    def _format_history(history: list[dict]) -> str:
        """Render conversation turns as 'Role: content' lines (agent-compatible idiom)."""
        return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in history)

    def _answer_with_history(self, question: str, history: list[dict]) -> str:
        """Retrieve using history-conditioned query, then answer grounded on the excerpts."""
        recent = history[-self._history_window:] if self._history_window else []
        history_text = self._format_history(recent)

        # 1) History-aware retrieval: fold recent turns into the search query so
        #    follow-ups (pronouns, ellipsis) resolve against prior context.
        search_query = f"{history_text}\nUser: {question}".strip() if history_text else question
        context = self._hybrid.retrievedContext(search_query)

        # 2) History-aware grounded answer (no planning).
        parts = []
        if history_text:
            parts.append(f"Prior conversation:\n{history_text}")
        parts.append(f"Retrieved Wikipedia excerpts:\n{context}")
        parts.append(f"Question: {question}")
        prompt_text = ANSWER_SYSTEM + "\n\n" + "\n\n".join(parts)
        response = self._llm.invoke([
            SystemMessage(content=ANSWER_SYSTEM),
            HumanMessage(content="\n\n".join(parts)),
        ])
        answer = response.content if hasattr(response, "content") else str(response)
        answer_usage = usage_from_message(response, prompt_text)
        return answer, answer_usage

    def query(self, question: str) -> str:
        """Stateless single-shot answer (drop-in for the RAGAS evaluation harness).

        Evaluation questions are independent, so no conversation history is applied
        here; this mirrors plain single-shot retrieval for a fair comparison."""
        answer, answer_usage = self._answer_with_history(question, [])
        self._eval_answer_usage.add(answer_usage)
        return answer

    def chat(self) -> None:
        """Interactive multi-turn chat that keeps conversation history across turns.

        After each turn, prints the estimated tokens used by the single answer call."""
        banner("Starting chat with AI")
        print("Wikipedia context-aware retriever (chat with history). Type your question; 'exit'/'quit' to stop.\n")
        conversation_history: list[dict] = []
        # This engine never plans, so the session plan total stays empty.
        session_plan = TokenUsage()
        session_answer = TokenUsage()
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break
            if user_input.lower() in {"exit", "quit"}:
                print("Goodbye.")
                break
            if not user_input:
                continue
            try:
                answer, answer_usage = self._answer_with_history(user_input, conversation_history)
            except Exception as exc:
                console_log(f"Error: {exc}", "ERROR")
                continue
            print(f"\nAssistant: {answer}\n")
            session_answer.add(answer_usage)
            # This engine never plans, so the plan row always reports zero usage.
            print(format_plan_answer_usage(TokenUsage(), answer_usage))
            if answer.strip() == "Exiting":
                print("Goodbye.")
                break
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": answer})

        print(format_session_total(session_plan, session_answer))
