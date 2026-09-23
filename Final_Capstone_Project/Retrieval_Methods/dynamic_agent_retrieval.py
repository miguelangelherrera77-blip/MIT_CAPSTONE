#################################################################################
# 
# Massachusetts Institute of Technology - xPRO
# RAG and Context Engineering: Designing and Building Production-Grade AI Systems
# 
# Student: Miguel Herrera
# Section: B
# Date and Time: 2026-09-22 19:31:56 -07:00
#
# Description: Dynamic ReAct tool-using agent for the Wikipedia RAG engine. A
#              LangGraph plan node selects one action per step (retrieve,
#              graph_expand, clarify, or answer) with no upfront plan, reusing the
#              shared HybridRetriever. The graph_expand tool is enabled ONLY for the
#              graph-enabled search method; lexical/semantic/hybrid run graph-free.
#              The plan/answer reasoning uses the configurable [agent] agent_model
#              (falling back to the [llm] model). Exposes query() (batch, drop-in for
#              the RAGAS harness) and chat() (interactive, with conversation history),
#              and tracks per-turn and cumulative evaluation token usage.
#
#################################################################################

"""Dynamic (ReAct) tool-using agent retrieval over the Wikipedia corpus."""
from __future__ import annotations

import configparser
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from Final_Capstone_Project.Retrieval_Methods.hybrid_retrieval import HybridRetriever, get_embeddings
from Final_Capstone_Project.Utility_Scripts.token_usage import (
    TokenUsage,
    banner,
    format_plan_answer_usage,
    format_session_total,
    usage_from_message,
)


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Search methods that enable the agent's graph_expand tool. For all other methods
# (lexical / semantic / hybrid) the agent runs graph-free.
GRAPH_ENABLED_METHODS = {"graph", "graph_enabled", "all"}
CONFIG_PATH = Path(__file__).resolve().parents[1] / "retrieval.conf"
_config = configparser.ConfigParser()
_config.read(CONFIG_PATH)
LLM_MODEL = _config.get("llm", "model", fallback="openai/gpt-5.4-mini")
EMBEDDING_MODEL = _config.get("embedding", "model", fallback="openai/text-embedding-3-small")
TEMPERATURE = _config.getfloat("llm", "temperature", fallback=0.2)
MAX_TOKENS = _config.getint("llm", "max_tokens", fallback=1024)

# The Chroma directory is resolved relative to this module unless the caller overrides it.
CHROMA_DIR = str(CONFIG_PATH.parent / "Capstone_Database" / "Capstone_Chroma_DB")

# ── agent limits (configurable via the [agent] section of retrieval.conf) ────
# The reasoning model for the agent's plan/answer nodes. Falls back to the shared
# [llm] model when [agent] agent_model is unset. This governs ONLY the agent's own
# LLM; the retrieval stack (HybridRetriever) keeps using the [llm] model.
AGENT_MODEL = _config.get("agent", "agent_model", fallback=LLM_MODEL)
TOP_K = _config.getint("agent", "top_k", fallback=5)                    # chunks pulled per retrieval action
SEED_K = _config.getint("agent", "seed_k", fallback=2)                  # seed chunks fed to graph expansion
MAX_ITERATIONS = _config.getint("agent", "max_iterations", fallback=5)  # safety cap on plan/act rounds
DEBUG = _config.getboolean("agent", "debug", fallback=True)             # per-step agent tracing to the console

ANSWER_SYSTEM = (
    "You are a helpful assistant for a Wikipedia retrieval engine. Answer the question "
    "using ONLY the provided Wikipedia article excerpts, and quote from them where you "
    "can. Consider any clarifications, which may add important details. If the retrieved "
    "excerpts do not contain enough information, say so explicitly and do not speculate "
    "or use outside knowledge. If the user is simply asking to exit, answer exactly: Exiting"
)

PLAN_SYSTEM = (
    "You are a research agent for a Wikipedia retrieval engine. You help the user find "
    "information about significant people, places, and topics using a search database of "
    "Wikipedia article chunks.\n\n"
    "You are given the user's question (and prior conversation), the queries already "
    "executed, the chunks retrieved so far, and any clarifications from the user. Decide "
    "what to do next and respond with a JSON object ONLY:\n"
    "{\n"
    '  "action": "retrieve" | "graph_expand" | "clarify" | "answer",\n'
    '  "queries": ["query1", "query2"],   // 1-3 NEW queries; only for action=="retrieve"; do not repeat executed ones\n'
    '  "clarification": "question text",    // only for action=="clarify"\n'
    '  "reasoning": "brief explanation"\n'
    "}\n\n"
    "Guidelines:\n"
    '- "retrieve": you need more information via hybrid BM25 + vector search.\n'
    '- "graph_expand": the question needs neighboring or linked article context; expands '
    "the most relevant seed chunks with adjacent and referenced-article chunks.\n"
    '- "clarify": the message is not really a question and you must ask for more input. '
    "Use ONLY as a last resort, and only AFTER attempting to query the database.\n"
    '- "answer": you have enough information, or the user asked to exit."'
)

# Graph-free variant of the plan prompt: used when the selected search method does not
# enable graph expansion, so the planner is never offered the "graph_expand" action.
PLAN_SYSTEM_NO_GRAPH = (
    "You are a research agent for a Wikipedia retrieval engine. You help the user find "
    "information about significant people, places, and topics using a search database of "
    "Wikipedia article chunks.\n\n"
    "You are given the user's question (and prior conversation), the queries already "
    "executed, the chunks retrieved so far, and any clarifications from the user. Decide "
    "what to do next and respond with a JSON object ONLY:\n"
    "{\n"
    '  "action": "retrieve" | "clarify" | "answer",\n'
    '  "queries": ["query1", "query2"],   // 1-3 NEW queries; only for action=="retrieve"; do not repeat executed ones\n'
    '  "clarification": "question text",    // only for action=="clarify"\n'
    '  "reasoning": "brief explanation"\n'
    "}\n\n"
    "Guidelines:\n"
    '- "retrieve": you need more information via the configured search method.\n'
    '- "clarify": the message is not really a question and you must ask for more input. '
    "Use ONLY as a last resort, and only AFTER attempting to query the database.\n"
    '- "answer": you have enough information, or the user asked to exit."'
)


def console_log(message: str, level: str = "INFO") -> None:
    """Write a timestamped diagnostic message to the console."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}", flush=True)


# ── agent state ──────────────────────────────────────────────────────────────
class _AgentState(TypedDict):
    conversation_history: list[dict]
    clarification_history: list[str]
    pending_queries: list[str]
    pending_graph_expand: bool
    executed_queries: list[str]
    doc_bodies: dict[str, str]
    iterations: int
    mode: str
    next_action: str
    clarification_question: str
    answer: str
    done: bool


# ── ReAct tool-using agent (LangGraph: plan -> retrieve/graph_expand/clarify/answer) ──
class ToolUsingAgent:
    """A ReAct tool-using agent that dynamically selects a retrieval or answer action
    each step over the Wikipedia corpus, reusing the shared HybridRetriever. Exposes a
    query() method (drop-in for the RAGAS harness) and a chat() loop (with history)."""

    def __init__(
        self,
        docs: list[dict],
        *,
        search_method: str = "hybrid",
        llm_model: str = LLM_MODEL,
        agent_model: str = AGENT_MODEL,
        embedding_model: str = EMBEDDING_MODEL,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
        top_k: int = TOP_K,
        seed_k: int = SEED_K,
        max_iterations: int = MAX_ITERATIONS,
        debug: bool = DEBUG,
        api_key: str | None = None,
        base_url: str = OPENROUTER_BASE_URL,
        chroma_dir: str | None = None,
    ):
        self._search_method = str(search_method).lower()
        resolved_api_key = api_key or os.environ["OPENROUTER_API_KEY"]
        resolved_chroma_dir = chroma_dir or CHROMA_DIR
        # The agent's plan/answer reasoning uses agent_model; the retrieval stack
        # below keeps using llm_model (the shared [llm] model).
        self._llm = ChatOpenAI(
            model=agent_model,
            api_key=resolved_api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        embedding_function = get_embeddings(embedding_model)
        # Graph expansion is available ONLY when the selected method is graph-enabled
        # ("Hybrid with Graph Enabled"). For lexical/semantic/hybrid the agent has no
        # graph_expand tool, so those settings are a fair, graph-free comparison.
        self._graph_enabled = self._search_method in GRAPH_ENABLED_METHODS
        # The "retrieve" tool uses the selected base search method (graph maps to hybrid
        # seeds; the graph work happens in the separate graph_expand tool below).
        retrieve_method = "hybrid" if self._search_method in {"all", "graph", "graph_enabled"} else self._search_method
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
        # Only build the graph-augmented retriever when graph expansion is enabled;
        # this also avoids loading the knowledge graph for non-graph runs.
        self._graph_retriever = None
        if self._graph_enabled:
            self._graph_retriever = HybridRetriever(
                docs,
                search_method="graph",
                llm_model=llm_model,
                api_key=resolved_api_key,
                base_url=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                embedding_function=embedding_function,
                chroma_dir=resolved_chroma_dir,
            )
        self._top_k = top_k
        self._seed_k = seed_k
        self._max_iterations = max_iterations
        self._debug = debug
        # Per-turn token accounting: plan-node calls vs the final answer-node call.
        # Reset at the start of each _run() and read by chat() after the turn.
        self._plan_usage = TokenUsage()
        self._answer_usage = TokenUsage()
        # Cumulative token accounting across an evaluation run (every query() adds into
        # these). Independent of the per-turn accumulators above; reset via
        # reset_eval_usage() before an eval and read via get_eval_usage() after.
        self._eval_plan_usage = TokenUsage()
        self._eval_answer_usage = TokenUsage()
        self._graph = self._build_graph()

    def reset_eval_usage(self) -> None:
        """Clear the cumulative evaluation token accounting before an eval run."""
        self._eval_plan_usage = TokenUsage()
        self._eval_answer_usage = TokenUsage()

    def get_eval_usage(self) -> tuple[TokenUsage, TokenUsage]:
        """Return the cumulative (plan_usage, answer_usage) tallied across queries."""
        return self._eval_plan_usage, self._eval_answer_usage

    def _build_graph(self):
        def retrieve_node(state: _AgentState) -> dict:
            doc_bodies = dict(state["doc_bodies"])
            executed = list(state["executed_queries"])
            for query in state["pending_queries"]:
                if self._debug:
                    console_log(f"[agent] retrieving for: {query!r}")
                for doc_id, content, _ in self._hybrid.getTopK(query, self._top_k):
                    doc_bodies[doc_id] = content
                executed.append(query)
            return {
                "doc_bodies": doc_bodies,
                "executed_queries": executed,
                "pending_queries": [],
                "iterations": state["iterations"] + 1,
            }

        def graph_expand_node(state: _AgentState) -> dict:
            # Safety net: if graph expansion is disabled, do nothing and re-plan.
            if self._graph_retriever is None:
                return {"pending_graph_expand": False, "iterations": state["iterations"] + 1}
            doc_bodies = dict(state["doc_bodies"])
            executed = list(state["executed_queries"])
            query = state["conversation_history"][-1]["content"] if state["conversation_history"] else ""
            if not query:
                query = state["clarification_history"][0] if state["clarification_history"] else ""
            if self._debug:
                console_log(f"[agent] graph-expanding seeds for: {query!r}")
            context = self._graph_retriever.retrievedContext(query)
            if context:
                doc_bodies["GRAPH_EXPANDED_CONTEXT"] = context
            executed.append(f"GRAPH_EXPAND:{query}")
            return {
                "doc_bodies": doc_bodies,
                "executed_queries": executed,
                "pending_graph_expand": False,
                "iterations": state["iterations"] + 1,
            }

        def plan_node(state: _AgentState) -> dict:
            """The agent's brain: pick the next action as a JSON object."""
            if state["iterations"] >= self._max_iterations:
                if self._debug:
                    console_log("[agent] max iterations — forcing answer.")
                return {
                    "next_action": "answer",
                    "pending_queries": [],
                    "pending_graph_expand": False,
                    "clarification_question": "",
                }

            doc_summary = "\n\n---\n\n".join(
                f"[{doc_id}]\n{content}" for doc_id, content in list(state["doc_bodies"].items())[:20]
            )
            executed_str = "\n".join(f"- {q}" for q in state["executed_queries"]) or "(none)"
            clarif_str = "\n".join(state["clarification_history"]) or "(none)"
            conv_str = "\n".join(
                f"{m['role'].capitalize()}: {m['content']}"
                for m in state["conversation_history"][-6:]
            ) or "(none)"
            user_content = (
                f"Prior conversation:\n{conv_str}\n\n"
                f"Interactions so far:\n{clarif_str}\n\n"
                f"Queries already executed:\n{executed_str}\n\n"
                f"Chunks retrieved ({len(state['doc_bodies'])} total):\n\n{doc_summary}"
            )
            plan_system = PLAN_SYSTEM if self._graph_enabled else PLAN_SYSTEM_NO_GRAPH
            response = self._llm.invoke([
                SystemMessage(content=plan_system),
                HumanMessage(content=user_content),
            ])
            self._plan_usage.add(usage_from_message(response, plan_system + "\n\n" + user_content))
            raw = (response.content if hasattr(response, "content") else str(response)).strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
            try:
                result = json.loads(raw)
                action = result.get("action", "answer")
            except (json.JSONDecodeError, ValueError):
                action, result = "answer", {}
            # Graph expansion is unavailable unless graph-enabled; treat any stray
            # graph_expand action as "answer" so a disabled tool is never invoked.
            if action == "graph_expand" and not self._graph_enabled:
                if self._debug:
                    console_log("[agent] graph_expand requested but disabled — answering instead.")
                action = "answer"
            if self._debug:
                console_log(f"[agent] action={action} reasoning={result.get('reasoning', '')!r}")
            if action == "retrieve":
                return {
                    "next_action": "retrieve",
                    "pending_queries": result.get("queries", []),
                    "pending_graph_expand": False,
                    "clarification_question": "",
                }
            if action == "graph_expand":
                return {
                    "next_action": "graph_expand",
                    "pending_queries": [],
                    "pending_graph_expand": True,
                    "clarification_question": "",
                }
            if action == "clarify":
                return {
                    "next_action": "clarify",
                    "pending_queries": [],
                    "pending_graph_expand": False,
                    "clarification_question": result.get("clarification", "Could you clarify your question?"),
                }
            return {
                "next_action": "answer",
                "pending_queries": [],
                "pending_graph_expand": False,
                "clarification_question": "",
            }

        def clarify_node(state: _AgentState) -> dict:
            question = state["clarification_question"]
            history = list(state["clarification_history"])
            if state["mode"] == "chat":
                print(f"\nAssistant: {question}")
                history.append(f"Q: {question}\nA: {input('You: ').strip()}")
            else:  # batch mode never blocks on input
                history.append(f"Q: {question}\n[no clarification available in batch mode]")
            return {"clarification_history": history, "clarification_question": ""}

        def answer_node(state: _AgentState) -> dict:
            doc_context = "\n\n---\n\n".join(
                f"[{doc_id}]\n{content}" for doc_id, content in state["doc_bodies"].items()
            )
            conv_str = "\n".join(
                f"{m['role'].capitalize()}: {m['content']}"
                for m in state["conversation_history"][-6:]
            )
            clarif_str = "\n".join(state["clarification_history"])
            parts = []
            if conv_str:
                parts.append(f"Prior conversation:\n{conv_str}")
            if clarif_str:
                parts.append(f"Interactions so far:\n{clarif_str}")
            parts.append(f"Retrieved Wikipedia excerpts:\n{doc_context}")
            answer_prompt = ANSWER_SYSTEM + "\n\n" + "\n\n".join(parts)
            response = self._llm.invoke([
                SystemMessage(content=ANSWER_SYSTEM),
                HumanMessage(content="\n\n".join(parts)),
            ])
            self._answer_usage.add(usage_from_message(response, answer_prompt))
            answer = response.content if hasattr(response, "content") else str(response)
            return {"answer": answer, "done": True}

        def route_from_plan(state: _AgentState) -> str:
            action = state.get("next_action", "answer")
            if action == "retrieve" and state["pending_queries"]:
                return "retrieve"
            if action == "graph_expand" and state.get("pending_graph_expand"):
                return "graph_expand"
            if action == "clarify":
                return "clarify"
            return "answer"

        graph = StateGraph(_AgentState)
        graph.add_node("retrieve", retrieve_node)
        graph.add_node("graph_expand", graph_expand_node)
        graph.add_node("plan", plan_node)
        graph.add_node("clarify", clarify_node)
        graph.add_node("answer", answer_node)
        graph.set_entry_point("plan")
        graph.add_edge("retrieve", "plan")
        graph.add_edge("graph_expand", "plan")
        graph.add_conditional_edges("plan", route_from_plan, {
            "retrieve": "retrieve",
            "graph_expand": "graph_expand",
            "clarify": "clarify",
            "answer": "answer",
        })
        graph.add_edge("clarify", "plan")
        graph.add_edge("answer", END)
        return graph.compile()

    def _run(self, question: str, mode: str, conversation_history: Optional[list[dict]] = None) -> str:
        # Reset per-turn token accounting before this run's plan/answer calls.
        self._plan_usage = TokenUsage()
        self._answer_usage = TokenUsage()
        initial: _AgentState = {
            "conversation_history": conversation_history or [],
            "clarification_history": [question],
            "pending_queries": [question],
            "pending_graph_expand": False,
            "executed_queries": [],
            "doc_bodies": {},
            "iterations": 0,
            "mode": mode,
            "next_action": "plan",
            "clarification_question": "",
            "answer": "",
            "done": False,
        }
        answer = self._graph.invoke(initial)["answer"]
        # Fold this turn's usage into the cumulative evaluation totals.
        self._eval_plan_usage.add(self._plan_usage)
        self._eval_answer_usage.add(self._answer_usage)
        return answer

    def _print_turn_usage(self) -> None:
        """Print a read-friendly per-turn token panel: plan vs answer tokens and total."""
        print(format_plan_answer_usage(self._plan_usage, self._answer_usage))

    def query(self, question: str) -> str:
        """Batch mode — no clarification (drop-in for the RAGAS evaluation harness)."""
        return self._run(question, "batch")

    def chat(self) -> None:
        """Interactive multi-turn chat that keeps conversation history across turns."""
        banner("Starting chat with AI")
        print("Wikipedia research agent (chat with history). Type your question; 'exit'/'quit' to stop.\n")
        conversation_history: list[dict] = []
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
                answer = self._run(user_input, "chat", conversation_history)
            except Exception as exc:
                console_log(f"Error: {exc}", "ERROR")
                continue
            print(f"\nAssistant: {answer}\n")
            self._print_turn_usage()
            session_plan.add(self._plan_usage)
            session_answer.add(self._answer_usage)
            if answer.strip() == "Exiting":
                print("Goodbye.")
                break
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": answer})

        print(format_session_total(session_plan, session_answer))
