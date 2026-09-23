#################################################################################
# 
# Massachusetts Institute of Technology - xPRO
# RAG and Context Engineering: Designing and Building Production-Grade AI Systems
# 
# Student: Miguel Herrera
# Section: B
# Date and Time: 2026-09-22 19:31:56 -07:00
#
# Description: Shared interactive console menus for the capstone RAGAS harness.
#              Provides the mode, retrieval-engine (Context-Aware / Agentic Dynamic /
#              Compare Both), dataset, question-set, search-method (full menu plus an
#              agent-specific base-retriever menu), size, and sampling prompts, along
#              with their answer-normalization maps. Each prompt prints its options,
#              validates the entry, and returns a normalized value (or None on quit /
#              EOF). Lifted out of the Checkpoint 5.1 agent solution so the menu layer
#              is reusable and the checkpoint keeps only its CLI orchestration.
#
#################################################################################

"""Reusable interactive console menus for the RAGAS evaluation harness."""
from __future__ import annotations


MODE_MAP = {
    "1": "chat",
    "chat": "chat",
    "interactive": "chat",
    "2": "eval",
    "eval": "eval",
    "evaluation": "eval",
    "3": "quit",
    "quit": "quit",
}

ENGINE_MAP = {
    "1": "context",
    "context": "context",
    "context_aware": "context",
    "context-aware": "context",
    "2": "agent",
    "agent": "agent",
    "agentic": "agent",
    "dynamic": "agent",
    "3": "both",
    "both": "both",
    "compare": "both",
    "4": "quit",
    "quit": "quit",
}

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
    "3": "hybrid",
    "hybrid": "hybrid",
    "4": "graph",
    "graph": "graph",
    "graph_enabled": "graph",
    "hybrid_graph": "graph",
    "5": "quit",
    "quit": "quit",
}

# The Dynamic Agent always has a graph-expansion tool available regardless of the base
# method, so its menu offers only the base retriever (lexical / semantic / hybrid) and
# never a standalone "graph" option.
AGENT_BASE_RETRIEVAL_MAP = {
    "1": "lexical",
    "lexical": "lexical",
    "bm25": "lexical",
    "2": "semantic",
    "semantic": "semantic",
    "vector": "semantic",
    "3": "hybrid",
    "hybrid": "hybrid",
    "4": "quit",
    "quit": "quit",
}


def prompt_mode() -> str | None:
    """Prompt for the top-level mode: RAGAS evaluation or interactive chat."""
    print("\nSelect Mode:")
    print("  1. Interactive Chat (multi-turn conversation with history)")
    print("  2. RAGAS Evaluation (score the agent on the test datasets)")
    print("  3. Quit")
    while True:
        try:
            choice = input("Enter mode selection [1-3]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return None
        mapped = MODE_MAP.get(choice)
        if mapped == "quit":
            print("Quit selected. Exiting without running.")
            return None
        if mapped:
            return mapped
        print("Invalid choice. Enter 1 (chat), 2 (evaluation), or 3 (quit).")


def prompt_engine(allow_compare: bool = True) -> str | None:
    """Prompt for the retrieval engine.

    ``allow_compare`` controls whether the eval-only "Compare Both" option is offered;
    it is hidden in interactive chat, where comparison does not apply."""
    print("\nSelect Retrieval Engine:")
    print("  1. Context-Aware Retriever   (grounded RAG; folds recent conversation into each query; no autonomous planning)")
    print("  2. Agentic Dynamic Retriever (ReAct agent: dynamically plans retrieve / graph-expand / clarify / answer)")
    if allow_compare:
        print("  3. Compare Both              (eval only: run both engines and log a side-by-side comparison)")
        print("  4. Quit")
        quit_choice, prompt_range = "4", "[1-4]"
    else:
        print("  3. Quit")
        quit_choice, prompt_range = "3", "[1-3]"
    while True:
        try:
            choice = input(f"Enter engine selection {prompt_range}: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return None
        if choice == quit_choice:
            print("Quit selected. Exiting without running.")
            return None
        mapped = ENGINE_MAP.get(choice)
        if mapped == "quit":
            print("Quit selected. Exiting without running.")
            return None
        # "Compare Both" is only selectable when comparison is allowed (eval mode).
        if mapped == "both" and not allow_compare:
            mapped = None
        if mapped:
            return mapped
        if allow_compare:
            print("Invalid choice. Enter 1 (context-aware), 2 (agentic dynamic), 3 (compare both), or 4 (quit).")
        else:
            print("Invalid choice. Enter 1 (context-aware), 2 (agentic dynamic), or 3 (quit).")


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
    """Prompt the user to select a retrieval method from options 1-5."""
    print("\nSelect Search Method:")
    print("  1. Lexical Search (Only): BM25 matches exact terms, names, dates, and phrases")
    print("  2. Semantic Search (Only): Chroma dense vectors find conceptual meaning and paraphrases")
    print("  3. Hybrid Search: BM25 + Chroma vectors combine exact-term and deep-meaning retrieval")
    print("  4. Hybrid with Graph Enabled: Hybrid Search plus Graph DB expansion of neighboring and linked article context")
    print("  5. Quit (Exit without running)")
    while True:
        try:
            choice = input("Enter search method selection [1-5]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return None
        if choice in RETRIEVAL_METHOD_MAP:
            mapped = RETRIEVAL_METHOD_MAP[choice]
            if mapped == "quit":
                print("Quit selected. Exiting without running.")
                return None
            return mapped
        print("Invalid choice. Please enter 1 (lexical), 2 (semantic), 3 (hybrid), 4 (hybrid with graph), or 5 (quit).")


def prompt_agent_base_retrieval_method() -> str | None:
    """Prompt for the Dynamic Agent's base retriever.

    The agent always has a graph-expansion tool available on demand, so the choice
    here only sets the base retrieve tool (lexical / semantic / hybrid); there is no
    standalone graph option."""
    print("\nSelect Base Search Method (the agent adds graph expansion on demand):")
    print("  1. Lexical base: BM25 exact-term retrieval, with graph expansion available on demand")
    print("  2. Semantic base: Chroma dense-vector retrieval, with graph expansion available on demand")
    print("  3. Hybrid base: BM25 + Chroma vectors, with graph expansion available on demand")
    print("  4. Quit (Exit without running)")
    while True:
        try:
            choice = input("Enter base search method selection [1-4]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return None
        if choice in AGENT_BASE_RETRIEVAL_MAP:
            mapped = AGENT_BASE_RETRIEVAL_MAP[choice]
            if mapped == "quit":
                print("Quit selected. Exiting without running.")
                return None
            return mapped
        print("Invalid choice. Please enter 1 (lexical), 2 (semantic), 3 (hybrid), or 4 (quit).")


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
