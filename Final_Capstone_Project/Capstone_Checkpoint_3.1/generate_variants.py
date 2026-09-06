r"""Capstone Checkpoint 3.1 — paraphrase variant generator.

Standalone version of the Lab 3.2 paraphrase logic, wired to the capstone folders.
For each main question it asks the LLM (OpenRouter, openai/gpt-5.4-mini) for a few
paraphrases a real user might type, preserving meaning so the SAME answer applies.
Each generated variant reuses its original's grading_notes and sources.

Defaults (all overridable via CLI):
    input   test_variables/test_main_questions.json
    n       2   (paraphrases per question)
    output  testinputs_variant_questions.json   (next to this script)

Run:
    python generate_variants.py
    python generate_variants.py [input.json] [n_variants] [out.json]

Setup:
    1. pip install langchain-openai langchain-core python-dotenv
    2. Put OPENROUTER_API_KEY in a .env file (repo root or this folder).

IMPORTANT: Automatic paraphrases can drift. After generating, review the output and
delete any variant whose meaning changed, introduced ambiguity, or duplicates another.
"""
from __future__ import annotations

import os
import sys
import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"   # same model used across the capstone

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "test_variables" / "test_main_questions.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "testinputs_variant_questions.json"
DEFAULT_N = 2

VARIANT_SYSTEM_PROMPT = (
    "You rephrase questions. Given a question, produce alternative phrasings a "
    "real user might type, preserving the meaning so the SAME answer applies. "
    "Return one paraphrase per line, with no numbering or bullets."
)


def require_api_key() -> None:
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit(
            "\n[setup] OPENROUTER_API_KEY is not set.\n"
            "  Create a file named '.env' with one line:\n"
            "      OPENROUTER_API_KEY=sk-or-your-key-here\n"
        )


def make_variants(llm: ChatOpenAI, question: str, n: int) -> list[str]:
    """Ask the LLM for n paraphrases of `question`, one per line."""
    messages = [
        SystemMessage(content=VARIANT_SYSTEM_PROMPT),
        HumanMessage(content=f"Question: {question}\n\nProduce {n} paraphrases."),
    ]
    response = llm.invoke(messages)
    text = response.content if hasattr(response, "content") else str(response)
    variants = [line.strip("-*. ").strip() for line in text.splitlines() if line.strip()]
    return variants[:n]


def main() -> None:
    in_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    n = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_N
    out_path = Path(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_OUTPUT

    require_api_key()

    if not in_path.exists():
        raise SystemExit(f"Input questions file not found: {in_path}")

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.3,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
    )

    print(f"Generating up to {n} paraphrase(s) for each of {len(data)} question(s)...\n")
    augmented: list[dict] = []
    variant_count = 0
    for i, entry in enumerate(data, 1):
        augmented.append(entry)  # keep the original
        print(f"[{i}] original: {entry['question']}")
        for variant in make_variants(llm, entry["question"], n):
            augmented.append({
                "question": variant,
                "grading_notes": entry["grading_notes"],
                "sources": entry.get("sources", []),
            })
            print(f"      -> variant: {variant}")
            variant_count += 1
        print()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(augmented, f, indent=2)

    print("=" * 72)
    print(f"Done: {len(data)} original(s) + {variant_count} variant(s) = {len(augmented)} total.")
    print(f"Written to: {out_path}")
    print("Each variant reuses its original's grading_notes and sources.")
    print("Review the file and delete any variant whose meaning drifted before evaluating.")


if __name__ == "__main__":
    main()
