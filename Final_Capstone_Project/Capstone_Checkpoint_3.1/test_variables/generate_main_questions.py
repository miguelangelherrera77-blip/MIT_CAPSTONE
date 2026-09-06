r"""Generate grounded main test questions from the Wikipedia capstone corpus.

For each sampled article it extracts the lead text and asks the LLM (OpenRouter,
openai/gpt-5.4-mini) to produce ONE question plus grading_notes grounded in that text.
Writes the results to test_main_questions.json with the source filename.

Run:  python generate_main_questions.py [n_questions] [out.json]
Default: 100 questions -> test_main_questions.json (next to this script).
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import os
import re
import sys
import json
import random
from pathlib import Path

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"

SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = SCRIPT_DIR.parent
FINAL_CAPSTONE_DIR = CHECKPOINT_DIR.parent
WIKI_DIR = FINAL_CAPSTONE_DIR / "Capstone_Database" / "Wikipedia"

# Questions from Checkpoint 2.1 are intentionally EXCLUDED.
EXCLUDE_ARTICLES = {
    "Albert_Einstein",
    "Toronto_Maple_Leafs",
    "GoldenEye_007_(1997_video_game)",
    "Abraham_Lincoln",
}

SYSTEM_PROMPT = (
    "You write factual test questions for a Wikipedia retrieval system. Given the lead "
    "text of a Wikipedia article, produce ONE clear, self-contained question a real user "
    "might ask that is answerable from the text, plus concise grading notes describing "
    "what a correct answer must contain. Return STRICT JSON with exactly two keys: "
    '"question" and "grading_notes". No markdown, no extra text.'
)


def require_api_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit(
            "\n[setup] OPENROUTER_API_KEY is not set. Put it in a .env file and rerun.\n"
        )
    return key


def lead_text(html_path: Path, max_chars: int = 2000) -> str:
    """Extract cleaned lead text from a Wikipedia HTML article."""
    with open(html_path, "r", encoding="utf-8", errors="replace") as fh:
        soup = BeautifulSoup(fh.read(), "html.parser")
    for tag in soup(["script", "style", "table"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator=" ", strip=True).split())
    # trim boilerplate up to the encyclopedia marker when present
    marker = "From Wikipedia, the free encyclopedia"
    idx = text.find(marker)
    if idx >= 0:
        text = text[idx + len(marker):]
    return text[:max_chars].strip()


def make_item(llm: ChatOpenAI, title: str, text: str, filename: str) -> dict | None:
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Article title: {title}\n\nLead text:\n{text}"),
    ]
    try:
        raw = llm.invoke(messages).content
    except Exception as exc:
        print(f"  LLM error for {filename}: {exc}")
        return None
    # strip code fences if present
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        obj = json.loads(raw)
        q = obj["question"].strip()
        notes = obj["grading_notes"].strip()
    except Exception:
        print(f"  Could not parse JSON for {filename}; skipping.")
        return None
    if not q or not notes:
        return None
    return {"question": q, "grading_notes": notes, "sources": [filename]}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else SCRIPT_DIR / "test_main_questions.json"

    require_api_key()
    if not WIKI_DIR.is_dir():
        raise SystemExit(f"Wikipedia directory not found: {WIKI_DIR}")

    all_files = sorted(f for f in os.listdir(WIKI_DIR) if f.endswith(".html"))
    candidates = [f for f in all_files if f.replace(".html", "") not in EXCLUDE_ARTICLES]
    random.seed(42)
    random.shuffle(candidates)

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.3,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
    )

    print(f"Generating {n} grounded questions from {len(candidates)} candidate articles...\n")
    items: list[dict] = []
    for filename in candidates:
        if len(items) >= n:
            break
        title = filename.replace(".html", "").replace("_", " ")
        text = lead_text(WIKI_DIR / filename)
        if len(text) < 150:   # skip stubs / near-empty pages
            continue
        item = make_item(llm, title, text, filename)
        if item:
            items.append(item)
            print(f"[{len(items)}/{n}] {filename}: {item['question'][:80]}")

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(items, fh, indent=2)

    print(f"\nDone: wrote {len(items)} questions to {out_path}")


if __name__ == "__main__":
    main()
