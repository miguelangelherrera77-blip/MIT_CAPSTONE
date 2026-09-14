#################################################################################
# 
# Massachusetts Institute of Technology - xPRO
# RAG and Context Engineering: Designing and Building Production-Grade AI Systems
# 
# Student: Miguel Herrera
# Section: B
# Date and Time: 2026-09-12 12:32:34 -07:00
#
# Description: This script samples Wikipedia article leads, asks an LLM for
#              grounded question and grading_notes records, validates them
#              against the main-question schema, and writes JSON and JSONL output;
#              an explicit Y/N override controls existing-file regeneration.
#
#################################################################################

r"""Generate grounded main test questions from the Wikipedia capstone corpus.

For each sampled article it extracts the lead text and asks the LLM (OpenRouter,
openai/gpt-5.4-mini) to produce ONE question plus grading_notes grounded in that text.
Writes the results to both Test_Main_Questions_LLM_Generated.json and its JSONL sibling.

Run:  python generate_main_questions.py [n_questions] [out.json] [override]
Default: 100 questions -> Final_Capstone_Project/Test_Variables/Test_Main_Questions_LLM_Generated.json.
Override: Y regenerates existing output; N preserves it (default).
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
from tqdm import tqdm
from utility_logging import ask_integer, ask_override, ask_yes_no, log_run

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"

SCRIPT_DIR = Path(__file__).resolve().parent
FINAL_CAPSTONE_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = FINAL_CAPSTONE_DIR / "Capstone_Checkpoint_4.1"
WIKI_DIR = FINAL_CAPSTONE_DIR / "Capstone_Database" / "Wikipedia"
TEST_VARIABLES_DIR = FINAL_CAPSTONE_DIR / "Test_Variables"
DEFAULT_OUTPUT = TEST_VARIABLES_DIR / "Test_Main_Questions_LLM_Generated.json"
SCHEMA_PATH = FINAL_CAPSTONE_DIR / "JSON_Schemas" / "Test_Main_Questions.schema.json"

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


def load_schema() -> dict:
    """Load the JSON Schema used to validate generated records."""
    print(f"[schema] Loading schema: {SCHEMA_PATH}")
    if not SCHEMA_PATH.is_file():
        raise FileNotFoundError(f"Main-question schema not found: {SCHEMA_PATH}")
    with SCHEMA_PATH.open("r", encoding="utf-8") as schema_file:
        schema = json.load(schema_file)
    if schema.get("type") != "array":
        raise ValueError("The main-question schema root must be an array schema.")
    print("[schema] Main-question schema loaded successfully.")
    return schema


def validate_records(records: list[dict], schema: dict) -> None:
    """Validate generated records against the configured JSON Schema."""
    record_schema = schema.get("$defs", {}).get("questionRecord", {})
    properties = record_schema.get("properties", {})
    required_fields = set(record_schema.get("required", []))
    if not properties or not required_fields:
        raise ValueError("The main-question schema must define properties and required fields.")

    for index, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise ValueError(f"Generated record {index} must be a JSON object.")
        missing = required_fields - record.keys()
        if missing:
            raise ValueError(f"Generated record {index} is missing: {', '.join(sorted(missing))}")
        unexpected = set(record) - set(properties)
        if unexpected and record_schema.get("additionalProperties") is False:
            raise ValueError(f"Generated record {index} has unexpected fields: {', '.join(sorted(unexpected))}")
        for field, field_schema in properties.items():
            if field not in record:
                continue
            value = record[field]
            if field_schema.get("type") == "string":
                if not isinstance(value, str) or len(value) < field_schema.get("minLength", 0) or not value.strip():
                    raise ValueError(f"Generated record {index} has an invalid '{field}' field.")
            elif field_schema.get("type") == "array":
                item_schema = field_schema.get("items", {})
                if not isinstance(value, list) or not all(
                    isinstance(item, str) and len(item) >= item_schema.get("minLength", 0) and item.strip()
                    for item in value
                ):
                    raise ValueError(f"Generated record {index} has an invalid '{field}' field.")


def write_json_outputs(items: list[dict], out_path: Path) -> Path:
    """Write the generated records as formatted JSON and one-object-per-line JSONL."""
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(items, fh, indent=2, ensure_ascii=False)
    jsonl_path = out_path.with_suffix(".jsonl")
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    return jsonl_path


def main():
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    else:
        print("[input] number_of_questions argument was not provided; a value is required.")
        n = ask_integer("Number of questions to generate")
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT
    override_arg = sys.argv[3].upper() if len(sys.argv) > 3 else (
        "Y" if ask_yes_no("Override existing output?") else "N"
    )
    print("=== Main question generator ===")
    print(f"[args] number_of_questions={n}")
    print(f"[args] output_json={out_path}")
    print(f"[args] override={override_arg or 'prompt when output exists'}")
    if n <= 0:
        raise ValueError("number_of_questions must be greater than zero.")
    if override_arg is not None and override_arg not in {"Y", "N"}:
        raise ValueError("override must be Y or N.")
    print("[args] Arguments validated successfully.")

    output_exists = out_path.is_file()
    if output_exists and override_arg is None:
        print("[output] Existing JSON output found; requesting override decision.")
        override_arg = "Y" if ask_override() else "N"
    if output_exists and override_arg == "N":
        print("[skip] Main question JSON output already exists; preserving it.")
        print(f"[skip] Output file: {out_path}")
        print("[skip] No LLM calls or output changes were made.")
        return
    if output_exists and override_arg == "Y":
        print("[override] Existing main question output will be regenerated.")
    else:
        print("[output] No existing JSON output found; generation will proceed.")

    schema = load_schema()
    print(f"[input] Checking Wikipedia source directory: {WIKI_DIR}")
    require_api_key()
    if not WIKI_DIR.is_dir():
        raise SystemExit(f"Wikipedia directory not found: {WIKI_DIR}")
    print("[input] Wikipedia source directory is available.")

    all_files = sorted(f for f in os.listdir(WIKI_DIR) if f.endswith(".html"))
    candidates = [f for f in all_files if f.replace(".html", "") not in EXCLUDE_ARTICLES]
    random.seed(42)
    random.shuffle(candidates)
    print(f"[input] Found {len(all_files)} HTML files; {len(candidates)} candidates remain after exclusions.")

    print(f"[api] Creating OpenRouter LLM client: {LLM_MODEL}")
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.3,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
    )
    print("[api] LLM client created successfully.")

    print(f"Generating {n} grounded questions from {len(candidates)} candidate articles...\n")
    items: list[dict] = []
    with tqdm(total=n, desc="Generating Main questions", unit="question") as question_bar:
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
                question_bar.update(1)

    validate_records(items, schema)
    jsonl_path = write_json_outputs(items, out_path)
    print(f"[validate] Validated {len(items)} generated records against the schema.")
    print(f"[write] JSON output written: {out_path}")
    print(f"[write] JSONL output written: {jsonl_path}")
    print(f"[complete] Generation completed successfully with {len(items)} question(s).")


if __name__ == "__main__":
    with log_run(__file__):
        main()
