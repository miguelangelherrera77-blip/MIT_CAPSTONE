#################################################################################
# 
# Massachusetts Institute of Technology - xPRO
# RAG and Context Engineering: Designing and Building Production-Grade AI Systems
# 
# Student: Miguel Herrera
# Section: B
# Date and Time: 2026-09-13 18:18:45 -07:00
#
# Description: Generates up to two LLM paraphrases per Main Question using the
#              configured OpenRouter model, preserves question/grading/source records,
#              validates them against the LLM paraphrase schema, writes the primary
#              JSON/JSONL dataset, and creates the evaluator-ready JSONL derivative.
#
#################################################################################

r"""Capstone Checkpoint 4.1 — paraphrase variant generator.

Standalone version of the Lab 3.2 paraphrase logic, wired to the capstone folders.
For each main question it asks the LLM (OpenRouter, openai/gpt-5.4-mini) for a few
paraphrases a real user might type, preserving meaning so the SAME answer applies.
Each generated variant reuses its original's grading_notes and sources.

Defaults (all overridable via CLI):
    input   Test_Variables/Test_Main_Questions_LLM_Generated.json
    n       2   (paraphrases per question)
    output  Test_Variables/Test_Variant_Questions_LLM_Generated.json and its JSONL sibling

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
from tqdm import tqdm
from utility_logging import ask_override, log_run

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"   # same model used across the capstone

SCRIPT_DIR = Path(__file__).resolve().parent
FINAL_CAPSTONE_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = FINAL_CAPSTONE_DIR / "Capstone_Checkpoint_4.1"
TEST_VARIABLES_DIR = FINAL_CAPSTONE_DIR / "Test_Variables"
DEFAULT_INPUT = TEST_VARIABLES_DIR / "Test_Questions_LLM_Generated.json"
DEFAULT_OUTPUT = TEST_VARIABLES_DIR / "Test_Questions_LLM_Paraphrased.json"
EVALUATION_OUTPUT = TEST_VARIABLES_DIR / "Test_Questions_LLM_Paraphrased_Evaluation.jsonl"
SCHEMA_PATH = FINAL_CAPSTONE_DIR / "JSON_Schemas" / "Test_Paraphrased_Questions.schema.json"
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


def write_json_outputs(records: list[dict], out_path: Path) -> Path:
    """Write generated records as formatted JSON and one-object-per-line JSONL."""
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    jsonl_path = out_path.with_suffix(".jsonl")
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return jsonl_path


def write_evaluation_jsonl(records: list[dict]) -> Path:
    """Write the flattened LLM paraphrase records for the evaluator."""
    with EVALUATION_OUTPUT.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return EVALUATION_OUTPUT


def evaluation_jsonl_is_valid(records: list[dict]) -> bool:
    """Check that the existing evaluator JSONL matches the primary JSON records."""
    if not EVALUATION_OUTPUT.is_file():
        return False
    try:
        with EVALUATION_OUTPUT.open("r", encoding="utf-8") as evaluation_file:
            existing = [json.loads(line) for line in evaluation_file if line.strip()]
    except (OSError, json.JSONDecodeError):
        return False
    return existing == records


def load_schema() -> dict:
    """Load the JSON Schema used to validate generated paraphrase records."""
    print(f"[schema] Loading schema: {SCHEMA_PATH}")
    if not SCHEMA_PATH.is_file():
        raise FileNotFoundError(f"LLM paraphrase schema not found: {SCHEMA_PATH}")
    with SCHEMA_PATH.open("r", encoding="utf-8") as schema_file:
        schema = json.load(schema_file)
    if schema.get("type") != "array":
        raise ValueError("The LLM paraphrase schema root must be an array schema.")
    print("[schema] Paraphrase schema loaded successfully.")
    return schema


def validate_records(records: list[dict], schema: dict) -> None:
    """Validate generated records against the configured JSON Schema."""
    record_schema = schema.get("$defs", {}).get("llmRecord", {})
    properties = record_schema.get("properties", {})
    required_fields = set(record_schema.get("required", []))
    if not properties or not required_fields:
        raise ValueError("The LLM paraphrase schema must define properties and required fields.")

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


def main() -> None:
    in_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    n = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_N
    out_path = Path(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_OUTPUT
    print("=== LLM paraphrase generator ===")
    print(f"[args] input_json={in_path}")
    print(f"[args] paraphrases_per_question={n}")
    print(f"[args] output_json={out_path}")

    if out_path.is_file():
        if not ask_override():
            schema = load_schema()
            with out_path.open("r", encoding="utf-8") as json_file:
                existing_records = json.load(json_file)
            validate_records(existing_records, schema)
            if evaluation_jsonl_is_valid(existing_records):
                print("[skip] Existing JSON and evaluation JSONL are valid; no files were changed.")
            else:
                write_evaluation_jsonl(existing_records)
                print(f"[repair] Evaluation JSONL rebuilt: {EVALUATION_OUTPUT}")
            return
        print("[override] Existing JSON output will be regenerated.")

    require_api_key()

    if not in_path.exists():
        raise SystemExit(f"Input questions file not found: {in_path}")
    print(f"[input] Reading source questions: {in_path}")

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    schema = load_schema()

    print(f"[api] Creating OpenRouter LLM client: {LLM_MODEL}")
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.3,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
    )
    print("[api] LLM client created successfully.")

    print(f"Generating up to {n} paraphrase(s) for each of {len(data)} question(s)...\n")
    augmented: list[dict] = []
    variant_count = 0
    with tqdm(
        total=len(data),
        desc="Processing Source questions",
        unit="question",
    ) as question_bar, tqdm(
        total=len(data) * n,
        desc="Generating Paraphrased variants",
        unit="variant",
    ) as variant_bar:
        for i, entry in enumerate(data, 1):
            source_question = entry.get("llm_question", entry.get("question"))
            if not isinstance(source_question, str) or not source_question.strip():
                raise ValueError(f"Input record {i} has no llm_question or question field.")
            original = {
                "question": source_question,
                "grading_notes": entry["grading_notes"],
                "sources": entry.get("sources", []),
            }
            augmented.append(original)
            variants = make_variants(llm, source_question, n)
            for variant in variants:
                augmented.append({
                    "question": variant,
                    "grading_notes": entry["grading_notes"],
                    "sources": entry.get("sources", []),
                })
                variant_count += 1
                variant_bar.update(1)
            if len(variants) < n:
                variant_bar.update(n - len(variants))
            question_bar.update(1)

    validate_records(augmented, schema)
    print(f"[validate] Validated {len(augmented)} generated records.")
    jsonl_path = write_json_outputs(augmented, out_path)
    evaluation_jsonl_path = write_evaluation_jsonl(augmented)

    print("=" * 72)
    print(f"Done: {len(data)} original(s) + {variant_count} variant(s) = {len(augmented)} total.")
    print(f"[write] JSON output written: {out_path}")
    print(f"[write] JSONL output written: {jsonl_path}")
    print(f"[write] Evaluation JSONL output written: {evaluation_jsonl_path}")
    print(f"[complete] Generated {len(augmented)} paraphrase record(s).")
    print("Each variant reuses its original's grading_notes and sources.")
    print("Review the file and delete any variant whose meaning drifted before evaluating.")


if __name__ == "__main__":
    with log_run(__file__):
        main()
