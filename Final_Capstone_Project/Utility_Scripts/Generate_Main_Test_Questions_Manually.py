#################################################################################
# 
# Massachusetts Institute of Technology - xPRO
# RAG and Context Engineering: Designing and Building Production-Grade AI Systems
# 
# Student: Miguel Herrera
# Section: B
# Date and Time: 2026-09-12 12:23:08 -07:00
#
# Description: This script validates manually authored question, grading_notes,
#              and sources records against Test_Main_Questions.schema.json, then
#              writes deterministic JSON and JSONL datasets without an LLM; an
#              explicit Y/N override controls existing-file regeneration.
#
#################################################################################

r"""Write manually entered evaluation questions as JSON and JSONL.

Add questions to ``MANUAL_QUESTIONS`` below. Each record should contain:

    {
        "question": "...",
        "grading_notes": "...",
        "evaluation_category": "factual_retrieval",
        "sources": ["Article_Name.html"],
    }

The ``question`` and ``grading_notes`` fields are required by the Checkpoint 4.1
evaluation. ``sources`` is optional, but useful for tracing an answer to the
Wikipedia corpus.

Run from the repository root:

    python Final_Capstone_Project/Utility_Scripts/Generate_Main_Test_Questions_Manually.py [output.json] [override]

The JSON and JSONL output files are written to the Test_Variables directory by
default. Override is N by default; use Y to replace existing output files.
This script is deterministic and does not use an LLM or API key.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm
from utility_logging import ask_override, log_run


SCRIPT_DIR = Path(__file__).resolve().parent
TEST_VARIABLES_DIR = SCRIPT_DIR.parent / "Test_Variables"
DEFAULT_OUTPUT = TEST_VARIABLES_DIR / "Test_Questions_Manually_Generated.json"
SCHEMA_PATH = SCRIPT_DIR.parent / "JSON_Schemas" / "Test_Main_Questions.schema.json"


# Add manually authored records here. Keep the list non-empty before running.
MANUAL_QUESTIONS: list[dict[str, Any]] = [
    {
        "question": "What is the capital and largest city of Mali?",
        "grading_notes": "A correct answer must name Bamako as both the capital and largest city of Mali.",
        "evaluation_category": "factual_retrieval",
        "sources": ["Mali.html"],
    },
    {
        "question": "Who was the last native pharaoh of Egypt mentioned in the lead text?",
        "grading_notes": "A correct answer must identify Nectanebo II. It is acceptable to mention that he was the last native pharaoh and/or that he belonged to the short-lived 30th Dynasty.",
        "evaluation_category": "factual_retrieval",
        "sources": ["List_of_pharaohs.html"],
    },
    {
        "question": "According to the lead text, what were the two main coalitions that fought in World War II?",
        "grading_notes": "A correct answer must name both the Allies and the Axis powers as the two main coalitions.",
        "evaluation_category": "multi_fact",
        "sources": ["World_War_II.html"],
    },
    {
        "question": "Who created the Norman Gunston character, and what kind of TV character was he?",
        "grading_notes": "A correct answer must identify Wendy Skelcher and Garry McDonald as the creators and describe Norman Gunston as a satirical fictional television character or persona.",
        "evaluation_category": "obscure_knowledge",
        "sources": ["Norman_Gunston.html"],
    },
    {
        "question": "What was Joseph Warren Stilwell's nickname, and during which World War II theater did he serve as a United States Army general?",
        "grading_notes": "A complete answer must give Stilwell's nickname, 'Vinegar Joe,' and identify the China-Burma-India theater. Award partial credit only when one of these two required details is correct.",
        "evaluation_category": "multi_fact",
        "sources": ["Joseph_Stilwell.html"],
    },
    {
        "question": "Quote the opening sentence of the Wikipedia article titled '13 (2010 film)' exactly as written.",
        "grading_notes": "A correct answer must reproduce the article's opening sentence verbatim, preserving its meaning and identifying details. Paraphrases or invented quotations are not fully correct.",
        "evaluation_category": "quotation_fidelity",
        "sources": ["13_(2010_film).html"],
    },
    {
        "question": "Compare Albert Einstein and Joseph Warren Stilwell. What was each person best known for, and what major historical context shaped their work? Support each answer with a quotation from the relevant Wikipedia article.",
        "grading_notes": "A complete answer must address both Einstein and Stilwell, explain what each was best known for, describe relevant historical context, and include one attributable quotation from each person's article. Do not award full credit for discussing only one person or for unsupported quotations.",
        "evaluation_category": "cross_document_synthesis",
        "sources": ["Albert_Einstein.html", "Joseph_Stilwell.html"],
    },
    {
        "question": "What is the current population of Mars, according to the latest available information? If this fact is not supported by the Wikipedia corpus, clearly state that the corpus does not provide the answer.",
        "grading_notes": "A grounded answer must recognize that Mars has no human population and that the corpus does not provide a current population figure. Do not award full credit for inventing a current population or presenting an unsupported external estimate as a corpus answer.",
        "evaluation_category": "out_of_corpus_abstention",
        "sources": [],
    },
]


def load_schema() -> dict[str, Any]:
    """Load the shared main-question JSON Schema without modifying it."""
    if not SCHEMA_PATH.is_file():
        raise FileNotFoundError(f"Main-question schema not found: {SCHEMA_PATH}")
    with SCHEMA_PATH.open("r", encoding="utf-8") as schema_file:
        schema = json.load(schema_file)
    if schema.get("type") != "array":
        raise ValueError("The main-question schema root must be an array schema.")
    return schema


def validate_records(records: list[dict[str, Any]], schema: dict[str, Any]) -> None:
    """Validate manually authored records against the shared main schema."""
    if not records:
        raise ValueError(
            "MANUAL_QUESTIONS is empty. Add at least one question before running."
        )

    record_schema = schema.get("$defs", {}).get("questionRecord", {})
    properties = record_schema.get("properties", {})
    required_fields = set(record_schema.get("required", []))
    if not properties or not required_fields:
        raise ValueError("The main-question schema must define properties and required fields.")
    for index, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise ValueError(f"Question {index} must be a JSON object.")
        missing = required_fields - record.keys()
        if missing:
            fields = ", ".join(sorted(missing))
            raise ValueError(f"Question {index} is missing required field(s): {fields}")
        unexpected = set(record) - set(properties)
        if unexpected and record_schema.get("unevaluatedProperties") is False:
            fields = ", ".join(sorted(unexpected))
            raise ValueError(f"Question {index} has unexpected field(s): {fields}")
        for field in sorted(required_fields - {"sources"}):
            value = record[field]
            field_schema = properties[field]
            if field_schema.get("type") != "string" or not isinstance(value, str):
                raise ValueError(
                    f"Question {index} has an empty or non-text '{field}' field."
                )
            if len(value) < field_schema.get("minLength", 0) or not value.strip():
                raise ValueError(
                    f"Question {index} has an empty or non-text '{field}' field."
                )
        if "sources" in record:
            sources = record["sources"]
            source_schema = properties["sources"]
            item_schema = source_schema.get("items", {})
            if not isinstance(sources, list) or not all(
                isinstance(source, str)
                and len(source) >= item_schema.get("minLength", 0)
                and source.strip()
                for source in sources
            ):
                raise ValueError(
                    f"Question {index} has invalid 'sources'; use a list of non-empty strings."
                )


def write_json_outputs(records: list[dict[str, Any]], out_path: Path) -> Path:
    """Write formatted JSON and one JSON object per line in JSONL."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as json_file:
        json.dump(records, json_file, indent=2, ensure_ascii=False)
        json_file.write("\n")

    jsonl_path = out_path.with_suffix(".jsonl")
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as jsonl_file:
        for record in records:
            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return jsonl_path


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT
    override_arg = sys.argv[2].upper() if len(sys.argv) > 2 else None
    print("=== Manual main-question generator ===")
    print(f"[args] output_json={out_path}")
    print(f"[args] override={override_arg or 'prompt when output exists'}")
    if override_arg is not None and override_arg not in {"Y", "N"}:
        raise ValueError("override must be Y or N.")
    output_exists = out_path.is_file()
    if output_exists and override_arg is None:
        override_arg = "Y" if ask_override() else "N"
    if output_exists and override_arg == "N":
        print("[skip] Existing JSON output preserved; no files were changed.")
        return
    if output_exists and override_arg == "Y":
        print("[override] Existing manual main-question output will be regenerated.")

    schema = load_schema()
    with tqdm(
        total=len(MANUAL_QUESTIONS),
        desc="Generating Manual questions",
        unit="question",
    ) as question_bar:
        validate_records(MANUAL_QUESTIONS, schema)
        jsonl_path = write_json_outputs(MANUAL_QUESTIONS, out_path)
        question_bar.update(len(MANUAL_QUESTIONS))

    print(f"[validate] Validated {len(MANUAL_QUESTIONS)} manual question record(s).")
    print(f"[write] JSON output written: {out_path}")
    print(f"[write] JSONL output written: {jsonl_path}")
    print(f"[complete] Generated {len(MANUAL_QUESTIONS)} manual question record(s).")


if __name__ == "__main__":
    with log_run(__file__):
        main()
