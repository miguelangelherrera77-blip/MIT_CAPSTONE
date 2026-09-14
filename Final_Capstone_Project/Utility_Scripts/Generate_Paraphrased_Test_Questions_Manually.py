#################################################################################
# 
# Massachusetts Institute of Technology - xPRO
# RAG and Context Engineering: Designing and Building Production-Grade AI Systems
# 
# Student: Miguel Herrera
# Section: B
# Date and Time: 2026-09-13 18:18:45 -07:00
#
# Description: Validates manually authored grouped paraphrase records, including
#              original_question, paraphrased_question1/2, grading_notes, and sources;
#              prevents duplicate wording; writes grouped JSON/JSONL authoring files;
#              and creates the flattened evaluator JSONL derivative without an LLM.
#
#################################################################################

r"""Write manually entered paraphrased questions as JSON and JSONL.

Add paraphrases to ``MANUAL_PARAPHRASES`` below. Each record should contain:

    {
        "original_question": "What is the capital and largest city of Mali?",
        "paraphrased_question1": "Which city is both the capital and largest city of Mali?",
        "paraphrased_question2": "What city serves as Mali's capital and largest urban center?",
        "grading_notes": "A correct answer must name Bamako as both the capital and largest city of Mali.",
        "sources": ["Mali.html"],
    }

``original_question`` identifies the source wording. The two paraphrase fields are
the manually authored alternatives written to the output. This script performs no
LLM or API calls.

Run from the repository root:

    python Final_Capstone_Project/Utility_Scripts/Generate_Paraphrased_Test_Questions_Manually.py

By default, output is written to ``Final_Capstone_Project/Test_Variables``. Existing
output is preserved and skipped; use ``--rebuild`` to overwrite it intentionally.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm
from utility_logging import ask_override, log_run


SCRIPT_DIR = Path(__file__).resolve().parent
TEST_VARIABLES_DIR = SCRIPT_DIR.parent / "Test_Variables"
DEFAULT_OUTPUT = TEST_VARIABLES_DIR / "Test_Questions_Manually_Paraphrased.json"
SCHEMA_PATH = SCRIPT_DIR.parent / "JSON_Schemas" / "Test_Paraphrased_Questions.schema.json"


# Add one or more manually authored paraphrases for each original question here.
MANUAL_PARAPHRASES: list[dict[str, Any]] = [
    {
        "original_question": "What is the capital and largest city of Mali?",
        "paraphrased_question1": "Which city is both the capital and largest city of Mali?",
        "paraphrased_question2": "What city serves as Mali's capital and largest urban center?",
        "grading_notes": "A correct answer must name Bamako as both the capital and largest city of Mali.",
        "sources": ["Mali.html"],
    },
]


def load_schema() -> dict[str, Any]:
    """Load the JSON Schema used to validate manual paraphrase records."""
    print(f"[schema] Loading schema: {SCHEMA_PATH}")
    if not SCHEMA_PATH.is_file():
        raise FileNotFoundError(f"Paraphrase schema not found: {SCHEMA_PATH}")
    with SCHEMA_PATH.open("r", encoding="utf-8") as schema_file:
        schema = json.load(schema_file)
    if schema.get("type") != "array":
        raise ValueError("The paraphrase schema root must be an array schema.")
    print("[schema] Paraphrase schema loaded successfully.")
    return schema


def validate_records(records: list[dict[str, Any]], schema: dict[str, Any]) -> None:
    """Validate paraphrase records against the configured schema and local rules."""
    if not records:
        raise ValueError(
            "MANUAL_PARAPHRASES is empty. Add at least one paraphrase before running."
        )

    record_schema = schema.get("$defs", {}).get("manualRecord", {})
    properties = record_schema.get("properties", {})
    required_fields = set(record_schema.get("required", []))
    if not required_fields or not properties:
        raise ValueError("The paraphrase schema must define record properties and required fields.")
    questions: set[str] = set()
    for index, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise ValueError(f"Paraphrase {index} must be a JSON object.")
        missing = required_fields - record.keys()
        if missing:
            fields = ", ".join(sorted(missing))
            raise ValueError(f"Paraphrase {index} is missing required field(s): {fields}")
        unexpected = set(record) - set(properties)
        if unexpected and record_schema.get("additionalProperties") is False:
            fields = ", ".join(sorted(unexpected))
            raise ValueError(f"Paraphrase {index} has unexpected field(s): {fields}")
        for field in sorted(required_fields - {"sources"}):
            value = record[field]
            field_schema = properties.get(field, {})
            if field_schema.get("type") == "string" and (
                not isinstance(value, str)
                or len(value) < field_schema.get("minLength", 0)
                or not value.strip()
            ):
                raise ValueError(
                    f"Paraphrase {index} has an empty or non-text '{field}' field."
                )
        original = record["original_question"].strip()
        paraphrases = [
            record["paraphrased_question1"].strip(),
            record["paraphrased_question2"].strip(),
        ]
        normalized_original = original.casefold()
        normalized_paraphrases = [question.casefold() for question in paraphrases]
        if normalized_original in normalized_paraphrases:
            raise ValueError(f"Paraphrase {index} must differ from its original question.")
        if len(set(normalized_paraphrases)) != len(normalized_paraphrases):
            raise ValueError(f"Paraphrase {index} contains duplicate paraphrases.")
        for question in normalized_paraphrases:
            if question in questions:
                raise ValueError(f"Duplicate paraphrase question at record {index}.")
            questions.add(question)

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
                f"Paraphrase {index} has invalid 'sources'; use a list of non-empty strings."
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


def write_evaluation_jsonl(records: list[dict[str, Any]], jsonl_path: Path) -> Path:
    """Write one normalized evaluation record for each authored paraphrase."""
    evaluation_path = jsonl_path.with_name(f"{jsonl_path.stem}_Evaluation.jsonl")
    evaluation_records = []
    for record in records:
        for field in ("paraphrased_question1", "paraphrased_question2"):
            evaluation_records.append({
                "question": record[field],
                "grading_notes": record["grading_notes"],
                "sources": record["sources"],
            })
    with evaluation_path.open("w", encoding="utf-8", newline="\n") as evaluation_file:
        for record in evaluation_records:
            evaluation_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return evaluation_path


def evaluation_jsonl_is_valid(records: list[dict[str, Any]], evaluation_path: Path) -> bool:
    """Check that the existing evaluation JSONL matches the grouped records."""
    if not evaluation_path.is_file():
        return False
    expected = []
    for record in records:
        for field in ("paraphrased_question1", "paraphrased_question2"):
            expected.append({
                "question": record[field],
                "grading_notes": record["grading_notes"],
                "sources": record["sources"],
            })
    try:
        with evaluation_path.open("r", encoding="utf-8") as evaluation_file:
            actual = [json.loads(line) for line in evaluation_file if line.strip()]
    except (OSError, json.JSONDecodeError):
        return False
    return actual == expected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write manually authored paraphrased questions to JSON and JSONL."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Overwrite existing JSON and JSONL output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output = args.output.expanduser().resolve()
    print("=== Manual paraphrase generator ===")
    print(f"[args] output_json={args.output}")
    print(f"[args] rebuild={args.rebuild}")
    print(f"[output] JSON output path: {args.output}")
    output_exists = args.output.is_file()
    if output_exists and not args.rebuild:
        if not ask_override(str(args.output.resolve())):
            schema = load_schema()
            with args.output.open("r", encoding="utf-8") as json_file:
                existing_records = json.load(json_file)
            validate_records(existing_records, schema)
            existing_jsonl = args.output.with_suffix(".jsonl")
            evaluation_jsonl = existing_jsonl.with_name(f"{existing_jsonl.stem}_Evaluation.jsonl")
            if evaluation_jsonl_is_valid(existing_records, evaluation_jsonl):
                print("[skip] Existing JSON and evaluation JSONL are valid; no files were changed.")
            else:
                write_evaluation_jsonl(existing_records, existing_jsonl)
                print(f"[repair] Evaluation JSONL rebuilt: {evaluation_jsonl}")
            return
        args.rebuild = True
    if output_exists and args.rebuild:
        print("[override] Existing JSON output will be regenerated.")

    schema = load_schema()
    with tqdm(
        total=len(MANUAL_PARAPHRASES),
        desc="Generating Manual paraphrases",
        unit="record",
    ) as paraphrase_bar:
        validate_records(MANUAL_PARAPHRASES, schema)
        jsonl_path = write_json_outputs(MANUAL_PARAPHRASES, args.output)
        evaluation_jsonl_path = write_evaluation_jsonl(MANUAL_PARAPHRASES, jsonl_path)
        paraphrase_bar.update(len(MANUAL_PARAPHRASES))

    print(f"[validate] Validated {len(MANUAL_PARAPHRASES)} manual paraphrase record(s).")
    print(f"[write] JSON output written: {args.output}")
    print(f"[write] JSONL output written: {jsonl_path}")
    print(f"[write] Evaluation JSONL output written: {evaluation_jsonl_path}")
    print(f"[complete] Generated {len(MANUAL_PARAPHRASES)} manual paraphrase record(s).")


if __name__ == "__main__":
    with log_run(__file__):
        main()
