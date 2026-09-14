#################################################################################
# 
# Massachusetts Institute of Technology - xPRO
# RAG and Context Engineering: Designing and Building Production-Grade AI Systems
# 
# Student: Miguel Herrera
# Section: B
# Date and Time: 2026-09-12 11:54:44 -07:00
#
# Description: This script loads or creates Wikipedia JSONL chunks, embeds them
#              with OpenRouter, and builds the persisted Checkpoint 4.1 Chroma
#              database while logging skip and rebuild decisions.
#
#################################################################################

"""Build the Checkpoint 4.1 Chroma database from Wikipedia JSONL chunks."""
from __future__ import annotations

import argparse
import configparser
import json
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from Chunk_Wikipedia_HTML_To_JSONL import (
    OUTPUT_DIR as DEFAULT_JSONL_DIR,
    WIKI_DIR,
    chunk_article,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
from utility_logging import log_run


SCRIPT_DIR = Path(__file__).resolve().parent
FINAL_CAPSTONE_DIR = SCRIPT_DIR.parent
DEFAULT_DB_DIR = FINAL_CAPSTONE_DIR / "Capstone_Database" / "Capstone_Chroma_DB"
CONFIG_PATH = FINAL_CAPSTONE_DIR / "retrieval.conf"
_config = configparser.ConfigParser()
_config.read(CONFIG_PATH)
DEFAULT_EMBEDDING_MODEL = _config.get("embedding", "model", fallback="openai/text-embedding-3-small")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
BATCH_SIZE = 50
PLACEHOLDER_FILENAMES = {"readme.txt", "readme.md", ".gitkeep"}


def validate_database_directory(db_dir: Path) -> Path:
    """Allow ChromaDB creation only in the canonical project database directory."""
    canonical_dir = DEFAULT_DB_DIR.resolve()
    requested_dir = db_dir.resolve()
    if requested_dir != canonical_dir:
        raise SystemExit(
            f"[db] ChromaDB must be created at {canonical_dir}; received {requested_dir}"
        )
    return canonical_dir


def database_has_data(db_dir: Path) -> bool:
    """Return True only when a real persisted Chroma database with embedded vectors is present."""
    if not db_dir.is_dir():
        return False

    sqlite_path = db_dir / "chroma.sqlite3"
    if not sqlite_path.is_file() or sqlite_path.stat().st_size == 0:
        return False

    # Chroma only writes a collection segment directory once vectors are actually persisted;
    # chroma.sqlite3 alone can exist from an interrupted run with no embedded data.
    return any(
        path.is_dir() and any(child.is_file() for child in path.iterdir())
        for path in db_dir.iterdir()
    )


def ensure_jsonl(jsonl_dir: Path) -> list[Path]:
    """Build JSONL files from HTML when the requested JSONL corpus is absent."""
    jsonl_files = sorted(jsonl_dir.glob("*.jsonl")) if jsonl_dir.is_dir() else []
    if jsonl_files:
        print(f"[jsonl] Found {len(jsonl_files)} JSONL files in {jsonl_dir}")
        return jsonl_files

    if not WIKI_DIR.is_dir():
        raise SystemExit(f"[jsonl] Wikipedia HTML directory not found: {WIKI_DIR}")

    html_files = sorted(WIKI_DIR.glob("*.html"))
    if not html_files:
        raise SystemExit(f"[jsonl] No HTML files found in {WIKI_DIR}")

    print(f"[jsonl] No JSONL files found. Chunking {len(html_files)} HTML files...")
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=True,
    )
    for index, html_path in enumerate(html_files, start=1):
        chunk_count = chunk_article(html_path, jsonl_dir, splitter)
        print(f"[jsonl] {index}/{len(html_files)}: {chunk_count} chunks")

    jsonl_files = sorted(jsonl_dir.glob("*.jsonl"))
    print(f"[jsonl] Created {len(jsonl_files)} JSONL files in {jsonl_dir}")
    return jsonl_files


def load_jsonl_documents(jsonl_files: list[Path]) -> tuple[list[Document], list[str]]:
    """Load JSONL chunk records as LangChain documents and stable Chroma IDs."""
    documents: list[Document] = []
    ids: list[str] = []
    print(f"[load] Reading {len(jsonl_files)} JSONL files...")
    with tqdm(total=len(jsonl_files), desc="Reading JSONL chunk files", unit="file") as load_bar:
        for file_index, jsonl_path in enumerate(jsonl_files, start=1):
            with jsonl_path.open("r", encoding="utf-8") as jsonl_file:
                for line_number, line in enumerate(jsonl_file, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSON in {jsonl_path}:{line_number}: {exc}") from exc
                    text = str(record.get("text", "")).strip()
                    record_id = str(record.get("id", "")).strip()
                    if not text or not record_id:
                        raise ValueError(f"Missing id or text in {jsonl_path}:{line_number}")
                    metadata = dict(record.get("metadata") or {})
                    metadata.update(
                        {
                            "source": record.get("source", metadata.get("source", jsonl_path.name)),
                            "document_id": record.get("document_id", metadata.get("document_id", jsonl_path.stem)),
                            "chunk_index": int(record.get("chunk_index", metadata.get("chunk_index", 0))),
                        }
                    )
                    documents.append(Document(page_content=text, metadata=metadata))
                    ids.append(record_id)
            load_bar.update(1)
    if not documents:
        raise SystemExit("[load] JSONL files contained no usable chunk records")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk IDs found in the JSONL corpus")
    print(f"[load] Loaded {len(documents)} chunks")
    return documents, ids


def get_embeddings(model: str, route: str) -> OpenAIEmbeddings:
    load_dotenv()
    if route != "openrouter":
        raise ValueError("Only the 'openrouter' embedding route is supported.")

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("[embed] OPENROUTER_API_KEY is not set")
    embedding_kwargs = {
        "model": model,
        "api_key": api_key,
        "check_embedding_ctx_length": False,
    }
    embedding_kwargs["base_url"] = OPENROUTER_BASE_URL
    return OpenAIEmbeddings(**embedding_kwargs)


def build_database(
    jsonl_dir: Path,
    db_dir: Path,
    embedding_model: str,
    route: str,
    force_rebuild: bool = False,
) -> None:
    db_dir = validate_database_directory(db_dir)
    if force_rebuild and db_dir.exists():
        print(f"[db] Decision: rebuild requested; removing existing database at {db_dir}")
        shutil.rmtree(db_dir)
    elif database_has_data(db_dir):
        print(
            f"[db] Decision: existing Chroma database found at {db_dir}; "
            "skipping rebuild and preserving existing data"
        )
        return
    elif db_dir.exists():
        print(f"[db] Directory exists but does not contain a valid Chroma database: {db_dir}")
        print("[db] Cleaning placeholder/scaffold content before rebuilding the database.")
        for path in sorted(db_dir.iterdir(), reverse=True):
            if path.is_dir():
                shutil.rmtree(path)
            elif path.is_file():
                path.unlink()

    jsonl_files = ensure_jsonl(jsonl_dir)
    documents, ids = load_jsonl_documents(jsonl_files)
    db_dir.mkdir(parents=True, exist_ok=True)
    print(f"[db] Creating Chroma database at {db_dir}")
    db = Chroma(
        persist_directory=str(db_dir),
        embedding_function=get_embeddings(embedding_model, route),
    )
    with tqdm(total=len(documents), desc="Embedding and storing chunks", unit="chunk") as embed_bar:
        for start in range(0, len(documents), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(documents))
            db.add_documents(documents[start:end], ids=ids[start:end])
            embed_bar.update(end - start)
    print(f"[db] Complete: {len(documents)} chunks stored in {db_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action_or_route",
        nargs="?",
        choices=("rebuild", "openrouter"),
        help="Use 'rebuild' to recreate, or 'openrouter' to select the embedding route.",
    )
    parser.add_argument(
        "route",
        nargs="?",
        choices=("openrouter",),
        default=None,
        help="Embedding API route. Only 'openrouter' is supported.",
    )
    parser.add_argument("--jsonl-dir", type=Path, default=DEFAULT_JSONL_DIR)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    action = "rebuild" if args.action_or_route == "rebuild" else None
    route = "openrouter"
    print("=== Capstone Chroma database builder ===")
    print(f"[config] action: {action or 'use existing database when available'}")
    print(f"[config] embedding route: {route}")
    print(f"[config] JSONL source: {args.jsonl_dir.resolve()}")
    print(f"[config] Chroma target: {args.db_dir.resolve()}")
    build_database(
        args.jsonl_dir,
        args.db_dir,
        args.embedding_model,
        route,
        force_rebuild=action == "rebuild",
    )


if __name__ == "__main__":
    with log_run(__file__):
        main()