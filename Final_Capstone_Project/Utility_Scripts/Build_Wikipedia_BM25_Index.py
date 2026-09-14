#################################################################################
# 
# Massachusetts Institute of Technology - xPRO
# RAG and Context Engineering: Designing and Building Production-Grade AI Systems
# 
# Student: Miguel Herrera
# Section: B
# Date and Time: 2026-09-13 18:50:00 -07:00
#
# Description: Builds and persists the bm25s lexical index for Wikipedia JSONL
#              chunks, preserving stable chunk IDs and text metadata for retrieval.
#
#################################################################################

"""Build the persisted bm25s lexical index used by Checkpoint 4.1."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import bm25s
from tqdm import tqdm
from utility_logging import log_run

SCRIPT_DIR = Path(__file__).resolve().parent
FINAL_CAPSTONE_DIR = SCRIPT_DIR.parent
DEFAULT_JSONL_DIR = FINAL_CAPSTONE_DIR / "Capstone_Database" / "Wikipedia_JSONL"
DEFAULT_INDEX_DIR = FINAL_CAPSTONE_DIR / "Capstone_Database" / "Capstone_BM25_Lexical_Indexes"
METADATA_FILE = "documents.json"
VALID_INDEX_FILES = {"params.index.json", "vocab.index.json", METADATA_FILE}
LEGACY_VALID_INDEX_FILES = {"index.json", METADATA_FILE}
PLACEHOLDER_FILENAMES = {"readme.txt", "readme.md", ".gitkeep"}


def has_valid_index(index_dir: Path) -> bool:
    """Return True only when a real persisted bm25s index is present."""
    if not index_dir.is_dir():
        return False
    files = {path.name.lower() for path in index_dir.iterdir() if path.is_file()}
    normalized = {name.lower() for name in files}
    return VALID_INDEX_FILES.issubset(normalized) or LEGACY_VALID_INDEX_FILES.issubset(normalized)


def load_documents(jsonl_dir: Path) -> list[dict[str, str]]:
    documents = []
    for jsonl_path in sorted(jsonl_dir.glob("*.jsonl")):
        with jsonl_path.open("r", encoding="utf-8") as jsonl_file:
            for line_number, line in enumerate(jsonl_file, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                record_id = str(record.get("id", "")).strip()
                text = str(record.get("text", "")).strip()
                if not record_id or not text:
                    raise ValueError(f"Missing id or text in {jsonl_path}:{line_number}")
                documents.append({"id": record_id, "text": text})
    if not documents:
        raise ValueError(f"No JSONL chunks found in {jsonl_dir}")
    return documents


def tokenize(text: str) -> list[str]:
    import re
    stopwords = {
        "a", "an", "the", "and", "but", "or", "nor", "so", "yet", "for",
        "in", "on", "at", "to", "of", "by", "with", "from", "into", "onto", "upon",
        "about", "above", "below", "between", "through", "during", "before", "after",
        "under", "over", "around", "along", "across", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "i", "we", "you", "he", "she", "it", "they", "me", "us", "him", "her", "them",
        "my", "our", "your", "his", "its", "their", "this", "that", "these", "those",
        "as", "if", "up", "out", "not", "no",
    }
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in stopwords]


def build_index(jsonl_dir: Path, index_dir: Path, rebuild: bool = False) -> None:
    if has_valid_index(index_dir) and not rebuild:
        print(f"[skip] Existing bm25s index preserved: {index_dir}")
        return
    if index_dir.exists():
        if rebuild:
            print(f"[rebuild] Removing existing bm25s index: {index_dir}")
            shutil.rmtree(index_dir)
        else:
            placeholder_files = [
                path for path in sorted(index_dir.iterdir())
                if path.is_file() and path.name.lower() in PLACEHOLDER_FILENAMES
            ]
            if placeholder_files:
                print(f"[cleanup] Removing placeholder scaffolding before building index: {index_dir}")
                for path in placeholder_files:
                    path.unlink()
            if any(index_dir.iterdir()):
                print(f"[rebuild] Existing directory is not a valid bm25s index; rebuilding: {index_dir}")
                for path in sorted(index_dir.iterdir(), reverse=True):
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()

    documents = load_documents(jsonl_dir)
    corpus_tokens = [tokenize(document["text"]) for document in documents]
    index_dir.mkdir(parents=True, exist_ok=True)
    print(f"[build] Indexing {len(documents)} Wikipedia chunks with bm25s...")
    retriever = bm25s.BM25(method="lucene")
    retriever.index(corpus_tokens, show_progress=True)
    retriever.save(str(index_dir))
    with (index_dir / METADATA_FILE).open("w", encoding="utf-8") as metadata_file:
        json.dump(documents, metadata_file, ensure_ascii=False)
    print(f"[complete] bm25s index saved: {index_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl-dir", type=Path, default=DEFAULT_JSONL_DIR)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_index(args.jsonl_dir, args.index_dir, args.rebuild)


if __name__ == "__main__":
    with log_run(__file__):
        main()
