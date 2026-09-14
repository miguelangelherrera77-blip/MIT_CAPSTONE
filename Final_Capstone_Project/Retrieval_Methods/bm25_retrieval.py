#################################################################################
# 
# Massachusetts Institute of Technology - xPRO
# RAG and Context Engineering: Designing and Building Production-Grade AI Systems
# 
# Student: Miguel Herrera
# Section: B
# Date and Time: 2026-09-13 18:50:00 -07:00
#
# Description: This module loads Wikipedia JSONL chunks, loads or builds the
#              persisted bm25s lexical index, preserves stable chunk metadata,
#              and returns top-k lexical matches without rebuilding every run.
#
#################################################################################

"""BM25 keyword retrieval for the Wikipedia corpus."""
from __future__ import annotations

import configparser
import json
import re
import subprocess
import sys
from pathlib import Path

import bm25s


CONFIG_PATH = Path(__file__).resolve().parents[1] / "retrieval.conf"
_config = configparser.ConfigParser()
_config.read(CONFIG_PATH)
BM25_CANDIDATES = _config.getint("retrieval", "bm25_candidates", fallback=10)
BM25_TOP_K = BM25_CANDIDATES
WIKIPEDIA_JSONL_DIR = CONFIG_PATH.parent / "Capstone_Database" / "Wikipedia_JSONL"
CHUNKER_SCRIPT = CONFIG_PATH.parent / "Utility_Scripts" / "Chunk_Wikipedia_HTML_To_JSONL.py"
BM25_INDEX_DIR = CONFIG_PATH.parent / "Capstone_Database" / "Capstone_BM25_Lexical_Indexes"
BM25_BUILDER = CONFIG_PATH.parent / "Utility_Scripts" / "Build_Wikipedia_BM25_Index.py"
BM25_METADATA_FILE = "documents.json"


_STOPWORDS = {
    "a", "an", "the", "and", "but", "or", "nor", "so", "yet", "for",
    "in", "on", "at", "to", "of", "by", "with", "from", "into", "onto", "upon",
    "about", "above", "below", "between", "through", "during", "before", "after",
    "under", "over", "around", "along", "across", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "i", "we", "you", "he", "she", "it", "they", "me", "us", "him", "her", "them",
    "my", "our", "your", "his", "its", "their", "this", "that", "these", "those",
    "as", "if", "up", "out", "not", "no",
}


def tokenize(text: str) -> list[str]:
    """Tokenize text using the Checkpoint 4.1 BM25 normalization rules."""
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in _STOPWORDS]


class BM25Retriever:
    """Retrieve document IDs, text, and BM25 scores for a query."""

    def __init__(self, documents: list[dict] | None = None, index_dir: Path = BM25_INDEX_DIR):
        documents = documents if documents is not None else load_wikipedia_chunks()
        self._index_dir = Path(index_dir)
        self._ensure_index()
        self._index = bm25s.BM25.load(str(self._index_dir), mmap=True)
        metadata_path = self._index_dir / BM25_METADATA_FILE
        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            indexed_documents = json.load(metadata_file)
        self._document_ids = [document["id"] for document in indexed_documents]
        self._document_texts = [document["text"] for document in indexed_documents]

    def _ensure_index(self) -> None:
        required_files = (
            self._index_dir / BM25_METADATA_FILE,
            self._index_dir / "params.index.json",
            self._index_dir / "vocab.index.json",
        )
        if all(path.is_file() for path in required_files):
            return
        legacy_required_files = (self._index_dir / BM25_METADATA_FILE, self._index_dir / "index.json")
        if all(path.is_file() for path in legacy_required_files):
            return
        if not BM25_BUILDER.is_file():
            raise FileNotFoundError(f"BM25 builder not found: {BM25_BUILDER}")
        result = subprocess.run(
            [sys.executable, str(BM25_BUILDER), "--jsonl-dir", str(WIKIPEDIA_JSONL_DIR), "--index-dir", str(self._index_dir)],
            cwd=str(BM25_BUILDER.parent),
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"bm25s index builder failed with exit code {result.returncode}")

    def get_top_k(self, query: str, k: int = BM25_CANDIDATES) -> list[tuple[str, str, float]]:
        query_tokens = [tokenize(query)]
        top_indices, scores = self._index.retrieve(query_tokens, k=k, show_progress=False)
        top_indices = top_indices[0].tolist()
        top_scores = scores[0].tolist()
        return [
            (self._document_ids[index], self._document_texts[index], score)
            for index, score in zip(top_indices, top_scores)
        ]


def load_wikipedia_chunks() -> list[dict]:
    """Load Wikipedia JSONL chunks, creating them with the shared chunker if absent."""
    jsonl_files = sorted(WIKIPEDIA_JSONL_DIR.glob("*.jsonl")) if WIKIPEDIA_JSONL_DIR.is_dir() else []
    if not jsonl_files:
        if not CHUNKER_SCRIPT.is_file():
            raise FileNotFoundError(f"Wikipedia chunker not found: {CHUNKER_SCRIPT}")
        result = subprocess.run(
            [sys.executable, str(CHUNKER_SCRIPT)],
            cwd=str(CHUNKER_SCRIPT.parent),
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Wikipedia chunker failed with exit code {result.returncode}")
        jsonl_files = sorted(WIKIPEDIA_JSONL_DIR.glob("*.jsonl"))

    documents = []
    for jsonl_path in jsonl_files:
        with jsonl_path.open("r", encoding="utf-8") as jsonl_file:
            for line_number, line in enumerate(jsonl_file, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                text = str(record.get("text", "")).strip()
                record_id = str(record.get("id", "")).strip()
                if not text or not record_id:
                    raise ValueError(f"Missing id or text in {jsonl_path}:{line_number}")
                documents.append({"id": record_id, "text": text})
    if not documents:
        raise ValueError(f"No usable Wikipedia chunks found in {WIKIPEDIA_JSONL_DIR}")
    return documents
