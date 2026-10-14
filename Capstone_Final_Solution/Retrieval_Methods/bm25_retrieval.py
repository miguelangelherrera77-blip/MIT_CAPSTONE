#################################################################################
# 
# Massachusetts Institute of Technology - xPRO
# RAG and Context Engineering: Designing and Building Production-Grade AI Systems
# 
# Student: Miguel Herrera
# Section: B
# Date and Time: 2026-09-13 18:50:00 -07:00
# Header reviewed: 2026-10-12 11:34:17 -07:00
#
# Description: Solution-local BM25 retrieval over an existing persisted bm25s index.
#              Preserves indexed chunk IDs and text metadata; missing artifacts raise
#              an explicit error. Never creates or rebuilds the external database.
#
#################################################################################

"""BM25 keyword retrieval for the Wikipedia corpus."""
from __future__ import annotations

import configparser
import json
import re
from pathlib import Path

import bm25s
from solution_config import DATABASE_DIR


CONFIG_PATH = Path(__file__).resolve().parents[1] / "retrieval.conf"
_config = configparser.ConfigParser()
_config.read(CONFIG_PATH)
BM25_CANDIDATES = _config.getint("retrieval", "bm25_candidates", fallback=10)
BM25_TOP_K = BM25_CANDIDATES
WIKIPEDIA_JSONL_DIR = DATABASE_DIR / "Wikipedia_JSONL"
BM25_INDEX_DIR = DATABASE_DIR / "Capstone_BM25_Lexical_Indexes"
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
        raise FileNotFoundError(
            f"Existing BM25 index required at {self._index_dir}; automatic builds are disabled."
        )

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
    """Read existing Wikipedia JSONL chunks without creating database artifacts."""
    jsonl_files = sorted(WIKIPEDIA_JSONL_DIR.glob("*.jsonl")) if WIKIPEDIA_JSONL_DIR.is_dir() else []
    if not jsonl_files:
        raise FileNotFoundError(f"Existing Wikipedia JSONL corpus required at {WIKIPEDIA_JSONL_DIR}")

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
                documents.append({
                    **record,
                    "id": record_id,
                    "text": text,
                    "title": str(record.get("document_id", record_id.split("::")[0])).replace("_", " "),
                })
    if not documents:
        raise ValueError(f"No usable Wikipedia chunks found in {WIKIPEDIA_JSONL_DIR}")
    return documents
