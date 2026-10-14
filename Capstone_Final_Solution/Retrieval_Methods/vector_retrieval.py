#################################################################################
# 
# Massachusetts Institute of Technology - xPRO
# RAG and Context Engineering: Designing and Building Production-Grade AI Systems
# 
# Student: Miguel Herrera
# Section: B
# Date and Time: 2026-09-12 11:54:44 -07:00
# Header reviewed: 2026-10-12 11:34:17 -07:00
#
# Description: Solution-local loading and similarity search of existing Chroma data.
#              Preserves document IDs and requires existing database artifacts;
#              never constructs a missing database or updates stored documents.
#              Similarity queries can still make embedding API calls.
#
#################################################################################

"""Read-only loading and similarity retrieval of existing Chroma data."""
from __future__ import annotations

import configparser
import os
from collections.abc import Callable
from pathlib import Path

from langchain_chroma import Chroma


EmbeddingFactory = Callable[[], object]
CONFIG_PATH = Path(__file__).resolve().parents[1] / "Final_Solution_Retrieval.conf"
_config = configparser.ConfigParser()
_config.read(CONFIG_PATH)
VECTOR_CANDIDATES = _config.getint("retrieval", "vector_candidates", fallback=10)
VECTOR_TOP_K = VECTOR_CANDIDATES


def load_docs_from_chroma(chroma_dir: str, embedding_function: object) -> list[dict]:
    """Load document text and IDs from a persisted Chroma database."""
    if not os.path.isfile(os.path.join(chroma_dir, "chroma.sqlite3")):
        raise FileNotFoundError(f"Existing Chroma database required at {chroma_dir}")
    db = Chroma(persist_directory=chroma_dir, embedding_function=embedding_function)
    results = db.get(include=["documents", "metadatas"])
    docs = []
    for doc_id, text, metadata in zip(
        results.get("ids", []), results.get("documents", []), results.get("metadatas", [])
    ):
        if not text:
            continue
        docs.append({"id": str((metadata or {}).get("id", doc_id)), "text": text})
    print(f"Loaded {len(docs)} documents from existing Chroma DB at {chroma_dir}/")
    return docs


def build_or_load_db(
    docs: list[dict],
    chroma_dir: str,
    embedding_function: object,
    batch_size: int = 50,
) -> Chroma:
    """Open an existing Chroma database; never build or update stored documents."""
    if os.path.isfile(os.path.join(chroma_dir, "chroma.sqlite3")):
        print(f"Loading existing vector DB from {chroma_dir}/")
        return Chroma(persist_directory=chroma_dir, embedding_function=embedding_function)

    raise FileNotFoundError(
        f"Existing Chroma database required at {chroma_dir}; automatic builds are disabled."
    )


def get_top_k(db: Chroma, query: str, k: int = VECTOR_CANDIDATES) -> list[tuple[str, str, float]]:
    """Return document ID, text, and vector distance for the top-k results."""
    results = db.similarity_search_with_score(query, k=k)
    return [(document.metadata.get("id", "unknown"), document.page_content, score)
            for document, score in results]
