#################################################################################
# 
# Massachusetts Institute of Technology - xPRO
# RAG and Context Engineering: Designing and Building Production-Grade AI Systems
# 
# Student: Miguel Herrera
# Section: B
# Date and Time: 2026-09-12 11:54:44 -07:00
#
# Description: This module loads persisted Chroma documents, delegates missing
#              database construction, preserves document IDs, and runs similarity search.
#
#################################################################################

"""Chroma vector-store construction, loading, and similarity retrieval."""
from __future__ import annotations

import configparser
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from langchain_chroma import Chroma


EmbeddingFactory = Callable[[], object]
CONFIG_PATH = Path(__file__).resolve().parents[1] / "retrieval.conf"
_config = configparser.ConfigParser()
_config.read(CONFIG_PATH)
VECTOR_CANDIDATES = _config.getint("retrieval", "vector_candidates", fallback=10)
VECTOR_TOP_K = VECTOR_CANDIDATES
BUILD_UTILITY = CONFIG_PATH.parent / "Utility_Scripts" / "Build_Wikipedia_Article_ChromaDB.py"


def load_docs_from_chroma(chroma_dir: str, embedding_function: object) -> list[dict]:
    """Load document text and IDs from a persisted Chroma database."""
    if not os.path.isdir(chroma_dir) or not os.listdir(chroma_dir):
        return []
    db = Chroma(persist_directory=chroma_dir, embedding_function=embedding_function)
    try:
        results = db.get(include=["documents", "metadatas"])
    except Exception as exc:
        print(f"Warning: could not read persisted Chroma docs: {exc}")
        return []
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
    """Load a persisted Chroma DB or build it with the database utility."""
    if os.path.isdir(chroma_dir) and os.listdir(chroma_dir):
        print(f"Loading existing vector DB from {chroma_dir}/")
        return Chroma(persist_directory=chroma_dir, embedding_function=embedding_function)

    if not BUILD_UTILITY.is_file():
        raise FileNotFoundError(f"Chroma build utility not found: {BUILD_UTILITY}")

    print(f"Chroma DB not found. Delegating construction to {BUILD_UTILITY}")
    result = subprocess.run(
        [sys.executable, str(BUILD_UTILITY), "openrouter", "--db-dir", str(chroma_dir)],
        cwd=str(BUILD_UTILITY.parent),
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Chroma build utility failed with exit code {result.returncode}: {BUILD_UTILITY}"
        )
    if not os.path.isdir(chroma_dir) or not os.listdir(chroma_dir):
        raise RuntimeError(f"Chroma build utility completed without creating: {chroma_dir}")
    return Chroma(persist_directory=chroma_dir, embedding_function=embedding_function)


def get_top_k(db: Chroma, query: str, k: int = VECTOR_CANDIDATES) -> list[tuple[str, str, float]]:
    """Return document ID, text, and vector distance for the top-k results."""
    results = db.similarity_search_with_score(query, k=k)
    return [(document.metadata.get("id", "unknown"), document.page_content, score)
            for document, score in results]
