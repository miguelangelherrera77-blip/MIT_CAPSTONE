#################################################################################
# 
# Massachusetts Institute of Technology - xPRO
# RAG and Context Engineering: Designing and Building Production-Grade AI Systems
# 
# Student: Miguel Herrera
# Section: B
# Date and Time: 2026-09-12 11:54:44 -07:00
#
# Description: This module normalizes BM25 relevance and vector-distance scores,
#              applies configured weights, merges candidates by document ID, and
#              returns the fused top-k retrieval results.
#
#################################################################################

"""Weighted score fusion for BM25 and vector retrieval results."""
from __future__ import annotations

import configparser
from pathlib import Path


CONFIG_PATH = Path(__file__).resolve().parents[1] / "retrieval.conf"
_config = configparser.ConfigParser()
_config.read(CONFIG_PATH)
WEIGHT_BM25 = _config.getfloat("ranking", "weight_bm25", fallback=0.5)
WEIGHT_VECTOR = _config.getfloat("ranking", "weight_vector", fallback=0.5)
FUSED_TOP_K = _config.getint("retrieval", "fused_top_k", fallback=4)


def normalize_scores(scores: list[float], invert: bool = False) -> list[float]:
    """Normalize scores to 0..1, optionally inverting distance-like scores."""
    if not scores:
        return []
    low, high = min(scores), max(scores)
    if high == low:
        return [0.5] * len(scores)
    normalized = [(score - low) / (high - low) for score in scores]
    return [1.0 - value for value in normalized] if invert else normalized


def weighted_hybrid_rank(
    bm25_results: list[tuple[str, str, float]],
    vector_results: list[tuple[str, str, float]],
    k: int,
    bm25_weight: float = WEIGHT_BM25,
    vector_weight: float = WEIGHT_VECTOR,
) -> list[tuple[str, str, float]]:
    """Fuse BM25 scores and vector distances into a ranked top-k result list."""
    content_by_id: dict[str, str] = {}
    bm25_scores: dict[str, float] = {}
    vector_scores: dict[str, float] = {}

    if bm25_results:
        normalized = normalize_scores([score for _, _, score in bm25_results])
        for (document_id, content, _), score in zip(bm25_results, normalized):
            content_by_id[document_id] = content
            bm25_scores[document_id] = score

    if vector_results:
        normalized = normalize_scores(
            [distance for _, _, distance in vector_results],
            invert=True,
        )
        for (document_id, content, _), score in zip(vector_results, normalized):
            content_by_id[document_id] = content
            vector_scores[document_id] = score

    fused = [
        (
            document_id,
            content,
            bm25_weight * bm25_scores.get(document_id, 0.0)
            + vector_weight * vector_scores.get(document_id, 0.0),
        )
        for document_id, content in content_by_id.items()
    ]
    fused.sort(key=lambda result: result[2], reverse=True)
    return fused[:k]
