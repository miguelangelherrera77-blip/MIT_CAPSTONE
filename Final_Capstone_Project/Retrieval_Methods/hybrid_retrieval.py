#################################################################################
# 
# Massachusetts Institute of Technology - xPRO
# RAG and Context Engineering: Designing and Building Production-Grade AI Systems
# 
# Student: Miguel Herrera
# Section: B
# Date and Time: 2026-09-13 18:18:45 -07:00
#
# Description: Unified retrieval and answer-generation engine for lexical BM25,
#              semantic Chroma vector, hybrid fusion, graph-expanded, and combined
#              Lexical+Semantic+Graph modes. Loads only required backends, ranks context,
#              expands graph neighbors when selected, and calls the grounded answer LLM.
#
#################################################################################

"""Hybrid BM25 and vector retrieval orchestration."""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from Final_Capstone_Project.Retrieval_Methods.bm25_retrieval import BM25Retriever, BM25_CANDIDATES
from Final_Capstone_Project.Retrieval_Methods.vector_retrieval import (
    VECTOR_CANDIDATES,
    build_or_load_db,
    get_top_k as get_vector_top_k,
)
from Final_Capstone_Project.Ranking_Techniques.weighted_fusion_ranking import (
    FUSED_TOP_K,
    WEIGHT_BM25,
    WEIGHT_VECTOR,
    weighted_hybrid_rank,
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
SUPPORTED_SEARCH_METHODS = {"lexical", "semantic", "graph", "graph_enabled", "hybrid", "all"}
FUSED_SEARCH_METHODS = {"graph", "graph_enabled", "hybrid", "all"}
GRAPH_SEARCH_METHODS = {"graph", "graph_enabled", "all"}
ANSWER_SYSTEM = (
    "You are a helpful assistant for a Wikipedia retrieval engine. Answer the question "
    "using ONLY the provided Wikipedia article excerpts, and quote from them where you "
    "can. If the documents do not contain the answer, say so rather than guessing."
)


def get_embeddings(model: str) -> OpenAIEmbeddings:
    """Create the configured embedding function for the Chroma vector DB."""
    return OpenAIEmbeddings(
        model=model,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
        check_embedding_ctx_length=False,
    )


class BaseRetriever(ABC):
    """Shared RAG retriever interface used by the evaluation harness."""

    def __init__(
        self,
        llm_model: str,
        api_key: str | None = None,
        base_url: str = OPENROUTER_BASE_URL,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ):
        self._llm = ChatOpenAI(
            model=llm_model,
            api_key=api_key or os.environ["OPENROUTER_API_KEY"],
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    @abstractmethod
    def retrievedContext(self, query: str) -> str:
        ...

    def query(self, question: str) -> str:
        context = self.retrievedContext(question)
        messages = [
            SystemMessage(content=ANSWER_SYSTEM),
            HumanMessage(content=f"Context (Wikipedia excerpts):\n{context}\n\nQuestion: {question}"),
        ]
        response = self._llm.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)


def retrieve_hybrid(
    query: str,
    k: int,
    bm25_top_k: int,
    vector_top_k: int,
    bm25_search: Callable[[str, int], list[tuple[str, str, float]]],
    vector_search: Callable[[str, int], list[tuple[str, str, float]]],
    rank_results: Callable[
        [list[tuple[str, str, float]], list[tuple[str, str, float]], int],
        list[tuple[str, str, float]],
    ],
) -> list[tuple[str, str, float]]:
    """Retrieve BM25/vector candidates and delegate weighted ranking."""
    bm25_results = bm25_search(query, bm25_top_k)
    vector_results = vector_search(query, vector_top_k)
    return rank_results(bm25_results, vector_results, k)


class HybridRetriever(BaseRetriever):
    """Hybrid retriever that combines BM25 and vector search with weighted fusion,
    with optional graph-augmented neighborhood expansion."""

    def __init__(
        self,
        docs: list[dict],
        search_method: str = "hybrid",
        *,
        llm_model: str = "openai/gpt-5.4-mini",
        api_key: str | None = None,
        base_url: str = OPENROUTER_BASE_URL,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        embedding_function: object | None = None,
        chroma_dir: str | None = None,
    ):
        super().__init__(
            llm_model=llm_model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._search_method = str(search_method).lower()
        if self._search_method not in SUPPORTED_SEARCH_METHODS:
            raise ValueError(
                f"Unsupported search method '{search_method}'. "
                f"Choose one of: {', '.join(sorted(SUPPORTED_SEARCH_METHODS))}."
            )
        self._doc_ids = [d["id"] for d in docs]
        self._doc_texts = [d["text"] for d in docs]
        self._bm25 = BM25Retriever() if self._search_method == "lexical" or self._search_method in FUSED_SEARCH_METHODS else None
        self._db = None
        if self._search_method == "semantic" or self._search_method in FUSED_SEARCH_METHODS:
            if embedding_function is None:
                raise ValueError("embedding_function is required for hybrid, semantic, graph, and all search.")
            self._db = build_or_load_db(
                docs,
                chroma_dir or str(Path(__file__).resolve().parents[1] / "Capstone_Database" / "Capstone_Chroma_DB"),
                embedding_function,
            )
        self._graph = None
        if self._search_method in GRAPH_SEARCH_METHODS:
            graph_file = Path(__file__).resolve().parents[1] / "Capstone_Database" / "Capstone_Graph_DB" / "wikipedia_articles.graphml"
            if graph_file.is_file():
                import networkx as nx
                print(f"[graph] Loading knowledge graph from {graph_file.name}...")
                self._graph = nx.read_graphml(graph_file)
                print(f"[graph] Loaded {self._graph.number_of_nodes()} nodes and {self._graph.number_of_edges()} edges.")
            else:
                print(f"[warn] Graph file not found at {graph_file}; falling back to hybrid retrieval.")
        print(f"{self._search_method.capitalize()} retriever ready over {len(docs)} documents.")

    def _bm25_topk(self, query: str, k: int):
        if self._bm25 is None:
            raise RuntimeError("BM25 retrieval is unavailable for semantic search mode")
        return self._bm25.get_top_k(query, k)

    def _vector_topk(self, query: str, k: int):
        if self._db is None:
            raise RuntimeError("Vector retrieval is unavailable for lexical search mode")
        return get_vector_top_k(self._db, query, k)

    def getTopK(self, query: str, k: int):
        if self._search_method == "semantic":
            return self._vector_topk(query, k)
        if self._search_method == "lexical":
            return self._bm25_topk(query, k)
        if self._search_method in FUSED_SEARCH_METHODS:
            return retrieve_hybrid(
                query,
                k,
                BM25_CANDIDATES,
                VECTOR_CANDIDATES,
                self._bm25_topk,
                self._vector_topk,
                lambda bm, vec, limit: weighted_hybrid_rank(
                    bm, vec, limit, WEIGHT_BM25, WEIGHT_VECTOR
                ),
            )
        raise RuntimeError(f"No retrieval dispatch exists for '{self._search_method}'.")

    def _graph_expand(self, seed_results: list[tuple[str, str, float]]) -> str:
        """Expand seed chunks with adjacent sequence chunks and referenced article leads."""
        if self._graph is None:
            return "\n\n---\n\n".join(f"[{doc_id}]\n{content}" for doc_id, content, _ in seed_results)
        sections = []
        seen_docs = set()
        for doc_id, content, _ in seed_results:
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                sections.append(f"=== PRIMARY RETRIEVED CHUNK: {doc_id} ===\n{content}")
            node_id = f"chunk:{doc_id}"
            if self._graph.has_node(node_id):
                for _, target, data in self._graph.out_edges(node_id, data=True):
                    edge_type = data.get("edge_type")
                    if edge_type in {"next_chunk", "previous_chunk"}:
                        target_chunk = target.removeprefix("chunk:")
                        if target_chunk not in seen_docs and self._graph.has_node(target):
                            seen_docs.add(target_chunk)
                            t_text = self._graph.nodes[target].get("text", "")
                            if t_text:
                                sections.append(f"--- ADJACENT CONTEXT ({edge_type}): {target_chunk} ---\n{t_text}")
                    elif edge_type == "links_to":
                        target_art = target.removeprefix("article:")
                        lead_chunk = f"chunk:{target_art}::chunk-0000"
                        lead_id = f"{target_art}::chunk-0000"
                        if lead_id not in seen_docs and self._graph.has_node(lead_chunk):
                            seen_docs.add(lead_id)
                            l_text = self._graph.nodes[lead_chunk].get("text", "")
                            l_label = data.get("link_label", target_art)
                            if l_text:
                                sections.append(f"--- REFERENCED ARTICLE CONTEXT ({target_art} via '{l_label}'): {lead_id} ---\n{l_text}")
        return "\n\n---\n\n".join(sections)

    def retrievedContext(self, query: str) -> str:
        if self._search_method in {"graph", "graph_enabled", "all"}:
            seeds = self.getTopK(query, 2)
            return self._graph_expand(seeds)
        results = self.getTopK(query, FUSED_TOP_K)
        return "\n\n---\n\n".join(f"[{doc_id}]\n{content}" for doc_id, content, _ in results)
