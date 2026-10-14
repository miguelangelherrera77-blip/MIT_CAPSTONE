#################################################################################
# 
# Massachusetts Institute of Technology - xPRO
# RAG and Context Engineering: Designing and Building Production-Grade AI Systems
# 
# Student: Miguel Herrera
# Section: B
# Date and Time: 2026-09-13 18:18:45 -07:00
# Header reviewed: 2026-10-12 11:34:17 -07:00
#
# Description: Solution-local lexical, vector, hybrid, and graph retrieval orchestration.
#              Requires existing external indexes; never builds database artifacts.
#              Reranks candidates using title/term relevance and clean article leads,
#              bounds context, and labels retrieved sources for grounded answers.
#              The read-only HTML evidence overlay does not modify the indexed corpus.
#
#################################################################################

"""Hybrid BM25 and vector retrieval orchestration."""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Callable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from Retrieval_Methods.bm25_retrieval import BM25Retriever, BM25_CANDIDATES
from Retrieval_Methods.vector_retrieval import (
    VECTOR_CANDIDATES,
    build_or_load_db,
    get_top_k as get_vector_top_k,
)
from Ranking_Techniques.weighted_fusion_ranking import (
    FUSED_TOP_K,
    WEIGHT_BM25,
    WEIGHT_VECTOR,
    weighted_hybrid_rank,
    normalize_scores,
)
from Retrieval_Methods.article_evidence import (
    ArticleEvidence, format_evidence, limit_context, sources_footer,
)
from solution_config import CONFIG, DATABASE_DIR

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
SUPPORTED_SEARCH_METHODS = {"lexical", "semantic", "graph", "graph_enabled", "hybrid", "all"}
FUSED_SEARCH_METHODS = {"graph", "graph_enabled", "hybrid", "all"}
GRAPH_SEARCH_METHODS = {"graph", "graph_enabled", "all"}
ANSWER_SYSTEM = (
    "You are a helpful assistant for a Wikipedia retrieval engine. Answer the question "
    "using ONLY the provided Wikipedia article excerpts, and quote from them where you "
    "can. If the documents do not contain the answer, say so rather than guessing. "
    "Cite supporting excerpt IDs in square brackets and name their Wikipedia article "
    "titles. Never invent a source or use prior assistant answers as evidence."
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
        answer = response.content if hasattr(response, "content") else str(response)
        return answer + sources_footer(context)


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
        self._articles = ArticleEvidence(docs)
        self._seed_k = CONFIG.getint("retrieval", "graph_seed_k")
        self._context_max_chars = CONFIG.getint("retrieval", "context_max_chars")
        self._debug = CONFIG.getboolean("retrieval", "debug", fallback=False)
        if self._seed_k <= 0 or self._context_max_chars < 1000:
            raise ValueError("graph_seed_k must be positive and context_max_chars must be at least 1000.")
        self._bm25 = BM25Retriever() if self._search_method == "lexical" or self._search_method in FUSED_SEARCH_METHODS else None
        self._db = None
        if self._search_method == "semantic" or self._search_method in FUSED_SEARCH_METHODS:
            if embedding_function is None:
                raise ValueError("embedding_function is required for hybrid, semantic, graph, and all search.")
            self._db = build_or_load_db(
                docs,
                chroma_dir or str(DATABASE_DIR / "Capstone_Chroma_DB"),
                embedding_function,
            )
        self._graph = None
        if self._search_method in GRAPH_SEARCH_METHODS:
            graph_file = DATABASE_DIR / "Capstone_Graph_DB" / "wikipedia_articles.graphml"
            if graph_file.is_file():
                import networkx as nx
                print(f"[graph] Loading knowledge graph from {graph_file.name}...")
                self._graph = nx.read_graphml(graph_file)
                print(f"[graph] Loaded {self._graph.number_of_nodes()} nodes and {self._graph.number_of_edges()} edges.")
            else:
                raise FileNotFoundError(f"Graph-enabled retrieval requires existing graph: {graph_file}")
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
        if k <= 0:
            raise ValueError("Retrieval result count must be positive.")
        if self._search_method == "semantic":
            raw = self._vector_topk(query, max(k, VECTOR_CANDIDATES))
            scores = normalize_scores([score for _, _, score in raw], invert=True)
            candidates = [(doc_id, text, score) for (doc_id, text, _), score in zip(raw, scores)]
        elif self._search_method == "lexical":
            raw = self._bm25_topk(query, max(k, BM25_CANDIDATES))
            scores = normalize_scores([score for _, _, score in raw])
            candidates = [(doc_id, text, score) for (doc_id, text, _), score in zip(raw, scores)]
        elif self._search_method in FUSED_SEARCH_METHODS:
            candidates = retrieve_hybrid(
                query,
                max(k, BM25_CANDIDATES + VECTOR_CANDIDATES),
                BM25_CANDIDATES,
                VECTOR_CANDIDATES,
                self._bm25_topk,
                self._vector_topk,
                lambda bm, vec, limit: weighted_hybrid_rank(
                    bm, vec, limit, WEIGHT_BM25, WEIGHT_VECTOR
                ),
            )
        else:
            raise RuntimeError(f"No retrieval dispatch exists for '{self._search_method}'.")
        results = self._articles.rerank(query, candidates, k)
        if self._debug:
            print(f"[retrieval] query={query!r}; selected={[doc_id for doc_id, _, _ in results]}")
        return results

    def _graph_expand(self, seed_results: list[tuple[str, str, float]]) -> str:
        """Expand seed chunks with adjacent sequence chunks and referenced article leads."""
        if self._graph is None:
            return limit_context(
                [format_evidence(doc_id, content) for doc_id, content, _ in seed_results],
                self._context_max_chars,
            )
        # Keep all ranked seeds ahead of neighbors so expansion cannot crowd them out.
        sections = [format_evidence(doc_id, content) for doc_id, content, _ in seed_results]
        seen_docs = {doc_id for doc_id, _, _ in seed_results}
        for doc_id, _, _ in seed_results:
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
                                sections.append(format_evidence(target_chunk, t_text))
                    elif edge_type == "links_to":
                        target_art = target.removeprefix("article:")
                        lead_id = f"{target_art}::intro"
                        if lead_id not in seen_docs and target_art in self._articles.article_ids:
                            seen_docs.add(lead_id)
                            sections.append(format_evidence(lead_id, self._articles.introduction(target_art)))
        return limit_context(sections, self._context_max_chars)

    def retrievedContext(self, query: str) -> str:
        if self._search_method in {"graph", "graph_enabled", "all"}:
            seeds = self.getTopK(query, self._seed_k)
            return self._graph_expand(seeds)
        results = self.getTopK(query, FUSED_TOP_K)
        return limit_context(
            [format_evidence(doc_id, content) for doc_id, content, _ in results],
            self._context_max_chars,
        )
