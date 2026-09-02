r"""Capstone Checkpoint 2.1 — Retrieval Strategy Design and Baseline Implementation (starter).
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code / PyCharm / Jupytext.
"""

# %% [markdown]
# # Capstone Checkpoint 2.1 — Retrieval Strategy Design and Baseline Implementation
# **MO-LLM Module 2 / Required Capstone Checkpoint (120 minutes)**
#
# ## What this checkpoint is
#
# In Checkpoint 1.1, you showed that a plain LLM can't reliably answer questions about
# your corpus. Now, you will **add retrieval**: Design a retrieval strategy for your scenario
# and build a **baseline retrieval system** that finds the most relevant documents for
# a query, so the model can ground its answers in them.
#
# This mirrors the Module 2 labs — keyword (BM25), vector (semantic), and hybrid
# retrieval — applied to your own capstone corpus. The graded deliverable is the completed 
# Capstone Checkpoint 2.1 worksheet, which includes your written responses and evidence of your 
# retrieval system implementation and testing. This script provides a small working example of 
# baseline retrieval. Use it to understand the retrieval workflow, then adapt the code to implement 
# and test a baseline retriever using your selected capstone dataset.
#
# **Learning outcomes (Module 2):**
# 1. Design a retrieval strategy appropriate for a given dataset and query type.
# 2. Implement and test a baseline retrieval system using structured and/or semantic
#    approaches.

# %% [markdown]
# ## Step 1 — Keep your capstone scenario
#
# Use the **same scenario** you chose in Checkpoint 1.1.
#
# | Scenario | Corpus | Retrieval considerations |
# |---|---|---|
# | **Research Paper Navigator** | ~150 research-paper PDFs (`Labs/CapstoneDatasets/ResearchPapers/`) | long documents; you'll likely chunk them; questions often name a specific paper or compare papers. |
# | **Wikipedia Retrieval Engine** | ~2,400 Wikipedia HTML articles (`Labs/CapstoneDatasets/Wikipedia/`) | many short-to-medium articles; questions name a figure/place or span several articles. |
#
# A good baseline is keyword (BM25), semantic (embeddings + vector search), or a
# hybrid of both — exactly what you built in Labs 1.2–2.2.

# %% [markdown]
# ## Setup (~5 min)
#
# 1. **Python 3.11 or 3.12**
# 2. `pip install langchain-openai langchain-core python-dotenv`
# 3. Use the OpenRouter API key provided for this program. This checkpoint uses
#  the `openai/gpt-5.4-mini` model, with usage covered by the course credits. (this uses the paid gpt-5.4-mini chat model — covered by your course credits — and a keyword retriever, no embeddings).
# 4. Create a `.env` file next to this script: `OPENROUTER_API_KEY=sk-or-v1-...`
#
# This runs on a tiny built-in sample corpus, so you do not need to prepare your own
# dataset. It still requires an OpenRouter API key to run the LLM (it is not offline or
# free of API calls). Your real baseline (over your full corpus) is what you describe in
# the writeup.

# %%
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import os
import re
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from rank_bm25 import BM25Okapi

# %%
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"  # latest small OpenAI model, fast; covered by course credits
EMBEDDING_MODEL = "openai/text-embedding-3-small"
TEMPERATURE = 0.2
TOP_K = 4                  # Wikipedia pages sent to the LLM as context.
CANDIDATE_POOL = 10        # Candidates pulled from EACH retriever before fusion
WEIGHT_BM25 = 0.5
WEIGHT_VECTOR = 0.5
CHECKPOINT_DIR = Path(__file__).resolve().parent
FINAL_CAPSTONE_DIR = CHECKPOINT_DIR.parent
CHROMA_DIR = str(FINAL_CAPSTONE_DIR / "Capstone_Database" / "Capstone_Chroma_DB")
LOG_PATH = CHECKPOINT_DIR  / "checkpoint_2_1_retrieval.log"

# === SET THIS to the scenario you chose in Checkpoint 1.1 ===
SCENARIO = "wikipedia"   # "research_papers" or "wikipedia"

ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the question using ONLY the provided "
    "documents, and quote from them where you can. If the documents do not contain "
    "the answer, say so rather than guessing."
)

# %%
def check_api_key() -> str:
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Use the OpenRouter API key "
            "provided for this course, put it in a .env file next to this "
            "script, and rerun."
        )
    return key

def make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        api_key=check_api_key(),
        base_url=OPENROUTER_BASE_URL,
    )

def log(label: str, text: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {label}\n{text}\n{'-' * 72}\n")


# %% [markdown]
# ## Load Wikipedia corpus
#
# Load HTML articles from the Capstone_Database/Wikipedia folder.
# Each article is parsed to extract text content.


# %%
def load_wikipedia_docs(wiki_dir: str = None) -> list[dict]:
    """Load Wikipedia HTML articles from the Capstone_Database/Wikipedia folder.
    
    Returns a list of dicts with {"id": filename, "text": extracted_text}.
    """
    if wiki_dir is None:
        wiki_dir = str(FINAL_CAPSTONE_DIR / "Capstone_Database" / "Wikipedia")
    
    if not os.path.isdir(wiki_dir):
        print(f"Warning: Wikipedia directory not found at {wiki_dir}")
        return []
    
    docs = []
    html_files = sorted([f for f in os.listdir(wiki_dir) if f.endswith(".html")])
    total_files = len(html_files)
    print(f"Loading {total_files} Wikipedia articles from {wiki_dir}/...")
    
    for idx, filename in enumerate(html_files, 1):
        filepath = os.path.join(wiki_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
                html_content = fh.read()
            
            # Parse HTML and extract text
            soup = BeautifulSoup(html_content, "html.parser")
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text(separator=" ", strip=True)
            
            # Clean up whitespace
            text = " ".join(text.split())
            
            if text:  # Only add if there's content
                docs.append({
                    "id": filename.replace(".html", ""),
                    "text": text
                })
            
            # Progress update every 100 files or at the end
            if idx % 100 == 0 or idx == total_files:
                print(f"  Progress: {idx}/{total_files} articles processed ({len(docs)} valid)")
        except Exception as e:
            print(f"  Error reading {filename}: {e}")
            continue
    
    print(f"✓ Loaded {len(docs)} articles successfully.")
    return docs


# ─── Vector side (embeddings) ──────────────────────────────────────
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=check_api_key(),
        base_url=OPENROUTER_BASE_URL,
    )


def load_docs_from_chroma(chroma_dir: str = CHROMA_DIR) -> list[dict]:
    """Load document text from an existing Chroma DB instead of re-scanning the source corpus."""
    if not os.path.isdir(chroma_dir) or not os.listdir(chroma_dir):
        return []

    db = Chroma(persist_directory=chroma_dir, embedding_function=get_embeddings())
    try:
        results = db.get(include=["documents", "metadatas"])
    except Exception as exc:  # pragma: no cover - defensive fallback
        print(f"Warning: Could not read persisted Chroma docs: {exc}")
        return []

    docs = []
    for doc_id, text, meta in zip(
        results.get("ids", []),
        results.get("documents", []),
        results.get("metadatas", []),
    ):
        if not text:
            continue
        doc_meta = meta or {}
        docs.append({
            "id": str(doc_meta.get("id", doc_id)),
            "text": text,
        })

    print(f"✓ Loaded {len(docs)} documents from existing Chroma DB at {chroma_dir}/")
    return docs


# Load Wikipedia documents: skip the HTML scan if the persisted vector DB is already available.
if os.path.isdir(CHROMA_DIR) and os.listdir(CHROMA_DIR):
    print(f"✓ Chroma DB already exists at {CHROMA_DIR}/ — loading persisted documents instead of scanning Wikipedia HTML files.")
    DOCS = load_docs_from_chroma(CHROMA_DIR)
else:
    print(f"⚠ Chroma DB not found at {CHROMA_DIR}/ — building a fresh vector database from Wikipedia HTML files.")
    DOCS = load_wikipedia_docs()
DOC_BY_ID = {d["id"]: d for d in DOCS}


# %% [markdown]
# ## Step 2 — The hybrid retriever (BM25 + Vector Search)
#
# Hybrid retrieval combines keyword search (BM25) with semantic search (embeddings).
# BM25 scores are "higher = better", while vector distances are "lower = better",
# so scores are normalized and the vector side is inverted before combining with a
# weighted sum. This gives you the best of both approaches.

# %%
# ─── Keyword side (BM25) ─────────────────────────────────────
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
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


def build_or_load_db(docs: list[dict], chroma_dir: str = CHROMA_DIR) -> Chroma:
    """Build or load a persisted Chroma vector database from documents."""
    if os.path.isdir(chroma_dir) and os.listdir(chroma_dir):
        print(f"✓ Loading existing vector DB from {chroma_dir}/")
        return Chroma(persist_directory=chroma_dir, embedding_function=get_embeddings())
    
    print(f"\n⏳ Building vector DB (first run — embedding {len(docs)} documents)...")
    print(f"   This may take 10 minutes to several hours depending on document count.")
    print(f"   API endpoint: {OPENROUTER_BASE_URL}")
    print(f"   Embedding model: {EMBEDDING_MODEL}")
    print(f"   Persistence directory: {chroma_dir}/\n")
    
    os.makedirs(chroma_dir, exist_ok=True)
    doc_objs = [
        Document(page_content=d["text"], metadata={"id": d["id"]})
        for d in docs
    ]

    batch_size = 50
    total_docs = len(doc_objs)
    total_batches = (total_docs + batch_size - 1) // batch_size
    print(f"   Starting embedding process in batches of {batch_size} documents...")

    db = Chroma(persist_directory=chroma_dir, embedding_function=get_embeddings())
    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total_docs)
        batch = doc_objs[start:end]
        batch_ids = [doc.metadata["id"] for doc in batch]
        print(
            f"   Embedding batch {batch_idx + 1}/{total_batches} "
            f"({start + 1}-{end}/{total_docs})..."
        )
        db.add_documents(batch, ids=batch_ids)
        print(f"     Completed batch {batch_idx + 1}/{total_batches}.")

    print(f"\n✓ Indexed {len(docs)} documents into {chroma_dir}/")
    return db


def _normalize(scores: list[float], invert: bool = False) -> list[float]:
    """Min-max scale a list of scores to [0, 1]. If invert is True, flip the scores
    so a LOW raw value (e.g., a small vector distance = very similar) becomes a HIGH
    normalized score."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [0.5] * len(scores)  # all equal → neutral
    norm = [(s - lo) / (hi - lo) for s in scores]
    return [1.0 - n for n in norm] if invert else norm


# ─── Base and Hybrid Retriever Classes ─────────────────────────────────────
class BaseRetriever(ABC):
    def __init__(self, llm_model: str = LLM_MODEL):
        self._llm = ChatOpenAI(
            model=llm_model,
            api_key=check_api_key(),
            base_url=OPENROUTER_BASE_URL,
            temperature=TEMPERATURE,
        )

    @abstractmethod
    def retrievedContext(self, query: str) -> str:
        ...

    def _build_user_message(self, query: str, context: str) -> str:
        return f"Context (documents):\n{context}\n\nQuestion: {query}"

    def query(self, question: str) -> str:
        context = self.retrievedContext(question)
        user_message = HumanMessage(content=self._build_user_message(question, context))
        messages = [SystemMessage(content=ANSWER_SYSTEM), user_message]

        try:
            response = self._llm.invoke(messages)
            answer = response.content if hasattr(response, "content") else str(response)
        except Exception:
            raise
        return answer


class HybridRetriever(BaseRetriever):
    """Hybrid retriever fusing BM25 (keyword) and vector (semantic) search."""

    def __init__(self, docs: list[dict], **kwargs):
        super().__init__(**kwargs)
        # Store docs as both dicts and as text for indexing
        self._docs = docs
        self._doc_ids = [d["id"] for d in docs]
        self._doc_texts = [d["text"] for d in docs]
        
        # Build BM25 index
        self._bm25 = BM25Okapi([tokenize(text) for text in self._doc_texts])
        
        # Build or load persisted vector DB from docs
        self._db = build_or_load_db(docs, CHROMA_DIR)
        print(f"Hybrid retriever ready over {len(docs)} documents.")

    def _bm25_topk(self, query: str, k: int) -> list[tuple[str, str, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self._doc_ids[i], self._doc_texts[i], scores[i]) for i in top]

    def _vector_topk(self, query: str, k: int) -> list[tuple[str, str, float]]:
        results = self._db.similarity_search_with_score(query, k=k)
        return [
            (d.metadata.get("id", "unknown"), d.page_content, s)
            for d, s in results
        ]

    def getTopK(self, query: str, k: int) -> list[tuple[str, str, float]]:
        """Fuse the two retrievers: Pull a candidate pool from each, normalize, and
        combine with a weighted sum. BM25 is higher-is-better and vector distance is
        lower-is-better, so the vector side is normalized with invert=True."""
        bm = self._bm25_topk(query, CANDIDATE_POOL)
        vec = self._vector_topk(query, CANDIDATE_POOL)

        content_by_id: dict[str, str] = {}
        bm_norm: dict[str, float] = {}
        vec_norm: dict[str, float] = {}

        if bm:
            for (doc_id, content, _), val in zip(bm, _normalize([s for _, _, s in bm])):
                content_by_id[doc_id] = content
                bm_norm[doc_id] = val
        if vec:
            for (doc_id, content, _), val in zip(
                vec, _normalize([d for _, _, d in vec], invert=True)
            ):
                content_by_id[doc_id] = content
                vec_norm[doc_id] = val

        fused = [
            (
                doc_id,
                content,
                WEIGHT_BM25 * bm_norm.get(doc_id, 0.0)
                + WEIGHT_VECTOR * vec_norm.get(doc_id, 0.0),
            )
            for doc_id, content in content_by_id.items()
        ]
        fused.sort(key=lambda t: t[2], reverse=True)
        return fused[:k]

    def retrievedContext(self, query: str) -> str:
        results = self.getTopK(query, TOP_K)
        return "\n\n---\n\n".join(
            f"[{doc_id}]\n{content}" for doc_id, content, _ in results
        )


# %% [markdown]
# ## Step 3 — Your representative queries (TODO)
#
# Submission item #2 asks for **3-5 representative queries** for your scenario and the
# results your system retrieves for each. Write those queries here. Some good ones include questions that:
#
# - Are answerable from **one** document (tests precision),
# - Need **several** documents (tests recall / aggregation),
# - Have wording that **differs** from the document's wording (i.e., tests whether
#   keyword vs. semantic retrieval matters for your corpus)
#
# Return a list of 3-5 query strings.

# %%
def my_representative_queries() -> list[str]:
    """Return 3-5 representative queries for YOUR chosen scenario.

    TODO — your turn. See the guidance above. Each item is a query string. Pick
    queries that a real user of your system would ask and that require different
    retrieval behaviors (single-doc, multi-doc, paraphrased).

    Delete the raise NotImplementedError line once your code works.
    """
    return [
        "Who was Albert Einstein?", #Lexical Question
        "How is Ace Bailey related to Toronto Maple Leafs?", #Semantic Question
        "According to the Wikipedia pages, how many weapons are featured in the video game 'GoldenEye 007'?", #Paraphrased Question
        "Provide a short summary of the accomplishments of Abraham Lincoln?" #Summary Question
    ]
    #raise NotImplementedError("my_representative_queries() — see the TODO above.")


# %% [markdown]
# ## Step 4 — Run the hybrid retriever and capture the evidence
#
# This runs each query through the hybrid retriever (BM25 + vector search) and the LLM,
# printing the retrieved document ids/scores and the grounded answer, and logging
# everything to `checkpoint_2_1_retrieval.log`. The retrieved documents from the output
# are the evidence for submission item #2.

# %%
def run() -> None:
    retriever = HybridRetriever(DOCS)
    queries = my_representative_queries()
    print(f"Checkpoint 2.1 — hybrid retrieval  |  scenario: {SCENARIO}\n")
    for i, query in enumerate(queries, 1):
        hits = retriever.getTopK(query, TOP_K)
        print("=" * 72)
        print(f"QUERY {i}: {query}")
        if not hits:
            print("  (nothing matched — note this in your writeup)")
            log(
                f"QUERY {i}: {query}",
                "retrieved=[]\nanswer=(no documents retrieved)",
            )
            continue
        try:
            ans = retriever.query(query)
        except Exception as exc:
            print(f"  LLM error: {exc}")
            log(
                f"QUERY {i}: {query}",
                f"retrieved={[(doc_id, score) for doc_id, _, score in hits]}\nerror={exc}",
            )
            continue
        print(f"  answer: {ans}\n")
        log(
            f"QUERY {i}: {query}",
            f"retrieved={[(doc_id, score) for doc_id, _, score in hits]}\nanswer={ans}",
        )
    print("=" * 72)
    print(
        "Done. Use the retrieved document results above as evidence in your writeup, and "
        "describe your REAL baseline (over your full corpus) in the submission."
    )


run()

# %% [markdown]
# ## Step 5 — Your written submission (the graded deliverable)
#
# Use your completed retrieval implementation and test results to complete the Capstone Checkpoint 2.1
# worksheet. In the worksheet, you will document your retrieval approach, provide evidence that your 
# system is functioning, include 3–5 representative queries and retrieved results, and reflect on where
# your approach performs well and where it struggles.  
 
# Save your completed Python file in the appropriate checkpoint folder in your GitHub repository. 
# Upload the completed worksheet only to the learning platform as your graded submission.
