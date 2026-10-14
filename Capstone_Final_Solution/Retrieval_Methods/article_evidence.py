# Header reviewed: 2026-10-12 11:34:17 -07:00
"""Read-only article introductions, title matching, reranking, and source labels."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

from bs4 import BeautifulSoup

from solution_config import DATABASE_DIR


Result = tuple[str, str, float]
_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "how", "in", "is", "it", "of", "on", "the", "to", "was",
    "were", "what", "when", "where", "which", "who", "why", "with",
    "tell", "me", "about", "give", "overview", "known",
}
_OVERVIEW = re.compile(
    r"^\s*(?:who\s+(?:is|was|are|were)\b|what\s+(?:is|was|are|were)\b|"
    r"tell\s+me\s+about\b|give\s+(?:me\s+)?(?:an?\s+)?overview\b)", re.I
)


def words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower())) - _STOPWORDS


def article_id(doc_id: str) -> str:
    return doc_id.split("::", 1)[0]


def source_label(doc_id: str) -> str:
    name = article_id(doc_id)
    return f"Wikipedia: {name.replace('_', ' ')} (https://en.wikipedia.org/wiki/{quote(name)})"


def format_evidence(doc_id: str, text: str) -> str:
    return f"[{doc_id}]\nSource: {source_label(doc_id)}\n{text}"


def sources_footer(context: str) -> str:
    labels = list(dict.fromkeys(re.findall(r"^Source: (.+)$", context, re.M)))
    return "\n\nSources (retrieved evidence):\n" + "\n".join(f"- {label}" for label in labels) if labels else ""


def limit_context(sections: list[str], max_chars: int) -> str:
    included: list[str] = []
    remaining = max_chars
    separator = "\n\n---\n\n"
    for section in sections:
        if included:
            remaining -= len(separator)
        if remaining <= 0:
            break
        if len(section) > remaining:
            lines = section.split("\n", 2)
            header_length = len("\n".join(lines[:2])) + 1
            if remaining <= header_length:
                break
            included.append(section[:remaining])
            break
        included.append(section)
        remaining -= len(section)
    return separator.join(included)


def _boilerplate(text: str) -> bool:
    markers = ("Jump to content", "Main menu", "Toggle the table of contents",
               "Edit links Article Talk", "Tools move to sidebar")
    return any(marker in text for marker in markers)


class ArticleEvidence:
    def __init__(self, docs: list[dict], html_dir: Path | None = None):
        self.html_dir = html_dir or DATABASE_DIR / "Wikipedia"
        self.article_ids = sorted({article_id(doc["id"]) for doc in docs})
        self._by_words: dict[frozenset[str], list[str]] = {}
        self._by_token: dict[str, set[str]] = {}
        for name in self.article_ids:
            title_words = words(name.replace("_", " "))
            self._by_words.setdefault(frozenset(title_words), []).append(name)
            for token in title_words:
                self._by_token.setdefault(token, set()).add(name)

    def match_titles(self, query: str) -> list[str]:
        query_words = words(query)
        exact = self._by_words.get(frozenset(query_words), [])
        if exact:
            return exact
        candidates: set[str] = set()
        for token in query_words:
            candidates.update(self._by_token.get(token, set()))
        full = [
            name for name in sorted(candidates)
            if words(name.replace("_", " ")) <= query_words
        ]
        if full:
            return full
        # A surname-only query is resolved only when one article ends in that name.
        subject = re.sub(r"^(?:who|what)\s+(?:is|was|are|were)\s+", "", query.strip(), flags=re.I)
        subject = re.sub(r"^tell\s+me\s+about\s+", "", subject, flags=re.I)
        subject_words = [word for word in _WORD.findall(subject.lower()) if word not in _STOPWORDS]
        suffix_matches = [
            name for name in sorted(candidates)
            if subject_words and [
                word for word in _WORD.findall(name.replace("_", " ").lower()) if word not in _STOPWORDS
            ][-len(subject_words):] == subject_words
        ]
        return suffix_matches if len(suffix_matches) == 1 else []

    @lru_cache(maxsize=256)
    def introduction(self, name: str) -> str:
        if name not in self.article_ids:
            raise ValueError(f"Article is not in the loaded corpus: {name}")
        path = self.html_dir / f"{name}.html"
        if not path.is_file():
            raise FileNotFoundError(f"Article HTML required for clean introductory evidence: {path}")
        soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
        root = soup.select_one("#mw-content-text .mw-parser-output")
        if root is None:
            root = soup.select_one(".mw-parser-output")
        if root is None:
            raise ValueError(f"Wikipedia article content container not found: {path}")
        for tag in root.select(
            "script, style, nav, table, figure, .toc, .mw-editsection, .hatnote, "
            ".navbox, .metadata, .reflist, .mw-empty-elt, sup.reference"
        ):
            tag.decompose()
        paragraphs: list[str] = []
        for element in root.find_all(["p", "h2"]):
            if element.name == "h2" and paragraphs:
                break
            if element.name != "p":
                continue
            text = " ".join(element.get_text(" ", strip=True).split())
            if len(text) >= 80 and not _boilerplate(text):
                paragraphs.append(text)
            if sum(len(paragraph) for paragraph in paragraphs) >= 3500:
                break
        if not paragraphs:
            raise ValueError(f"No substantive introductory paragraphs found: {path}")
        return "\n\n".join(paragraphs)[:4500]

    def rerank(self, query: str, candidates: list[Result], k: int) -> list[Result]:
        matches = set(self.match_titles(query))
        query_words = words(query)
        overview = bool(_OVERVIEW.search(query))
        by_id = {doc_id: (doc_id, text, score) for doc_id, text, score in candidates}
        for name in sorted(matches):
            doc_id = f"{name}::intro"
            by_id[doc_id] = (doc_id, self.introduction(name), 0.0)
        ranked: list[Result] = []
        for doc_id, text, score in by_id.values():
            title_match = article_id(doc_id) in matches
            passage_overlap = len(query_words & words(text)) / max(len(query_words), 1)
            bonus = (0.6 if title_match else 0.0) + 0.2 * passage_overlap
            if doc_id.endswith("::intro"):
                bonus += 1.5 if overview else 0.3
            if _boilerplate(text):
                bonus -= 1.0
            ranked.append((doc_id, text, float(score) + bonus))
        ranked.sort(key=lambda row: (-row[2], row[0]))
        return ranked[:k]
