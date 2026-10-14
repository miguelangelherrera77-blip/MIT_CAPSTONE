# Header reviewed: 2026-10-12 11:34:17 -07:00
"""Local configuration and external, existing database paths."""
from __future__ import annotations

import configparser
from pathlib import Path


SOLUTION_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SOLUTION_DIR / "retrieval.conf"
CONFIG = configparser.ConfigParser()
if not CONFIG_PATH.is_file():
    raise FileNotFoundError(f"Solution configuration not found: {CONFIG_PATH}")
CONFIG.read(CONFIG_PATH, encoding="utf-8")
DATABASE_DIR = (SOLUTION_DIR / CONFIG.get("paths", "database_dir")).resolve()


def validate_database() -> None:
    required = [
        DATABASE_DIR / "Wikipedia",
        DATABASE_DIR / "Wikipedia_JSONL",
        DATABASE_DIR / "Capstone_BM25_Lexical_Indexes" / "documents.json",
        DATABASE_DIR / "Capstone_Chroma_DB" / "chroma.sqlite3",
        DATABASE_DIR / "Capstone_Graph_DB" / "wikipedia_articles.graphml",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Existing database artifacts are required; this solution never builds or "
            "copies them. Missing:\n" + "\n".join(missing)
        )
