#################################################################################
# 
# Massachusetts Institute of Technology - xPRO
# RAG and Context Engineering: Designing and Building Production-Grade AI Systems
# 
# Student: Miguel Herrera
# Section: B
# Date and Time: 2026-09-12 11:54:44 -07:00
#
# Description: This script extracts visible Wikipedia HTML text, splits articles
#              into indexed chunks, writes one JSONL file per article, and skips
#              existing output unless --rebuild is explicitly supplied.
#
#################################################################################

"""Extract and chunk the Wikipedia HTML corpus into one JSONL per article."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
from utility_logging import log_run


SCRIPT_DIR = Path(__file__).resolve().parent
WIKI_DIR = SCRIPT_DIR.parent / "Capstone_Database" / "Wikipedia"
OUTPUT_DIR = SCRIPT_DIR.parent / "Capstone_Database" / "Wikipedia_JSONL"


def html_to_text(html_path: Path) -> str:
    """Return visible, whitespace-normalized text from one HTML file."""
    with html_path.open("r", encoding="utf-8", errors="replace") as html_file:
        soup = BeautifulSoup(html_file.read(), "html.parser")

    for tag in soup(["script", "style"]):
        tag.decompose()
    return " ".join(soup.get_text(separator=" ", strip=True).split())


def chunk_article(
    html_path: Path,
    output_dir: Path,
    splitter: RecursiveCharacterTextSplitter,
) -> int:
    """Write one JSONL file for an HTML article and return its chunk count."""
    text = html_to_text(html_path)
    chunks = splitter.create_documents(
        [text],
        metadatas=[{"source": html_path.name, "document_id": html_path.stem}],
    )
    output_path = output_dir / f"{html_path.stem}.jsonl"

    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for chunk_index, document in enumerate(chunks):
            record = {
                "id": f"{html_path.stem}::chunk-{chunk_index:04d}",
                "source": html_path.name,
                "document_id": html_path.stem,
                "chunk_index": chunk_index,
                "text": document.page_content,
                "metadata": document.metadata,
            }
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return len(chunks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk Wikipedia HTML files into one JSONL file per article."
    )
    parser.add_argument("--input-dir", type=Path, default=WIKI_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Regenerate JSONL files even when the output directory already contains them.",
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    args = parse_args()

    existing_jsonl = sorted(args.output_dir.glob("*.jsonl")) if args.output_dir.is_dir() else []
    if existing_jsonl and not args.rebuild:
        print(f"[skip] Output directory already contains {len(existing_jsonl)} JSONL file(s).")
        print("[skip] No HTML files were processed and existing JSONL files were preserved.")
        return
    if existing_jsonl and args.rebuild:
        print(
            f"[rebuild] Rebuild requested; regenerating {len(existing_jsonl)} existing JSONL file(s)."
        )

    if not args.input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {args.input_dir}")
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be greater than zero")
    if args.chunk_overlap < 0 or args.chunk_overlap >= args.chunk_size:
        raise SystemExit("--chunk-overlap must be at least zero and less than --chunk-size")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        add_start_index=True,
    )
    html_files = sorted(args.input_dir.glob("*.html"))
    if not html_files:
        raise SystemExit(f"No HTML files found in: {args.input_dir}")

    total_chunks = 0
    with tqdm(total=len(html_files), desc="Chunking HTML articles", unit="article") as article_bar:
        for article_number, html_path in enumerate(html_files, start=1):
            chunk_count = chunk_article(html_path, args.output_dir, splitter)
            total_chunks += chunk_count
            article_bar.update(1)

    print(f"Wrote {len(html_files)} JSONL files ({total_chunks} chunks) to {args.output_dir}")


if __name__ == "__main__":
    with log_run(__file__):
        main()