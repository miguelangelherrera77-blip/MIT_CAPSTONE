#################################################################################
# 
# Massachusetts Institute of Technology - xPRO
# RAG and Context Engineering: Designing and Building Production-Grade AI Systems
# 
# Student: Miguel Herrera
# Section: B
# Date and Time: 2026-09-12 16:12:28 -07:00
#
# Description: This script scans Wikipedia HTML files and JSONL chunks, creates
#              Article and Chunk nodes with contains, next_chunk, previous_chunk,
#              and links_to (with link_label) edges, and persists the NetworkX
#              graph as GraphML without using an LLM or API calls.
#
#################################################################################

r"""Build a NetworkX article graph from the Wikipedia HTML corpus.

The graph contains one Article node per HTML file and Chunk nodes from the
Wikipedia_JSONL records. Each Article points to its chunks with a ``contains``
edge, adjacent chunks within an article are linked with ``next_chunk`` and
``previous_chunk`` edges, and chunks referencing other articles in the corpus
are connected via ``links_to`` directed edges annotated with their anchor text
``link_label``. Later graph-enrichment steps can add entities, categories,
and typed relationships without rebuilding the article identity layer.

Run from the repository root:

    python Final_Capstone_Project/Utility_Scripts/Build_Wikipedia_Article_GraphDB.py

Use ``--rebuild`` to replace an existing GraphML graph. Without ``--rebuild``,
an existing graph is preserved and the build is skipped.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import unquote

import networkx as nx
from tqdm import tqdm

from utility_logging import log_run


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR.parent / "Capstone_Database" / "Wikipedia"
DEFAULT_JSONL_DIR = SCRIPT_DIR.parent / "Capstone_Database" / "Wikipedia_JSONL"
DEFAULT_GRAPH_DIR = SCRIPT_DIR.parent / "Capstone_Database" / "Capstone_Graph_DB"
DEFAULT_GRAPH_PATH = DEFAULT_GRAPH_DIR / "wikipedia_articles.graphml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one NetworkX Article node per Wikipedia HTML file."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--jsonl-dir", type=Path, default=DEFAULT_JSONL_DIR)
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--graph-file", type=Path, default=DEFAULT_GRAPH_PATH)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Replace an existing GraphML graph instead of skipping it.",
    )
    return parser.parse_args()


def count_jsonl_records(jsonl_paths: list[Path]) -> int:
    """Count non-empty JSONL records so the Chunk progress bar has a total."""
    total = 0
    for jsonl_path in jsonl_paths:
        with jsonl_path.open("r", encoding="utf-8") as jsonl_file:
            total += sum(1 for line in jsonl_file if line.strip())
    return total


LINK_PATTERN = re.compile(
    r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
TAG_STRIP_PATTERN = re.compile(r"<[^>]+>")
STRIP_BLOCKS_PATTERN = re.compile(
    r"<(?:script|style|table|nav)[^>]*>.*?</(?:script|style|table|nav)>",
    re.IGNORECASE | re.DOTALL,
)


def extract_content_links(
    html_path: Path, valid_article_ids: set[str]
) -> list[tuple[str, str]]:
    """Extract (anchor_text, target_article_id) pairs linking to articles in the corpus."""
    with html_path.open("r", encoding="utf-8", errors="replace") as html_file:
        html_text = html_file.read()

    body_match = re.search(
        r'<div[^>]+id=["\']bodyContent["\'][^>]*>(.*)',
        html_text,
        re.IGNORECASE | re.DOTALL,
    )
    if body_match:
        html_text = body_match.group(1)

    html_text = STRIP_BLOCKS_PATTERN.sub("", html_text)
    seen: set[tuple[str, str]] = set()
    links: list[tuple[str, str]] = []
    source_id = html_path.stem

    for href, inner_html in LINK_PATTERN.findall(html_text):
        href = href.strip()
        if not href or href.startswith("#"):
            continue
        anchor_text = TAG_STRIP_PATTERN.sub("", inner_html).strip()
        if not anchor_text:
            continue
        anchor_text = " ".join(anchor_text.split())
        target_name = href.split("/")[-1].split("#")[0].replace(".html", "")
        target_id = unquote(target_name)
        if target_id in valid_article_ids and target_id != source_id:
            key = (anchor_text, target_id)
            if key not in seen:
                seen.add(key)
                links.append(key)
    return links


def build_article_graph(input_dir: Path, jsonl_dir: Path) -> nx.DiGraph:
    """Create Article nodes, Chunk nodes, and Article-to-Chunk edges."""
    print(f"[graph] Scanning HTML input directory: {input_dir.resolve()}")
    html_files = sorted(input_dir.glob("*.html"))
    if not html_files:
        raise FileNotFoundError(f"No Wikipedia HTML files found in: {input_dir}")
    print(f"[graph] Found {len(html_files)} Wikipedia HTML file(s).")
    if not jsonl_dir.is_dir():
        raise FileNotFoundError(f"Wikipedia JSONL directory not found: {jsonl_dir}")
    jsonl_paths = [jsonl_dir / f"{html_path.stem}.jsonl" for html_path in html_files]
    existing_jsonl_paths = [path for path in jsonl_paths if path.is_file()]
    chunk_total = count_jsonl_records(existing_jsonl_paths)
    print(
        f"[input] Found {len(existing_jsonl_paths)} matching JSONL file(s) and "
        f"{chunk_total} chunk record(s)."
    )

    print("[graph] Initializing NetworkX directed graph.")
    graph = nx.DiGraph(name="Wikipedia Article Graph")
    valid_article_ids = {html_path.stem for html_path in html_files}

    # Pre-populate all Article nodes
    for html_path in html_files:
        article_id = html_path.stem
        node_id = f"article:{article_id}"
        graph.add_node(
            node_id,
            node_type="Article",
            article_id=article_id,
            title=article_id.replace("_", " "),
            source=html_path.name,
            source_suffix=html_path.suffix,
            file_size_bytes=html_path.stat().st_size,
        )

    chunk_count = 0
    contains_edge_count = 0
    seq_edge_count = 0
    link_edge_count = 0

    with tqdm(total=len(html_files), desc="Processing Articles and Links", unit="article") as article_bar, tqdm(
        total=chunk_total,
        desc="Creating Chunk nodes",
        unit="chunk",
    ) as chunk_bar:
      for article_number, html_path in enumerate(html_files, 1):
          article_id = html_path.stem
          node_id = f"article:{article_id}"
          article_links = extract_content_links(html_path, valid_article_ids)
          jsonl_path = jsonl_dir / f"{article_id}.jsonl"
          if not jsonl_path.is_file():
              print(f"\n[warn] No JSONL file found for Article node: {article_id}")
              article_bar.update(1)
              continue

          previous_chunk_node: str | None = None
          with jsonl_path.open("r", encoding="utf-8") as jsonl_file:
              for line_number, line in enumerate(jsonl_file, 1):
                  if not line.strip():
                      continue
                  try:
                      record = json.loads(line)
                  except json.JSONDecodeError as exc:
                      raise ValueError(f"Invalid JSONL at {jsonl_path}:{line_number}: {exc}") from exc
                  chunk_id = str(record.get("id", "")).strip()
                  text = str(record.get("text", "")).strip()
                  if not chunk_id or not text:
                      raise ValueError(f"Missing id or text at {jsonl_path}:{line_number}")
                  chunk_node = f"chunk:{chunk_id}"
                  metadata = record.get("metadata") or {}
                  graph.add_node(
                      chunk_node,
                      node_type="Chunk",
                      chunk_id=chunk_id,
                      document_id=str(record.get("document_id", article_id)),
                      source=str(record.get("source", html_path.name)),
                      chunk_index=int(record.get("chunk_index", 0)),
                      text=text,
                      start_index=int(metadata.get("start_index", 0)),
                  )
                  graph.add_edge(node_id, chunk_node, edge_type="contains")
                  contains_edge_count += 1
                  if previous_chunk_node is not None:
                      graph.add_edge(previous_chunk_node, chunk_node, edge_type="next_chunk")
                      graph.add_edge(chunk_node, previous_chunk_node, edge_type="previous_chunk")
                      seq_edge_count += 2
                  previous_chunk_node = chunk_node

                  for anchor_text, target_id in article_links:
                      if anchor_text in text:
                          target_node = f"article:{target_id}"
                          if graph.has_edge(chunk_node, target_node):
                              existing_label = graph[chunk_node][target_node].get("link_label", "")
                              if anchor_text not in existing_label.split("; "):
                                  combined = f"{existing_label}; {anchor_text}" if existing_label else anchor_text
                                  graph[chunk_node][target_node]["link_label"] = combined
                          else:
                              graph.add_edge(
                                  chunk_node,
                                  target_node,
                                  edge_type="links_to",
                                  link_label=anchor_text,
                              )
                              link_edge_count += 1

                  chunk_count += 1
                  chunk_bar.update(1)

          article_bar.update(1)

    print(
        f"[graph] Construction complete: {graph.number_of_nodes()} node(s), "
        f"{graph.number_of_edges()} edge(s) ({contains_edge_count} contains, "
        f"{seq_edge_count} sequence, {link_edge_count} links_to)."
    )
    return graph


def main() -> None:
    args = parse_args()
    print("=== Wikipedia Article graph builder ===")
    print(f"[args] input_dir={args.input_dir.resolve()}")
    print(f"[args] jsonl_dir={args.jsonl_dir.resolve()}")
    print(f"[args] graph_dir={args.graph_dir.resolve()}")
    print(f"[args] graph_file={args.graph_file.resolve()}")
    print(f"[args] rebuild={args.rebuild}")
    print("[args] Arguments parsed successfully.")

    print("[input] Validating Wikipedia HTML input directory.")
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Wikipedia HTML input directory not found: {args.input_dir}")
    print("[input] Wikipedia HTML input directory is available.")

    graph_file_exists = args.graph_file.is_file() and args.graph_file.stat().st_size > 0
    if graph_file_exists and not args.rebuild:
        print(f"[skip] Existing GraphML output found: {args.graph_file.resolve()}")
        print("[skip] Graph preserved; no HTML files were processed.")
        return
    if args.graph_file.exists() and not graph_file_exists:
        print(f"[cleanup] Existing graph path is empty or invalid: {args.graph_file.resolve()}")
        args.graph_file.unlink()
    if args.graph_file.exists() and args.rebuild:
        print("[override] Existing GraphML output will be replaced.")
    else:
        print("[output] No existing GraphML output found; graph construction will proceed.")

    print(f"[input] Reading Wikipedia HTML files from: {args.input_dir}")
    graph = build_article_graph(args.input_dir, args.jsonl_dir)
    print(f"[write] Creating graph directory: {args.graph_dir.resolve()}")
    args.graph_dir.mkdir(parents=True, exist_ok=True)
    print(f"[write] Writing GraphML file: {args.graph_file.resolve()}")
    nx.write_graphml(graph, args.graph_file)
    print(f"[write] GraphML graph written: {args.graph_file}")
    print(f"[complete] Created {graph.number_of_nodes()} Article node(s) and {graph.number_of_edges()} edge(s).")


if __name__ == "__main__":
    with log_run(__file__):
        main()
