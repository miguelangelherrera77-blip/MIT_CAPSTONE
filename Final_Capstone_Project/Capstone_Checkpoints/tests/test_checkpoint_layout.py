# Header reviewed: 2026-10-12 11:34:17 -07:00
"""Offline regressions for checkpoint relocation; no model calls or DB builders."""
from __future__ import annotations

import ast
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

CHECKPOINTS = Path(__file__).resolve().parents[1]
PROJECT = CHECKPOINTS.parent
REPO = PROJECT.parent
DATABASE = PROJECT / "Capstone_Database"
sys.path.insert(0, str(REPO))

PATH_NAMES = {
    "SCRIPT_DIR", "CHECKPOINT_DIR", "FINAL_CAPSTONE_DIR", "CHECKPOINTS_DIR",
    "PROJECT_DIR", "REPO_ROOT", "WORKSPACE_DIR", "WORKSPACE_ROOT", "CONFIG_PATH",
    "DEDICATED_CONFIG_PATH", "CHROMA_DIR", "WIKI_DIR", "LOG_PATH", "LOG_DIR",
    "TEST_VARIABLES_DIR", "ORIGINALS_PATH", "MANUAL_ORIGINALS_PATH", "VARIANTS_PATH",
    "MANUAL_PARAPHRASED_EVALUATION_PATH", "RAGAS_ROOT", "TEST_RESULTS_LOG",
    "WIKIPEDIA_JSONL_DIR", "BM25_INDEX_DIR", "BM25_BUILDER", "CHUNKER_SCRIPT",
    "BUILD_UTILITY", "DEFAULT_INPUT_DIR", "DEFAULT_JSONL_DIR", "DEFAULT_GRAPH_DIR",
    "DEFAULT_INDEX_DIR", "DEFAULT_DB_DIR", "OUTPUT_DIR", "SCHEMA_PATH",
    "VENV_DIR", "REQUIREMENTS_PATH",
}


def path_constants(path: Path) -> dict:
    # Early notebooks execute probes or open Chroma at import time, so evaluate
    # only their path declarations rather than invoking those runtime workflows.
    namespace = {"__file__": str(path), "Path": Path, "str": str}
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if names and all(name in PATH_NAMES for name in names):
                value = eval(compile(ast.Expression(node.value), str(path), "eval"), namespace)
                for name in names:
                    namespace[name] = value
    return namespace


class CheckpointLayoutTests(unittest.TestCase):
    def test_all_checkpoints_moved_without_nested_database(self):
        for number in range(1, 9):
            name = f"Capstone_Checkpoint_{number}.1"
            self.assertTrue((CHECKPOINTS / name).is_dir())
            self.assertFalse((PROJECT / name).exists())
        self.assertTrue(DATABASE.is_dir())
        self.assertEqual(list(CHECKPOINTS.rglob("Capstone_Database")), [])

    def test_all_python_parses_and_uses_relocated_package(self):
        for path in CHECKPOINTS.rglob("*.py"):
            with self.subTest(file=str(path.relative_to(CHECKPOINTS))):
                tree = ast.parse(path.read_text(encoding="utf-8-sig"))
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                        if module.startswith("Final_Capstone_Project."):
                            self.assertTrue(module.startswith(
                                "Final_Capstone_Project.Capstone_Checkpoints."
                            ), module)

    def test_launcher_paths_and_datasets(self):
        for path in CHECKPOINTS.glob("Capstone_Checkpoint_*\\MHERRERA*.py"):
            with self.subTest(file=path.name):
                values = path_constants(path)
                self.assertEqual(values["CHECKPOINT_DIR"], path.parent)
                self.assertEqual(Path(values["LOG_PATH"]).parent, path.parent)
                if "CHROMA_DIR" in values:
                    self.assertEqual(Path(values["CHROMA_DIR"]),
                                     DATABASE / "Capstone_Chroma_DB")
                if "WIKI_DIR" in values:
                    self.assertEqual(Path(values["WIKI_DIR"]), DATABASE / "Wikipedia")
                for key in ("ORIGINALS_PATH", "VARIANTS_PATH", "MANUAL_ORIGINALS_PATH",
                            "MANUAL_PARAPHRASED_EVALUATION_PATH", "DEDICATED_CONFIG_PATH"):
                    if key in values:
                        self.assertTrue(Path(values[key]).is_file(), values[key])
                if "RAGAS_ROOT" in values:
                    self.assertTrue(Path(values["RAGAS_ROOT"]).is_dir())

    def test_early_notebook_imports_without_running_live_probes(self):
        for number in range(1, 4):
            path = next((CHECKPOINTS / f"Capstone_Checkpoint_{number}.1").glob("MHERRERA*.py"))
            with self.subTest(checkpoint=number):
                if number == 3:
                    values = runpy.run_path(str(path), run_name="checkpoint_import_smoke")
                    self.assertTrue(callable(values["main"]))
                    continue
                tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
                body = []
                for node in tree.body:
                    if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                            and isinstance(node.value.func, ast.Name)
                            and node.value.func.id in {"run", "run_probes"}):
                        continue
                    if (isinstance(node, ast.If) and any(
                        isinstance(target, ast.Name) and target.id == "DOCS"
                        for child in ast.walk(node) if isinstance(child, ast.Assign)
                        for target in child.targets
                    )):
                        continue
                    body.append(node)
                tree.body = body
                namespace = {"__file__": str(path), "__name__": "checkpoint_import_smoke",
                             "DOCS": []}
                exec(compile(tree, str(path), "exec"), namespace)
                entrypoint = "run_probes" if number == 1 else "run"
                self.assertTrue(callable(namespace[entrypoint]))

    def test_support_database_paths_and_setup_roots(self):
        paths = {
            "Retrieval_Methods\\bm25_retrieval.py": {
                "WIKIPEDIA_JSONL_DIR": DATABASE / "Wikipedia_JSONL",
                "BM25_INDEX_DIR": DATABASE / "Capstone_BM25_Lexical_Indexes",
                "BM25_BUILDER": CHECKPOINTS / "Utility_Scripts" / "Build_Wikipedia_BM25_Index.py",
                "CHUNKER_SCRIPT": CHECKPOINTS / "Utility_Scripts" / "Chunk_Wikipedia_HTML_To_JSONL.py",
            },
            "Utility_Scripts\\Setup.py": {
                "CHECKPOINTS_DIR": CHECKPOINTS, "PROJECT_DIR": PROJECT, "REPO_ROOT": REPO,
                "VENV_DIR": REPO / ".venv", "REQUIREMENTS_PATH": REPO / "venv_requirements.txt",
            },
            "Utility_Scripts\\utility_logging.py": {"WORKSPACE_ROOT": REPO},
            "Utility_Scripts\\Build_Wikipedia_Article_ChromaDB.py": {
                "DEFAULT_DB_DIR": DATABASE / "Capstone_Chroma_DB",
                "CONFIG_PATH": CHECKPOINTS / "retrieval.conf",
            },
            "Utility_Scripts\\Build_Wikipedia_Article_GraphDB.py": {
                "DEFAULT_INPUT_DIR": DATABASE / "Wikipedia",
                "DEFAULT_JSONL_DIR": DATABASE / "Wikipedia_JSONL",
                "DEFAULT_GRAPH_DIR": DATABASE / "Capstone_Graph_DB",
            },
            "Utility_Scripts\\Build_Wikipedia_BM25_Index.py": {
                "DEFAULT_JSONL_DIR": DATABASE / "Wikipedia_JSONL",
                "DEFAULT_INDEX_DIR": DATABASE / "Capstone_BM25_Lexical_Indexes",
            },
            "Utility_Scripts\\Chunk_Wikipedia_HTML_To_JSONL.py": {
                "WIKI_DIR": DATABASE / "Wikipedia", "OUTPUT_DIR": DATABASE / "Wikipedia_JSONL",
            },
            "Utility_Scripts\\Generate_Main_Test_Questions_LLM.py": {
                "WIKI_DIR": DATABASE / "Wikipedia",
                "SCHEMA_PATH": CHECKPOINTS / "JSON_Schemas" / "Test_Main_Questions.schema.json",
            },
            "Capstone_Checkpoint_3.1\\test_variables\\generate_main_questions.py": {
                "WIKI_DIR": DATABASE / "Wikipedia",
            },
            "Utility_Scripts\\security_cost_audit_demo.py": {"WORKSPACE_DIR": REPO},
        }
        for module in ("context_aware_retrieval", "dynamic_agent_retrieval"):
            paths[f"Retrieval_Methods\\{module}.py"] = {
                "CHROMA_DIR": DATABASE / "Capstone_Chroma_DB",
                "CONFIG_PATH": CHECKPOINTS / "retrieval.conf",
            }
        for relative, expected in paths.items():
            values = path_constants(CHECKPOINTS / relative)
            for name, value in expected.items():
                with self.subTest(file=relative, constant=name):
                    self.assertEqual(Path(values[name]), value)
                    self.assertTrue(value.exists(), value)

    def test_retriever_loads_existing_bm25_and_graph_without_builder_or_llm(self):
        from Final_Capstone_Project.Capstone_Checkpoints.Retrieval_Methods.bm25_retrieval import (
            BM25Retriever, load_wikipedia_chunks,
        )
        from Final_Capstone_Project.Capstone_Checkpoints.Retrieval_Methods.hybrid_retrieval import (
            HybridRetriever,
        )
        docs = load_wikipedia_chunks()
        with patch("subprocess.run", side_effect=AssertionError("Builder must not run")):
            bm25 = BM25Retriever(docs)
            hits = bm25.get_top_k("Albert Einstein", 4)
            self.assertEqual(len(hits), 4)
            module = "Final_Capstone_Project.Capstone_Checkpoints.Retrieval_Methods.hybrid_retrieval"
            with (
                patch(f"{module}.ChatOpenAI"),
                patch(f"{module}.build_or_load_db"),
                patch(f"{module}.get_vector_top_k", return_value=hits),
            ):
                retriever = HybridRetriever(
                    docs, search_method="graph", api_key="offline",
                    embedding_function=object(),
                )
                self.assertIsNotNone(retriever._graph)
                self.assertGreater(retriever._graph.number_of_nodes(), 0)
                self.assertTrue(retriever.retrievedContext("Albert Einstein"))

    def test_modern_launcher_help_from_root_and_unrelated_directory(self):
        launchers = [
            path for path in CHECKPOINTS.glob("Capstone_Checkpoint_*\\MHERRERA*.py")
            if path.parent.name not in {"Capstone_Checkpoint_1.1", "Capstone_Checkpoint_2.1",
                                        "Capstone_Checkpoint_3.1"}
        ]
        self.assertEqual(len(launchers), 6)
        environment = dict(os.environ, ANONYMIZED_TELEMETRY="False", HF_HUB_OFFLINE="1",
                           PYTHONIOENCODING="utf-8")
        with tempfile.TemporaryDirectory() as temporary:
            for cwd in (REPO, Path(temporary)):
                for path in launchers:
                    with self.subTest(file=path.name, cwd=str(cwd)):
                        result = subprocess.run(
                            [sys.executable, str(path), "--help"], cwd=cwd,
                            env=environment, capture_output=True, text=True,
                            encoding="utf-8", timeout=90,
                        )
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        self.assertIn("usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()
