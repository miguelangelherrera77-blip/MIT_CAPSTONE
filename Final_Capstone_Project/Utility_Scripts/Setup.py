#################################################################################
# 
# Massachusetts Institute of Technology - xPRO
# RAG and Context Engineering: Designing and Building Production-Grade AI Systems
# 
# Student: Miguel Herrera
# Section: B
# Date and Time: 2026-09-13 20:00:00 -07:00
#
# Description: Creates the local runtime scaffolding expected by the capstone
#              project on a fresh clone, including required data directories,
#              environment placeholders, and optional database builders for the
#              Chroma, graph, and BM25 indexes.
#
#################################################################################

"""Create the on-disk scaffolding required to run the capstone solution."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from utility_logging import log_run

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
REPO_ROOT = PROJECT_DIR.parent
VENV_DIR = REPO_ROOT / ".venv"
REQUIREMENTS_PATH = REPO_ROOT / "venv_requirements.txt"


def virtual_environment_python() -> Path:
    """Return the venv Python path for the current host operating system."""
    executable_name = "python.exe" if os.name == "nt" else "python"
    scripts_dir = "Scripts" if os.name == "nt" else "bin"
    return VENV_DIR / scripts_dir / executable_name


def ensure_directory(path: Path, created: list[Path]) -> None:
    """Create a directory if it is missing and record the new directory."""
    if path.exists():
        return

    path.mkdir(parents=True, exist_ok=True)
    created.append(path)


def ensure_env_file(created: list[Path]) -> None:
    """Create a .env template if the root environment file is absent."""
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        return

    env_path.write_text("OPENROUTER_API_KEY=your_openrouter_key_here\n", encoding="utf-8")
    created.append(env_path)


def ensure_gitignore(created: list[Path]) -> None:
    """Create or extend .gitignore with local setup and runtime artifacts."""
    gitignore_path = REPO_ROOT / ".gitignore"
    was_present = gitignore_path.exists()
    existing = gitignore_path.read_text(encoding="utf-8") if was_present else ""
    entries = [
        ".gitignore",
        ".venv/",
        ".env",
        "__pycache__/",
        "*.py[cod]",
        "Final_Capstone_Project/Utility_Scripts/Logs/",
        "Final_Capstone_Project/Capstone_Database/Wikipedia/",
        "Final_Capstone_Project/Capstone_Database/Wikipedia_JSONL/",
        "Final_Capstone_Project/Capstone_Database/Capstone_Chroma_DB/",
        "Final_Capstone_Project/Capstone_Database/Capstone_Graph_DB/",
        "Final_Capstone_Project/Capstone_Database/Capstone_BM25_Lexical_Indexes/",
    ]
    missing_entries = [entry for entry in entries if entry not in existing.splitlines()]
    if not missing_entries:
        return

    separator = "" if not existing or existing.endswith("\n") else "\n"
    gitignore_path.write_text(
        existing + separator + "\n".join(missing_entries) + "\n",
        encoding="utf-8",
    )
    if not was_present:
        created.append(gitignore_path)


def ensure_virtual_environment(created: list[Path]) -> Path:
    """Create the host-specific venv and install the project requirements once."""
    python_path = virtual_environment_python()
    if python_path.exists():
        print(f"[venv] Using existing virtual environment at {VENV_DIR}")
        return python_path

    print(f"[venv] Creating virtual environment at {VENV_DIR} for {sys.platform}")
    subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
    created.append(VENV_DIR)
    if not REQUIREMENTS_PATH.exists():
        raise FileNotFoundError(f"Requirements file not found: {REQUIREMENTS_PATH}")

    print(f"[venv] Installing dependencies from {REQUIREMENTS_PATH.name}")
    subprocess.run(
        [str(python_path), "-m", "pip", "install", "-r", str(REQUIREMENTS_PATH)],
        check=True,
    )
    return python_path


def ensure_expected_directories(created: list[Path]) -> None:
    """Create the directory structure used by the project runtime and databases."""
    required_dirs = [
        PROJECT_DIR / "Capstone_Database",
        PROJECT_DIR / "Capstone_Database" / "Capstone_Chroma_DB",
        PROJECT_DIR / "Capstone_Database" / "Capstone_Graph_DB",
        PROJECT_DIR / "Capstone_Database" / "Wikipedia",
        PROJECT_DIR / "Capstone_Database" / "Wikipedia_JSONL",
        PROJECT_DIR / "Capstone_Database" / "Capstone_BM25_Lexical_Indexes",
        PROJECT_DIR / "Capstone_Database" / "Capstone_Graph_DB" / "GraphML",
        PROJECT_DIR / "Ragas_Experiments" / "datasets",
        PROJECT_DIR / "Ragas_Experiments" / "experiments",
        PROJECT_DIR / "Test_Variables",
        PROJECT_DIR / "JSON_Schemas",
        PROJECT_DIR / "Utility_Scripts" / "Logs",
    ]

    for directory in required_dirs:
        ensure_directory(directory, created)

    # Ensure the checkpoint-specific evaluation directories that are expected by the app exist.
    checkpoint_dirs = [
        PROJECT_DIR / "Capstone_Checkpoint_3.1" / "test_variables",
        PROJECT_DIR / "Capstone_Checkpoint_3.1" / "ragas_experiments_3_1" / "datasets",
        PROJECT_DIR / "Capstone_Checkpoint_3.1" / "ragas_experiments_3_1" / "experiments",
        PROJECT_DIR / "Capstone_Checkpoint_4.1",
    ]
    for directory in checkpoint_dirs:
        ensure_directory(directory, created)


def ensure_placeholder_files(created: list[Path]) -> None:
    """Create lightweight placeholders for required local directories when they are empty."""
    placeholder_files = [
        PROJECT_DIR / "Capstone_Database" / "Wikipedia" / "README.txt",
        PROJECT_DIR / "Capstone_Database" / "Wikipedia_JSONL" / "README.txt",
        PROJECT_DIR / "Capstone_Database" / "Capstone_Chroma_DB" / "README.txt",
        PROJECT_DIR / "Capstone_Database" / "Capstone_Graph_DB" / "README.txt",
        PROJECT_DIR / "Capstone_Database" / "Capstone_Graph_DB" / "GraphML" / "README.txt",
    ]

    for path in placeholder_files:
        if not path.exists():
            if path.parent.exists() or path.parent.mkdir(parents=True, exist_ok=True):
                path.write_text(
                    "This directory is created by Setup.py for the local runtime state of the capstone project.\n"
                    "Add the corpus or generated databases here before running retrieval or evaluation.\n",
                    encoding="utf-8",
                )
                created.append(path)

    bm25_dir = PROJECT_DIR / "Capstone_Database" / "Capstone_BM25_Lexical_Indexes"
    if bm25_dir.exists() and not any(bm25_dir.iterdir()):
        (bm25_dir / "README.txt").write_text(
            "This directory is created by Setup.py and will be populated by the BM25 builder when the corpus is available.\n",
            encoding="utf-8",
        )
        created.append(bm25_dir / "README.txt")


def build_database_if_requested(
    build: bool,
    python_path: Path,
    rebuild: bool = False,
    created: list[Path] | None = None,
) -> None:
    """Optionally run the project utility scripts to populate the local databases."""
    if not build:
        print("[skip] Database build step not requested. Use --build to generate local indexes and databases.")
        return

    wikipedia_dir = PROJECT_DIR / "Capstone_Database" / "Wikipedia"
    html_files = sorted(wikipedia_dir.glob("*.html")) if wikipedia_dir.is_dir() else []
    if not html_files:
        print("[info] No Wikipedia HTML corpus found at Final_Capstone_Project/Capstone_Database/Wikipedia/.")
        print("[info] Add the HTML corpus first, then rerun Setup.py --build to populate the local databases.")
        return

    commands = [
        (
            "Wikipedia_JSONL",
            [
                str(python_path),
                str(SCRIPT_DIR / "Chunk_Wikipedia_HTML_To_JSONL.py"),
                *( ["--rebuild"] if rebuild else [] ),
            ],
        ),
        (
            "ChromaDB",
            [
                str(python_path),
                str(SCRIPT_DIR / "Build_Wikipedia_Article_ChromaDB.py"),
                "openrouter",
            ],
        ),
        (
            "GraphDB",
            [
                str(python_path),
                str(SCRIPT_DIR / "Build_Wikipedia_Article_GraphDB.py"),
                *( ["--rebuild"] if rebuild else [] ),
            ],
        ),
        (
            "BM25",
            [
                str(python_path),
                str(SCRIPT_DIR / "Build_Wikipedia_BM25_Index.py"),
                *( ["--rebuild"] if rebuild else [] ),
            ],
        ),
    ]

    generated_files_before = {
        path
        for generated_dir in (
            PROJECT_DIR / "Capstone_Database" / "Capstone_Chroma_DB",
            PROJECT_DIR / "Capstone_Database" / "Capstone_Graph_DB",
            PROJECT_DIR / "Capstone_Database" / "Capstone_BM25_Lexical_Indexes",
            PROJECT_DIR / "Capstone_Database" / "Wikipedia_JSONL",
        )
        if generated_dir.exists()
        for path in generated_dir.rglob("*")
        if path.is_file()
    }
    for database_name, command in commands:
        print(f"[build] Running {database_name}")
        try:
            subprocess.run(command, check=True, cwd=str(SCRIPT_DIR), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as exc:
            print(f"[error] Command failed with exit code {exc.returncode}: {' '.join(str(part) for part in command)}")
            print("[info] The directory scaffolding was still created; the database build requires the corpus and runtime dependencies.")
            raise SystemExit(exc.returncode)
    if created is not None:
        generated_files_after = {
            path
            for generated_dir in (
                PROJECT_DIR / "Capstone_Database" / "Capstone_Chroma_DB",
                PROJECT_DIR / "Capstone_Database" / "Capstone_Graph_DB",
                PROJECT_DIR / "Capstone_Database" / "Capstone_BM25_Lexical_Indexes",
                PROJECT_DIR / "Capstone_Database" / "Wikipedia_JSONL",
            )
            if generated_dir.exists()
            for path in generated_dir.rglob("*")
            if path.is_file()
        }
        created.extend(sorted(generated_files_after - generated_files_before))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the local runtime scaffolding expected by the capstone retrieval project. "
            "Optional database build steps can generate Chroma, graph, and BM25 files from the Wikipedia corpus."
        )
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Also run the project DB builders if the local Wikipedia corpus is present.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force a rebuild of generated Chroma/Graph/BM25 files when --build is used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=== Capstone Project Setup ===")
    print(f"Repository root: {REPO_ROOT}")
    print(f"Project root: {PROJECT_DIR}")

    created: list[Path] = []
    ensure_env_file(created)
    ensure_gitignore(created)
    python_path = ensure_virtual_environment(created)
    ensure_expected_directories(created)
    ensure_placeholder_files(created)

    build_database_if_requested(
        build=args.build,
        python_path=python_path,
        rebuild=args.rebuild,
        created=created,
    )

    if created:
        print("\nSetup built the following missing items:")
        for path in created:
            print(f"- {path.relative_to(REPO_ROOT).as_posix()}")


if __name__ == "__main__":
    with log_run(__file__):
        main()
