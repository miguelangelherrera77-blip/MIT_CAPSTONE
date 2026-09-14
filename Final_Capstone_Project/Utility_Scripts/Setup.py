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


PLACEHOLDER_OPENROUTER_KEY = "your_openrouter_key_here"


def _openrouter_key_status() -> str:
    """Return 'set', 'placeholder', or 'missing' for the configured OpenRouter key."""
    env_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if env_key:
        return "placeholder" if env_key == PLACEHOLDER_OPENROUTER_KEY else "set"

    env_path = REPO_ROOT / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("OPENROUTER_API_KEY"):
                _, _, value = line.partition("=")
                value = value.strip()
                if value:
                    return "placeholder" if value == PLACEHOLDER_OPENROUTER_KEY else "set"
    return "missing"


def _prompt_choice(prompt: str) -> str:
    """Read a line of input, failing with a clear message when no TTY is available."""
    try:
        return input(prompt).strip().lower()
    except EOFError:
        raise SystemExit(
            "[error] No input available to answer the confirmation prompt above. "
            "Re-run Setup.py from an interactive terminal, or resolve the condition without --build."
        ) from None
    except KeyboardInterrupt:
        raise SystemExit("\n[quit] Setup cancelled by user.") from None


def confirm_continue_without_key(key_status: str) -> bool:
    """Checkpoint: warn that the OpenRouter key looks unusable before spending an API call."""
    print(f"[checkpoint] OPENROUTER_API_KEY is {key_status} in the root .env file.")
    print("[checkpoint] Update .env with your real OpenRouter key, e.g.: OPENROUTER_API_KEY=sk-or-v1-...")
    while True:
        answer = _prompt_choice("[confirm] Continue with ChromaDB anyway (it will fail with a 401 error)? [y]es/[n]o/[q]uit: ")
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        if answer in ("q", "quit"):
            raise SystemExit("[quit] Setup cancelled by user.")
        print("[input] Please answer y, n, or q.")


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


def _bm25_index_is_valid(index_dir: Path) -> bool:
    """Mirror Build_Wikipedia_BM25_Index.py's own validity check."""
    if not index_dir.is_dir():
        return False
    names = {path.name.lower() for path in index_dir.iterdir() if path.is_file()}
    modern = {"params.index.json", "vocab.index.json", "documents.json"}
    legacy = {"index.json", "documents.json"}
    return modern.issubset(names) or legacy.issubset(names)


def _chroma_db_has_data(db_dir: Path) -> bool:
    """Mirror Build_Wikipedia_Article_ChromaDB.py's own validity check."""
    if not db_dir.is_dir():
        return False
    sqlite_path = db_dir / "chroma.sqlite3"
    if not sqlite_path.is_file() or sqlite_path.stat().st_size == 0:
        return False
    # chroma.sqlite3 alone can exist from an interrupted run with no embedded data;
    # a collection segment directory is only written once vectors are persisted.
    return any(
        path.is_dir() and any(child.is_file() for child in path.iterdir())
        for path in db_dir.iterdir()
    )


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


def confirm_rebuild(job_name: str, target: str) -> bool:
    """Ask the user whether to rebuild existing output; 'q' exits Setup.py immediately."""
    while True:
        answer = _prompt_choice(f"[confirm] Rebuild {job_name}? Existing output found at {target}. [y]es/[n]o/[q]uit: ")
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        if answer in ("q", "quit"):
            raise SystemExit("[quit] Setup cancelled by user.")
        print("[input] Please answer y, n, or q.")


def _chroma_dir_has_existing_content(db_dir: Path) -> bool:
    """Return True if the directory holds anything beyond the placeholder scaffold file."""
def _dir_has_existing_content(path: Path) -> bool:
    """Return True if the directory holds anything beyond the placeholder scaffold file."""
    if not path.is_dir():
        return False
    placeholder_names = {"readme.txt", "readme.md", ".gitkeep"}
    return any(child.name.lower() not in placeholder_names for child in path.iterdir())


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
    jsonl_dir = PROJECT_DIR / "Capstone_Database" / "Wikipedia_JSONL"
    jsonl_files = sorted(jsonl_dir.glob("*.jsonl")) if jsonl_dir.is_dir() else []
    if not html_files and not jsonl_files:
        print("[info] No Wikipedia HTML or JSONL corpus found.")
        print("[info] Add HTML files or JSONL files first, then rerun Setup.py --build.")
        return

    # Local, non-API jobs run first so they still complete when no API key is configured.
    # Each job is only invoked when its output is missing, incomplete, or explicitly --rebuild.
    graph_file = PROJECT_DIR / "Capstone_Database" / "Capstone_Graph_DB" / "wikipedia_articles.graphml"
    graph_file_valid = graph_file.is_file() and graph_file.stat().st_size > 0
    bm25_dir = PROJECT_DIR / "Capstone_Database" / "Capstone_BM25_Lexical_Indexes"
    chroma_dir = PROJECT_DIR / "Capstone_Database" / "Capstone_Chroma_DB"

    commands: list[tuple[str, list[str]]] = []
    if html_files and jsonl_files and rebuild:
        if confirm_rebuild("HTML chunking (Wikipedia_JSONL)", str(jsonl_dir)):
            commands.append(
                (
                    "Wikipedia_JSONL",
                    [str(python_path), str(SCRIPT_DIR / "Chunk_Wikipedia_HTML_To_JSONL.py"), "--rebuild"],
                )
            )
        else:
            print(f"[skip] Keeping {len(jsonl_files)} existing JSONL file(s); rebuild declined.")
    elif html_files and not jsonl_files:
        commands.append(
            (
                "Wikipedia_JSONL",
                [str(python_path), str(SCRIPT_DIR / "Chunk_Wikipedia_HTML_To_JSONL.py")],
            )
        )
    else:
        print(f"[skip] Wikipedia_JSONL skipped: {len(jsonl_files)} JSONL file(s) already exist at {jsonl_dir}.")

    if html_files and graph_file_valid and rebuild:
        if confirm_rebuild("GraphDB", str(graph_file)):
            commands.append(
                (
                    "GraphDB",
                    [str(python_path), str(SCRIPT_DIR / "Build_Wikipedia_Article_GraphDB.py"), "--rebuild"],
                )
            )
        else:
            print(f"[skip] Keeping existing GraphML output; rebuild declined: {graph_file}")
    elif html_files and not graph_file_valid and graph_file.exists():
        if confirm_rebuild("GraphDB (existing output appears invalid or incomplete)", str(graph_file)):
            commands.append(
                (
                    "GraphDB",
                    [str(python_path), str(SCRIPT_DIR / "Build_Wikipedia_Article_GraphDB.py"), "--rebuild"],
                )
            )
        else:
            print(f"[skip] Keeping invalid GraphML output as-is; rebuild declined: {graph_file}")
    elif html_files and not graph_file_valid:
        commands.append(
            (
                "GraphDB",
                [str(python_path), str(SCRIPT_DIR / "Build_Wikipedia_Article_GraphDB.py")],
            )
        )
    elif html_files:
        print(f"[skip] GraphDB skipped: GraphML output already exists at {graph_file}.")
    else:
        print("[skip] GraphDB requires HTML files; no GraphDB job will run.")

    bm25_valid = _bm25_index_is_valid(bm25_dir)
    if bm25_valid and rebuild:
        if confirm_rebuild("BM25 index", str(bm25_dir)):
            commands.append(
                (
                    "BM25",
                    [str(python_path), str(SCRIPT_DIR / "Build_Wikipedia_BM25_Index.py"), "--rebuild"],
                )
            )
        else:
            print(f"[skip] Keeping existing bm25s index; rebuild declined: {bm25_dir}")
    elif not bm25_valid and _dir_has_existing_content(bm25_dir):
        if confirm_rebuild("BM25 index (existing content appears invalid or incomplete)", str(bm25_dir)):
            commands.append(
                (
                    "BM25",
                    [str(python_path), str(SCRIPT_DIR / "Build_Wikipedia_BM25_Index.py"), "--rebuild"],
                )
            )
        else:
            print(f"[skip] Keeping invalid bm25s index as-is; rebuild declined: {bm25_dir}")
    elif not bm25_valid:
        commands.append(
            (
                "BM25",
                [str(python_path), str(SCRIPT_DIR / "Build_Wikipedia_BM25_Index.py")],
            )
        )
    else:
        print(f"[skip] BM25 skipped: index already exists at {bm25_dir}.")

    # ChromaDB calls the OpenRouter embedding API; run it last, and only when data is missing.
    # Setup.py never forces a Chroma rebuild via --rebuild, to avoid re-spending on embeddings
    # unintentionally, but an existing incomplete/corrupt directory still needs confirmation
    # before its contents are replaced.
    if _chroma_db_has_data(chroma_dir):
        print(f"[skip] ChromaDB skipped: database already exists at {chroma_dir}.")
    else:
        proceed = True
        if _dir_has_existing_content(chroma_dir):
            proceed = confirm_rebuild("ChromaDB (incomplete or corrupt database found)", str(chroma_dir))
            if not proceed:
                print(f"[skip] Keeping incomplete Chroma database as-is; rebuild declined: {chroma_dir}")

        if proceed:
            key_status = _openrouter_key_status()
            if key_status != "set":
                proceed = confirm_continue_without_key(key_status)
                if not proceed:
                    print("[skip] ChromaDB build skipped until OPENROUTER_API_KEY is configured.")

        if proceed:
            commands.append(
                (
                    "ChromaDB",
                    [
                        str(python_path),
                        str(SCRIPT_DIR / "Build_Wikipedia_Article_ChromaDB.py"),
                        "openrouter",
                    ],
                )
            )

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
    total_jobs = len(commands)
    for job_number, (database_name, command) in enumerate(commands, start=1):
        print(f"[build] Running {database_name} (job {job_number}/{total_jobs})")
        try:
            # Inherit stdout/stderr so each script's own tqdm progress bar is visible.
            subprocess.run(command, check=True, cwd=str(SCRIPT_DIR))
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
