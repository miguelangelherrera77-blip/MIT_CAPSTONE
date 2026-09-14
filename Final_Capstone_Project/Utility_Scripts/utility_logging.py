#################################################################################
# 
# Massachusetts Institute of Technology - xPRO
# RAG and Context Engineering: Designing and Building Production-Grade AI Systems
# 
# Student: Miguel Herrera
# Section: B
# Date and Time: 2026-09-12 12:27:00 -07:00
#
# Description: This module tees console and error output into append-only logs,
#              adds timestamps, converts repository paths to relative paths, and
#              redacts external absolute paths for Utility scripts.
#
#################################################################################

"""Shared append-only logging for scripts in the Utility_Scripts directory."""
from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
import re
import sys
from typing import Iterator, TextIO


SCRIPT_DIR = Path(__file__).resolve().parent
LOG_DIR = SCRIPT_DIR / "Logs"
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
ABSOLUTE_PATH_PATTERN = re.compile(r"(?<![\w])(?:[A-Za-z]:[\\/]|\\\\)[^,;\r\n]+")
def ask_override(output_path: str | None = None) -> bool:
    """Ask whether an existing generated output should be replaced."""
    if output_path:
        print(f"Existing JSON output found: {output_path}")
    else:
        print("Existing JSON output found.")
    print("Override existing output? [y/N]: ", end="", flush=True)
    try:
        answer = sys.stdin.readline()
    except (EOFError, TypeError):
        print()
        return False
    if answer is None:
        return False
    answer = answer.strip().lower()
    return answer in {"y", "yes"}


def ask_yes_no(prompt: str) -> bool:
    """Ask a yes/no question and default safely to no when input is unavailable."""
    print(f"{prompt} [y/N]: ", end="", flush=True)
    try:
        answer = sys.stdin.readline()
    except (EOFError, TypeError):
        print()
        return False
    return answer is not None and answer.strip().lower() in {"y", "yes"}


def ask_integer(prompt: str) -> int:
    """Require a positive integer from the terminal without applying a default."""
    print(f"{prompt}: ", end="", flush=True)
    try:
        answer = sys.stdin.readline()
    except (EOFError, TypeError):
        print()
        raise ValueError("number_of_questions is required; no default is applied.")
    if answer is None or not answer.strip():
        raise ValueError("number_of_questions is required; no default is applied.")
    try:
        value = int(answer.strip())
    except ValueError as exc:
        raise ValueError("number_of_questions must be a positive integer.") from exc
    if value <= 0:
        raise ValueError("number_of_questions must be greater than zero.")
    return value


def _format_absolute_path(match: re.Match[str]) -> str:
    """Return a workspace-relative path or redact an external absolute path."""
    raw_path = match.group(0).rstrip(".,:)]}")
    try:
        candidate = Path(raw_path).resolve()
        relative_path = candidate.relative_to(WORKSPACE_ROOT)
    except (OSError, ValueError):
        return "<absolute-path>"
    return relative_path.as_posix()


def _sanitize_log_text(text: str) -> str:
    """Relativize repository paths and redact external paths before logging."""
    return ABSOLUTE_PATH_PATTERN.sub(_format_absolute_path, text)


class _TimestampedTee(TextIO):
    """Write output to the console and an append-only timestamped log."""

    def __init__(self, console: TextIO, log_file: TextIO):
        self._console = console
        self._log_file = log_file

    def write(self, text: str) -> int:
        self._console.write(text)
        self._console.flush()
        if text:
            timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
            lines = text.splitlines(keepends=True)
            for line in lines:
                if line.strip():
                    self._log_file.write(f"[{timestamp}] {_sanitize_log_text(line)}")
                else:
                    self._log_file.write(line)
            self._log_file.flush()
        return len(text)

    def flush(self) -> None:
        self._console.flush()
        self._log_file.flush()

    def isatty(self) -> bool:
        return self._console.isatty()

    def fileno(self) -> int:
        return self._console.fileno()

    @property
    def encoding(self) -> str:
        return self._console.encoding or "utf-8"


@contextmanager
def log_run(script_path: str | Path) -> Iterator[Path]:
    """Append one timestamped run to the log named after ``script_path``."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    script_name = Path(script_path).stem
    log_path = LOG_DIR / f"{script_name}.log"
    started = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    with log_path.open("a", encoding="utf-8", newline="") as log_file:
        log_file.write(f"\n=== RUN START: {started} ===\n")
        log_file.flush()
        with redirect_stdout(_TimestampedTee(__import__("sys").stdout, log_file)), redirect_stderr(
            _TimestampedTee(__import__("sys").stderr, log_file)
        ):
            try:
                yield log_path
            except BaseException as exc:
                finished = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
                error_text = _sanitize_log_text(str(exc))
                log_file.write(f"[{finished}] RUN FAILED: {type(exc).__name__}: {error_text}\n")
                log_file.flush()
                raise
            else:
                finished = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
                log_file.write(f"=== RUN END: {finished} ===\n")
                log_file.flush()
