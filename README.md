# CAPSTONE PROJECT
### RAG and Context Engineering: Designing and Building Production-Grade AI Systems

**Miguel Herrera - Section B**

## Overview

A Wikipedia retrieval-augmented generation (RAG) system combining BM25 lexical
search, Chroma vector retrieval, and optional graph expansion.

- **Context-Aware:** single-pass, history-aware retrieval for predictable cost.
- **Agentic Dynamic:** iterative planning and retrieval for complex questions.

Both engines support baseline and hardened operation. Sanitization and tagged
trust boundaries are defense-in-depth, not security or factual-correctness guarantees.

## Start Here

| Goal | Resource |
| --- | --- |
| Set up and run checkpoints 1.1-7.1 | [Checkpoint setup and run guide](Capstone_Checkpoints/Checkpoint_Readme.md) |
| Review Final Solution results, checkpoint experiments, and engineering trade-offs | [Testing results](Capstone_Checkpoints/Checkpoint_Testing_Results.md) |
| Browse implementations and shared utilities | [Capstone_Checkpoints](Capstone_Checkpoints/) and [Capstone_Utilities](Capstone_Utilities/) |
| Review raw reports and evaluation CSVs | [Capstone_Ragas_Experiments](Capstone_Ragas_Experiments/) |

The checkpoint guide contains Windows PowerShell and macOS Zsh/Bash commands,
dataset instructions, troubleshooting, and offline checks. Historical results
are preserved separately and are not certification of the current implementation.

## Project Organization

| Area | Purpose |
| --- | --- |
| [Capstone_Checkpoints](Capstone_Checkpoints/) | Checkpoints 1.1-7.1, checkpoint configuration, question datasets, and documentation |
| [Capstone_Final_Solution](Capstone_Final_Solution/) | Automatic hardened, graph-enabled final Wikipedia RAG solution |
| [Capstone_Utilities](Capstone_Utilities/) | Shared checkpoint retrieval/ranking modules, utility scripts, and JSON schemas |
| [Capstone_Utilities/Retrieval_Methods](Capstone_Utilities/Retrieval_Methods/) | Retrieval modules shared by checkpoints 4.1-7.1 |
| [Capstone_Utilities/Ranking_Techniques](Capstone_Utilities/Ranking_Techniques/) | Weighted ranking shared by checkpoint retrieval workflows |
| [Capstone_Utilities/JSON_Schemas](Capstone_Utilities/JSON_Schemas/) | JSON schemas used by question-generation utilities |
| [Capstone_Ragas_Experiments](Capstone_Ragas_Experiments/) | Local evaluation datasets, reports, experiment logs, and CSV results |
| [venv_requirements.txt](venv_requirements.txt) | Direct Python dependencies for the documented project workflows |

## Requirements and Operational Warnings

- Use a local Python environment with `venv` and `pip`. The dependency file
  lists package requirements but does not declare a project-wide minimum Python
  version; compatibility depends on the selected package versions. Offline
  validation used **Python 3.12.10 on Windows**. Other versions and macOS/Linux
  execution are not certified.
- [venv_requirements.txt](venv_requirements.txt) lists direct dependencies for
  checkpoints, the final solution, and setup/evaluation utilities. Pip resolves
  transitive dependencies automatically.
- Follow the checkpoint guide from the repository root. Setup installs
  dependencies only when creating the virtual environment; it does not refresh
  an existing environment or populate indexes without a build request.
- Set `OPENROUTER_API_KEY` in the environment or the `.env` location documented
  by the applicable guide. **Never commit credentials.**
- Network access is needed for installation and API calls. Answer generation,
  judging, and embeddings can incur charges; allow adequate local storage.
- HTML is required for chunking and graph creation; JSONL suffices for BM25
  and Chroma creation. A fresh clone needs suitable corpus/index artifacts.
- Checkpoints 4.1-7.1 invoke setup/build checks before normal menus and quit
  selections. Missing artifacts can trigger builders and embedding costs;
  `--help` exits before bootstrap.
- Checkpoints 1.1 and 2.1 execute live workflows at module level; do not import
  them as offline smoke tests.
- Do not rebuild databases merely to launch a checkpoint. Explicit rebuilds
  can overwrite generated artifacts after confirmation.

## Evaluation Summary

Historical experiments show a cost/accuracy trade-off:

- Context-Aware generally uses fewer model calls and tokens.
- Agentic can improve synthesis and multi-fact answers at higher cost and latency.
- Pure semantic retrieval and exact quotation fidelity remain limitations.
- Small, LLM-judged datasets do not establish general reliability.

[Testing results](Capstone_Checkpoints/Checkpoint_Testing_Results.md)
preserves the detailed tables, dates, methodology, output references, model
configurations, embedding budget, and limitations. Historical hardening scores
predate the native shared Agentic fix and have not been rerun against it.
Reported model costs exclude judge and embedding usage where specified.