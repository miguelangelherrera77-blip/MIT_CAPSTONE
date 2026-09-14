# CAPSTONE PROJECT
RAG and Context Engineering: Designing and Building Production-Grade AI Systems

Miguel Herrera - Section B

## Project Overview
This repository contains the capstone RAG project for a Wikipedia Retrieval Engine scenario. It includes the main project implementation, checkpoint solution files, evaluation utilities, local database directories, course labs, and supporting scripts used for retrieval, ranking, and analysis.

## Quick Links

| Section | Description | Link |
| --- | --- | --- |
| Project Overview | Repository purpose and system context | [Overview](#project-overview) |
| Checkpoint 1.1 | Evaluating when retrieval is required | [Checkpoint 1.1](#capstone-checkpoint-11) |
| Checkpoint 2.1 | Retrieval strategy and baseline implementation | [Checkpoint 2.1](#capstone-checkpoint-21) |
| Checkpoint 3.1 | RAGAS evaluation and paraphrase robustness | [Checkpoint 3.1](#capstone-checkpoint-31) |
| Checkpoint 4.1 | Advanced retrieval and evaluation harness | [Checkpoint 4.1](#capstone-checkpoint-41) |
| Setup and Local Data | Bootstrap commands and generated local data | [Setup](#setup-and-local-data) |
| Key Project Areas | Source modules and supporting project areas | [Project Areas](#key-project-areas) |

## Capstone Checkpoints

### Capstone Checkpoint 1.1
**Evaluating when retrieval is required.** This checkpoint evaluates how an LLM performs without retrieval and determines whether retrieval is required for the selected Wikipedia scenario. The solution is in `Final_Capstone_Project/Capstone_Checkpoint_1.1/`.

Add `OPENROUTER_API_KEY` to the root `.env` file before running the solution. The checkpoint uses `python-dotenv`, `langchain-openai`, and `langchain-core`.

### Capstone Checkpoint 2.1
**Retrieval strategy design and baseline implementation.** This checkpoint implements Retrieval-Augmented Generation over the Wikipedia corpus using vector retrieval through ChromaDB and lexical retrieval through BM25. The solution is in `Final_Capstone_Project/Capstone_Checkpoint_2.1/`.

Place the Wikipedia HTML corpus in `Final_Capstone_Project/Capstone_Database/Wikipedia/` and add `OPENROUTER_API_KEY` to `.env`. The required packages are listed in `venv_requirements.txt`.

### Capstone Checkpoint 3.1
**Evaluation infrastructure and baseline diagnosis.** This checkpoint evaluates the retrieval system with RAGAS and compares original questions with paraphrased variants to measure robustness to rephrasing. The solution and validation utilities are in `Final_Capstone_Project/Capstone_Checkpoint_3.1/`.

The evaluation workflow generates or loads grounded questions, creates paraphrased variants, evaluates answers with a RAGAS `DiscreteMetric` judge, and reports original-versus-paraphrase pass rates and their difference. Datasets and experiment outputs are stored under `test_variables/` and `ragas_experiments_3_1/`. The Wikipedia corpus and a usable local Chroma database are required for a complete evaluation run.

Run the framework validation utility from the repository root with `python Final_Capstone_Project/Utility_Scripts/run_framework_validation.py`. Add `OPENROUTER_API_KEY` to `.env` before running LLM-backed evaluation.

### Capstone Checkpoint 4.1
**Advanced retrieval and evaluation harness.** This checkpoint combines persisted vector, graph, BM25 lexical, and hybrid retrieval strategies in an interactive evaluation workflow. The solution is in `Final_Capstone_Project/Capstone_Checkpoint_4.1/MHERRERA_Capstone_Checkpoint_4_1_Solution.py`.

On startup, the solution runs the local preflight setup before displaying the menu. It invokes `Setup.py --build`, which requires the Wikipedia HTML corpus under `Final_Capstone_Project/Capstone_Database/Wikipedia/`. The build creates or reuses the Wikipedia JSONL corpus, ChromaDB, GraphDB, and BM25 indexes, each with a separate progress stage. Existing valid generated artifacts are reused; `--rebuild` regenerates JSONL, GraphDB, and BM25 outputs when supplied directly to Setup.py.

To run the solution directly from the repository root:

```powershell
.\.venv\Scripts\python.exe .\Final_Capstone_Project\Capstone_Checkpoint_4.1\MHERRERA_Capstone_Checkpoint_4_1_Solution.py
```

Setup.py creates missing runtime directories and a root `.env` template when needed. Setup output is logged to `Final_Capstone_Project/Utility_Scripts/Logs/Setup.log`. Generated databases, corpus files, and logs remain local.

## Setup and Local Data
Run the setup utility from the repository root:

```powershell
.\.venv\Scripts\python.exe .\Final_Capstone_Project\Utility_Scripts\Setup.py
```

Use `--build` to generate the local Wikipedia JSONL corpus and retrieval databases after adding the HTML corpus. Use `--rebuild` with `--build` when regeneration is explicitly required:

```powershell
.\.venv\Scripts\python.exe .\Final_Capstone_Project\Utility_Scripts\Setup.py --build
.\.venv\Scripts\python.exe .\Final_Capstone_Project\Utility_Scripts\Setup.py --build --rebuild
```

Setup creates or verifies the following local paths:
- `Final_Capstone_Project/Capstone_Database/Wikipedia/`
- `Final_Capstone_Project/Capstone_Database/Wikipedia_JSONL/`
- `Final_Capstone_Project/Capstone_Database/Capstone_Chroma_DB/`
- `Final_Capstone_Project/Capstone_Database/Capstone_Graph_DB/`
- `Final_Capstone_Project/Capstone_Database/Capstone_BM25_Lexical_Indexes/`
- `Final_Capstone_Project/Utility_Scripts/Logs/`

If the Wikipedia HTML corpus is absent, setup still creates the runtime scaffolding but skips JSONL and database generation. The generated database folders must contain real artifacts before retrieval can use them; placeholder README files are only scaffolding.

## Key Project Areas
- `Final_Capstone_Project/Capstone_Checkpoint_1.1/` contains the checkpoint 1.1 solution and supporting artifacts.
- `Final_Capstone_Project/Capstone_Checkpoint_2.1/` contains the retrieval strategy and baseline implementation files.
- `Final_Capstone_Project/Capstone_Checkpoint_3.1/` contains the evaluation harness, validation utilities, datasets, and experiment results.
- `Final_Capstone_Project/Capstone_Checkpoint_4.1/` contains the advanced retrieval starter and final solution files.
- `Final_Capstone_Project/Capstone_Database/` stores the local corpus and database artifacts used for retrieval.
- `Final_Capstone_Project/Retrieval_Methods/` contains BM25, vector, and hybrid retrieval logic.
- `Final_Capstone_Project/Ranking_Techniques/` contains ranking and fusion logic.
- `Final_Capstone_Project/Ragas_Experiments/` stores evaluation logic and experiment outputs.
- `lab_*` directories contain the course lab scripts, starter files, and requirements for guided work.

## Notes
- The repository is intended to be used with a local Python environment and project-specific data directories.
- Large corpus and database artifacts may be stored locally and are not always intended for version control.
- This inventory reflects the current workspace contents and project structure.

--------------------------
