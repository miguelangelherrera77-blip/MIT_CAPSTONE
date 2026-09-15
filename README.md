# CAPSTONE PROJECT
RAG and Context Engineering: Designing and Building Production-Grade AI Systems

Miguel Herrera - Section B

## Project Overview
This repository contains the capstone Wikipedia RAG project. It includes the main implementation, checkpoint solutions, evaluation utilities, local database directories, course lab materials, and supporting retrieval scripts.

## Project System Requirements

- **Operating system:** Windows, macOS, or Linux.
- **Python:** Python 3.10 or newer.
- **Python tooling:** `venv` and `pip` available in the system Python environment.
- **Internet access:** Required for dependency installation and OpenRouter API calls.
- **OpenRouter credentials:** Set `OPENROUTER_API_KEY` in the root `.env` file or the active environment.
- **Local storage:** Sufficient disk space for the corpus, generated JSONL files, ChromaDB, GraphDB, BM25 indexes, and logs.
- **Corpus input:** HTML files are needed for chunking and GraphDB creation; JSONL files are sufficient for ChromaDB and BM25.
- **Terminal access:** Run setup commands from the repository root so all relative paths resolve correctly.

## Quick Links

| Section | Description | Link |
| --- | --- | --- |
| Project Overview | Repository purpose and system context | [Overview](#project-overview) |
| System Requirements | Host, Python, and runtime prerequisites | [Requirements](#project-system-requirements) |
| Checkpoint 1.1 | Evaluating when retrieval is required | [Checkpoint 1.1](#capstone-checkpoint-11) |
| Checkpoint 2.1 | Retrieval strategy and baseline implementation | [Checkpoint 2.1](#capstone-checkpoint-21) |
| Checkpoint 3.1 | RAGAS evaluation and paraphrase robustness | [Checkpoint 3.1](#capstone-checkpoint-31) |
| Checkpoint 4.1 | Advanced retrieval and evaluation harness | [Checkpoint 4.1](#capstone-checkpoint-41) |
| Checkpoint 4.1 Testing | Category-based testing results and analysis | [Testing results and analysis](#testing-results-and-analysis) |
| Checkpoint 4.1 Test Summary | Human-readable summary of Checkpoint 4.1 test runs | [detailed_test_results.log](Final_Capstone_Project/Ragas_Experiments/detailed_test_results.log) |
| Setup and Local Data | Bootstrap commands and generated local data | [Setup](#setup-and-local-data) |
| ChromaDB Cost Estimate | Token count and estimated embedding cost | [Cost Estimate](#chromadb-token-and-cost-estimate) |
| Key Project Areas | Source modules and supporting project areas | [Project Areas](#key-project-areas) |

## Capstone Checkpoints

### Capstone Checkpoint 1.1
**Evaluating when retrieval is required.** This checkpoint measures baseline LLM performance without retrieval and determines whether retrieval is needed for the Wikipedia scenario. The solution is in [Final_Capstone_Project/Capstone_Checkpoint_1.1/](Final_Capstone_Project/Capstone_Checkpoint_1.1/).

Add `OPENROUTER_API_KEY` to the root `.env` file before running the solution.

### Capstone Checkpoint 2.1
**Retrieval strategy design and baseline implementation.** This checkpoint builds a Wikipedia RAG workflow using ChromaDB vector retrieval and BM25 lexical retrieval. The solution is in [Final_Capstone_Project/Capstone_Checkpoint_2.1/](Final_Capstone_Project/Capstone_Checkpoint_2.1/).

Place the corpus under [Final_Capstone_Project/Capstone_Database/Wikipedia/](Final_Capstone_Project/Capstone_Database/Wikipedia/) and keep the OpenRouter key in the project `.env` file.

### Capstone Checkpoint 3.1
**Evaluation infrastructure and baseline diagnosis.** This checkpoint evaluates retrieval quality with RAGAS and compares original questions to paraphrased variants to detect robustness issues. The implementation is in [Final_Capstone_Project/Capstone_Checkpoint_3.1/](Final_Capstone_Project/Capstone_Checkpoint_3.1/).

#### What Checkpoint 3.1 evaluates
The solution in [Final_Capstone_Project/Capstone_Checkpoint_3.1/MHERRERA_Capstone_Checkpoint_3_1_Solution.py](Final_Capstone_Project/Capstone_Checkpoint_3.1/MHERRERA_Capstone_Checkpoint_3_1_Solution.py) evaluates the hybrid retriever using a RAGAS correctness judge and compares original vs. paraphrased performance.

- Loads the API key from the environment or root `.env`.
- Uses the persisted ChromaDB when available; otherwise it scans the HTML corpus.
- Combines BM25 and vector retrieval, then scores original and paraphrase sets separately.
- Measures pass rate and delta: `paraphrase rate - original rate`.
- Verifies the judge with a manipulated-answer control test.

#### First-time setup
```powershell
python .\Final_Capstone_Project\Utility_Scripts\Setup.py
```

On macOS or Linux:

```bash
python3 ./Final_Capstone_Project/Utility_Scripts/Setup.py
```

Create a root `.env` file:

```dotenv
OPENROUTER_API_KEY=sk-or-your-key-here
```

Place the Wikipedia HTML corpus in:

```text
Final_Capstone_Project/Capstone_Database/Wikipedia/
```

#### Required input files
The committed files are under [Final_Capstone_Project/Capstone_Checkpoint_3.1/test_variables/](Final_Capstone_Project/Capstone_Checkpoint_3.1/test_variables/):

- [test_main_questions.json](Final_Capstone_Project/Capstone_Checkpoint_3.1/test_variables/test_main_questions.json)
- [testinputs_variant_questions.json](Final_Capstone_Project/Capstone_Checkpoint_3.1/test_variables/testinputs_variant_questions.json)

If missing, generate them in order:

```powershell
cd .\Final_Capstone_Project\Capstone_Checkpoint_3.1\test_variables
..\..\..\.venv\Scripts\python.exe .\generate_main_questions.py 100 .\test_main_questions.json
..\..\..\.venv\Scripts\python.exe .\generate_variants.py .\test_main_questions.json 2 .\testinputs_variant_questions.json
```

#### Run the Checkpoint 3.1 solution
```powershell
.\.venv\Scripts\python.exe .\Final_Capstone_Project\Capstone_Checkpoint_3.1\MHERRERA_Capstone_Checkpoint_3_1_Solution.py
```

The script evaluates both datasets in one run and prints the original rate, paraphrase rate, delta, verdict, and validation result.

#### Outputs
Each run writes or appends results under [Final_Capstone_Project/Capstone_Checkpoint_3.1/](Final_Capstone_Project/Capstone_Checkpoint_3.1/):

- [detailed_test_results.log](Final_Capstone_Project/Capstone_Checkpoint_3.1/detailed_test_results.log)
- [ragas_experiments_3_1/datasets/wiki_eval_originals.csv](Final_Capstone_Project/Capstone_Checkpoint_3.1/ragas_experiments_3_1/datasets/wiki_eval_originals.csv)
- [ragas_experiments_3_1/datasets/wiki_eval_paraphrases.csv](Final_Capstone_Project/Capstone_Checkpoint_3.1/ragas_experiments_3_1/datasets/wiki_eval_paraphrases.csv)
- [ragas_experiments_3_1/experiments/](Final_Capstone_Project/Capstone_Checkpoint_3.1/ragas_experiments_3_1/experiments/)

#### Interpreting the result
A delta below `-5%` indicates brittleness to rephrasing; a delta above `+5%` suggests the paraphrases scored unusually high and should be reviewed. Otherwise, the system is treated as robust.

#### Common problems
- `OPENROUTER_API_KEY is not set`
- Originals or variants file missing
- Wikipedia corpus not found
- ChromaDB load or embedding error
- Empty datasets or API rate limit issues

### Capstone Checkpoint 4.1
**Advanced retrieval and evaluation harness.** This checkpoint combines vector, graph, BM25 lexical, and hybrid retrieval strategies in an interactive evaluation workflow. The solution is in [Final_Capstone_Project/Capstone_Checkpoint_4.1/MHERRERA_Capstone_Checkpoint_4_1_Solution.py](Final_Capstone_Project/Capstone_Checkpoint_4.1/MHERRERA_Capstone_Checkpoint_4_1_Solution.py).

It runs local preflight setup on startup and invokes [Setup.py](Final_Capstone_Project/Utility_Scripts/Setup.py) with `--build` as needed. The build uses the Wikipedia HTML corpus and creates or reuses JSONL, ChromaDB, GraphDB, and BM25 artifacts. The project also includes a small `ragas` import shim in [Final_Capstone_Project/Utility_Scripts/ragas_vertexai_shim.py](Final_Capstone_Project/Utility_Scripts/ragas_vertexai_shim.py) to avoid an upstream import issue.

Run it from the repository root:

```powershell
.\.venv\Scripts\python.exe .\Final_Capstone_Project\Capstone_Checkpoint_4.1\MHERRERA_Capstone_Checkpoint_4_1_Solution.py
```

#### Testing results and analysis
Checkpoint 4.1 records one result row per evaluated question in [Final_Capstone_Project/Ragas_Experiments/experiments/](Final_Capstone_Project/Ragas_Experiments/experiments/). Across all logged evaluations, `78/120` questions passed (`65%`).

| Retriever | Original questions | Paraphrased questions | Combined |
| --- | ---: | ---: | ---: |
| Lexical | 6/8 (75%) | 9/16 (56%) | 15/24 (63%) |
| Semantic | 6/8 (75%) | 11/16 (69%) | 17/24 (71%) |
| Hybrid | 6/8 (75%) | 11/16 (69%) | 17/24 (71%) |
| Graph | 5/8 (62%) | 10/16 (62%) | 15/24 (63%) |
| All | 5/8 (62%) | 9/16 (56%) | 14/24 (58%) |
| **All runs** | **28/40 (70%)** | **50/80 (62.5%)** | **78/120 (65%)** |

Category summary:

| Evaluation category | Passed | Total | Pass rate |
| --- | ---: | ---: | ---: |
| `factual_retrieval` | 26 | 30 | 87% |
| `obscure_knowledge` | 15 | 15 | 100% |
| `multi_fact` | 26 | 30 | 87% |
| `cross_document_synthesis` | 3 | 15 | 20% |
| `quotation_fidelity` | 0 | 15 | 0% |
| `out_of_corpus_abstention` | 8 | 15 | 53% |
| **Overall** | **78** | **120** | **65%** |

The log indicates a `7.5` percentage-point drop from original to paraphrased questions, with the biggest weakness in `cross_document_synthesis` and `quotation_fidelity`.

#### ChromaDB token and cost estimate
| Measure | Current workspace | Estimate or formula |
| --- | ---: | --- |
| JSONL files | 2,419 | Files read by the Chroma builder |
| JSONL chunks/records | 159,301 | Records embedded |
| Input tokens | 41,923,588 | Exact tokenizer count of each `text` field |
| Embedding batches | 3,187 | `ceil(159,301 / 50)` |
| Assumed input rate | $0.02 / 1M tokens | Pricing assumption |
| Estimated first-build embedding cost | **$0.84** | `41,923,588 / 1,000,000 * $0.02` |
| Existing valid ChromaDB | **$0.00** | Reused when valid |

## Setup and Local Data
Run the setup utility from the repository root:

```powershell
python .\Final_Capstone_Project\Utility_Scripts\Setup.py
```

On macOS or Linux:

```bash
python3 ./Final_Capstone_Project/Utility_Scripts/Setup.py
```

Optional build commands:

```powershell
python .\Final_Capstone_Project\Utility_Scripts\Setup.py --build
python .\Final_Capstone_Project\Utility_Scripts\Setup.py --build --rebuild
```

```bash
python3 ./Final_Capstone_Project/Utility_Scripts/Setup.py --build
python3 ./Final_Capstone_Project/Utility_Scripts/Setup.py --build --rebuild
```

Setup creates or verifies the following local paths:
- [Final_Capstone_Project/Capstone_Database/Wikipedia/](Final_Capstone_Project/Capstone_Database/Wikipedia/)
- [Final_Capstone_Project/Capstone_Database/Wikipedia_JSONL/](Final_Capstone_Project/Capstone_Database/Wikipedia_JSONL/)
- [Final_Capstone_Project/Capstone_Database/Capstone_Chroma_DB/](Final_Capstone_Project/Capstone_Database/Capstone_Chroma_DB/)
- [Final_Capstone_Project/Capstone_Database/Capstone_Graph_DB/](Final_Capstone_Project/Capstone_Database/Capstone_Graph_DB/)
- [Final_Capstone_Project/Capstone_Database/Capstone_BM25_Lexical_Indexes/](Final_Capstone_Project/Capstone_Database/Capstone_BM25_Lexical_Indexes/)
- [Final_Capstone_Project/Utility_Scripts/Logs/](Final_Capstone_Project/Utility_Scripts/Logs/)

The workspace currently includes the Wikipedia HTML and JSONL corpus, question files, and evaluation outputs. Generated databases and local runtime artifacts are created on first setup and are not always committed to version control.

## Key Project Areas
- [Final_Capstone_Project/Capstone_Checkpoint_1.1/](Final_Capstone_Project/Capstone_Checkpoint_1.1/) contains the checkpoint 1.1 solution and supporting artifacts.
- [Final_Capstone_Project/Capstone_Checkpoint_2.1/](Final_Capstone_Project/Capstone_Checkpoint_2.1/) contains the retrieval strategy and baseline implementation files.
- [Final_Capstone_Project/Capstone_Checkpoint_3.1/](Final_Capstone_Project/Capstone_Checkpoint_3.1/) contains the evaluation harness, validation utilities, datasets, and experiment results.
- [Final_Capstone_Project/Capstone_Checkpoint_4.1/](Final_Capstone_Project/Capstone_Checkpoint_4.1/) contains the advanced retrieval starter and final solution files.
- [Final_Capstone_Project/Capstone_Database/](Final_Capstone_Project/Capstone_Database/) stores the local corpus and database artifacts used for retrieval.
- [Final_Capstone_Project/Retrieval_Methods/](Final_Capstone_Project/Retrieval_Methods/) contains BM25, vector, and hybrid retrieval logic.
- [Final_Capstone_Project/Ranking_Techniques/](Final_Capstone_Project/Ranking_Techniques/) contains ranking and fusion logic.
- [Final_Capstone_Project/Ragas_Experiments/](Final_Capstone_Project/Ragas_Experiments/) stores evaluation logic and experiment outputs.
- `lab_*` directories contain the course lab scripts, starter files, and requirements for guided work.

## Notes
- The project is designed for a local Python environment and project-specific data directories.
- Large corpus and database artifacts are often stored locally rather than committed to version control.
- This README reflects the current workspace structure and the major setup and evaluation workflows.

--------------------------
