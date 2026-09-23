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
| Checkpoint 5.1 | Dual-engine chat and agentic evaluation | [Checkpoint 5.1](#capstone-checkpoint-51) |
| Checkpoint 5.1 Test Summary | Per-engine and comparison evaluation results | [detailed_test_results_agentic.log](Final_Capstone_Project/Ragas_Experiments/detailed_test_results_agentic.log) |
| Checkpoint 4.1 | Advanced retrieval and evaluation harness | [Checkpoint 4.1](#capstone-checkpoint-41) |
| Checkpoint 4.1 Testing | Category-based testing results and analysis | [Testing results and analysis](#testing-results-and-analysis) |
| Checkpoint 4.1 Test Summary | Human-readable summary of Checkpoint 4.1 test runs | [detailed_test_results.log](Final_Capstone_Project/Ragas_Experiments/detailed_test_results.log) |
| Checkpoint 3.1 | RAGAS evaluation and paraphrase robustness | [Checkpoint 3.1](#capstone-checkpoint-31) |
| Checkpoint 2.1 | Retrieval strategy and baseline implementation | [Checkpoint 2.1](#capstone-checkpoint-21) |
| Checkpoint 1.1 | Evaluating when retrieval is required | [Checkpoint 1.1](#capstone-checkpoint-11) |
| Setup and Local Data | Bootstrap commands and generated local data | [Setup](#setup-and-local-data) |
| ChromaDB Cost Estimate | Token count and estimated embedding cost | [Cost Estimate](#chromadb-token-and-cost-estimate) |
| Key Project Areas | Source modules and supporting project areas | [Project Areas](#key-project-areas) |

## Capstone Checkpoints

Checkpoints are listed newest first.

### Capstone Checkpoint 5.1
**Agentic tool-using retrieval with a dual-engine, comparison-ready workflow.** This checkpoint adds an interactive solution that lets the user choose a retrieval engine and either chat with it or score it through RAGAS. The solution is in [Final_Capstone_Project/Capstone_Checkpoint_5.1/MHERRERA_Capstone_Checkpoint_5_1_Agent_Solution.py](Final_Capstone_Project/Capstone_Checkpoint_5.1/MHERRERA_Capstone_Checkpoint_5_1_Agent_Solution.py).

#### Retrieval engines
- **Context-Aware Retriever** — a history-aware, non-agentic retriever. It folds recent conversation turns into both the retrieval query and the grounded answer prompt, but performs no autonomous planning.
- **Agentic Dynamic Retriever** — a LangGraph ReAct agent that dynamically selects one action per step (`plan → retrieve / graph_expand / clarify / answer`) with no upfront plan. Graph expansion is enabled only for the graph-enabled search method; otherwise the agent runs graph-free.

#### Search methods
The interactive menu exposes four base search methods, reused by both engines:

- **Lexical (Only)** — BM25 exact-term retrieval.
- **Semantic (Only)** — Chroma dense-vector retrieval.
- **Hybrid** — weighted BM25 + vector fusion.
- **Hybrid with Graph Enabled** — hybrid retrieval plus Graph DB expansion of neighboring and linked article context.

#### Modes
- **Interactive Chat** — multi-turn conversation with history, printing per-turn token usage (plan vs answer) and a cumulative session total. Token counts are read from the provider when reported (OpenRouter model named) or estimated locally with a labeled tokenizer.
- **RAGAS Evaluation** — scores the selected engine with the DiscreteMetric correctness judge, over originals, paraphrases, or both.
- **Compare Both** (evaluation only) — runs both engines on the same dataset and logs a side-by-side comparison (correctness delta, LLM call/token cost, passes-per-1K-tokens, and category deltas).

#### Configuration
Behavior is driven by [Final_Capstone_Project/retrieval.conf](Final_Capstone_Project/retrieval.conf): the `[agent]` section sets the agent reasoning model (`agent_model`) and limits (`top_k`, `seed_k`, `max_iterations`, `debug`), `[context_aware]` sets the chat `history_window`, and `[ranking]` sets the BM25/vector fusion weights recorded in each result entry.

#### Run the Checkpoint 5.1 solution
```powershell
.\.venv\Scripts\python.exe .\Final_Capstone_Project\Capstone_Checkpoint_5.1\MHERRERA_Capstone_Checkpoint_5_1_Agent_Solution.py
```

The script runs local preflight setup on startup, then prompts for mode, engine, and (for evaluation) dataset, search method, size, and sampling.

#### Outputs
All Checkpoint 5.1 evaluation results are written newest-first to [Final_Capstone_Project/Ragas_Experiments/detailed_test_results_agentic.log](Final_Capstone_Project/Ragas_Experiments/detailed_test_results_agentic.log), with per-question CSVs under [Final_Capstone_Project/Ragas_Experiments/experiments/](Final_Capstone_Project/Ragas_Experiments/experiments/). Each entry records the engine, search method, fusion-ranking weights, result, and LLM call/token effort; comparison runs add a side-by-side section.

#### Testing results and analysis
Results below cover all eight logged "Compare Both" runs — two batches of four search methods — each scoring both engines on the same 16 paraphrased manual questions (RAGAS DiscreteMetric; answer/judge `openai/gpt-5.4-mini`). Batch A used the default fusion weights (BM25 0.5 / Vector 0.5); Batch B used tuned weights (BM25 0.4 / Vector 0.6). `Δ` is agentic correctness minus context-aware correctness.

**Correctness by search method (both batches):**

| Search method | Batch | Context-Aware | Agentic Dynamic | Δ (agent − context) |
| --- | --- | ---: | ---: | ---: |
| Lexical | A (0.5/0.5) | 11/16 (69%) | 15/16 (94%) | +25% |
| Lexical | B (0.4/0.6) | 11/16 (69%) | 13/16 (81%) | +12% |
| Semantic | A (0.5/0.5) | 13/16 (81%) | 4/16 (25%) | −56% |
| Semantic | B (0.4/0.6) | 11/16 (69%) | 7/16 (44%) | −25% |
| Hybrid | A (0.5/0.5) | 10/16 (62%) | 14/16 (88%) | +25% |
| Hybrid | B (0.4/0.6) | 10/16 (62%) | 15/16 (94%) | +31% |
| Hybrid + Graph | A (0.5/0.5) | 9/16 (56%) | 14/16 (88%) | +31% |
| Hybrid + Graph | B (0.4/0.6) | 9/16 (56%) | 14/16 (88%) | +31% |

**Model-call and token/cost comparison (per 16-question run):**

Both engines make exactly 16 answer calls (one per question). The difference is the agent's planner: it makes one extra model call per reasoning step, so total calls and tokens scale with how much the agent iterates. Cost tracks tokens directly (billing is per token), so the token multiplier is the cost multiplier at a fixed model price.

| Search method | Batch | Context calls | Context tokens | Agent calls (plan + answer) | Agent tokens | Token cost multiple (agent ÷ context) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Lexical | A | 16 (0 + 16) | 18.3K | 48 (32 + 16) | 99.7K | 5.4x |
| Lexical | B | 16 (0 + 16) | 18.3K | 47 (31 + 16) | 92.6K | 5.1x |
| Semantic | A | 16 (0 + 16) | 18.5K | 69 (53 + 16) | 35.6K | 1.9x |
| Semantic | B | 16 (0 + 16) | 18.5K | 71 (55 + 16) | 38.2K | 2.1x |
| Hybrid | A | 16 (0 + 16) | 18.3K | 49 (33 + 16) | 104.7K | 5.7x |
| Hybrid | B | 16 (0 + 16) | 18.2K | 48 (32 + 16) | 93.0K | 5.1x |
| Hybrid + Graph | A | 16 (0 + 16) | 35.4K | 48 (32 + 16) | 94.0K | 2.7x |
| Hybrid + Graph | B | 16 (0 + 16) | 35.4K | 48 (32 + 16) | 100.1K | 2.8x |

Efficiency as correctness per 1K tokens (higher is cheaper per correct answer):

| Engine | Passes per 1K tokens (range across runs) |
| --- | ---: |
| Context-Aware | ~0.25–0.70 |
| Agentic Dynamic | ~0.11–0.18 |

The agent makes roughly **3–4.5x the model calls** and **2–5.7x the tokens** of the context-aware engine for the same 16 questions, and is about **3–5x less cost-efficient** per correct answer in every configuration. Note the Semantic rows: the agent's low token multiple there (1.9–2.1x) is not a saving — it reflects short, low-context planning loops that also produced its worst accuracy, so it spent more calls to retrieve less useful context.

#### Iterative retrieval vs. the fixed pipeline
The two engines differ structurally in how retrieval happens, which explains the accuracy/cost split above:

- **Context-Aware = a fixed, single-pass pipeline.** For each question it runs exactly one retrieve → answer cycle: build one search query (folding in recent conversation), fetch the top-k chunks once, and generate the answer. The retrieval depth is fixed regardless of question difficulty, which is why its cost is flat (16 calls, ~18–35K tokens) and predictable.
- **Agentic Dynamic = an iterative, plan-driven loop.** A plan node decides after each step whether to `retrieve` again (with new sub-queries), `graph_expand`, `clarify`, or `answer` — so a hard question can trigger several retrieval rounds while an easy one answers quickly. This adaptive depth is what lifts `cross_document_synthesis` (0/2 → 2/2) and `multi_fact` (3/4 → 4/4): the agent re-queries to assemble facts a single pass misses. It is also the direct cause of the higher cost (the 31–55 plan calls above) and of the Semantic failure mode, where the loop iterated without a lexical anchor and drifted off-target.

In short, the fixed pipeline trades recall on multi-step questions for flat, low cost and stable behavior; the agent trades cost and predictability for adaptive, multi-round retrieval that wins on synthesis-heavy questions but can spiral on weak retrieval signals.

Category pass rates for the strongest agentic configuration (Hybrid, Batch B, agentic 15/16):

| Evaluation category | Context-Aware | Agentic Dynamic |
| --- | ---: | ---: |
| `factual_retrieval` | 4/4 (100%) | 4/4 (100%) |
| `multi_fact` | 3/4 (75%) | 4/4 (100%) |
| `cross_document_synthesis` | 0/2 (0%) | 2/2 (100%) |
| `obscure_knowledge` | 2/2 (100%) | 2/2 (100%) |
| `out_of_corpus_abstention` | 1/2 (50%) | 2/2 (100%) |
| `quotation_fidelity` | 0/2 (0%) | 1/2 (50%) |

Observations:
- The Agentic Dynamic engine improved correctness on Lexical, Hybrid, and Hybrid + Graph in both batches (deltas of +12% to +31%), with its biggest gains on `cross_document_synthesis` and `multi_fact`.
- Pure Semantic is the agent's clear failure mode: it dropped to 4/16 (Batch A) and 7/16 (Batch B), well below the context-aware baseline — the planning loop iterated without the lexical signal to anchor it.
- Cost is the decisive tradeoff (see the model-call/token table above): choose the agent when accuracy on synthesis/multi-fact questions matters more than cost, and prefer the fixed context-aware pipeline for Semantic-only retrieval or cost-sensitive runs.
- `quotation_fidelity` (verbatim opening-sentence questions) stayed weak across both engines, batches, and methods, so it is a corpus/prompt limitation rather than an engine choice.

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

### Capstone Checkpoint 2.1
**Retrieval strategy design and baseline implementation.** This checkpoint builds a Wikipedia RAG workflow using ChromaDB vector retrieval and BM25 lexical retrieval. The solution is in [Final_Capstone_Project/Capstone_Checkpoint_2.1/](Final_Capstone_Project/Capstone_Checkpoint_2.1/).

Place the corpus under [Final_Capstone_Project/Capstone_Database/Wikipedia/](Final_Capstone_Project/Capstone_Database/Wikipedia/) and keep the OpenRouter key in the project `.env` file.

### Capstone Checkpoint 1.1
**Evaluating when retrieval is required.** This checkpoint measures baseline LLM performance without retrieval and determines whether retrieval is needed for the Wikipedia scenario. The solution is in [Final_Capstone_Project/Capstone_Checkpoint_1.1/](Final_Capstone_Project/Capstone_Checkpoint_1.1/).

Add `OPENROUTER_API_KEY` to the root `.env` file before running the solution.

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
- [Final_Capstone_Project/Capstone_Checkpoint_5.1/](Final_Capstone_Project/Capstone_Checkpoint_5.1/) contains the dual-engine (Context-Aware and Agentic Dynamic) chat and evaluation solution.
- [Final_Capstone_Project/Capstone_Checkpoint_4.1/](Final_Capstone_Project/Capstone_Checkpoint_4.1/) contains the advanced retrieval starter and final solution files.
- [Final_Capstone_Project/Capstone_Checkpoint_3.1/](Final_Capstone_Project/Capstone_Checkpoint_3.1/) contains the evaluation harness, validation utilities, datasets, and experiment results.
- [Final_Capstone_Project/Capstone_Checkpoint_2.1/](Final_Capstone_Project/Capstone_Checkpoint_2.1/) contains the retrieval strategy and baseline implementation files.
- [Final_Capstone_Project/Capstone_Checkpoint_1.1/](Final_Capstone_Project/Capstone_Checkpoint_1.1/) contains the checkpoint 1.1 solution and supporting artifacts.
- [Final_Capstone_Project/Capstone_Database/](Final_Capstone_Project/Capstone_Database/) stores the local corpus and database artifacts used for retrieval.
- [Final_Capstone_Project/Retrieval_Methods/](Final_Capstone_Project/Retrieval_Methods/) contains BM25, vector, hybrid, context-aware, and agentic retrieval logic.
- [Final_Capstone_Project/Ranking_Techniques/](Final_Capstone_Project/Ranking_Techniques/) contains ranking and fusion logic.
- [Final_Capstone_Project/Utility_Scripts/](Final_Capstone_Project/Utility_Scripts/) contains shared setup, evaluation, dataset-loading, menu, token-usage, and report-writing helpers.
- [Final_Capstone_Project/Ragas_Experiments/](Final_Capstone_Project/Ragas_Experiments/) stores evaluation logic and experiment outputs.
- `lab_*` directories contain the course lab scripts, starter files, and requirements for guided work.

## Notes
- The project is designed for a local Python environment and project-specific data directories.
- Large corpus and database artifacts are often stored locally rather than committed to version control.
- This README reflects the current workspace structure and the major setup and evaluation workflows.

--------------------------
