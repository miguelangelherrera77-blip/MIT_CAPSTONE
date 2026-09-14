# CAPSTONE PROJECT
RAG and Context Engineering: Designing and Building Production-Grade AI Systems

Miguel Herrera - Section B

## Project Overview
This repository contains the capstone RAG project for a Wikipedia Retrieval Engine scenario. It includes the main project implementation, checkpoint solution files, evaluation utilities, local database directories, course labs, and supporting scripts used for retrieval, ranking, and analysis.

## Project System Requirements

- **Operating system:** Windows, macOS, or Linux. `Setup.py` detects the host platform and creates the virtual environment using the appropriate `Scripts` or `bin` layout.
- **Python:** Python 3.10 or newer. Python 3.14 is supported by the current setup and logging code.
- **Python tooling:** `venv` and `pip` must be available in the system Python installation so Setup.py can create `.venv` and install [venv_requirements.txt](venv_requirements.txt).
- **Internet access:** Required during first-time dependency installation. It is also required for OpenRouter API calls used by embeddings, answer generation, paraphrase generation, and RAGAS judging.
- **OpenRouter credentials:** An `OPENROUTER_API_KEY` is required for AI-backed features and ChromaDB embedding creation. Store it in the root `.env` file or provide it through the process environment.
- **Local storage:** Sufficient free disk space is required for the `.venv`, Wikipedia HTML or JSONL corpus, generated JSONL files, ChromaDB, GraphDB, BM25 indexes, and logs. Storage needs grow with corpus size.
- **Corpus input:** HTML files are required for HTML chunking and GraphDB creation. Existing JSONL files are sufficient for ChromaDB and BM25 creation. No corpus is required to create the initial runtime scaffolding.
- **Terminal access:** Setup.py should be run from the repository root so it can locate the requirements file, `.env`, and project directories.

## Quick Links

| Section | Description | Link |
| --- | --- | --- |
| Project Overview | Repository purpose and system context | [Overview](#project-overview) |
| System Requirements | Host, Python, and runtime prerequisites | [Requirements](#project-system-requirements) |
| Checkpoint 1.1 | Evaluating when retrieval is required | [Checkpoint 1.1](#capstone-checkpoint-11) |
| Checkpoint 2.1 | Retrieval strategy and baseline implementation | [Checkpoint 2.1](#capstone-checkpoint-21) |
| Checkpoint 3.1 | RAGAS evaluation and paraphrase robustness | [Checkpoint 3.1](#capstone-checkpoint-31) |
| Checkpoint 4.1 | Advanced retrieval and evaluation harness | [Checkpoint 4.1](#capstone-checkpoint-41) |
| Setup and Local Data | Bootstrap commands and generated local data | [Setup](#setup-and-local-data) |
| Key Project Areas | Source modules and supporting project areas | [Project Areas](#key-project-areas) |

## Capstone Checkpoints

### Capstone Checkpoint 1.1
**Evaluating when retrieval is required.** This checkpoint evaluates how an LLM performs without retrieval and determines whether retrieval is required for the selected Wikipedia scenario. The solution is in [Final_Capstone_Project/Capstone_Checkpoint_1.1/](Final_Capstone_Project/Capstone_Checkpoint_1.1/).

Add `OPENROUTER_API_KEY` to the root `.env` file before running the solution. The checkpoint uses `python-dotenv`, `langchain-openai`, and `langchain-core`.

### Capstone Checkpoint 2.1
**Retrieval strategy design and baseline implementation.** This checkpoint implements Retrieval-Augmented Generation over the Wikipedia corpus using vector retrieval through ChromaDB and lexical retrieval through BM25. The solution is in [Final_Capstone_Project/Capstone_Checkpoint_2.1/](Final_Capstone_Project/Capstone_Checkpoint_2.1/).

Place the Wikipedia HTML corpus in [Final_Capstone_Project/Capstone_Database/Wikipedia/](Final_Capstone_Project/Capstone_Database/Wikipedia/) and add `OPENROUTER_API_KEY` to [.env](.env). The required packages are listed in [venv_requirements.txt](venv_requirements.txt).

### Capstone Checkpoint 3.1
**Evaluation infrastructure and baseline diagnosis.** This checkpoint evaluates the retrieval system with RAGAS and compares original questions with paraphrased variants to measure robustness to rephrasing. The solution and validation utilities are in [Final_Capstone_Project/Capstone_Checkpoint_3.1/](Final_Capstone_Project/Capstone_Checkpoint_3.1/).

#### What Checkpoint 3.1 evaluates
The solution in [Final_Capstone_Project/Capstone_Checkpoint_3.1/MHERRERA_Capstone_Checkpoint_3_1_Solution.py](Final_Capstone_Project/Capstone_Checkpoint_3.1/MHERRERA_Capstone_Checkpoint_3_1_Solution.py) evaluates the Checkpoint 2.1 hybrid retriever with a RAGAS `DiscreteMetric` correctness judge. It performs the following sequence:

1. Loads `OPENROUTER_API_KEY` from the process environment or the root `.env` file.
2. Loads documents from the persisted Chroma database at [Final_Capstone_Project/Capstone_Database/Capstone_Chroma_DB/](Final_Capstone_Project/Capstone_Database/Capstone_Chroma_DB/) when it contains data. If Chroma is unavailable or cannot be read, it scans the Wikipedia HTML corpus at [Final_Capstone_Project/Capstone_Database/Wikipedia/](Final_Capstone_Project/Capstone_Database/Wikipedia/).
3. Builds the hybrid retriever. BM25 provides lexical candidates and Chroma provides semantic candidates. Their normalized scores are fused with equal weights (`0.5` BM25 and `0.5` vector), using a candidate pool of `10` and returning the top `4` documents.
4. Loads the original questions from [test_variables/test_main_questions.json](Final_Capstone_Project/Capstone_Checkpoint_3.1/test_variables/test_main_questions.json).
5. Loads the combined original-plus-paraphrase file from [test_variables/testinputs_variant_questions.json](Final_Capstone_Project/Capstone_Checkpoint_3.1/test_variables/testinputs_variant_questions.json), then separates paraphrase rows by excluding questions that also appear in the originals file.
6. Answers each question with the configured OpenRouter answer model and scores the answer against its `grading_notes` with the configured RAGAS judge model.
7. Evaluates originals and paraphrases separately, reports pass rates and the delta (`paraphrase rate - original rate`), and classifies the retriever as robust or brittle to rephrasing.
8. Runs a manipulated-answer probe to verify that the judge accepts a correct control answer and rejects a deliberately false answer.

The answer model, judge model, and embedding model are currently `openai/gpt-5.4-mini`, `openai/gpt-5.4-mini`, and `openai/text-embedding-3-small`, accessed through OpenRouter at `https://openrouter.ai/api/v1`. The answer temperature is `0.2`.

#### First-time setup
Run `Setup.py` from the repository root. It creates the host-specific `.venv`, installs [venv_requirements.txt](venv_requirements.txt) when the environment is new, creates the local runtime directories, and appends generated runtime paths to the local ignore file.

```powershell
python .\Final_Capstone_Project\Utility_Scripts\Setup.py
```

On macOS or Linux:

```bash
python3 ./Final_Capstone_Project/Utility_Scripts/Setup.py
```

Create a root `.env` file containing your own key:

```dotenv
OPENROUTER_API_KEY=sk-or-your-key-here
```

Do not commit `.env` or expose the key in source control. The key is required for embeddings, answer generation, paraphrase generation, and RAGAS judging.

Place the Wikipedia HTML files in:

```text
Final_Capstone_Project/Capstone_Database/Wikipedia/
```

The solution can read an existing Chroma database, but a complete fresh setup should prepare the local directories first. The command below is only needed if Setup.py has not already been run:

```powershell
python .\Final_Capstone_Project\Utility_Scripts\Setup.py
```

On macOS or Linux:

```bash
python3 ./Final_Capstone_Project/Utility_Scripts/Setup.py
```

For Checkpoint 3.1, the HTML corpus is the important source prerequisite. The 3.1 solution can scan that corpus directly and can create a Chroma database on first use if no persisted Chroma data is available. Generated databases and logs are local runtime artifacts.

#### Required input files
The standard committed question files are already under [Final_Capstone_Project/Capstone_Checkpoint_3.1/test_variables/](Final_Capstone_Project/Capstone_Checkpoint_3.1/test_variables/):

- [test_main_questions.json](Final_Capstone_Project/Capstone_Checkpoint_3.1/test_variables/test_main_questions.json): original questions with `question`, `grading_notes`, and `sources` fields.
- [testinputs_variant_questions.json](Final_Capstone_Project/Capstone_Checkpoint_3.1/test_variables/testinputs_variant_questions.json): the combined file containing originals and paraphrased questions. Each paraphrase must retain the original `grading_notes` and `sources`.

If either file is missing, the evaluator stops with a file-not-found error. Generate the files in this order:

1. Generate grounded originals from the HTML corpus. The first argument is the requested number of questions; the second optional argument is the output path.

```powershell
cd .\Final_Capstone_Project\Capstone_Checkpoint_3.1\test_variables
..\..\..\.venv\Scripts\python.exe .\generate_main_questions.py 100 .\test_main_questions.json
```

The generator samples candidate articles with a fixed random seed, excludes the four Checkpoint 2.1 articles, asks `openai/gpt-5.4-mini` for one grounded question per article, and writes the grading notes and source filename. It requires the HTML corpus and the API key.

2. Generate paraphrases from the originals. The arguments are input JSON, number of paraphrases per question, and optional output JSON.

```powershell
..\..\..\.venv\Scripts\python.exe .\generate_variants.py .\test_main_questions.json 2 .\testinputs_variant_questions.json
```

This produces up to two paraphrases per original, preserves the original rows, and writes one combined file. Review the generated paraphrases before evaluation and remove any that change the meaning, add ambiguity, or duplicate another question.

#### Run the Checkpoint 3.1 solution
The main solution has no required command-line arguments. Run it from the repository root so the relative environment and project paths are unambiguous:

```powershell
.\.venv\Scripts\python.exe .\Final_Capstone_Project\Capstone_Checkpoint_3.1\MHERRERA_Capstone_Checkpoint_3_1_Solution.py
```

The script prints progress for document loading and each evaluated question. It evaluates both datasets in one run; it does not provide a dataset-selection CLI option. At the end it prints the original pass rate, paraphrase pass rate, delta, robustness verdict, and framework-validation result.

#### Outputs
Each run writes or appends the following files under [Final_Capstone_Project/Capstone_Checkpoint_3.1/](Final_Capstone_Project/Capstone_Checkpoint_3.1/):

- [detailed_test_results.log](Final_Capstone_Project/Capstone_Checkpoint_3.1/detailed_test_results.log): structured comparison and framework-validation messages, failures, CSV paths, pass rates, delta, verdict, and manipulated-answer probe.
- [ragas_experiments_3_1/datasets/wiki_eval_originals.csv](Final_Capstone_Project/Capstone_Checkpoint_3.1/ragas_experiments_3_1/datasets/wiki_eval_originals.csv): RAGAS local dataset for original questions.
- [ragas_experiments_3_1/datasets/wiki_eval_paraphrases.csv](Final_Capstone_Project/Capstone_Checkpoint_3.1/ragas_experiments_3_1/datasets/wiki_eval_paraphrases.csv): RAGAS local dataset for paraphrase questions.
- [ragas_experiments_3_1/experiments/](Final_Capstone_Project/Capstone_Checkpoint_3.1/ragas_experiments_3_1/experiments/): scored experiment CSV results. Filenames are generated by RAGAS for each run.
- [Final_Capstone_Project/Capstone_Database/Capstone_Chroma_DB/](Final_Capstone_Project/Capstone_Database/Capstone_Chroma_DB/): persisted embeddings and Chroma data when the solution builds or uses the vector database.

The CSV results include the question, grading notes, retrieved response, retriever label, and RAGAS verdict. Existing logs are appended rather than replaced.

#### Interpreting the result
The final comparison is calculated as:

```text
delta = paraphrase pass rate - original pass rate
```

A delta below `-5%` is reported as brittle to rephrasing. A delta above `+5%` is reported as paraphrases scoring higher and should be reviewed for lucky wording or judge leniency. Otherwise, the retriever is reported as robust because the pass rates are comparable. A failed manipulated-answer probe indicates that the evaluation judge configuration should be investigated before trusting the aggregate result.

#### Common problems
- `OPENROUTER_API_KEY is not set`: create the root `.env` file or set the environment variable in the active shell.
- `Originals not found`: create [test_variables/test_main_questions.json](Final_Capstone_Project/Capstone_Checkpoint_3.1/test_variables/test_main_questions.json) or run [generate_main_questions.py](Final_Capstone_Project/Capstone_Checkpoint_3.1/test_variables/generate_main_questions.py).
- `Variants file not found`: run [generate_variants.py](Final_Capstone_Project/Capstone_Checkpoint_3.1/test_variables/generate_variants.py) after the originals file exists.
- `Wikipedia directory not found`: place the HTML corpus under [Final_Capstone_Project/Capstone_Database/Wikipedia/](Final_Capstone_Project/Capstone_Database/Wikipedia/).
- Chroma load or embedding errors: verify the active virtual environment, `langchain-chroma`, `langchain-openai`, `chromadb`, and the OpenRouter key; remove only a corrupted local Chroma directory before rebuilding it.
- Empty datasets: inspect the JSON files and ensure each row contains non-empty `question` and `grading_notes` values.
- API rate limits or timeout errors: reduce the number of generated questions or paraphrases, retry later, and review partial output before rerunning.

### Capstone Checkpoint 4.1
**Advanced retrieval and evaluation harness.** This checkpoint combines persisted vector, graph, BM25 lexical, and hybrid retrieval strategies in an interactive evaluation workflow. The solution is in [Final_Capstone_Project/Capstone_Checkpoint_4.1/MHERRERA_Capstone_Checkpoint_4_1_Solution.py](Final_Capstone_Project/Capstone_Checkpoint_4.1/MHERRERA_Capstone_Checkpoint_4_1_Solution.py).

On startup, the solution runs the local preflight setup before displaying the menu. It invokes [Setup.py](Final_Capstone_Project/Utility_Scripts/Setup.py) with `--build`, which requires the Wikipedia HTML corpus under [Final_Capstone_Project/Capstone_Database/Wikipedia/](Final_Capstone_Project/Capstone_Database/Wikipedia/). The build creates or reuses the Wikipedia JSONL corpus, ChromaDB, GraphDB, and BM25 indexes, each with a separate progress stage. Existing valid generated artifacts are reused; `--rebuild` regenerates JSONL, GraphDB, and BM25 outputs when supplied directly to Setup.py.

To run the solution directly from the repository root:

```powershell
.\.venv\Scripts\python.exe .\Final_Capstone_Project\Capstone_Checkpoint_4.1\MHERRERA_Capstone_Checkpoint_4_1_Solution.py
```

Setup.py creates the host-specific `.venv`, installs missing dependencies when the venv is first created, creates missing runtime directories and a root [.env](.env) template when needed, and appends generated artifact paths to the local ignore file. Setup output is logged to [Final_Capstone_Project/Utility_Scripts/Logs/Setup.log](Final_Capstone_Project/Utility_Scripts/Logs/Setup.log). The `.env` file, generated databases, and logs are local runtime artifacts; the current repository also contains the Wikipedia HTML corpus.

The `--build` jobs have different network behavior. HTML chunking, GraphDB creation, and BM25 index creation process local files only. The ChromaDB builder sends chunks to OpenRouter for embeddings when it must create a database; an existing valid Chroma database is reused. `--build --rebuild` forces new JSONL, GraphDB, and BM25 outputs, but the Chroma builder is invoked without its rebuild action by this setup script. The Checkpoint 1.1-4.1 solutions and the Checkpoint 3.1 question generators also use OpenRouter for chat responses, embeddings, or evaluation when run.

## Setup and Local Data
Run the setup utility from the repository root. This creates `.venv`, installs the dependencies from [venv_requirements.txt](venv_requirements.txt), and creates local runtime directories.

```powershell
python .\Final_Capstone_Project\Utility_Scripts\Setup.py
```

On macOS or Linux:

```bash
python3 ./Final_Capstone_Project/Utility_Scripts/Setup.py
```

Use `--build` to generate the local Wikipedia JSONL corpus and retrieval databases after adding the HTML or JSONL corpus. Existing JSONL files are used directly by the ChromaDB and BM25 jobs. Use `--rebuild` with `--build` when regeneration is explicitly required:

```powershell
python .\Final_Capstone_Project\Utility_Scripts\Setup.py --build
python .\Final_Capstone_Project\Utility_Scripts\Setup.py --build --rebuild
```

On macOS or Linux:

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

The current workspace contains 2,419 Wikipedia HTML files and 2,419 matching JSONL files. If both corpora are absent, setup creates the runtime scaffolding but skips database generation. With JSONL files but no HTML files, setup runs ChromaDB and BM25 from JSONL, while skipping HTML chunking and GraphDB. The generated database folders must contain real artifacts before retrieval can use them; placeholder README files are only scaffolding.

ChromaDB is created only in [Final_Capstone_Project/Capstone_Database/Capstone_Chroma_DB/](Final_Capstone_Project/Capstone_Database/Capstone_Chroma_DB/). The Chroma builder rejects alternate database paths, including backup directories.

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
- The repository is intended to be used with a local Python environment and project-specific data directories.
- Large corpus and database artifacts may be stored locally and are not always intended for version control.
- This inventory reflects the current workspace contents and project structure.

--------------------------
