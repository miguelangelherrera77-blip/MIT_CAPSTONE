# Capstone Checkpoints: Setup and Run Guide

This guide covers checkpoints 1.1 through 7.1: prerequisites, one launch
command per checkpoint, available command-line options, and where to find
results. Each launch command starts at the repository root and intentionally
has **no checkpoint-specific parameters**. Use the tables to understand optional
CLI parameters; add only the options you need when running a checkpoint.

See [Testing Results](Checkpoint_Testing_Results.md) for historical scores, evaluation
methodology, model-cost comparisons, embedding budgets, and limitations.

## Before you start

- The dependency file lists package requirements but does not declare a
  project-wide minimum Python version. Offline validation used Python 3.12.10
  on Windows; other versions and operating systems are not certified.
- Run commands from the `MIT_CAPSTONE` repository root. Windows examples use
  PowerShell; macOS examples use Terminal with Zsh or Bash. Linux can use the
  macOS-style commands.
- Use the repository virtual environment. A Windows virtual environment cannot
  be reused on macOS; create a platform-local environment there.
- Set `OPENROUTER_API_KEY` in the root `.env` or process environment. Never
  commit credentials.
- Network access is needed for model calls and query embeddings. Answer
  generation, judging, and embeddings can incur charges.
- The shared corpus and indexes remain in [Capstone_Database](../Capstone_Database/).
- Checkpoint launchers and configuration are in this folder. Shared utility
  scripts, retrieval modules, and ranking techniques are in
  [Capstone_Utilities](../Capstone_Utilities/). Datasets and evaluation outputs are in the repository-root
  [Capstone_Ragas_Experiments](../Capstone_Ragas_Experiments/) folder.

## First-time setup

From the repository root, run the setup utility:

**Windows (PowerShell)**

```powershell
python .\Capstone_Utilities\Utility_Scripts\Setup.py
```

**macOS (Zsh/Bash)**

```bash
python3 ./Capstone_Utilities/Utility_Scripts/Setup.py
```

[Setup.py](../Capstone_Utilities/Utility_Scripts/Setup.py) creates a missing root `.env` template,
local directories/placeholders, and the repository `.venv`. It installs
[venv_requirements.txt](../venv_requirements.txt) only when creating that
environment; an existing environment is reused without updating dependencies.
If the environment is incomplete, install the declared requirements with pip
using the environment's Python interpreter.

**Windows (PowerShell)**

```powershell
& .\.venv\Scripts\python.exe -m pip install -r .\venv_requirements.txt
```

**macOS (Zsh/Bash)**

```bash
./.venv/bin/python -m pip install -r ./venv_requirements.txt
```

Replace the placeholder in `.env` with your real key:

```dotenv
OPENROUTER_API_KEY=your-real-openrouter-key
```

Setup does not populate the corpus or build indexes unless requested. Provide
Wikipedia HTML under [Wikipedia](../Capstone_Database/Wikipedia/) or existing
JSONL under [Wikipedia_JSONL](../Capstone_Database/Wikipedia_JSONL/). HTML is
needed for graph generation; JSONL is sufficient for BM25 and Chroma generation.
Checkpoint 1.1 does not require a corpus.

Setup build options:

| Option | Purpose | Caution |
|---|---|---|
| `--build` | Build missing or incomplete corpus/index artifacts. | Chroma creation makes embedding API calls and may incur charges. |
| `--rebuild` | With `--build`, request rebuilds of generated Chroma, Graph, and BM25 artifacts. | Can overwrite generated artifacts; confirmation prompts may appear. Do not use just to launch a checkpoint. |

Existing artifacts are normally reused. Review the database directories before
requesting a build.

## Running checkpoints

Run each command from the repository root. Commands use the repository
virtual-environment interpreter and omit optional CLI parameters.

### Checkpoint 1.1: no-retrieval baseline

[Solution](Capstone_Checkpoint_1.1/MHERRERA_Capstone_Checkpoint_1_1_Solution.py)

**Windows (PowerShell)**

```powershell
& .\.venv\Scripts\python.exe .\Capstone_Checkpoints\Capstone_Checkpoint_1.1\MHERRERA_Capstone_Checkpoint_1_1_Solution.py
```

**macOS (Zsh/Bash)**

```bash
./.venv/bin/python ./Capstone_Checkpoints/Capstone_Checkpoint_1.1/MHERRERA_Capstone_Checkpoint_1_1_Solution.py
```

Runs the example and authored probes against the configured model. It needs no
corpus or embeddings and is a live model run, not an interactive chat menu.
Responses append to `checkpoint_1_1_responses.log` in the checkpoint folder.
This checkpoint has no CLI parameters.

### Checkpoint 2.1: baseline hybrid retrieval

[Solution](Capstone_Checkpoint_2.1/MHERRERA_Capstone_Checkpoint_2_1_Solution.py)

**Windows (PowerShell)**

```powershell
& .\.venv\Scripts\python.exe .\Capstone_Checkpoints\Capstone_Checkpoint_2.1\MHERRERA_Capstone_Checkpoint_2_1_Solution.py
```

**macOS (Zsh/Bash)**

```bash
./.venv/bin/python ./Capstone_Checkpoints/Capstone_Checkpoint_2.1/MHERRERA_Capstone_Checkpoint_2_1_Solution.py
```

Loads existing Chroma documents when available; otherwise it scans Wikipedia
HTML. It runs representative queries with BM25/vector fusion and model answers.
Missing Chroma data may lead to embedding/build costs. Results append to
`checkpoint_2_1_retrieval.log`. There is no modern evaluation CLI.

### Checkpoint 3.1: originals versus paraphrases

[Solution](Capstone_Checkpoint_3.1/MHERRERA_Capstone_Checkpoint_3_1_Solution.py)

**Windows (PowerShell)**

```powershell
& .\.venv\Scripts\python.exe .\Capstone_Checkpoints\Capstone_Checkpoint_3.1\MHERRERA_Capstone_Checkpoint_3_1_Solution.py
```

**macOS (Zsh/Bash)**

```bash
./.venv/bin/python ./Capstone_Checkpoints/Capstone_Checkpoint_3.1/MHERRERA_Capstone_Checkpoint_3_1_Solution.py
```

The solution evaluates the existing [main questions](Capstone_Checkpoint_3.1/test_variables/test_main_questions.json)
and [paraphrased questions](Capstone_Checkpoint_3.1/test_variables/testinputs_variant_questions.json),
then runs a judge-control check. If either dataset is missing, the checkpoint's
question-generation utilities can create it using model calls. Their positional
arguments are:

| Utility | Positional argument | Meaning |
|---|---|---|
| `generate_main_questions.py` | `count` | Number of main questions to generate. |
| `generate_main_questions.py` | `output_path` | Destination JSON file for the generated main questions. |
| `generate_variants.py` | `input_path` | Main-question JSON file to paraphrase. |
| `generate_variants.py` | `variants_per_question` | Number of paraphrases to generate per main question. |
| `generate_variants.py` | `output_path` | Destination JSON file for paraphrases. |

Reuse existing datasets when available. The workflow uses persisted Chroma data
when available; otherwise corpus loading/building may be needed. Logs and RAGAS
CSVs are in the checkpoint's `ragas_experiments_3_1` folder. The solution
launcher has no CLI parameters.

### Checkpoint 4.1: advanced retrieval evaluation

[Solution](Capstone_Checkpoint_4.1/MHERRERA_Capstone_Checkpoint_4_1_Solution.py)

**Windows (PowerShell)**

```powershell
& .\.venv\Scripts\python.exe .\Capstone_Checkpoints\Capstone_Checkpoint_4.1\MHERRERA_Capstone_Checkpoint_4_1_Solution.py
```

**macOS (Zsh/Bash)**

```bash
./.venv/bin/python ./Capstone_Checkpoints/Capstone_Checkpoint_4.1/MHERRERA_Capstone_Checkpoint_4_1_Solution.py
```

Without options, the checkpoint opens its evaluation menus. It supports
`lexical`, `semantic`, `hybrid`, `graph`, and `all` retrieval. It has no
`--mode` or `--engine` option.

| Parameter | Values | Purpose |
|---|---|---|
| `--dataset` (`--input_type`) | `main`, `paraphrased`, `both` | Select evaluation questions. |
| `--retrieval` (`--search_method`) | `lexical`, `semantic`, `graph`, `hybrid`, `all` | Select retrieval strategy. |
| `--main-source` | `manual`, `llm` | Select the main-question dataset when evaluating `main` or `both`. |
| `--numbers` (`--number_questions`, `-n`) | Integer | Maximum questions per selected dataset; cannot exceed its size. |
| `--random` | `Y`, `N` | Sample questions randomly (`Y`) or take the first rows (`N`). |
| `--help` | Flag | Show CLI options and exit before setup runs. |

### Checkpoint 5.1: Context-Aware and Agentic Dynamic

[Solution](Capstone_Checkpoint_5.1/MHERRERA_Capstone_Checkpoint_5_1_Agent_Solution.py)

**Windows (PowerShell)**

```powershell
& .\.venv\Scripts\python.exe .\Capstone_Checkpoints\Capstone_Checkpoint_5.1\MHERRERA_Capstone_Checkpoint_5_1_Agent_Solution.py
```

**macOS (Zsh/Bash)**

```bash
./.venv/bin/python ./Capstone_Checkpoints/Capstone_Checkpoint_5.1/MHERRERA_Capstone_Checkpoint_5_1_Agent_Solution.py
```

Without options, the checkpoint opens menus. `both` is an evaluation-only
comparison, not a chat engine. This checkpoint does not offer hardening flags.

| Parameter | Values | Purpose |
|---|---|---|
| `--mode` | `chat`, `eval`, `quit` | Choose interactive chat, evaluation, or exit. |
| `--engine` | `context`, `agent`, `both` | Choose Context-Aware, Agentic, or both engines (`both` is evaluation-only). |
| `--dataset` (`--input_type`) | `main`, `paraphrased`, `both` | Select evaluation questions. |
| `--retrieval` (`--search_method`) | `lexical`, `semantic`, `hybrid`, `graph` | Select retrieval strategy. |
| `--main-source` | `manual`, `llm` | Select the main-question dataset for `main` or `both`. |
| `--numbers` (`--number_questions`, `-n`) | Integer | Maximum questions per selected dataset. |
| `--random` | `Y`, `N` | Randomly sample (`Y`) or take the first rows (`N`). |
| `--help` | Flag | Show CLI options and exit before setup runs. |

### Checkpoint 6.1: diagnostic evaluation and cost experiments

[Solution](Capstone_Checkpoint_6.1/MHERRERA_Capstone_Checkpoint_6_1_Agent_Solution.py)

**Windows (PowerShell)**

```powershell
& .\.venv\Scripts\python.exe .\Capstone_Checkpoints\Capstone_Checkpoint_6.1\MHERRERA_Capstone_Checkpoint_6_1_Agent_Solution.py
```

**macOS (Zsh/Bash)**

```bash
./.venv/bin/python ./Capstone_Checkpoints/Capstone_Checkpoint_6.1/MHERRERA_Capstone_Checkpoint_6_1_Agent_Solution.py
```

Without options, the checkpoint opens menus. The optional cost experiment makes
many model and judge calls. Its estimated model cost excludes embedding and
judge costs; its report is `checkpoint_6_1_security_performance.log`.

| Parameter | Values | Purpose |
|---|---|---|
| `--mode` | `chat`, `eval`, `quit` | Choose interactive chat, evaluation, or exit. |
| `--engine` | `context`, `agent`, `both` | Choose an engine; `both` is evaluation-only. |
| `--dataset` (`--input_type`) | `main`, `paraphrased`, `both`, `failure` | Select standard questions or the five diagnostic probes. |
| `--retrieval` (`--search_method`) | `lexical`, `semantic`, `hybrid`, `graph` | Select retrieval strategy. `graph` means hybrid retrieval with graph enabled. |
| `--main-source` | `manual`, `llm` | Select the main-question dataset for `main` or `both`. |
| `--numbers` (`--number_questions`, `-n`) | Integer | Maximum questions per selected dataset. |
| `--random` | `Y`, `N` | Randomly sample (`Y`) or take the first rows (`N`). |
| `--cost-experiment` | Flag | Run the opt-in multi-model cost experiment. |
| `--help` | Flag | Show CLI options and exit before setup runs. |

### Checkpoint 7.1: baseline and hardened engine workflows

[Solution](Capstone_Checkpoint_7.1/MHERRERA_Capstone_Checkpoint_7_1_Agent_Solution_Prod.py)

**Windows (PowerShell)**

```powershell
& .\.venv\Scripts\python.exe .\Capstone_Checkpoints\Capstone_Checkpoint_7.1\MHERRERA_Capstone_Checkpoint_7_1_Agent_Solution_Prod.py
```

**macOS (Zsh/Bash)**

```bash
./.venv/bin/python ./Capstone_Checkpoints/Capstone_Checkpoint_7.1/MHERRERA_Capstone_Checkpoint_7_1_Agent_Solution_Prod.py
```

Without options, the checkpoint opens menus and prompts for the hardening
setting. Scripted chat/evaluation requires exactly one of `--harden` or
`--no-harden`. `both` is an evaluation-only engine comparison.

| Parameter | Values | Purpose |
|---|---|---|
| `--mode` | `chat`, `eval`, `quit` | Choose interactive chat, evaluation, or exit. |
| `--engine` | `context`, `agent`, `both` | Choose Context-Aware, Agentic, or both (`both` is evaluation-only). |
| `--dataset` (`--input_type`) | `main`, `paraphrased`, `both`, `failure` | Select standard questions or the five diagnostic probes. |
| `--retrieval` (`--search_method`) | `lexical`, `semantic`, `hybrid`, `graph` | Select retrieval strategy. `graph` means hybrid retrieval with graph enabled. |
| `--main-source` | `manual`, `llm` | Select the main-question dataset for `main` or `both`. |
| `--numbers` (`--number_questions`, `-n`) | Integer | Maximum questions per selected dataset. |
| `--random` | `Y`, `N` | Randomly sample (`Y`) or take the first rows (`N`). |
| `--harden` / `--no-harden` | Mutually exclusive flags | Enable sanitization and tagged trust boundaries, or run the baseline. One is required for scripted chat/evaluation. |
| `--hardening-tests` | Flag | Replay diagnostic probes across both engines in baseline and hardened modes. |
| `--cost-experiment` | Flag | Run the opt-in model-ladder and cost experiment; does not compare hardening modes. |
| `--help` | Flag | Show CLI options and exit before checkpoint setup runs. |

Hardening is defense-in-depth, not a prompt-injection or factual-correctness
guarantee. The shared [ToolUsingAgent](../Capstone_Utilities/Retrieval_Methods/dynamic_agent_retrieval.py)
sanitizes user and conversation inputs, generated search queries, retrieved
passages, and graph context; hardened prompts keep instructions and untrusted
content in separate tagged sections.

## Shared datasets, configuration, and results

- [Checkpoint_Retrieval.conf](Checkpoint_Retrieval.conf) controls shared model settings, candidate
  counts, fusion weights, Agentic limits, and Context-Aware history. Checkpoints
  1.1-3.1 define model/retrieval constants in their scripts.
- Checkpoints 4.1-7.1 use [Test_Variables](../Capstone_Ragas_Experiments/Test_Variables/) for `main`,
  `paraphrased`, or `both` datasets; `--main-source` selects `manual` or `llm`.
  The `failure` dataset is available in 6.1-7.1. Question counts must fit the
  selected data. `--random N` takes the first rows; `--random Y` samples them.
- Shared evaluation CSVs are in [Capstone_Ragas_Experiments/experiments](../Capstone_Ragas_Experiments/experiments/).
  Checkpoint 4.1 writes `detailed_test_results.log`; later dual-engine
  workflows use `detailed_test_results_agentic.log` and their dedicated reports.
- Utility logs are in `Capstone_Utilities/Utility_Scripts/Logs`.

## Startup behavior

Checkpoints 4.1-7.1 invoke setup with `--build` on normal CLI startup, before
menus or quit selections. Setup decides whether builders are needed, so even
selecting quit may have startup side effects. `--help` exits before bootstrap
and is the safer import/CLI check for these launchers.

Checkpoint 1.1 executes probes when its script is run or imported. Checkpoint
2.1 loads the corpus/Chroma and executes its query workflow at module level. Do
not import either script as an offline smoke test. Prefer named CLI options over
numeric menu choices; numeric retrieval mappings differ between checkpoints.

## Troubleshooting

- **Missing imports:** use the repository virtual environment and install the
  declared requirements if needed. Setup does not update an existing environment.
- **Missing API key:** replace the root `.env` placeholder or set the process
  environment variable. Never place a real key in source code.
- **Missing corpus/indexes:** inspect the existing database directories before
  requesting a build. Chroma creation can incur embedding costs.
- **Dataset errors:** confirm the selected dataset exists and the question count
  does not exceed its size.
- **Provider or judge errors:** a successful CLI launch does not establish
  network access, model availability, quotation accuracy, or judge reliability.
- **Unsupported `harden` keyword:** make sure the updated shared checkpoint files
  are being used rather than an older copy.
