# Capstone Checkpoints: Setup and Run Guide

This guide documents checkpoints 1.1 through 7.1 and their current implementation.
It replaces the setup/run guidance in the [repository README](../../README.md)
for checkpoint execution; it does not replace historical experiment reports.

## Layout and prerequisites

- Python 3.10 or newer is declared by the project dependencies. Offline
  validation was performed with Python 3.12.10; other versions were not certified.
- Windows examples use PowerShell; macOS examples use Terminal with Zsh or Bash
  (and also apply to Linux). Start in the `MIT_CAPSTONE` repository
  root, not this folder. Use the repository virtual environment for every run.
- Set `OPENROUTER_API_KEY` in the root `.env` or the process environment. Do not
  commit credentials.
- Network access is required for dependency installation, model calls, and
  query embeddings. Answer generation, judging, and embeddings can incur charges.
- The corpus and indexes remain in [Capstone_Database](../Capstone_Database/).
  They were not moved or copied during checkpoint organization.
- Shared checkpoint code, schemas, question datasets, configuration, and
  evaluation outputs live here. Optional [architecture diagrams](../../Architecture_Diagrams/)
  and [backup questions](../../Backup_Questions/) live at the repository root.
- The [isolated final solution](../Capstone_Final_Solution/README.md) has its own
  code/configuration and separate run instructions. It is not interchangeable
  with the checkpoint launchers.

## 1. First-time setup

From the repository root, run setup with an installed Python interpreter:

**Windows (PowerShell)**

```powershell
python .\Final_Capstone_Project\Capstone_Checkpoints\Utility_Scripts\Setup.py
```

**macOS (Zsh/Bash)**

```bash
python3 ./Final_Capstone_Project/Capstone_Checkpoints/Utility_Scripts/Setup.py
```

[Setup.py](Utility_Scripts/Setup.py) creates a missing root `.env` template,
local directories/placeholders, and the repository `.venv`. It installs
[venv_requirements.txt](../../venv_requirements.txt) only when creating that
virtual environment. **An existing environment is reused without installing
or updating dependencies.** If it is incomplete, install the declared dependencies:

**Windows (PowerShell)**

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\venv_requirements.txt
```

**macOS (Zsh/Bash)**

```bash
./.venv/bin/python -m pip install -r ./venv_requirements.txt
```

Replace the placeholder key in the root `.env` with your real key:

```dotenv
OPENROUTER_API_KEY=your-real-openrouter-key
```

Setup without `--build` does not populate the corpus or build indexes. Supply
Wikipedia HTML under [Wikipedia](../Capstone_Database/Wikipedia/), or existing
JSONL under [Wikipedia_JSONL](../Capstone_Database/Wikipedia_JSONL/). HTML is
needed for graph generation; JSONL is sufficient for BM25 and Chroma generation.
Checkpoint 1.1 does not require a corpus.

If artifacts are missing, explicitly request a build:

**Windows (PowerShell)**

```powershell
.\.venv\Scripts\python.exe .\Final_Capstone_Project\Capstone_Checkpoints\Utility_Scripts\Setup.py --build
```

**macOS (Zsh/Bash)**

```bash
./.venv/bin/python ./Final_Capstone_Project/Capstone_Checkpoints/Utility_Scripts/Setup.py --build
```

This may chunk HTML, build BM25/GraphML, and create embeddings for Chroma.
**Chroma creation makes embedding API calls and can incur charges.**
Existing artifacts are normally reused; incomplete artifacts can trigger
confirmation prompts. `--build --rebuild` requests rebuilds with confirmations
and can overwrite generated artifacts. Do not use it merely to launch a checkpoint.

## 2. Command conventions and startup side effects

Define the variables for your shell from the repository root before using the
checkpoint examples below. No environment activation is required because the
commands invoke the virtual-environment interpreter explicitly.

**Windows (PowerShell)**

```powershell
$python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
$checkpoints = Join-Path (Get-Location) 'Final_Capstone_Project\Capstone_Checkpoints'
```

**macOS (Zsh/Bash)**

```bash
python="$PWD/.venv/bin/python"
checkpoints="$PWD/Final_Capstone_Project/Capstone_Checkpoints"
```

Every command below assumes these variables exist in the current terminal.
Do not use PowerShell's `&` invocation prefix in Zsh/Bash. Paths are quoted to
support repository locations containing spaces. A Windows virtual environment
cannot be reused on macOS; run setup there to create a platform-local environment.

**Checkpoints 4.1-7.1 invoke setup with `--build` on normal CLI startup, before
menus and quit selections.** Setup decides whether builders are needed. This
means even selecting quit is not a guarantee of zero startup side effects.
Their `--help` option exits before bootstrap and is the safer import/CLI check.

Checkpoint 1.1 executes probes when its script is run or imported. Checkpoint
2.1 loads the corpus/Chroma and executes its query workflow at module level.
Do not import those scripts as an offline smoke test.

Prefer named CLI options over numeric menu choices: numeric retrieval mappings
are not identical across checkpoints. For scripted evaluations, supply the
dataset, retrieval method, source, question count, and sampling explicitly.

## 3. Run each checkpoint

### Checkpoint 1.1: no-retrieval baseline

[Solution](Capstone_Checkpoint_1.1/MHERRERA_Capstone_Checkpoint_1_1_Solution.py)

**Windows (PowerShell)**

```powershell
& $python "$checkpoints\Capstone_Checkpoint_1.1\MHERRERA_Capstone_Checkpoint_1_1_Solution.py"
```

**macOS (Zsh/Bash)**

```bash
"$python" "$checkpoints/Capstone_Checkpoint_1.1/MHERRERA_Capstone_Checkpoint_1_1_Solution.py"
```

Runs the example and authored probes against the configured model. No corpus or
embedding is required. This is a live model run, not an interactive chat menu.
Responses append to the checkpoint's `checkpoint_1_1_responses.log`.

### Checkpoint 2.1: baseline hybrid retrieval

[Solution](Capstone_Checkpoint_2.1/MHERRERA_Capstone_Checkpoint_2_1_Solution.py)

**Windows (PowerShell)**

```powershell
& $python "$checkpoints\Capstone_Checkpoint_2.1\MHERRERA_Capstone_Checkpoint_2_1_Solution.py"
```

**macOS (Zsh/Bash)**

```bash
"$python" "$checkpoints/Capstone_Checkpoint_2.1/MHERRERA_Capstone_Checkpoint_2_1_Solution.py"
```

Loads existing Chroma documents when available, otherwise scans Wikipedia HTML.
Runs the built-in representative queries with BM25/vector fusion and model
answers. Missing Chroma can lead to embedding/build costs. Results append to
the checkpoint's `checkpoint_2_1_retrieval.log`. There is no modern evaluation CLI.

### Checkpoint 3.1: originals versus paraphrases

[Solution](Capstone_Checkpoint_3.1/MHERRERA_Capstone_Checkpoint_3_1_Solution.py)

Required datasets are [test_main_questions.json](Capstone_Checkpoint_3.1/test_variables/test_main_questions.json)
and [testinputs_variant_questions.json](Capstone_Checkpoint_3.1/test_variables/testinputs_variant_questions.json).
Reuse existing datasets. If missing, generation makes model calls:

**Windows (PowerShell)**

```powershell
& $python "$checkpoints\Capstone_Checkpoint_3.1\test_variables\generate_main_questions.py" 100 "$checkpoints\Capstone_Checkpoint_3.1\test_variables\test_main_questions.json"
& $python "$checkpoints\Capstone_Checkpoint_3.1\test_variables\generate_variants.py" "$checkpoints\Capstone_Checkpoint_3.1\test_variables\test_main_questions.json" 2 "$checkpoints\Capstone_Checkpoint_3.1\test_variables\testinputs_variant_questions.json"
& $python "$checkpoints\Capstone_Checkpoint_3.1\MHERRERA_Capstone_Checkpoint_3_1_Solution.py"
```

**macOS (Zsh/Bash)**

```bash
"$python" "$checkpoints/Capstone_Checkpoint_3.1/test_variables/generate_main_questions.py" 100 "$checkpoints/Capstone_Checkpoint_3.1/test_variables/test_main_questions.json"
"$python" "$checkpoints/Capstone_Checkpoint_3.1/test_variables/generate_variants.py" "$checkpoints/Capstone_Checkpoint_3.1/test_variables/test_main_questions.json" 2 "$checkpoints/Capstone_Checkpoint_3.1/test_variables/testinputs_variant_questions.json"
"$python" "$checkpoints/Capstone_Checkpoint_3.1/MHERRERA_Capstone_Checkpoint_3_1_Solution.py"
```

When datasets already exist, run only the last command in the appropriate block.

These commands do not change directories. The solution evaluates both sets and
runs a judge-control check. It uses a persisted Chroma database when available;
otherwise corpus loading/building may be needed. Logs and RAGAS CSVs are
checkpoint-local, under `ragas_experiments_3_1`, not the shared results folder.

### Checkpoint 4.1: advanced retrieval evaluation

[Solution](Capstone_Checkpoint_4.1/MHERRERA_Capstone_Checkpoint_4_1_Solution.py)

Interactive menus:

**Windows (PowerShell)**

```powershell
& $python "$checkpoints\Capstone_Checkpoint_4.1\MHERRERA_Capstone_Checkpoint_4_1_Solution.py"
```

**macOS (Zsh/Bash)**

```bash
"$python" "$checkpoints/Capstone_Checkpoint_4.1/MHERRERA_Capstone_Checkpoint_4_1_Solution.py"
```

Explicit evaluation:

**Windows (PowerShell)**

```powershell
& $python "$checkpoints\Capstone_Checkpoint_4.1\MHERRERA_Capstone_Checkpoint_4_1_Solution.py" --dataset main --main-source manual --retrieval hybrid --numbers 4 --random N
```

**macOS (Zsh/Bash)**

```bash
"$python" "$checkpoints/Capstone_Checkpoint_4.1/MHERRERA_Capstone_Checkpoint_4_1_Solution.py" --dataset main --main-source manual --retrieval hybrid --numbers 4 --random N
```

Supports `lexical`, `semantic`, `hybrid`, `graph`, and `all` retrieval.
No `--mode` or `--engine` option exists for this checkpoint.

### Checkpoint 5.1: Context-Aware and Agentic Dynamic

[Solution](Capstone_Checkpoint_5.1/MHERRERA_Capstone_Checkpoint_5_1_Agent_Solution.py)

**Windows (PowerShell)**

```powershell
& $python "$checkpoints\Capstone_Checkpoint_5.1\MHERRERA_Capstone_Checkpoint_5_1_Agent_Solution.py"
& $python "$checkpoints\Capstone_Checkpoint_5.1\MHERRERA_Capstone_Checkpoint_5_1_Agent_Solution.py" --mode chat --engine context --retrieval graph
& $python "$checkpoints\Capstone_Checkpoint_5.1\MHERRERA_Capstone_Checkpoint_5_1_Agent_Solution.py" --mode eval --engine both --dataset main --main-source manual --retrieval hybrid --numbers 4 --random N
```

**macOS (Zsh/Bash)**

```bash
"$python" "$checkpoints/Capstone_Checkpoint_5.1/MHERRERA_Capstone_Checkpoint_5_1_Agent_Solution.py"
"$python" "$checkpoints/Capstone_Checkpoint_5.1/MHERRERA_Capstone_Checkpoint_5_1_Agent_Solution.py" --mode chat --engine context --retrieval graph
"$python" "$checkpoints/Capstone_Checkpoint_5.1/MHERRERA_Capstone_Checkpoint_5_1_Agent_Solution.py" --mode eval --engine both --dataset main --main-source manual --retrieval hybrid --numbers 4 --random N
```

The first command opens menus; the other commands start chat and a paired
evaluation. Use `agent` for the Agentic engine. `both` is evaluation-only.
This checkpoint does not expose hardening-selection flags.

### Checkpoint 6.1: diagnostic evaluation and cost experiments

[Solution](Capstone_Checkpoint_6.1/MHERRERA_Capstone_Checkpoint_6_1_Agent_Solution.py)

**Windows (PowerShell)**

```powershell
& $python "$checkpoints\Capstone_Checkpoint_6.1\MHERRERA_Capstone_Checkpoint_6_1_Agent_Solution.py"
& $python "$checkpoints\Capstone_Checkpoint_6.1\MHERRERA_Capstone_Checkpoint_6_1_Agent_Solution.py" --mode eval --engine agent --dataset failure --retrieval graph --numbers 5 --random N
```

**macOS (Zsh/Bash)**

```bash
"$python" "$checkpoints/Capstone_Checkpoint_6.1/MHERRERA_Capstone_Checkpoint_6_1_Agent_Solution.py"
"$python" "$checkpoints/Capstone_Checkpoint_6.1/MHERRERA_Capstone_Checkpoint_6_1_Agent_Solution.py" --mode eval --engine agent --dataset failure --retrieval graph --numbers 5 --random N
```

Optional multi-model experiment; this makes many paid model/judge calls:

**Windows (PowerShell)**

```powershell
& $python "$checkpoints\Capstone_Checkpoint_6.1\MHERRERA_Capstone_Checkpoint_6_1_Agent_Solution.py" --cost-experiment --retrieval hybrid --numbers 4 --random N
```

**macOS (Zsh/Bash)**

```bash
"$python" "$checkpoints/Capstone_Checkpoint_6.1/MHERRERA_Capstone_Checkpoint_6_1_Agent_Solution.py" --cost-experiment --retrieval hybrid --numbers 4 --random N
```

The model-cost estimate excludes embedding and judge costs. The dedicated report
is `checkpoint_6_1_security_performance.log` in the shared results folder.

### Checkpoint 7.1: baseline and hardened engine workflows

[Solution](Capstone_Checkpoint_7.1/MHERRERA_Capstone_Checkpoint_7_1_Agent_Solution_Prod.py)

Start without arguments for interactive menus, or select Context-Aware chat
and evaluation explicitly:

**Windows (PowerShell)**

```powershell
& $python "$checkpoints\Capstone_Checkpoint_7.1\MHERRERA_Capstone_Checkpoint_7_1_Agent_Solution_Prod.py"
& $python "$checkpoints\Capstone_Checkpoint_7.1\MHERRERA_Capstone_Checkpoint_7_1_Agent_Solution_Prod.py" --mode chat --engine context --retrieval graph --harden
& $python "$checkpoints\Capstone_Checkpoint_7.1\MHERRERA_Capstone_Checkpoint_7_1_Agent_Solution_Prod.py" --mode eval --engine context --dataset failure --retrieval lexical --numbers 5 --random N --harden
```

**macOS (Zsh/Bash)**

```bash
"$python" "$checkpoints/Capstone_Checkpoint_7.1/MHERRERA_Capstone_Checkpoint_7_1_Agent_Solution_Prod.py"
"$python" "$checkpoints/Capstone_Checkpoint_7.1/MHERRERA_Capstone_Checkpoint_7_1_Agent_Solution_Prod.py" --mode chat --engine context --retrieval graph --harden
"$python" "$checkpoints/Capstone_Checkpoint_7.1/MHERRERA_Capstone_Checkpoint_7_1_Agent_Solution_Prod.py" --mode eval --engine context --dataset failure --retrieval lexical --numbers 5 --random N --harden
```

Use `--no-harden` for a baseline run. Both engines accept the hardening setting.
Use `--engine agent` for Agentic chat/evaluation or `--engine both` for a paired
evaluation; `both` cannot be used for chat. Standard scripted runs require
`--harden` or `--no-harden`; omitting both produces a CLI error. Interactive
runs prompt for the hardening choice.

Agentic chat and baseline/hardened paired evaluations:

**Windows (PowerShell)**

```powershell
& $python "$checkpoints\Capstone_Checkpoint_7.1\MHERRERA_Capstone_Checkpoint_7_1_Agent_Solution_Prod.py" --mode chat --engine agent --retrieval graph --harden
& $python "$checkpoints\Capstone_Checkpoint_7.1\MHERRERA_Capstone_Checkpoint_7_1_Agent_Solution_Prod.py" --mode eval --engine both --dataset failure --retrieval lexical --numbers 5 --random N --no-harden
& $python "$checkpoints\Capstone_Checkpoint_7.1\MHERRERA_Capstone_Checkpoint_7_1_Agent_Solution_Prod.py" --mode eval --engine both --dataset failure --retrieval lexical --numbers 5 --random N --harden
```

**macOS (Zsh/Bash)**

```bash
"$python" "$checkpoints/Capstone_Checkpoint_7.1/MHERRERA_Capstone_Checkpoint_7_1_Agent_Solution_Prod.py" --mode chat --engine agent --retrieval graph --harden
"$python" "$checkpoints/Capstone_Checkpoint_7.1/MHERRERA_Capstone_Checkpoint_7_1_Agent_Solution_Prod.py" --mode eval --engine both --dataset failure --retrieval lexical --numbers 5 --random N --no-harden
"$python" "$checkpoints/Capstone_Checkpoint_7.1/MHERRERA_Capstone_Checkpoint_7_1_Agent_Solution_Prod.py" --mode eval --engine both --dataset failure --retrieval lexical --numbers 5 --random N --harden
```

Each paired evaluation runs both engines using the same selected hardening mode.
The two evaluation commands above are separate baseline and hardened runs and
incur answer-generation and judging costs.

The shared [ToolUsingAgent](Retrieval_Methods/dynamic_agent_retrieval.py) now
implements native hardening: it sanitizes the current question, prior turns,
clarification replies, generated search queries, retrieved passages, and graph
context. Hardened planner/answer prompts separate data with tags and explicit
trust instructions. Hardened clarification rounds count toward the action limit.
Baseline remains the default for callers that omit the setting and preserves
the legacy prompts and clarification counting. Regex sanitization is
defense-in-depth, not a prompt-injection or factual-correctness guarantee.

To replay the diagnostic suite across both engines and both hardening modes
(multiple model/judge calls):

**Windows (PowerShell)**

```powershell
& $python "$checkpoints\Capstone_Checkpoint_7.1\MHERRERA_Capstone_Checkpoint_7_1_Agent_Solution_Prod.py" --hardening-tests --retrieval lexical --numbers 5 --random N
```

**macOS (Zsh/Bash)**

```bash
"$python" "$checkpoints/Capstone_Checkpoint_7.1/MHERRERA_Capstone_Checkpoint_7_1_Agent_Solution_Prod.py" --hardening-tests --retrieval lexical --numbers 5 --random N
```

The separate cost experiment uses the Agentic baseline (native hardening defaults
to off); it does not compare hardening modes:

**Windows (PowerShell)**

```powershell
& $python "$checkpoints\Capstone_Checkpoint_7.1\MHERRERA_Capstone_Checkpoint_7_1_Agent_Solution_Prod.py" --cost-experiment --retrieval lexical --numbers 5 --random N
```

**macOS (Zsh/Bash)**

```bash
"$python" "$checkpoints/Capstone_Checkpoint_7.1/MHERRERA_Capstone_Checkpoint_7_1_Agent_Solution_Prod.py" --cost-experiment --retrieval lexical --numbers 5 --random N
```

This command is not a hardening test. Cost/hardening reports use
`checkpoint_7_1_optimization_performance.log`. Existing results are historical
evidence, not a new live validation of this fix.

## 4. Configuration, datasets, and outputs

[retrieval.conf](retrieval.conf) controls shared checkpoint model settings,
candidate counts, fusion weights, Agentic limits, and Context-Aware history.
The early checkpoints 1.1-3.1 define model/retrieval constants in their scripts
instead.

Checkpoints 4.1-7.1 use [Test_Variables](Test_Variables/) for evaluation:
`main`, `paraphrased`, or `both`, with `--main-source manual` or `llm`.
The failure dataset is available in 6.1-7.1. Counts must fit the selected dataset;
`--random N` takes the first records and `--random Y` samples.

Shared evaluation CSVs are under [Ragas_Experiments/experiments](Ragas_Experiments/experiments/).
Checkpoint 4.1 summaries use `detailed_test_results.log`; later dual-engine
workflows use `detailed_test_results_agentic.log`, plus dedicated reports where
described above. Utility logs are under `Utility_Scripts/Logs`.

## 5. Safe offline checks and troubleshooting

Inspect a modern launcher's options without bootstrap or model calls:

**Windows (PowerShell)**

```powershell
& $python "$checkpoints\Capstone_Checkpoint_7.1\MHERRERA_Capstone_Checkpoint_7_1_Agent_Solution_Prod.py" --help
```

**macOS (Zsh/Bash)**

```bash
"$python" "$checkpoints/Capstone_Checkpoint_7.1/MHERRERA_Capstone_Checkpoint_7_1_Agent_Solution_Prod.py" --help
```

Run the relocation and shared hardening regressions from the repository root:

**Windows (PowerShell)**

```powershell
& $python -m unittest Final_Capstone_Project.Capstone_Checkpoints.tests.test_checkpoint_layout Final_Capstone_Project.Capstone_Checkpoints.Retrieval_Methods.test_context_aware_retrieval_hardening Final_Capstone_Project.Capstone_Checkpoints.Retrieval_Methods.test_dynamic_agent_hardening -v
```

**macOS (Zsh/Bash)**

```bash
"$python" -m unittest Final_Capstone_Project.Capstone_Checkpoints.tests.test_checkpoint_layout Final_Capstone_Project.Capstone_Checkpoints.Retrieval_Methods.test_context_aware_retrieval_hardening Final_Capstone_Project.Capstone_Checkpoints.Retrieval_Methods.test_dynamic_agent_hardening -v
```

For the checkpoints covered here, offline validation checked relocated resource
paths, early notebook imports with live workflows suppressed, four modern
launcher help screens from two working directories, and existing BM25/GraphML
access. Models/vector searches were mocked in retrieval integration tests.
This is not certification of every live execution branch or answer quality.
Those validation runs were on Windows. macOS command examples have not been
executed on a Mac; the relocation test suite contains Windows-specific path
patterns and must be made platform-neutral before it can certify macOS coverage.

Common failures:

- **Missing imports:** use the repository virtual environment and install the
  declared requirements if packages are missing. Setup does not refresh an
  existing environment.
- **Missing API key:** replace the root `.env` placeholder or set the environment
  variable; never put a real key in source code.
- **Missing corpus/indexes:** inspect the existing database directories before
  explicitly approving a build. Database creation can cost money.
- **Unsupported `harden` keyword:** the shared Agentic constructor now supports
  it. If this error persists, verify that you are running the updated relocated
  checkpoint files, not an older copy.
- **Dataset errors:** verify the selected files and choose a question count no
  larger than the smallest selected set.
- **Provider or judging errors:** CLI/import success does not establish model
  availability, network connectivity, quotation accuracy, or judge reliability.

Native Agentic hardening was validated with mocked models/retrievers; no live
answer-quality or security certification is implied. The latest Windows
validation passed 21 offline tests across the three suites above, and Pylance
confirmed compatibility for all 19 shared Agentic constructor call sites.
Tests exercise real LangGraph control flow with mocked model responses, input
and evidence sanitization, baseline prompt preservation, and hardened
clarification limits. No live model/embedding calls or database rebuilds were
performed; the database file count, sizes, and modification times were unchanged.
