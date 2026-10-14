# Capstone Final Solution

Local copy of the automatic hardened, graph-enabled Wikipedia RAG solution.
Retrieval, routing, sanitization, evaluation, and reporting code live in this
folder. Hybrid ranking is imported from this solution's local
[Ranking_Techniques](./Ranking_Techniques/) package and reads ranking settings
from this folder's [retrieval.conf](./retrieval.conf). Checkpoint launchers and
configuration are not imported.

The external database artifacts are separate and are not included. See the
[repository overview](../README.md) and the
[checkpoint guide](../Capstone_Checkpoints/Checkpoint_Readme.md)
for project setup and shared materials.

Offline validation used Python 3.12.10 on Windows. macOS instructions below have
not been executed on a Mac; package compatibility depends on selected versions.

## Run on Windows

From this folder, using an environment with the dependencies installed:

```powershell
python -m pip install -r requirements.txt
python .\MHERRERA_Capstone_8_1_Automatic_Hardened_Hybrid_Graph.py
```

The existing repository virtual environment can also be used:

```powershell
& ..\.venv\Scripts\python.exe .\MHERRERA_Capstone_8_1_Automatic_Hardened_Hybrid_Graph.py
```

## Run on macOS (Zsh/Bash)

From this solution folder, `[paths] database_dir` in
[retrieval.conf](./retrieval.conf) points to the sibling
`Capstone_Database` folder at the repository root. The Windows default uses
backslashes, which are not directory separators on macOS.
For macOS, set it to `../Capstone_Database`.

Use a platform-local environment; do not reuse a Windows virtual environment:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python ./MHERRERA_Capstone_8_1_Automatic_Hardened_Hybrid_Graph.py
```

If the repository already has a macOS virtual environment with dependencies:

```bash
../.venv/bin/python ./MHERRERA_Capstone_8_1_Automatic_Hardened_Hybrid_Graph.py
```

The root [venv_requirements.txt](../venv_requirements.txt) covers checkpoints
and this solution together. This folder's [requirements.txt](requirements.txt)
is the pinned standalone list. Pip resolves transitive dependencies in either
case; the two lists are alternatives, not separate runtime components.

## Credentials and defaults

Set `OPENROUTER_API_KEY` in your environment or a local `.env` file. Never commit
the key. Normal chat makes embedding and answer calls; the Agentic path also
makes planning calls. `--help` and the tests below make no model calls.

Defaults in [retrieval.conf](./retrieval.conf):

- Chat, automatic routing, graph-enabled retrieval, hardening enabled.
- Simple questions: Context-Aware with GPT-4o-mini.
- Complex questions: Agentic Dynamic with GPT-5.4-mini.
- Four graph seeds for each engine, with a 14,000-character retrieved-context limit.
- Retrieval diagnostics print selected IDs so insufficient-evidence answers can
  be investigated.

CLI overrides and the copied evaluation/comparison/cost/hardening workflows are
retained. Evaluation questions are read from the shared
[Capstone_Ragas_Experiments/Test_Variables](../Capstone_Ragas_Experiments/Test_Variables/) folder;
the comparison, hardening, and cost summary is written to
`final_solution_8_1_optimization_performance.log` in the shared
[Capstone_Ragas_Experiments](../Capstone_Ragas_Experiments/) folder. Other RAGAS
result logs and per-question CSVs are also written there.
The cost experiment uses only GPT-4o-mini and GPT-5.4-mini, as configured in
[final_solution_8_1_cost_experiment_prototype.py](./Utility_Scripts/final_solution_8_1_cost_experiment_prototype.py).

## External database artifacts

No database artifacts are included, copied, moved, or automatically rebuilt.
`[paths] database_dir` defaults to
`..\Capstone_Database`, resolved relative to this folder rather than the shell's
working directory. Change that path when relocating this folder or using
another existing corpus.

The launcher requires existing Wikipedia HTML/JSONL, BM25, Chroma, and graph
artifacts. Missing artifacts produce an explicit error. There is no dependency
on the shared `Setup.py` or database builders. Existing Chroma embeddings must
match the configured embedding model.

## Isolated retrieval improvements

- Title matching recognizes exact article titles and unambiguous surname queries.
  Ambiguous surname matches are not forced.
- A bounded, in-memory HTML extraction reads actual article introductory
  paragraphs, excluding menus, tables/infoboxes, hatnotes, and references.
  This is a read-only evidence overlay, not a new persisted corpus or index.
- Hybrid candidates are deterministically reranked using title match, passage
  word overlap, introductory relevance for overview questions, and navigation
  penalties. No reranker LLM call is added.
- Graph expansion uses clean linked-article introductions instead of assuming
  `chunk-0000` is an introduction. Ranked seeds precede expanded neighbors.
- Explicit new questions are searched independently of previous assistant
  answers. Pronoun follow-ups also use the preceding user question.
- Both answer prompts request excerpt-ID citations and article titles. A
  deterministic **Sources (retrieved evidence)** footer names only documents
  actually supplied to the model. This footer is provenance, not proof that
  every generated claim is supported; inline citations still need evaluation.
- The local Agentic engine natively accepts `harden`, sanitizes its inputs and
  documents, and uses tagged trust boundaries without patching module globals.
  It retrieves before planning, reports malformed planner output explicitly,
  bounds clarification rounds, and honors local quit commands.
- Generated answer text never controls session termination.

The existing indexed chunks still contain some extraction noise. The overlay
improves introductory evidence without changing those artifacts; it is not a
replacement for a future, separately approved clean-corpus rebuild or learned
reranker. Alias resolution and follow-up interpretation are deterministic
heuristics, not a general entity-linking or query-rewriting model.

## Offline validation

Run from the repository root using the same interpreter as the application.
For Windows:

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s .\Unit_Tests -p test_final_solution.py -v
& .\.venv\Scripts\python.exe .\Capstone_Final_Solution\MHERRERA_Capstone_8_1_Automatic_Hardened_Hybrid_Graph.py --help
```

For macOS:

```bash
./.venv/bin/python -m unittest discover -s ./Unit_Tests -p test_final_solution.py -v
./.venv/bin/python ./Capstone_Final_Solution/MHERRERA_Capstone_8_1_Automatic_Hardened_Hybrid_Graph.py --help
```

The tests use mocked LLM/vector calls and read the existing corpus, BM25, and
graph for integration checks. They verify Einstein overview evidence, source
labels, graph-context bounds, solution-local ranking configuration, hardening,
and quit behavior. They do not create, rebuild, or copy database artifacts.
