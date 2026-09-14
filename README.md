# CAPSTONE PROJECT
RAG and Context Engineering: Designing and Building Production-Grade AI Systems

Miguel Herrera - Section B

## Project Overview
This repository contains the capstone RAG project for a Wikipedia Retrieval Engine scenario. It includes the main project implementation, checkpoint solution files, evaluation utilities, local database directories, course labs, and supporting scripts used for retrieval, ranking, and analysis.

## Repository Inventory

```text
MIT_CAPSTONE/
├── .env                                      # Local environment configuration for API keys and secrets
├── README.md                                 # Project overview and repository inventory
├── venv_requirements.txt                     # Dependency list for the project Python environment
├── OpenRouter_API_Usage.py                   # OpenRouter API helper usage file
├── Backups/                                  # Backup artifacts and retained database snapshots
│   └── Capstone_Chroma_DB/
├── ENV_ANALYSIS/                             # Environment and dependency analysis outputs
│   └── environment_inventory.txt
├── EVALUATION/                               # Evaluation and reporting utilities
│   ├── build_answers_docx.py
│   └── generate_evaluation_report.py
├── Final_Capstone_Project/                   # Main capstone implementation and checkpoint work
│   ├── retrieval.conf                        # Retrieval configuration values
│   ├── Capstone_Checkpoint_1.1/
│   │   └── MHERRERA_Capstone_Checkpoint_1_1_Solution.py
│   ├── Capstone_Checkpoint_2.1/
│   │   └── MHERRERA_Capstone_Checkpoint_2_1_Solution.py
│   ├── Capstone_Checkpoint_3.1/
│   │   ├── MHERRERA_Capstone_Checkpoint_3_1_Solution.py
│   │   ├── run_framework_validation.py
│   │   ├── ragas_experiments_3_1/
│   │   └── test_variables/
│   ├── Capstone_Checkpoint_4.1/
│   │   ├── capstone_checkpoint_4_1_advanced_retrieval_starter.py
│   │   ├── MHERRERA_Capstone_Checkpoint_4_1_Solution.py
│   │   └── ...
│   ├── Capstone_Database/
│   │   ├── Capstone_Chroma_DB/
│   │   ├── Capstone_Graph_DB/
│   │   ├── GraphDB/
│   │   ├── Wikipedia/
│   │   └── Wikipedia_JSONL/
│   ├── JSON_Schemas/
│   │   ├── Test_Main_Questions.schema.json
│   │   ├── Test_Paraphrased_Questions.schema.json
│   │   └── Wikipedia_JSONL_Graph_Record.schema.json
│   ├── Ragas_Experiments/
│   │   ├── __init__.py
│   │   ├── ragas_experiment_logic.py
│   │   ├── datasets/
│   │   └── experiments/
│   ├── Ranking_Techniques/
│   │   ├── __init__.py
│   │   └── weighted_fusion_ranking.py
│   ├── Retrieval_Methods/
│   │   ├── __init__.py
│   │   ├── bm25_retrieval.py
│   │   ├── hybrid_retrieval.py
│   │   └── vector_retrieval.py
│   ├── Test_Variables/
│   ├── Utility_Scripts/
│   └── ...
├── lab_databases/
│   ├── lab_1_2/
│   ├── lab_2_1/
│   ├── lab_2_2/
│   ├── lab_3_1/
│   └── lab_3_2/
├── lab_my_files/
├── lab_requirements/
│   ├── lab_3_1_requirements.txt
│   └── lab_3_2_requirements.txt
├── lab_solution_files/
│   ├── lab_1_1_chatbot_solution.py
│   ├── lab_1_2_keyword_retrieval_solution.py
│   ├── lab_2_1_vector_retrieval_solution.py
│   ├── lab_2_2_hybrid_retrieval_solution.py
│   ├── lab_3_1_evaluation_solution.py
│   ├── lab_3_2_test_variants_solution.py
│   ├── lab_4_1_multistep_retrieval_solution.py
│   └── lab_4_2_solution_files/
├── lab_starter_files/
│   ├── lab_1_1_chatbot_starter.py
│   ├── lab_1_2_keyword_retrieval_starter.py
│   ├── lab_2_1_vector_retrieval_starter.py
│   ├── lab_2_2_hybrid_retrieval_starter.py
│   ├── lab_3_1_evaluation_starter.py
│   ├── lab_3_2_test_variants_starter.py
│   ├── lab_4_1_multistep_retrieval_starter.py
│   └── lab_4_2_starter_files/
└── ...
```

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
