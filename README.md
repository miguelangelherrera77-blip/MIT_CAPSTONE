# CAPSTONE PROJECT
RAG and Context Engineering: Designing and Building Production-Grade AI Systems
Repository for Capstone files.

--------------------------
Miguel Herrera - Section B
--------------------------

## TABLE OF CONTENTS
+ [Project Description](#project-description)
+ [Capstone Checkpoint 3.1](#capstone-checkpoint-31-evaluation-infrastructure-and-baseline-diagnosis)
+ [Capstone Checkpoint 2.1](#capstone-checkpoint-21-retrieval-strategy-design-and-baseline-implementation)
+ [Capstone Checkpoint 1.1](#capstone-checkpoint-11-evaluating-when-retrieval-is-required)
+ [Folder Structure](#folder-structure)

--------------------------

## PROJECT DESCRIPTION
+ **Selected Scenario**
    * Wikipedia Retrieval Engine
+ **Purpose of the System**
    * The purpose of the Wikipedia Retrieval Engine is to provide a conversational interface that answers questions about significant people, places, and topics using Retrieval-Augmented Generation (RAG) over a collection of Wikipedia articles. The system should support factual questions, single-document questions that require information from one article, and multi-document questions that require comparing or combining information from multiple sources. Users should also be able to ask follow-up questions and receive responses grounded in retrieved content. To improve reliability and reduce hallucinations, the system should provide direct quotations from the source articles, allowing users to understand where the information originated and verify the response. 

--------------------------

### Capstone Checkpoint 3.1: Evaluation infrastructure and baseline diagnosis.
Goal: Build a structured evaluation framework, establish baseline performance metrics, and identify strengths, weaknesses, and likely failure modes of the Wikipedia Retrieval Engine. This checkpoint reuses the Checkpoint 2.1 **hybrid retriever** and scores it with **RAGAS**, then runs a **side-by-side comparison of original questions against paraphrased variants** to measure how robust the retriever is to rephrasing.

+ **What the checkpoint does**
    * **Grounded question generation** — `test_variables/generate_main_questions.py` samples real articles from `Capstone_Database/Wikipedia`, extracts each article's lead text, and asks the LLM (`openai/gpt-5.4-mini`) to produce one grounded question plus grading notes per article. Results are written to `test_variables/test_main_questions.json` (100 questions, each with a verified source article).
    * **Paraphrase variant generation** — `test_variables/generate_variants.py` (a standalone version of the Lab 3.2 paraphrase logic) reads the main questions and asks the LLM for **2 paraphrases per question**, preserving meaning so the same answer applies. Each variant reuses its original's grading notes and sources. Results are written to `test_variables/testinputs_variant_questions.json` (100 originals + 200 paraphrases = 300 total).
    * **RAGAS evaluation** — `capstone_checkpoint_3_1_evaluation_starter.py` is the main entry point. It loads the persisted vector store from `Capstone_Database/Capstone_Chroma_DB` (falling back to scanning `Capstone_Database/Wikipedia`), builds the hybrid BM25 + vector retriever, and scores every answer with a RAGAS `DiscreteMetric` LLM judge (pass/fail against grading notes).
    * **Originals vs paraphrases comparison** — `load_split_datasets()` builds two RAGAS datasets from a single source (`test_variables/testinputs_variant_questions.json`): the **originals** (from `test_variables/test_main_questions.json`) and the **paraphrases** (every row in the variants file that is not one of the originals). Each dataset is evaluated separately and the script reports each pass rate, the delta, and a brittle/robust verdict. If paraphrases score materially lower than originals, the retriever is brittle to rephrasing.

+ **How to run**
    * Generate main questions (optional; a set is already provided):
        * `python Final_Capstone_Project/Capstone_Checkpoint_3.1/test_variables/generate_main_questions.py 100`
    * Generate paraphrase variants (2 per question):
        * `python Final_Capstone_Project/Capstone_Checkpoint_3.1/test_variables/generate_variants.py`
    * Run the RAGAS originals-vs-paraphrases evaluation:
        * `python Final_Capstone_Project/Capstone_Checkpoint_3.1/capstone_checkpoint_3_1_evaluation_starter.py`
    * Per-dataset CSV results are written under `Capstone_Checkpoint_3.1/ragas_experiments/experiments/`, and a run log to `checkpoint_3_1_evaluation.log`.

+ **Checkpoint 3.1 Configuration Requirements**
    * Add your own **OpenRouter API Key** to the _OPENROUTER_API_KEY_ variable in the **.env** file.
    * Ensure the Wikipedia corpus is at **Capstone_Database/Wikipedia** and (optionally) the persisted vector store at **Capstone_Database/Capstone_Chroma_DB**.
    * Model: `openai/gpt-5.4-mini` (answers, judge, and generation); embeddings: `openai/text-embedding-3-small`.
    * Packages:
        * ragas
        * beautifulsoup4
        * chromadb
        * langchain-chroma
        * langchain-core
        * langchain-openai
        * openai
        * python-dotenv
        * rank-bm25

--------------------------

### Capstone Checkpoint 2.1: Retrieval strategy design and baseline implementation.
Goal: Implement and evaluate a hybrid Retrieval-Augmented Generation (RAG) system that combines BM25 keyword retrieval and vector search over the Wikipedia article collection.

+ **Checkpoint 2.1 Configuration Requirements**
    * Add your own **OpenRouter API Key** to the _OPENROUTER_API_KEY_ variable in the **.env** file.
    * Create a folder named **Capstone_Database** in the root directory of the capstone project.
    * Place the folder containing the Wikipedia HTML files inside **Capstone_Database** as **Capstone_Database/Wikipedia**.
    * Packages:
        * beautifulsoup4
        * chromadb
        * langchain-chroma
        * langchain-core
        * langchain-openai
        * python-dotenv
        * rank-bm25

--------------------------

### Capstone Checkpoint 1.1: Evaluating when retrieval is required.
Goal: Evaluate how an LLM performs without retrieval and determine whether retrieval is required for the selected scenario.

+ **Checkpoint 1.1 Configuration Requirements**
    * Add your own **OpenRouter API Key** to the _OPENROUTER_API_KEY_ variable in the **.env** file.
    * Packages:
        * python-dotenv
        * langchain-openai
        * langchain-core

--------------------------

## FOLDER STRUCTURE
```text
MIT_CAPSTONE/
├── .env                                    # Environment configuration (API keys and secrets)
├── README.md                               # Project documentation and overview
├── venv_requirements.txt                   # Python dependencies for virtual environment
├── Capstone_Database/                      # Wikipedia articles and Chroma vector database
└── Final_Capstone_Project/                 # Main capstone project directory
    ├── Capstone_Checkpoint_1.1/            # Checkpoint 1.1: Evaluating when retrieval is required
    │   ├── MHERRERA_Capstone_Checkpoint_1_1_Solution.py
    │   ├── MHerrera_Capstone_Checkpoint_1_1_Worksheet.docx
    │   ├── MHerrera_Capstone_Checkpoint_1_1_Worksheet.pdf
    │   └── checkpoint_1_1_responses.log
    ├── Capstone_Checkpoint_2.1/            # Checkpoint 2.1: RAG implementation
    │   ├── MHERRERA_Capstone_Checkpoint_2_1_Solution.py
    │   └── MHerrera_Capstone_Checkpoint_2_1_Worksheet-1.docx
    └── Capstone_Checkpoint_3.1/            # Checkpoint 3.1: RAGAS evaluation (originals vs paraphrases)
        ├── capstone_checkpoint_3_1_evaluation_starter.py   # Main entry point: hybrid retriever + RAGAS comparison
        ├── Required_Capstone_Checkpoint_3_1_Worksheet.docx # Checkpoint worksheet
        ├── checkpoint_3_1_evaluation.log                   # Run log (created on evaluation)
        ├── ragas_experiments/                              # Per-dataset RAGAS CSV results (created on evaluation)
        └── test_variables/
            ├── generate_main_questions.py                  # Grounded main-question generator (reads Wikipedia corpus)
            ├── generate_variants.py                        # Standalone paraphrase generator (2 variants per question)
            ├── test_main_questions.json                    # 100 grounded originals with grading notes and sources
            └── testinputs_variant_questions.json           # 100 originals + 200 paraphrases (300 total)
```

--------------------------
