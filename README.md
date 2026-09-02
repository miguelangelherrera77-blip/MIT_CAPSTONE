# CAPSTONE PROJECT
RAG and Context Engineering: Designing and Building Production-Grade AI Systems
Repository for Capstone files.

--------------------------
Miguel Herrera - Section B
--------------------------

## PROJECT DESCRIPTION
+ **Selected Scenario**
    * Wikipedia Retrieval Engine
+ **Purpose of the System**
    * The purpose of the Wikipedia Retrieval Engine is to provide a conversational interface that answers questions about significant people, places, and topics using Retrieval-Augmented Generation (RAG) over a collection of Wikipedia articles. The system should support factual questions, single-document questions that require information from one article, and multi-document questions that require comparing or combining information from multiple sources. Users should also be able to ask follow-up questions and receive responses grounded in retrieved content. To improve reliability and reduce hallucinations, the system should provide direct quotations from the source articles, allowing users to understand where the information originated and verify the response. 

+ **Capstone Configuration Requirements**
    * Add your own **OpenRouter API Key** to the _OPENROUTER_API_KEY_ variable in the **.env** file.
    * Packages:
        * python-dotenv
        * langchain-openai
        * langchain-core

--------------------------

### Capstone Checkpoint 2.1: Retrieval strategy design and baseline implementation.
Goal: Implement and evaluate a hybrid Retrieval-Augmented Generation (RAG) system that combines BM25 keyword retrieval and vector search over the Wikipedia article collection.

+ **Checkpoint 2.1 Configuration Requirements**
    * Add your own **OpenRouter API Key** to the _OPENROUTER_API_KEY_ variable in the **.env** file.
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
└── Final_Capstone_Project/                 # Main capstone project directory
    ├── Capstone_Checkpoint_1.1/            # Checkpoint 1.1: Evaluating when retrieval is required
    │   ├── MHERRERA_Capstone_Checkpoint_1_1_Solution.py
    │   ├── MHerrera_Capstone_Checkpoint_1_1_Worksheet.docx
    │   ├── MHerrera_Capstone_Checkpoint_1_1_Worksheet.pdf
    │   └── checkpoint_1_1_responses.log
    ├── Capstone_Checkpoint_2.1/            # Checkpoint 2.1: RAG implementation
    │   ├── MHERRERA_Capstone_Checkpoint_2_1_Solution.py
    │   └── MHerrera_Capstone_Checkpoint_2_1_Worksheet-1.docx
    └── Capstone_Database/                  # Database directory with Wikipedia articles
        └── Wikipedia/
```

--------------------------
