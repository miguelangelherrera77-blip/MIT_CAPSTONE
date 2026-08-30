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

| Item | Description |
|------|-------------|
| `.env` | Environment configuration file (contains API keys and secrets) |
| `README.md` | Project documentation and overview |
| `venv_requirements.txt` | Python dependencies for virtual environment |
| `Final_Capstone_Project/` | Main capstone project directory |
| `Final_Capstone_Project/Capstone_Checkpoint_1.1/` | Checkpoint 1.1: Evaluating when retrieval is required |
| `Final_Capstone_Project/Capstone_Checkpoint_1.1/MHERRERA_Capstone_Checkpoint_1_1_Solution.py` | Solution script for checkpoint 1.1 |
| `Final_Capstone_Project/Capstone_Checkpoint_1.1/checkpoint_1_1_responses.log` | Log file for checkpoint 1.1 responses |
| `Final_Capstone_Project/Capstone_Checkpoint_1.1/MHerrera_Capstone_Checkpoint_1_1_Worksheet.*` | Checkpoint 1.1 worksheet (PDF and DOCX) |
| `Final_Capstone_Project/Capstone_Checkpoint_2.1/` | Checkpoint 2.1: Retrieval-Augmented Generation implementation |
| `Final_Capstone_Project/Capstone_Checkpoint_2.1/MHERRERA_Capstone_Checkpoint_2_1_Solution.py` | Solution script for checkpoint 2.1 |
| `Final_Capstone_Project/Capstone_Checkpoint_2.1/MHerrera_Capstone_Checkpoint_2_1_Worksheet-1.docx` | Checkpoint 2.1 worksheet |
| `Final_Capstone_Project/Capstone_Database/Wikipedia/` | Wikipedia articles database for RAG system |


--------------------------
