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
+ **MIT_CAPSTONE/**
    * README.md
    * venv_requirements.txt
    * lab_solution_files/
        * lab_1_1_chatbot_solution.py
        * lab_1_2_keyword_retrieval_solution.py
        * lab_2_1_vector_retrieval_solution.py
    * lab_starter_files/
        * lab_1_1_chatbot_starter.py
        * lab_2_1_vector_retrieval_starter.py
    * lab_databases/
        * lab_1_2/
            * detailedEmails/
        * lab_2.1/
    * Final_Capstone_Project/
        * Capstone_Checkpoint_1.1/
            * MHERRERA_Capstone_Checkpoint_1_1_Solution.py
        * Capstone_Checkpoint_2.1/
            * MHERRERA_Capstone_Checkpoint_2_1_Solution.py
        * Capstone_Database/

--------------------------
