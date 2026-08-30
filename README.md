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

### Repository Notes
+ **Project Scope**
    * This repository includes the capstone checkpoints, lab starter/solution files, and supporting dataset folders used for developing a retrieval-augmented system.
+ **Workflow**
    * Begin with the capstone checkpoint scripts to establish the no-retrieval baseline and evaluate the need for retrieval.
    * Continue to build the retrieval pipeline using the selected Wikipedia corpus and grounded question-answering logic.
+ **Expected Output**
    * The final system should answer factual questions using information retrieved from relevant Wikipedia articles and cite supporting quotations when appropriate.
+ **Best Practice**
    * Keep the environment variables in a local .env file and avoid committing any secrets or API keys to version control.

--------------------------
