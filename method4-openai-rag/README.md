# RAG implementation, Method 4
RAG implementation using OpenAI vector stores, assistants and file search utilities.

## Modules
- **`main.py`**: Manages text cleaning and testing functionality, ensuring that output is formatted correctly and stored appropriately.
- **`rag_queries.py`**: Handles the core functionality of submitting queries to OpenAI and validating responses.
- **`upload_docs.py`**: Configures and manages the document environment, enabling RAG searches.

## Key Features
- **Text Cleaning**: Removes unwanted formatting from text to ensure clarity and correctness.
- **Query Handling**: Processes queries with OpenAI's RAG to fetch relevant responses based on the document context.
- **Document Management**: Sets up and maintains a knowledge base for the RAG system.

## Usage
- Run `upload_docs.py` to manage the documents that serve as the knowledge base for queries. OpenAI API key, vector_store_id and assistant_id initialised here. Vector store and chat assistant created if not already. Set in .env for future use. 
- `rag_queries.py` used for querying OpenAI with questions and handling the responses.
- - Move/Copy the `data` directory into `method3-localGPT`.
- Run `main.py` to conduct tests and handle responses. Update language model if desired.
- Questions are pulled from `testquestions-final.py`. This file is carried over from a separate project, hence the additional unused parts within this script.

## Dependencies
- `os`
- `re`
- `openai`
- OpenAI API: Ensure your API key, and other IDs are configured correctly.
