# RAG implementation, Method 3
Repurposed for testing using the resources in:
<https://github.com/PromtEngineer/localGPT.git>

## Modules
- **`constants.py`**: Defines constants and configurations for the application.
- **`ingest.py`**: Handles the ingestion of documents, processing them individually or in batches for storage and retrieval.
- **`main.py`**: Serves as the application's entry point, coordinating the overall process flow and integrating other modules.
- **`utils.py`**: Provides utility functions for logging and embedding generation.

## Key Features
- **Document Ingestion**: Load and process documents from various formats.
- **Query Processing**: Utilise advanced language models to handle and respond to queries.
- **Logging**: Maintain logs of interactions for analysis and review.
- **Embedding Generation**: Generate document and query embeddings to facilitate efficient retrieval.

## Usage
- `utils.py` contains additional functionalities for logging and embedding generation.
- Configure `constants.py` with the desired embedding and language models.
- Execute `ingest.py` to load and process documents from your specified directories.
- Run `main.py` to conduct tests and handle responses. 


## Dependencies
- `os`
- `logging`
- `click`
- `torch`
- `csv`
- `datetime`
- `pypdf2`
- `langchain`
- `huggingface`

Install these using `pip install` or your preferred Python package manager.
