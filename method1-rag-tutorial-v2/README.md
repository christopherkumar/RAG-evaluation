# RAG implementation, Method 1
Repurposed for testing using the resources in:
<https://github.com/pixegami/rag-tutorial-v2.git>

## Features
- **Embedding Initialisation**: Initialise embedding models tailored for specific document processing needs.
- **Document Loading and Processing**: Load and split documents into chunks that can be efficiently managed and processed.
- **Document Chunking**: Organise large documents into smaller, manageable pieces with unique identifiers for efficient storage and retrieval.
- **Vector Storage**: Utilise the Chroma vector store to maintain embeddings of document chunks.
- **Query Processing**: Leverage the RAG framework and language models to generate responses based on the content stored in the vector database.

## Key Functions
1. **get_embedding_function**: Retrieve an embedding model for document processing.
2. **load_documents**: Load documents from a specified directory.
3. **split_documents**: Divide documents into smaller sections or chunks.
4. **calculate_chunk_ids**: Assign unique identifiers to each document chunk for tracking and retrieval.
5. **add_to_chroma**: Store document chunks in the Chroma vector database.
6. **query_rag**: Process queries using the stored document data to generate intelligent responses.

## Usage
- Ensure ollama is installed and running on your computer. To see how to do this, visit <https://github.com/ollama/ollama>.
- Ensure the desired language model(s) and embedding model(s) have already been pulled using `ollama pull <model>`.
- Set embedding models in `run_main.bat` using the syntax `@REM python %BASE_DIR%main.py --embedding_model <ollama embedding model>`.
- To run more embedding models, repeat the above line of text with a different embedding model in `run_main.bat`.
- Define language models in `main.py` in the array `llms=[<ollama language model(s)>]`.
- Move/Copy the `data` directory into `method1-rag-tutorial-v2`.
- Run `run_main.bat` to conduct tests and handle responses. This will iterate through `main.py` using the embedding models defined.
- Questions are pulled from `testquestions-final.txt`.

## Dependencies
- `tqdm`
- `argparse`
- `langchain_community`
- `chromadb`

Make sure to install these libraries using `pip install` or similar commands based on your Python environment setup.
