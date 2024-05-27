# RAG implementation, Method 2
Repurposed for testing using the resources in:
https://github.com/AllAboutAI-YT/easy-local-rag.git

## Features
- **PDF to Text Conversion**: Converts PDF documents to text, making them accessible for further processing.
- **Text File Handling**: Processes text files into manageable chunks for analysis.
- **Contextual Query Handling**: Uses embeddings to find relevant context for queries, enhancing the quality of responses.
- **Query Rewriting**: Improves user queries by integrating previous conversation contexts, making the interactions more relevant and detailed.
- **Chat Handling**: Manages chat interactions, dynamically integrating user inputs into the system's knowledge base.

## Key Functions
- **convert_pdf_to_text**: Extracts text from PDF files.
- **upload_txtfile**: Processes text files into smaller, manageable chunks.
- **get_relevant_context**: Finds the most relevant pieces of text based on input queries.
- **rewrite_query**: Enhances queries using AI to include more context and detail.
- **ollama_chat**: Manages a full chat session, processing and responding to user inputs.
- **delete_vault_file**: Cleans up by deleting specified files.
- **generate_embeddings**: Creates embeddings from text for use in machine learning models.

## Usage
- Ensure the desired language model(s) and embedding model(s) have already been pulled using `ollama pull <model>`.
- Run `main.py` to conduct tests and handle responses. 

## Dependencies
- `pypdf2`
- `torch`
- `ollama`
- `re`
- `json`
- `argparse`

Ensure these libraries are installed using `pip install` or a similar package manager as per your Python environment setup.
