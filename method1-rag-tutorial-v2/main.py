import os
import shutil
import time
import argparse
from tqdm import tqdm
from chromadb.api.client import SharedSystemClient
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
import pushbullet_helper as pb_helper


def get_embedding_function(embedding_model):
    """Initialize and return the embedding model."""
    embeddings = OllamaEmbeddings(model=embedding_model)
    # print(f"Initializing embedding model: {embedding_model}")  # Debug print
    return embeddings

def clear_database(chroma_path):
    """Clear the database if it exists, with retries for locked files."""
    max_retries = 3
    retry_delay = 5  # seconds
    for attempt in range(max_retries):
        if not os.path.exists(chroma_path):
            return
        try:
            # Attempt to open and close the database file to ensure it's not locked
            with open(os.path.join(chroma_path, 'chroma.sqlite3'), 'r+'):
                pass
            shutil.rmtree(chroma_path)
            print("Database cleared successfully.")
            break
        except PermissionError as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Failed to clear the database after several attempts.")

def load_documents(data_path):
    """Load documents from the specified path."""
    document_loader = PyPDFDirectoryLoader(data_path)
    return document_loader.load()

def split_documents(documents):
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def calculate_chunk_ids(chunks):
    """Calculate and assign unique IDs to each chunk."""
    last_page_id = None
    current_chunk_index = 0
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id
    return chunks

def add_to_chroma(chroma_path, chunks, embedding_function):
    """Add documents to the Chroma vector store."""
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)
    try:
        chunks_with_ids = calculate_chunk_ids(chunks)
        existing_items = db.get(include=[])
        existing_ids = set(existing_items["ids"])
        new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]
        if new_chunks:
            progress_bar = tqdm(enumerate(new_chunks, start=1), total=len(new_chunks), desc="Adding documents", unit="doc")
            for i, chunk in progress_bar:
                db.add_documents([chunk], ids=[chunk.metadata["id"]])
                if i % 100 == 0:
                    progress_bar.set_description(f"Adding documents (processed {i} documents)")
            db.persist()
    finally:
        # Custom cleanup for Chroma
        db._client._system.stop()
        SharedSystemClient._identifer_to_system.pop(db._client._identifier, None)
        db = None


def query_rag(query_text, chroma_path, llm):
    """Generate a response to the query using RAG and the specified llm."""
    db = Chroma(persist_directory=chroma_path, embedding_function=get_embedding_function('mxbai-embed-large'))
    try:
        results = db.similarity_search_with_score(query_text, k=5)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    finally:
        # Custom cleanup for Chroma
        db._client._system.stop()
        SharedSystemClient._identifer_to_system.pop(db._client._identifier, None)
        db = None
    
    prompt_template = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
""")
    prompt = prompt_template.format(context=context_text, question=query_text)
    model = Ollama(model=llm)
    response_text = model.invoke(prompt)
    return response_text


def run_tests(base_dir, questions, llms, embedding_models):
    data_path = os.path.join(base_dir, 'data')   # data directory
    chroma_path = os.path.join(base_dir, 'chroma')

    for embedding_model in embedding_models:
        clear_database(chroma_path)
        documents = load_documents(data_path)
        chunks = split_documents(documents)
        embedding_function = get_embedding_function(embedding_model)
        add_to_chroma(chroma_path, chunks, embedding_function)

        for llm in llms:
            responses_dir = os.path.join(base_dir, 'responses', f'{llm}_{embedding_model}')
            if not os.path.exists(responses_dir):
                os.makedirs(responses_dir)

            for i, question in enumerate(questions):
                response_text = query_rag(question, chroma_path, llm)
                file_path = os.path.join(responses_dir, f"response_question_{i+1}.txt")

                with open(file_path, 'w') as file:
                    file.write(response_text)

                print(f"Response saved to {file_path}")


def load_questions(filename):
    with open(filename, 'r') as file:
        questions = [line.strip() for line in file.readlines() if line.strip()]
    return questions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG assessment with specified LLM and embedding model.")
    parser.add_argument('--embedding_model', type=str, required=True, help='Embedding model to use.')
    args = parser.parse_args()
    embedding_model = args.embedding_model

    base_dir = "F:/RAGassessment/rag-tutorial-v2/"

    llms = ['llama2', 'llama3', 'mistral']
    # embedding_models = ['mxbai-embed-large', 'nomic-embed-text', 'snowflake-arctic-embed']
    questions_file = os.path.join(base_dir, "testquestions-final.txt")
    questions = load_questions(questions_file)

    run_tests(base_dir, questions, llms, [embedding_model])  # Only use the specified embedding model
