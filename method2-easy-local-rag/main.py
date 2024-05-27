import os
import PyPDF2
import re
import json
import torch
import ollama
import argparse
from openai import OpenAI

# Iterate through each combination of language models and embedding models
llms = ["llama2", "llama3", "mistral"]
embedding_models = ["snowflake-arctic-embed"]

base_dir = "F:/RAGassessment/easy-local-rag"

# System message given to LLM
system_message = "You are a helpful assistant that is an expert at extracting the most useful information from a given text. Also bring in extra relevant information to the user query from outside the given context."

# Function to convert PDF to text and append to vault.txt
def convert_pdf_to_text(file_path):
    if file_path.endswith(".pdf"):
        with open(file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)
            text = ''
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                if page.extract_text():
                    text += page.extract_text() + " "
            
            # Normalize whitespace and clean up text
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Split text into chunks by sentences, respecting a maximum chunk size
            sentences = re.split(r'(?<=[.!?]) +', text)  # split on spaces following sentence-ending punctuation
            chunks = []
            current_chunk = ""
            for sentence in sentences:
                # Check if the current sentence plus the current chunk exceeds the limit
                if len(current_chunk) + len(sentence) + 1 < 1000:  # +1 for the space
                    current_chunk += (sentence + " ").strip()
                else:
                    # When the chunk exceeds 1000 characters, store it and start a new one
                    chunks.append(current_chunk)
                    current_chunk = sentence + " "
            if current_chunk:  # Don't forget the last chunk!
                chunks.append(current_chunk)
            with open("vault.txt", "a", encoding="utf-8") as vault_file:
                for chunk in chunks:
                    # Write each chunk to its own line
                    vault_file.write(chunk.strip() + "\n")
            print(f"PDF content from {file_path} appended to vault.txt with each chunk on a separate line.")

# Function to upload a text file and append to vault.txt
def upload_txtfile(file_path):
    if file_path.endswith(".txt"):
        with open(file_path, 'r', encoding="utf-8") as txt_file:
            text = txt_file.read()
            
            # Normalize whitespace and clean up text
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Split text into chunks by sentences, respecting a maximum chunk size
            sentences = re.split(r'(?<=[.!?]) +', text)  # split on spaces following sentence-ending punctuation
            chunks = []
            current_chunk = ""
            for sentence in sentences:
                # Check if the current sentence plus the current chunk exceeds the limit
                if len(current_chunk) + len(sentence) + 1 < 1000:  # +1 for the space
                    current_chunk += (sentence + " ").strip()
                else:
                    # When the chunk exceeds 1000 characters, store it and start a new one
                    chunks.append(current_chunk)
                    current_chunk = sentence + " "
            if current_chunk:  # Don't forget the last chunk!
                chunks.append(current_chunk)
            with open("vault.txt", "a", encoding="utf-8") as vault_file:
                for chunk in chunks:
                    # Write each chunk to its own line
                    vault_file.write(chunk.strip() + "\n")
            print(f"Text file content from {file_path} appended to vault.txt with each chunk on a separate line.")

# Function to open a file and return its contents as a string
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# Function to get relevant context from the vault based on user input
def get_relevant_context(rewritten_input, vault_embeddings, vault_content, embedding_model, top_k=3):
    if vault_embeddings.nelement() == 0:  # Check if the tensor has any elements
        return []
    # Encode the rewritten input
    input_embedding = ollama.embeddings(model=embedding_model, prompt=rewritten_input)["embedding"]
    # Compute cosine similarity between the input and vault embeddings
    cos_scores = torch.cosine_similarity(torch.tensor(input_embedding).unsqueeze(0), vault_embeddings)
    # Adjust top_k if it's greater than the number of available scores
    top_k = min(top_k, len(cos_scores))
    # Sort the scores and get the top-k indices
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    # Get the corresponding context from the vault
    relevant_context = [vault_content[idx].strip() for idx in top_indices]
    return relevant_context

def rewrite_query(user_input_json, conversation_history, ollama_model):
    user_input = json.loads(user_input_json)["Query"]
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-2:]])
    prompt = f"""Rewrite the following query by incorporating relevant context from the conversation history.
    The rewritten query should:

    - Preserve the core intent and meaning of the original query
    - Expand and clarify the query to make it more specific and informative for retrieving relevant context
    - Avoid introducing new topics or queries that deviate from the original query
    - DONT EVER ANSWER the Original query, but instead focus on rephrasing and expanding it into a new query

    Return ONLY the rewritten query text, without any additional formatting or explanations.

    Conversation History:
    {context}

    Original query: [{user_input}]

    Rewritten query:
    """
    response = client.chat.completions.create(
        model=ollama_model,
        messages=[{"role": "system", "content": prompt}],
        max_tokens=200,
        n=1,
        temperature=0.1,
    )
    rewritten_query = response.choices[0].message.content.strip()
    return json.dumps({"Rewritten Query": rewritten_query})

def ollama_chat(user_input, system_message, vault_embeddings, vault_content, ollama_model, embedding_model, conversation_history):
    conversation_history.append({"role": "user", "content": user_input})

    if len(conversation_history) > 1:
        query_json = {
            "Query": user_input,
            "Rewritten Query": ""
        }
        rewritten_query_json = rewrite_query(json.dumps(query_json), conversation_history, ollama_model)
        rewritten_query_data = json.loads(rewritten_query_json)
        rewritten_query = rewritten_query_data["Rewritten Query"]
        # print(PINK + "Original Query: " + user_input)
        # print(PINK + "Rewritten Query: " + rewritten_query)
    else:
        rewritten_query = user_input

    relevant_context = get_relevant_context(rewritten_query, vault_embeddings, vault_content, embedding_model)
    if relevant_context:
        context_str = "\n".join(relevant_context)
        
    user_input_with_context = user_input
    if relevant_context:
        user_input_with_context = user_input + "\n\nRelevant Context:\n" + context_str

    conversation_history[-1]["content"] = user_input_with_context

    messages = [
        {"role": "system", "content": system_message},
        *conversation_history
    ]

    response = client.chat.completions.create(
        model=ollama_model,
        messages=messages,
        max_tokens=2000,
    )
    conversation_history.append({"role": "assistant", "content": response.choices[0].message.content})
    return response.choices[0].message.content

def delete_vault_file():
    if os.path.exists("vault.txt"):
        os.remove("vault.txt")
        print("Deleted vault.txt successfully.")

# Parse command-line arguments
print("Parsing command-line arguments...")
parser = argparse.ArgumentParser(description="Ollama Chat")
parser.add_argument("--model", default="llama3", help="Ollama model to use (default: llama3)")
args = parser.parse_args()

# Configuration for the Ollama API client
print("Initializing Ollama API client...")
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='llama3'
)

# Load the vault content
print("Loading vault content...")
vault_content = []
if os.path.exists("vault.txt"):
    with open("vault.txt", "r", encoding='utf-8') as vault_file:
        vault_content = vault_file.readlines()

# Generate embeddings for the vault content using Ollama
print("Generating embeddings for the vault content...")

def generate_embeddings(vault_content, embedding_model):
    vault_embeddings = []
    for content in vault_content:
        response = ollama.embeddings(model=embedding_model, prompt=content)
        vault_embeddings.append(response["embedding"])
    return torch.tensor(vault_embeddings)

# Load questions from the testquestions-final.txt file
questions_file_path = os.path.join(base_dir, "testquestions-final.txt")
questions = []
if os.path.exists(questions_file_path):
    with open(questions_file_path, 'r', encoding='utf-8') as questions_file:
        questions = questions_file.readlines()

# Directory to save responses
responses_dir = os.path.join(base_dir, "responses")

# Ensure the responses directory exists
os.makedirs(responses_dir, exist_ok=True)

# Process files from the data directory
data_directory = os.path.join(base_dir, "data")
for filename in os.listdir(data_directory):
    file_path = os.path.join(data_directory, filename)
    if filename.endswith(".pdf"):
        convert_pdf_to_text(file_path)
    elif filename.endswith(".txt"):
        upload_txtfile(file_path)

for embedding_model in embedding_models:
    print(f"Processing with Embedding Model: {embedding_model}")

    # Call the function to delete vault.txt here
    delete_vault_file()

    # Generate embeddings for the current embedding model
    vault_embeddings_tensor = generate_embeddings(vault_content, embedding_model)

    for llm in llms:
        print(f"Processing with LLM: {llm}")
        
        # Directory for current combination
        combination_dir = os.path.join(responses_dir, f"{llm}_{embedding_model}")
        os.makedirs(combination_dir, exist_ok=True)
        
        # Conversation history for the current combination
        conversation_history = []
        
        for idx, question in enumerate(questions, start=1):
            print(f"Processing Question {idx}: {question.strip()}")
            response = ollama_chat(question.strip(), system_message, vault_embeddings_tensor, vault_content, llm, embedding_model, conversation_history)
            
            # Save response to a separate TXT file
            response_file_path = os.path.join(combination_dir, f"response_{idx}.txt")
            with open(response_file_path, 'w', encoding='utf-8') as response_file:
                response_file.write(response)
            
            print(f"Response saved to {response_file_path}")

os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")