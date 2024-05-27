import os
from openai import OpenAI

# Initialize the OpenAI client
# client = openai.OpenAI(api_key='api-key-here')
client = OpenAI(api_key='api-key-here')

# Create an assistant with file search enabled
assistant = client.beta.assistants.create(
    name="RAG search assistant",
    instructions="You are a RAG search workflow. Use your knowledge base to answer questions about agricultural queries.",
    model="language-model-here",
    tools=[{"type": "file_search"}]
)

# Create a vector store for the documents
vector_store = client.beta.vector_stores.create(name="Agricultural Queries")

# List existing files to avoid duplicates
existing_files = {file.filename for file in client.beta.vector_stores.files.list(vector_store_id=vector_store.id)}

# Specify the directory path and prepare files for upload
directory_path = "F:/RAGassessment/openai-rag/data"
file_paths = [
    os.path.join(directory_path, file) for file in os.listdir(directory_path)
    if file.endswith(('.pdf', '.txt')) and file not in existing_files
]

# Upload files to the vector store and poll for completion if there are new files
if file_paths:
    file_streams = [open(path, "rb") for path in file_paths]
    file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector_store.id, files=file_streams
    )
    print(f"Batch status: {file_batch.status}")
    print(f"Files processed: {file_batch.file_counts}")
    # Close file streams after upload
    for fs in file_streams:
        fs.close()
else:
    print("No new files to upload.")

# Update the assistant to use the new vector store
client.beta.assistants.update(
    assistant_id=assistant.id,
    tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}}
)
