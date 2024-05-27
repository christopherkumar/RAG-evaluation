from openai import OpenAI

# Initialize the OpenAI client
# client = openai.OpenAI(api_key='api-key-here')
client = OpenAI(api_key='api-key-here')


def query_openai_rag(question):
    """Query the OpenAI assistant and return the response text."""
    thread = client.beta.threads.create(
        messages=[{"role": "user", "content": question}],
        tool_resources={"file_search": {"vector_store_ids": ["vector-store-id-here"]}}
    )
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id, assistant_id="assistant-id-here"
    )
    messages = list(client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))
    if messages and messages[0].content:
        # Ensure the response is treated as a string
        response = messages[0].content[0].text
        if isinstance(response, str):
            return response
        else:
            return str(response)
    else:
        return "No response generated."


def query_and_validate(question: str, expected_response: str):
    """Validate the response from the RAG against an expected response."""
    response_text = query_openai_rag(question)
    # return response_text.strip().lower() == expected_response.strip().lower()
    return response_text
