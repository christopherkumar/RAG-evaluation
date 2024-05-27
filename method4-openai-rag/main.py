import os
import re
import test_questions
from rag_queries import client


def clean_text(text):
    """Remove file citation markers, Python-like object structures, and other formatting characters from the text."""
    # Remove Python-like object descriptions starting with 'Text('
    text = re.sub(r"Text\(annotations=\[.*?\],\s*", "", text)
    # Remove specific leading characters 'value='
    text = re.sub(r"^value='", "", text)
    # Remove file citation markers
    text = re.sub(r"【\d+:\d+†\w+】", "", text)
    # Remove trailing characters
    text = re.sub(r"\)'$", "", text)
    # Remove newline characters
    text = re.sub(r"\\n", "", text)
    # Remove apostrophe and closing parenthesis at the end of the text
    text = re.sub(r"'\)$", "", text)
    return text


def write_actual_response_to_txt(response_file, actual_response):
    """Write the actual response to a .txt file."""
    actual_response = clean_text(actual_response)  # Clean before writing
    with open(response_file, 'w', encoding='utf-8') as f:
        f.write(actual_response)


def run_tests(model):
    for name in dir(test_questions):
        if name.startswith('test_'):
            test_func = getattr(test_questions, name)
            if callable(test_func):
                actual_response, _ = test_func()
                response_file = os.path.join(output_dir, f"openai_{model}_{name.replace('test_', '')}_response.txt")
                write_actual_response_to_txt(response_file, actual_response)
                print(f"{name} response written to {response_file}")


if __name__ == "__main__":
    my_updated_assistant = client.beta.assistants.update(
        "assistant-id-here",
        model="language-model-here"
    )

    output_dir = "F:/RAGassessment/openai-rag/responses"
    os.makedirs(output_dir, exist_ok=True)
    run_tests(my_updated_assistant.model)
