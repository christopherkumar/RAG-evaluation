import os
import logging
import click
import torch
import utils
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # for streaming response
from langchain.callbacks.manager import CallbackManager
from langchain.document_loaders import CSVLoader, PDFMinerLoader, TextLoader, UnstructuredExcelLoader, Docx2txtLoader
from langchain.document_loaders import UnstructuredFileLoader, UnstructuredMarkdownLoader, UnstructuredHTMLLoader

from chromadb.config import Settings
from prompt_template_utils import get_prompt_template
from utils import get_embeddings, log_to_csv
from load_models import load_quantized_model_awq, load_quantized_model_gguf_ggml, load_quantized_model_qptq, load_full_model
from transformers import GenerationConfig, pipeline

# Constants
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/data"
PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB"
MODELS_PATH = "D:/models/"
INGEST_THREADS = os.cpu_count() or 8
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
)

CONTEXT_WINDOW_SIZE = 4096
MAX_NEW_TOKENS = CONTEXT_WINDOW_SIZE
N_GPU_LAYERS = 100
N_BATCH = 512

# Default and alternative embedding and model options
EMBEDDING_MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"
MODEL_ID = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
MODEL_BASENAME = "mistral-7b-instruct-v0.2.Q8_0.gguf"

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

def read_questions(file_path):
    """
    Reads questions from a file with each question starting with a number followed by a dot.
    """
    questions = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            number, question = line.split('.', 1)
            questions.append((int(number.strip()), question.strip()))
    return questions

def save_response(model_id, question_number, question, response, base_dir, show_sources, source_documents=None):
    """
    Saves the response to a question in a specified directory.
    """
    model_id = model_id.split('/')[-1]
    directory = os.path.join(base_dir, f"{model_id}-responses")
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, f"localGPT_{model_id}-question_{question_number}.md")
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(response)
        if show_sources and source_documents:
            file.write("\n----------------------------------SOURCE DOCUMENTS---------------------------\n")
            for document in source_documents:
                file.write(f"\n> {document.metadata['source']}:\n{document.page_content}\n")
            file.write("----------------------------------SOURCE DOCUMENTS---------------------------\n")

def load_model(device_type, model_id, model_basename=None, LOGGING=logging):
    """
    Select a model for text generation using the HuggingFace library.
    """
    logging.info(f"Loading Model: {model_id}, on: {device_type}")
    if model_basename is not None:
        if ".gguf" in model_basename.lower():
            llm = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
        elif ".ggml" in model_basename.lower():
            model, tokenizer = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
        elif ".awq" in model_basename.lower():
            model, tokenizer = load_quantized_model_awq(model_id, LOGGING)
        else:
            model, tokenizer = load_quantized_model_qptq(model_id, model_basename, device_type, LOGGING)
    else:
        model, tokenizer = load_full_model(model_id, model_basename, device_type, LOGGING)

    generation_config = GenerationConfig.from_pretrained(model_id)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=MAX_NEW_TOKENS, temperature=0.2, repetition_penalty=1.15, generation_config=generation_config)
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

def retrieval_qa_pipline(device_type, use_history, promptTemplate_type="llama"):
    """
    Initializes and returns a retrieval-based Question Answering (QA) pipeline.
    """
    embeddings = get_embeddings(device_type)
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    prompt, memory = get_prompt_template(promptTemplate_type=promptTemplate_type, history=use_history)
    llm = load_model(device_type, model_id=MODEL_ID, model_basename=MODEL_BASENAME, LOGGING=logging)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, callbacks=callback_manager, chain_type_kwargs={"prompt": prompt, "memory": memory if use_history else None})
    return qa

@click.command()
@click.option("--device_type", default="cuda" if torch.cuda.is_available() else "cpu", type=click.Choice(["cpu", "cuda", "ipu", "xpu", "mkldnn", "opengl", "opencl", "ideep", "hip", "ve", "fpga", "ort", "xla", "lazy", "vulkan", "mps", "meta", "hpu", "mtia"]), help="Device to run on. (Default is cuda)")
@click.option("--show_sources", "-s", is_flag=True, help="Show sources along with answers (Default is False)")
@click.option("--use_history", "-h", is_flag=True, help="Use history (Default is False)")
@click.option("--model_type", default="llama", type=click.Choice(["llama", "mistral", "non_llama"]), help="model type, llama, mistral or non_llama")
@click.option("--save_qa", is_flag=True, help="whether to save Q&A pairs to a CSV file (Default is False)")
def main(device_type, show_sources, use_history, model_type, save_qa):
    """
    Implements the main information retrieval task for a localGPT.
    """
    qa = retrieval_qa_pipline(device_type, use_history, promptTemplate_type=model_type)
    base_dir = "F:/RAGassessment/localGPT"
    questions_file_path = os.path.join(base_dir, "testquestions-final.txt")
    questions = read_questions(questions_file_path)
    response_dir = os.path.join(base_dir, f"localGPT_{MODEL_ID.split('/')[-1]}-responses")

    if not os.path.exists(response_dir):
        os.makedirs(response_dir)

    for question_number, question in questions:
        res = qa(question)
        answer = res["result"]
        source_docs = res["source_documents"] if show_sources else None
        save_response(MODEL_ID, question_number, question, answer, response_dir, show_sources, source_docs)
        if save_qa:
            log_to_csv(question, answer)

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO)
    main()
