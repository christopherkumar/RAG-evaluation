@echo off
SET BASE_DIR=F:\RAGassessment\rag-tutorial-v2\

@REM echo Running tests with <ollama embedding model>
@REM python %BASE_DIR%main.py --embedding_model <ollama embedding model>

echo Running tests with nomic-embed-text
python %BASE_DIR%main.py --embedding_model nomic-embed-text

echo All tests completed.
