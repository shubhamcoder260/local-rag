Local Rag Knowledge Base

A fully local, self hosted Retrieval-Agumented Generation (RAG) system. Upload documents(pdf or text), ask question based on the document, and get answers grounded in your own files- all running on your own machine, no cloud services, no API keys, no data leaving your computer.

Built with FastAPI, ChromaDB, Ollama, and Streamlit.

Features
1. Ingest raw text or PDF documents
2. Ask questions with streaming answers
3. Per-document isolation (no cross-contamination between unrelated documents)
4. Works on Any locally-installed Ollama model
5. Testing mode for benchmarking analysis
6.Fully Dockerized

Prerequisites
Before running this project, you will need
1. Docker and Docker Compose installed
2. Ollama installed and running on your host machine(Not inside Docker)
3. Pull the required embedding model(hardcoded and necessary)



