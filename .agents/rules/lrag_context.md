---
description: "Context and architecture guidelines for the L-RAG project"
trigger: "always_on"
---

# L-RAG Context
This project is a local RAG application.
- **Stack:** FastAPI (`api.py`), Streamlit (`app.py`), ChromaDB, and Ollama.
- **Ingestion (`ingestion/`):** PyMuPDF (extract) -> `word-overlap-v2` (chunk) -> `nomic-embed-text` batch 50 (embed) -> ChromaDB (store).
- **Retrieval:** Top 5 chunks fetched from ChromaDB; strictly prompted to answer from context only.
- **Limitations:** LLM counting and strict formatting (e.g., exact word limits) are unreliable. Embedding is the slowest stage (CPU bottleneck).
- **Key Files:** `README.md` contains vital setup steps (Linux systemd Ollama overrides). Benchmarks are logged to `logs/benchmark_results.csv`.

## Advanced RAG Roadmap
We are currently focusing on algorithmic improvements for text ingestion and retrieval (no new data types yet):
1. **Semantic / Recursive Chunking (`chunker.py`)**: Replace naive character overlap with structure/sentence-aware splitting to preserve semantic meaning.
2. **Asynchronous Embedding (`embedder.py`)**: Implement `asyncio` to send concurrent embedding requests to Ollama, reducing the CPU embedding bottleneck.
3. **Two-Stage Retrieval (Re-ranking) (`api.py`)**: Fetch Top 15 chunks from ChromaDB, then algorithmically score/re-rank them down to the Top 5 before feeding to the LLM to improve answer precision.
