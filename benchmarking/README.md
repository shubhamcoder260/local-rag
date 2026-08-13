# Ingestion Benchmark

## Run

```bash
# 1. Start both containers
docker compose up -d --build

# 2. Pull the embedding model into the ollama container (one-time)
docker compose exec ollama ollama pull nomic-embed-text

# 3. Drop a PDF in ./data, then benchmark it
cp your_file.pdf data/
docker compose run --rm app data/your_file.pdf
```

Output is a per-stage timing breakdown (extract / chunk / embed / store) plus
throughput (chunks/sec) — printed to stdout, no dashboard, no extra services.

## Why only 2 containers

- `storage.py` uses `chromadb.PersistentClient` (embedded, file-based), not a
  ChromaDB server — so there's nothing to containerize there, just a volume
  (`chroma_storage`) so results persist across runs.
- `embedder.py` needs a real Ollama server to call, so that's the one actual
  service dependency.

## Notes

- `chroma_storage` and `ollama_models` are named volumes — data survives
  `docker compose down` (use `-v` to wipe them).
- The benchmark asserts `len(chunks) == len(embeddings)` before storing, per
  the alignment risk flagged during code review — it'll fail loudly instead
  of silently storing misaligned data.
