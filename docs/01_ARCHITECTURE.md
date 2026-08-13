# Architecture

This document describes the system as a set of running processes and how
they communicate — not the internal code structure (see
`02_DIRECTORY_STRUCTURE.md` for that) and not the step-by-step logic of
ingestion or querying (see `03_INGESTION_PIPELINE.md` and
`04_QUERY_PIPELINE.md`).

## System diagram

```
                                ┌────────────┐
                                │  Browser   │
                                └────────────┘
                                      │ http://localhost:8501
┌────────────────────────────────────────────────────────────────────────────┐
│  Your machine                                                              │
│                                                                            │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │  Docker compose network                                          │     │
│   │                                                                  │     │
│   │   ┌──────────────────┐            ┌──────────────────┐           │     │
│   │   │   api service    │◄─ HTTP ────│   app service    │           │     │
│   │   │  FastAPI :8000   │            │  Streamlit:8501  │           │     │
│   │   └──────────────────┘            └──────────────────┘           │     │
│   └──────────────────────────────────────────────────────────────────┘     │
│                │                                                           │
│                │ host.docker.internal:11434                                │
│                │                                                           │
│     ┌────────────────────────────────┐                                     │
│     │         Ollama (host)          │                                     │
│     │    embeddings + chat model     │                                     │
│     └────────────────────────────────┘                                     │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

Two boundaries in this picture matter more than anything else, because
they're the two places real bugs came from historically (see
`09_CHANGELOG.md`):

1. **Browser ↔ `app` ↔ `api`** — the browser never talks to `api` directly.
   It only ever talks to the Streamlit `app` service, which in turn calls
   `api` over Docker's internal network.
2. **`api` ↔ host Ollama** — Ollama is deliberately *not* Dockerized. It
   runs directly on the host machine, and the `api` container reaches out
   to it across the container/host boundary.

## Components

### `api` service (FastAPI)

- **What it is:** the backend. A FastAPI app, run via `uvicorn`, listening
  on port 8000 inside its container.
- **What it owns:** the entire `ingestion` package (chunking, embedding
  calls, PDF extraction, and the ChromaDB-backed vector store). It is the
  only component that talks to ChromaDB or to Ollama.
- **Built from:** `Dockerfile.api` — a `python:3.11-slim` base image,
  installs `requirements.txt`, copies in the application code, and runs
  `uvicorn api:app --host 0.0.0.0 --port 8000`.
- **Persistence:** ChromaDB is embedded *inside this container's process*
  (via `chromadb.PersistentClient`), not a separate database server. Its
  on-disk data lives at `./chroma_storage` inside the container, which is
  mounted out to the `chroma_data` Docker volume so it survives rebuilds.

### `app` service (Streamlit)

- **What it is:** the frontend/UI. A Streamlit app listening on port 8501.
- **What it owns:** file upload UI, the "ask a question" form, the
  model/document dropdowns, the "Clear Knowledge Base" button, and optional
  local benchmark CSV logging (`logs/benchmark_log.csv`, mounted out to the
  `benchmark_data` volume).
- **What it does *not* own:** it never touches ChromaDB or Ollama directly.
  Every action goes through an HTTP call to `api`, using
  `API_BASE_URL=http://api:8000` — `api` here is Docker Compose's internal
  DNS resolving the service name to the right container, not a real
  hostname on the network.
- **Built from:** `Dockerfile.app` — same base image pattern as `api`, but
  runs `streamlit run app.py --server.address=0.0.0.0 --server.port=8501`.

### Ollama (host, not containerized)

- **What it is:** the actual LLM runtime — both the embedding model
  (`nomic-embed-text`, hardcoded and required) and whichever chat model the
  user selects.
- **Why it's not in Docker:** running Ollama inside a container would mean
  either bundling model weights into the image (huge, wasteful, and
  redundant if you use Ollama for anything else on the same machine) or
  managing a separate persistent volume just for models — running it on the
  host and reaching into it from Docker was the simpler, more standard
  pattern, at the cost of needing the networking fix described next.
- **How `api` reaches it:** two pieces of Docker Compose configuration work
  together —
  - `environment: OLLAMA_HOST=http://host.docker.internal:11434` tells the
    `api` container where to send Ollama requests.
  - `extra_hosts: host.docker.internal:host-gateway` is what makes
    `host.docker.internal` actually resolve to the host machine's IP from
    inside the container — without this line, that hostname simply
    wouldn't exist on Linux (it's automatic on Docker Desktop for
    Mac/Windows, but has to be added explicitly on Linux).
  - On the host side, Ollama itself has to be listening on `0.0.0.0`, not
    the Linux default of `127.0.0.1` — otherwise it refuses connections
    from outside the host entirely, container or not. This is the systemd
    override documented in the README's Linux-specific setup section, and
    is exactly what `setup.sh` automates.

## Docker Compose topology

```yaml
services:
  api:
    ports: ["8000:8000"]
    volumes: ["chroma_data:/app/chroma_storage"]
    environment: [OLLAMA_HOST=http://host.docker.internal:11434]
    extra_hosts: ["host.docker.internal:host-gateway"]

  app:
    ports: ["8501:8501"]
    volumes: ["benchmark_data:/app/logs"]
    environment: [API_BASE_URL=http://api:8000]
    depends_on: [api]

volumes:
  chroma_data:
  benchmark_data:
```

A few details worth calling out explicitly:

- **`depends_on: [api]`** only controls *startup order* (Compose starts
  `api` first) — it does not wait for `api` to actually be ready to accept
  requests. If `app` makes a request before `api` has finished starting,
  that request fails; it isn't retried automatically at the Compose level.
- **Two separate named volumes**, not one — `chroma_data` and
  `benchmark_data` are independent, mounted into different containers, and
  can be inspected/backed up/cleared independently
  (`docker volume ls`, `docker volume rm ...`).
- **Ports 8000 and 8501 are both published to the host** (`"8000:8000"` and
  `"8501:8501"`), meaning `api` is directly reachable from outside Docker
  too (e.g. `curl http://localhost:8000/models` works from the host, not
  just from `app`). This is intentional — it's what makes `bulk_ingest.py`
  and direct API testing possible without going through the UI.

## Data flow, at the architecture level

The detailed step-by-step logic lives in `03_INGESTION_PIPELINE.md` and
`04_QUERY_PIPELINE.md`; at the architecture level, the two flows look like
this:

**Ingesting a document:**
`Browser → app (upload) → api (/ingest or /ingest/pdf) → ingestion.pipeline
→ [chunker → embedder → Ollama (host) → storage → ChromaDB]`

**Asking a question:**
`Browser → app (ask) → api (/ask) → Ollama (host, embed the question) →
ChromaDB (similarity search) → Ollama (host, generate streamed answer) →
api (streams response) → app (renders token-by-token) → Browser`

Note that `/ask` calls out to Ollama **twice** in one request — once to
embed the question, once to generate the answer — and both calls cross the
container/host boundary described above.

## Deployment

Two ways to bring the whole system up, both covered in more detail in their
own places:

1. **Manual**, per the README: install prerequisites, fix Ollama's host
   binding, pull models, then `docker compose up --build`.
2. **Scripted**, via `setup.sh` (Linux only currently): an idempotent,
   interactive installer that checks/installs Ollama, applies the systemd
   host-binding fix, detects hardware and recommends a chat model, pulls
   both required models, verifies Docker, and runs `docker compose up
   --build` itself.

Either path produces the same running system described above — `setup.sh`
doesn't change the architecture, it just automates getting to it.