# Directory Structure

This document maps every file in the project to what it's responsible for.
For *why* the ingestion package is split the way it is, see
`03_INGESTION_PIPELINE.md`; this doc is the reference for "what file do I
open to change X."

## Top-level layout

```
local-rag/
├── api.py                  FastAPI app — routes only, no business logic
├── app.py                  Streamlit frontend
├── bulk_ingest.py           CLI tool for bulk-ingesting a folder via the API
├── setup.sh                 Interactive Linux installer/launcher
├── Dockerfile.api            Backend container image
├── Dockerfile.app            Frontend container image
├── docker-compose.yml        Ties both containers together
├── requirements.txt          Shared Python deps for both containers
├── README.md                 Setup instructions, troubleshooting, limitations
├── TEST_PLAN.md               Formal functional/cross-platform/model test plan
├── .gitignore                 Excludes venv/, chroma_storage/, logs/, etc.
├── sample_documents/           Test documents used during development
├── docs/                       This documentation folder
└── ingestion/                  The core backend logic, as an installable package
    ├── __init__.py
    ├── chunker.py
    ├── embedder.py
    ├── extractor.py
    ├── storage.py
    ├── pipeline.py
    └── reader.py
```

Two directories exist only at runtime and are gitignored, not committed:

- `chroma_storage/` — ChromaDB's on-disk data (the `chroma_data` Docker
  volume mounts here inside the `api` container).
- `logs/` — `benchmark_log.csv`, written by `app.py` when "TESTER MODE" is
  enabled (the `benchmark_data` Docker volume mounts here inside the `app`
  container).

## `api.py` — the route layer

**Role:** translates HTTP requests into calls into the `ingestion` package,
and translates whatever comes back (a result dict, or an exception) into an
HTTP response. It deliberately contains **no chunking, embedding, or
storage logic of its own** — that discipline is what makes the rest of this
package testable and reusable outside of FastAPI.

**Endpoints:**

| Method & path | Calls into | Returns |
|---|---|---|
| `POST /ingest` | `pipeline.ingest_document()` | `{status, message}` |
| `POST /ingest/pdf` | `pipeline.ingest_pdf()` | `{status, message}` |
| `POST /ask` | `ollama` directly, `storage.get_collection()` | Streamed text |
| `GET /documents` | `storage.list_documents()` | `{documents: [...]}` |
| `DELETE /knowledgebase` | `storage.clear_database()` | `{status, message}` |
| `GET /models` | `ollama.list()` | `{models: [...]}` |

Note that `/ask` is the one endpoint that still calls `ollama` directly
from `api.py` rather than going through a dedicated module — see
`04_QUERY_PIPELINE.md` for the reasoning and whether that should change.

**Exception handling pattern**, used consistently across every endpoint:
```python
try:
    ...
except HTTPException:
    raise                                    # already the right shape, pass through
except pipeline.DocumentTooLargeError as e:
    raise HTTPException(status_code=413, ...)  # translate a known business error
except Exception as e:
    raise HTTPException(status_code=500, ...)  # anything unexpected
```
This ordering matters — Python checks `except` clauses top to bottom, so
the specific `DocumentTooLargeError` clause has to come *before* the
generic `Exception` clause, or it would never be reached.

## `ingestion/` package

### `chunker.py`

**Role:** pure text-splitting. Takes a string, returns a list of chunk
strings. No I/O, no network calls, no dependencies outside the standard
library — this module can be unit-tested with nothing but plain strings.

**Exports:** `chunk_text(text, max_chars=1500, overlap_chars=150)`

Character-bounded rather than word-count-bounded, with a hard-slice
fallback for any single whitespace-delimited "word" that alone exceeds
`max_chars` (e.g. a CSV row or an unbroken log line with no internal
spaces). See `09_CHANGELOG.md` for the bug this fixed and why word-counting
failed silently on non-prose input.

### `embedder.py`

**Role:** owns the single source of truth for which embedding model is
used, and batches embedding calls to Ollama rather than issuing one network
round-trip per chunk.

**Exports:**
- `EMBEDDING_MODEL = "nomic-embed-text"` — imported by both `pipeline.py`
  (for ingestion) and `api.py` (for embedding the incoming question in
  `/ask`), so there is exactly one place this model name is ever typed.
- `embed_in_batches(chunks, batch_size=50)` — calls `ollama.embed()` with a
  list input rather than looping one chunk at a time.

### `extractor.py`

**Role:** file-format-specific text extraction. Currently PDF only.

**Exports:** `extract_text_from_pdf(file_bytes)` — uses PyMuPDF (`fitz`) in
a context manager, page by page.

This is the module `reader.py` will likely absorb or dispatch to once more
file formats are supported — see the note on `reader.py` below.

### `storage.py`

**Role:** the only module that imports `chromadb` or knows what a
"collection" is. Everything else in the codebase talks to storage through
this module's functions, never through the ChromaDB client directly.

**Exports:**
- `sanitize_collection_name(name)` — turns an arbitrary filename into a
  valid, safe ChromaDB collection name (lowercased, non-alphanumeric
  characters replaced, length-clamped).
- `get_or_create_collection(source_name)` / `get_collection(collection_name)`
- `index_document(source_name, chunks, embeddings)` — the one place chunk
  IDs are generated and the bulk `upsert()` call happens. Because both
  `/ingest` and `/ingest/pdf` route through this single function (via
  `pipeline.py`), there is no way for their ID-naming or storage behavior
  to drift apart — a real inconsistency that existed before this
  centralization; see `09_CHANGELOG.md`.
- `list_documents()`, `clear_database()`

**Design decision:** one ChromaDB collection per source document, not one
shared collection across all documents. This is what makes `/ask` require
a `source_collection` and guarantees a question about one document can
never retrieve chunks from an unrelated one.

### `pipeline.py`

**Role:** orchestration. This is the layer that knows the *order* things
happen in — chunk, then embed, then store — without knowing anything about
HTTP, FastAPI, or file uploads.

**Exports:**
- `MAX_TEXT_LENGTH = 500_000` — the single source of truth for the size
  guard, deliberately placed here rather than in `api.py`.
- `DocumentTooLargeError` — a plain `Exception` subclass, not an
  `HTTPException`. This module has zero FastAPI imports on purpose: it
  should be usable from a test script, a CLI tool, or a future non-HTTP
  entry point without dragging a web framework along. Translating this
  exception into an actual HTTP `413` is `api.py`'s job, not this module's.
- `ingest_document(text, source_name)` — chunk → embed → store, with the
  size check as the very first thing that happens, before any chunking or
  embedding work is done.
- `ingest_pdf(file_bytes, source_name)` — extracts text via
  `extractor.extract_text_from_pdf()`, then calls `ingest_document()`
  internally. Because of this, the size guard in `ingest_document()`
  automatically protects the PDF path too — one check, two callers.

### `reader.py` *(currently empty — stub)*

Not yet implemented. Based on its name and position alongside
`extractor.py`, the likely intent is a higher-level "read this file,
whatever it is" entry point that dispatches to format-specific extraction
(`extractor.py` for PDF, something new for other formats) — but this
hasn't been confirmed or built yet. Treat this as a placeholder until its
scope is actually decided.

### `__init__.py` *(currently empty)*

Present so `ingestion` is importable as a package (`from ingestion import
chunker`). Empty is a valid and common choice here — it doesn't need to
re-export anything unless a flatter import style (e.g. `from ingestion
import ingest_document` instead of `from ingestion import pipeline;
pipeline.ingest_document`) is wanted later.

## Dependency direction

```
api.py
  └── ingestion.pipeline
        ├── ingestion.chunker
        ├── ingestion.embedder
        ├── ingestion.extractor
        └── ingestion.storage

api.py
  ├── ingestion.embedder   (EMBEDDING_MODEL constant, for /ask)
  └── ingestion.storage    (get_collection, list_documents, clear_database)
```

Nothing inside `ingestion/` imports from `api.py` — the dependency arrow
only ever points one way, from the route layer down into the package,
never back up. That's what makes it possible to reason about `chunker.py`
or `embedder.py` in complete isolation from FastAPI, and it's a constraint
worth deliberately preserving as the codebase grows.
