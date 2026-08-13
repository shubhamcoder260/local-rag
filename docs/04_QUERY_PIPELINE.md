# Query Pipeline

This is the deep dive into what happens between "a question arrives" and
"a grounded, streamed answer appears in the browser." For the ingestion
side (how the data being searched got there), see
`03_INGESTION_PIPELINE.md`.

## Overview

```
 {question, source_collection, llm_model}
      │
      ▼
 ┌────────────────────┐
 │ embed the question │  ollama.embed(), same EMBEDDING_MODEL as ingestion
 └─────────┬──────────┘
           ▼
 ┌────────────────────┐
 │ get the collection  │  storage.get_collection(source_collection)
 │                      │  404 if it doesn't exist
 └─────────┬──────────┘
           ▼
 ┌────────────────────┐
 │ similarity search   │  collection.query(query_embeddings=..., n_results=5)
 │                      │  404 if no results at all
 └─────────┬──────────┘
           ▼
 ┌────────────────────┐
 │ build system prompt │  retrieved chunks joined into CONTEXT, with
 │                      │  strict grounding instructions
 └─────────┬──────────┘
           ▼
 ┌────────────────────┐
 │ stream the answer   │  metadata line first, then raw token stream
 └────────────────────┘
```

Unlike ingestion, this entire flow currently lives in one place —
`/ask` in `api.py` — rather than being split into its own module. That's
called out explicitly in `02_DIRECTORY_STRUCTURE.md` as a known asymmetry
with the ingestion side, which is fully modularized into `chunker.py` /
`embedder.py` / `storage.py`. Whether `/ask`'s logic should be extracted
into something like `ingestion/query.py` (or a sibling `retrieval/`
package) is an open question — see `05_DEVELOPMENT_ROADMAP.md`.

## Step 1: embedding the question

```python
query_response = ollama.embed(model=embedder.EMBEDDING_MODEL, input=payload.question)
query_embedding = query_response["embeddings"][0]
```

This reuses `embedder.EMBEDDING_MODEL` — the exact same constant, and
therefore the exact same model, used to embed every chunk during
ingestion. This is not optional or a style preference: two embeddings are
only meaningfully comparable by distance if they came from the same model.
Note the `[0]` — `ollama.embed()`'s `input` parameter accepts a list (used
for batching during ingestion), and even a single string input still comes
back wrapped as `{"embeddings": [[...]]}`, a list of one vector, hence
indexing the first element.

## Step 2: resolving the collection

```python
try:
    doc_collection = storage.get_collection(payload.source_collection)
except Exception:
    raise HTTPException(status_code=404, detail="No document found with collection name '...'. Use GET /documents to see available options.")
```

`source_collection` is a required field on every `/ask` request — there is
no "search across all documents" mode, by design (see
`03_INGESTION_PIPELINE.md`'s note on per-document isolation). If the name
doesn't correspond to an existing collection, ChromaDB's `get_collection()`
raises, and that's caught here and turned into a `404` with a message that
points the caller at `GET /documents` to see what's actually available —
useful both for the Streamlit UI (which populates its document dropdown
from that same endpoint) and for anyone calling the API directly.

## Step 3: similarity search

```python
results = doc_collection.query(query_embeddings=[query_embedding], n_results=5)
if not results or not results["documents"] or not results["documents"][0]:
    raise HTTPException(status_code=404, detail="No relevant context found in database.")
retrieved_context = "\n\n".join(results["documents"][0])
```

`n_results=5` — the top 5 nearest chunks by embedding distance — is a fixed
constant, not currently configurable per-request. It hasn't been tuned or
benchmarked against retrieval quality; it's a reasonable starting default,
not an evidence-backed choice yet. If retrieval quality ever becomes a
focus area, this is one of the first knobs worth actually testing rather
than assuming.

The `results["documents"][0]` indexing exists because ChromaDB's `query()`
is itself designed for *batched* queries — it accepts a list of query
embeddings and returns a list of result-lists, one per query. Since exactly
one question is embedded and queried at a time here, only the first
(`[0]`) result set is ever used.

## Step 4: building the system prompt

```python
system_prompt = f"""You are a highly secure, private AI assistant.
Answer using ONLY the exact facts present in the context below. Do not add
details, descriptions, dates, or explanations that are not explicitly
written in the context, even if you know them from elsewhere. If the
context only lists an item without details, state only that it is listed,
without elaborating.

CONTEXT:
{retrieved_context}"""
```

This is the entire grounding mechanism — there's no separate re-ranking or
fact-checking step; the strategy is entirely "instruct the model firmly, in
the system prompt, to stick to the provided context." This has known,
documented limits (see `07_BENCHMARKS.md` and the README's Known
Limitations section): the model can still fabricate plausible-sounding
elaboration on open-ended "explain why" or "rate by quality" style
questions when the source material is sparse, even with this instruction
in place. Simple factual lookups are reliably grounded; open-ended
reasoning questions are not guaranteed to be.

## Step 5: streaming the response

```python
def stream_generator():
    metadata = {
        "question": payload.question,
        "source_collection": payload.source_collection,
        "context_used": retrieved_context
    }
    yield json.dumps(metadata) + "\n"

    response_stream = ollama.generate(
        model=payload.llm_model, prompt=payload.question,
        system=system_prompt, stream=True
    )
    for chunk in response_stream:
        yield chunk["response"]

return StreamingResponse(stream_generator(), media_type="text/plain")
```

**Why a metadata line first, then raw text.** The response body isn't pure
JSON and isn't pure plain text — it's a JSON object on the *first* line
only (containing the question, which collection was searched, and the
retrieved context that was used), followed by the raw, unstructured token
stream from the LLM for every line after that. This hybrid shape exists so
the client can get structured metadata (specifically the retrieved context,
useful for the "Show retrieved context" expander in the UI) without having
to wait for the entire streamed answer to finish and without needing a more
complex framing protocol (like SSE) for what's otherwise just a plain text
stream.

**`llm_model` comes from the request, not a constant.** Unlike
`EMBEDDING_MODEL`, there is no hardcoded chat model anywhere in the
codebase — the caller specifies which installed Ollama model to use for
generation on every request. This is what allows the Streamlit dropdown
(populated from `GET /models`) to offer a live choice of any locally
installed model.

## Client side: consuming the stream (`app.py`)

```python
lines = response.iter_lines(decode_unicode=True)
metadata_line = next(lines)              # pulls exactly the first line
metadata = json.loads(metadata_line)

answer_placeholder = st.empty()
full_answer = ""
for line in lines:                       # everything after the first line
    full_answer += line
    answer_placeholder.write(full_answer)
```

`next(lines)` is called exactly once, deliberately, to consume only the
first line (the JSON metadata) before the `for` loop begins — the loop
then only ever sees the raw answer text that follows. `st.empty()` +
repeatedly overwriting the same placeholder is what produces the
live-typing effect in the UI, rather than appending a new line to the page
for every streamed token.

If `record_data` (the sidebar's "TESTER MODE" checkbox) is enabled, the
completed question/answer/context/timing gets appended to
`logs/benchmark_log.csv` via `log_benchmark()` — this happens *after* the
stream fully completes, using the accumulated `full_answer` and the
`context_used` field pulled from that first metadata line.

## What's not here (yet)

- **No conversation memory.** Every question is answered independently;
  there's no chat history passed into the prompt, so follow-up questions
  ("what about the second one?") have no prior turn to refer to. This is a
  deliberate, tracked deferral — see `05_DEVELOPMENT_ROADMAP.md`.
- **No re-ranking or hybrid search.** Retrieval is pure vector similarity
  search (ChromaDB's default HNSW index) — no BM25/keyword blending, no
  cross-encoder re-ranking pass on the top-5 results before they're used.
- **No retry or timeout handling** around the two `ollama` calls in this
  flow (the embedding call and the generation call) — if Ollama is
  unreachable or hangs, the request currently hangs or fails with a plain
  `500`, without a more specific error message distinguishing "Ollama is
  down" from any other unexpected error.
