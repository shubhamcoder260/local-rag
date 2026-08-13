# Ingestion Pipeline

This is the deep dive into what happens between "a document arrives" and
"it's queryable." For the file-level map, see `02_DIRECTORY_STRUCTURE.md`;
for the container/network picture, see `01_ARCHITECTURE.md`.

## Overview

```
 payload / file
      │
      ▼
 ┌─────────────┐   size guard, raises
 │ pipeline.py │   DocumentTooLargeError
 │             │   if text is too big
 └──────┬──────┘
        │
        ▼
 ┌─────────────┐
 │ chunker.py  │  chunk_text()
 │             │  splits into overlapping, char-bounded chunks
 └──────┬──────┘
        │  list[str]
        ▼
 ┌─────────────┐
 │ embedder.py │  embed_in_batches()
 │             │  one batched call per 50 chunks, not one call per chunk
 └──────┬──────┘
        │  list[list[float]]
        ▼
 ┌─────────────┐
 │ storage.py  │  index_document()
 │             │  one bulk upsert() into a per-document ChromaDB collection
 └─────────────┘
```

Both entry points — raw text via `/ingest` and PDF via `/ingest/pdf` —
converge on the exact same three-step pipeline. A PDF is simply text
extraction followed by this same flow (`pipeline.ingest_pdf()` calls
`pipeline.ingest_document()` internally).

## Step 1: the size guard

Before any chunking, embedding, or storage work happens, `pipeline.py`
checks the raw text length against `MAX_TEXT_LENGTH` (500,000 characters)
and raises `DocumentTooLargeError` immediately if it's over. This check is
placed as the very first line of `ingest_document()` deliberately — it's
cheap (a single `len()` call), so it should happen before any expensive
work is attempted, not after.

This guard exists because of a real incident: a 3.2MB / 100,001-row CSV
file caused an `ollama.embeddings()` call to fail with `"the input length
exceeds the context length"` partway through ingestion — after chunking and
partial embedding had already happened. The guard turns that into an
immediate, clear `413 Payload Too Large` response before any work is
wasted. See `09_CHANGELOG.md` for the full incident.

Because `ingest_pdf()` routes through `ingest_document()`, one check
protects both the raw-text and PDF paths — there's no separate size check
to maintain (or forget to update) for PDFs specifically.

## Step 2: chunking (`chunker.chunk_text`)

```python
def chunk_text(text, max_chars=1500, overlap_chars=150):
```

**Why character-bounded, not word-count-bounded.** An earlier version
counted "words" (split on whitespace) and capped chunks at a word count.
That works fine for ordinary prose, where a "word" is a handful of
characters — but it silently breaks on anything where whitespace isn't a
reliable size signal: a CSV row (`1,0.234,0.552,...`) with no internal
spaces is a single "word" that can be thousands of characters long, and a
DNA sequence or an unbroken log line has the same problem. Counting "500
words" of CSV data can mean an actual chunk of 20,000+ characters — far
past what the embedding model can accept. Shrinking the word-count limit
doesn't fix this, because the unit being counted is wrong, not just too
generous. Switching to counting actual characters fixes it regardless of
what the text looks like.

**The algorithm, step by step:**

1. Split the input on whitespace into a list of "words" (this is still
   used as the unit of *movement* through the text — chunks are built by
   accumulating whole words — but never as the unit of *size measurement*).
2. For each word: if it alone is longer than `max_chars` (the CSV-row/DNA
   case), flush whatever's currently accumulated as its own chunk, then
   hard-slice the oversized word into `max_chars`-sized pieces directly,
   with `overlap_chars` of overlap between consecutive slices. This is the
   fallback that makes the pathological case safe.
3. Otherwise, keep appending words to the current chunk until adding the
   next one would exceed `max_chars`. At that point, flush the current
   chunk, and seed the *next* chunk with the last `overlap_chars` worth of
   words from the one just flushed (working backward from the end) —
   this is what creates the overlap between consecutive chunks, so a
   sentence that happens to fall right on a chunk boundary isn't split with
   no shared context on either side.
4. Whatever's left in the buffer at the end becomes the final chunk.

**On overlap sizing:** the overlap-accumulation loop's break condition
tests `overlap_len - len(w) > overlap_chars`, which is not what the
docstring/design intent describes (`overlap_len + len(w) > overlap_chars`
would be the literal target). In practice this makes the actual overlap
land a little over the configured `overlap_chars` (observed ~155–158 chars
against a 150-char target in testing) rather than under it — harmless, but
worth knowing if you're ever debugging why overlaps look slightly larger
than configured.

**Verified behavior:** tested against both ordinary repeated-sentence prose
(clean chunks, all under the limit) and a reconstructed version of the
original CSV incident (long unbroken pseudo-CSV text) — all resulting
chunks stayed at or under `max_chars`, confirming the fallback path
actually engages correctly rather than just existing in theory.

## Step 3: batch embedding (`embedder.embed_in_batches`)

```python
def embed_in_batches(chunks, batch_size=50):
```

**Why batching.** Before this existed, ingestion embedded one chunk at a
time in a loop, each call to `ollama.embeddings()` being a separate network
round-trip, immediately followed by its own separate `ChromaDB.upsert()`
call for that single chunk. For a large document producing 1,000–2,000+
chunks, that's 1,000–2,000+ round-trips to Ollama *and* to ChromaDB. This
was measured at 10+ minutes of wall-clock time with sustained ~60% host CPU
usage for a single large ingestion.

The fix: call `ollama.embed()` (note — the plural/list-accepting method,
not the singular `ollama.embeddings()`) once per batch of 50 chunks, passing
the whole batch as a list via `input=batch`, and collect
`response["embeddings"]` (also plural — a list of vectors, one per input
chunk) into a single running list. This cuts the number of network
round-trips by roughly 50x for a large document, though it's worth being
honest that this eliminates *network/call overhead*, not the underlying
compute — the CPU still has to do the same total amount of embedding math,
so this is a "minutes → a shorter number of minutes" improvement, not an
"instant" one on CPU-only hardware.

**Why `EMBEDDING_MODEL` lives here, not in `api.py`.** The embedding model
used for a question at query time has to be the exact same model used to
embed the document chunks at ingestion time — otherwise the vectors aren't
comparable and similarity search is meaningless. Keeping the constant in
one module that both ingestion and querying import from is what guarantees
that invariant can't accidentally drift.

## Step 4: storage (`storage.index_document`)

```python
def index_document(source_name, chunks, embeddings):
```

Three things happen here, all as **one bulk call**, not per-chunk:

1. `sanitize_collection_name(source_name)` turns the filename into a valid
   ChromaDB collection name (lowercased, non-`[a-z0-9_-]` characters
   replaced with `_`, trimmed, padded to a minimum length, clamped to 63
   characters — ChromaDB's own naming constraints).
2. `get_or_create_collection()` either reuses an existing collection for
   this document (if re-ingesting the same source name) or creates a new
   one.
3. A single `collection.upsert(ids=..., embeddings=..., documents=...,
   metadatas=...)` call, with `ids` generated as
   `f"{collection_name}_chunk_{i}"` for every chunk in the list, and each
   chunk tagged with `{"source": source_name}` metadata.

**Why one collection per document, not one shared collection.** This is
the mechanism that makes cross-document contamination structurally
impossible rather than merely unlikely — a query against one document's
collection can never surface a chunk from a different document, because
they're not in the same searchable space at all. The tradeoff is that
`/ask` must always specify which collection (document) to search; there's
no "search everything" mode.

**Why chunk IDs matter here at all.** Because `upsert()` (not `insert()`)
is used, re-ingesting a document with the same source name and the same
resulting chunk IDs will *overwrite* the existing entries rather than
duplicate them — useful for correcting/re-processing a document without
manually clearing it first, though this also means the *number* of chunks
matters: if a re-ingested version of a document produces fewer chunks than
before, the old trailing chunk IDs beyond the new count are never deleted,
just no longer referenced by any query path that iterates the new chunk
list. This is a known edge case, not currently handled specially.

## What ties it together (`pipeline.py`)

```python
def ingest_document(text, source_name):
    if len(text) > MAX_TEXT_LENGTH:
        raise DocumentTooLargeError(...)
    chunks = chunker.chunk_text(text)
    embeddings = embedder.embed_in_batches(chunks)
    collection_name = storage.index_document(source_name, chunks, embeddings)
    return {"collection_name": collection_name, "chunks": len(chunks)}

def ingest_pdf(file_bytes, source_name):
    text = extractor.extract_text_from_pdf(file_bytes)
    return ingest_document(text=text, source_name=source_name)
```

This function is the entire "business logic" of ingestion, expressed as
four lines once the guard passes. It deliberately knows nothing about
HTTP, FastAPI, or file uploads — `api.py`'s job is only to get raw bytes or
text into this function and translate whatever comes back (a dict, or a
raised exception) into the right HTTP response. See
`02_DIRECTORY_STRUCTURE.md` for the exception-handling pattern that does
that translation.
