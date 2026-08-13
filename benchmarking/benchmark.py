"""
benchmark.py

Times each ingestion stage independently (extract -> chunk -> embed -> store)
so bottlenecks are visible instead of hidden behind one end-to-end number.
"""

import statistics
import sys
import time
from pathlib import Path

from ingestion import chunker, embedder, extractor, storage


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start


def benchmark_pdf(pdf_path: str, source_name: str | None = None) -> dict:
    source_name = source_name or Path(pdf_path).stem
    file_bytes = Path(pdf_path).read_bytes()

    r = {"source": source_name, "file_size_kb": round(len(file_bytes) / 1024, 1)}

    text, r["extract_sec"] = timed(extractor.extract_text_from_pdf, file_bytes)
    r["text_chars"] = len(text)

    chunks, r["chunk_sec"] = timed(chunker.chunk_text, text)
    r["chunk_count"] = len(chunks)
    r["avg_chunk_chars"] = round(statistics.mean(len(c) for c in chunks), 1) if chunks else 0

    embeddings, r["embed_sec"] = timed(embedder.embed_in_batches, chunks)
    r["chunks_per_sec_embed"] = round(len(chunks) / r["embed_sec"], 2) if r["embed_sec"] > 0 else None

    # sanity check flagged in code review: nothing upstream guarantees this
    assert len(embeddings) == len(chunks), (
        f"Alignment broken: {len(chunks)} chunks vs {len(embeddings)} embeddings"
    )

    r["collection_name"], r["store_sec"] = timed(
        storage.index_document, source_name, chunks, embeddings
    )

    for k in ("extract_sec", "chunk_sec", "embed_sec", "store_sec"):
        r[k] = round(r[k], 3)
    r["total_sec"] = round(sum(r[k] for k in ("extract_sec", "chunk_sec", "embed_sec", "store_sec")), 3)

    return r


def print_report(r: dict):
    print(f"\n=== {r['source']} ({r['file_size_kb']} KB) ===")
    print(f"extract  {r['extract_sec']:>7.3f}s   {r['text_chars']} chars")
    print(f"chunk    {r['chunk_sec']:>7.3f}s   {r['chunk_count']} chunks, avg {r['avg_chunk_chars']} chars")
    print(f"embed    {r['embed_sec']:>7.3f}s   {r['chunks_per_sec_embed']} chunks/sec")
    print(f"store    {r['store_sec']:>7.3f}s   -> '{r['collection_name']}'")
    print(f"TOTAL    {r['total_sec']:>7.3f}s")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python benchmark.py <path_to_pdf> [source_name]")
        sys.exit(1)

    report = benchmark_pdf(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    print_report(report)
