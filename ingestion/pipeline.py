import csv
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import psutil

from ingestion import chunker
from ingestion import embedder
from ingestion import extractor
from ingestion import storage


MAX_TEXT_LENGTH = 500_000

LOG_PATH = Path(os.environ.get("BENCHMARK_LOG_PATH", "logs/benchmark_results.csv"))

# Superset of columns across all stages. Not every stage populates every
# column (e.g. only "chunk" rows have overlap stats) - csv.DictWriter fills
# any column missing from a given row with "" automatically, so this is
# safe to keep as one flat schema rather than per-stage files.
LOG_FIELDS = [
    "timestamp", "run_id", "source", "stage",
    "duration_sec", "cpu_percent", "ram_mb", "algorithm_version",
    # chunk-stage quality columns
    "avg_overlap_chars", "max_overlap_chars", "target_overlap_chars",
    # embed-stage quality columns
    "embedding_dimension", "zero_vector_count", "nan_count", "dimension_mismatch_count",
]

_process = psutil.Process(os.getpid())

ALGORITHM_VERSIONS = {
    "extract": extractor.VERSION,
    "chunk": chunker.VERSION,
    "embed": embedder.VERSION,
    "store": storage.VERSION,
}


class DocumentTooLargeError(Exception):
    """Raised when text submitted for ingestion exceeds MAX_TEXT_LENGTH.

    Deliberately a plain exception, not an HTTPException - this module has no FastAPI dependency, so the API layer is responsible for
    translating this into the appropriate HTTP response."""

    pass


def _sample_resources():
    return {
        "cpu_percent": _process.cpu_percent(interval=None),
        "ram_mb": round(_process.memory_info().rss / (1024 * 1024), 1),
    }


def _log_stage(run_id: str, source: str, stage: str, duration_sec: float, **extra) -> dict:
    """
    Appends one row per stage (long format, per Phase 0 schema). `extra`
    carries stage-specific quality metrics (e.g. overlap stats for "chunk",
    embedding sanity checks for "embed") - merged into the row, left blank
    for stages that don't produce them.
    """
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "source": source,
        "stage": stage,
        "duration_sec": round(duration_sec, 3),
        "algorithm_version": ALGORITHM_VERSIONS[stage],
        **_sample_resources(),
        **extra,
    }
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        file_exists = LOG_PATH.exists()
        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        print(f"[benchmark logging] failed to write row: {e}")
    return row


def ingest_document(text: str, source_name: str, run_id: str = None):
    """
    Core ingestion pipeline (blocking). Logs one CSV row per stage, with
    quality metrics attached to the chunk and embed rows.
    """
    run_id = run_id or str(uuid.uuid4())

    t0 = time.perf_counter()
    chunks = chunker.chunk_text(text)
    overlap_stats = chunker.measure_overlap_accuracy(chunks)
    _log_stage(
        run_id, source_name, "chunk", time.perf_counter() - t0,
        avg_overlap_chars=overlap_stats["avg_overlap_chars"],
        max_overlap_chars=overlap_stats["max_overlap_chars"],
        target_overlap_chars=overlap_stats["target_overlap_chars"],
    )

    t1 = time.perf_counter()
    embeddings = embedder.embed_in_batches(chunks)
    quality = embedder.check_embedding_quality(embeddings)
    _log_stage(
        run_id, source_name, "embed", time.perf_counter() - t1,
        embedding_dimension=quality["embedding_dimension"],
        zero_vector_count=quality["zero_vector_count"],
        nan_count=quality["nan_count"],
        dimension_mismatch_count=quality["dimension_mismatch_count"],
    )

    t2 = time.perf_counter()
    collection_name = storage.index_document(source_name, chunks, embeddings)
    _log_stage(run_id, source_name, "store", time.perf_counter() - t2)

    return {"collection_name": collection_name, "chunks": len(chunks), "run_id": run_id}


def ingest_pdf(file_bytes: bytes, source_name: str):
    """PDF ingestion pipeline (blocking)."""
    run_id = str(uuid.uuid4())

    t0 = time.perf_counter()
    text = extractor.extract_text_from_pdf(file_bytes)
    _log_stage(run_id, source_name, "extract", time.perf_counter() - t0)

    return ingest_document(text=text, source_name=source_name, run_id=run_id)


def ingest_pdf_live(file_bytes: bytes, source_name: str):
    """Generator version of ingest_pdf - yields each stage's row as it completes."""
    run_id = str(uuid.uuid4())

    t0 = time.perf_counter()
    text = extractor.extract_text_from_pdf(file_bytes)
    row = _log_stage(run_id, source_name, "extract", time.perf_counter() - t0)
    yield {**row, "progress": "1/4"}

    t1 = time.perf_counter()
    chunks = chunker.chunk_text(text)
    overlap_stats = chunker.measure_overlap_accuracy(chunks)
    row = _log_stage(
        run_id, source_name, "chunk", time.perf_counter() - t1,
        avg_overlap_chars=overlap_stats["avg_overlap_chars"],
        max_overlap_chars=overlap_stats["max_overlap_chars"],
        target_overlap_chars=overlap_stats["target_overlap_chars"],
    )
    yield {**row, "chunk_count": len(chunks), "progress": "2/4"}

    t2 = time.perf_counter()
    embeddings = embedder.embed_in_batches(chunks)
    quality = embedder.check_embedding_quality(embeddings)
    row = _log_stage(
        run_id, source_name, "embed", time.perf_counter() - t2,
        embedding_dimension=quality["embedding_dimension"],
        zero_vector_count=quality["zero_vector_count"],
        nan_count=quality["nan_count"],
        dimension_mismatch_count=quality["dimension_mismatch_count"],
    )
    yield {**row, "progress": "3/4"}

    t3 = time.perf_counter()
    collection_name = storage.index_document(source_name, chunks, embeddings)
    row = _log_stage(run_id, source_name, "store", time.perf_counter() - t3)
    yield {**row, "progress": "4/4"}

    yield {
        "stage": "done",
        "run_id": run_id,
        "collection_name": collection_name,
        "chunks": len(chunks),
    }