import ollama
import math
EMBEDDING_MODEL = "nomic-embed-text"
VERSION = "nomic-embed-text-v1"

def embed_in_batches(chunks, batch_size=50):
    """Embeds a list of text chunks in batches, drastically reducing the number of separate calls to Ollama compared to one-at-a-time embedding."""
    all_embeddings=[]
    for i in range(0, len(chunks),batch_size):
        batch = chunks[i:i +batch_size]
        response = ollama.embed(
            model=EMBEDDING_MODEL, 
            input=batch)
        all_embeddings.extend(response["embeddings"])
        print(f"Embedding batch {i// batch_size + 1} ({min(i+batch_size, len(chunks))}/{len(chunks)} chunks)")
    return all_embeddings


def check_embedding_quality(embeddings):
    """
    Sanity-checks a batch of embeddings for structural correctness. Catches
    failure modes that return a technically-valid-shaped result but are
    silently broken - zero vectors, NaNs, or inconsistent dimensionality -
    none of which raise an exception on their own, but all of which would
    corrupt search results with no visible error (see: alignment risk
    identified during code review).
    """
    if not embeddings:
        return {
            "embedding_dimension": 0,
            "zero_vector_count": 0,
            "nan_count": 0,
            "dimension_mismatch_count": 0,
        }
 
    dims = [len(e) for e in embeddings]
    expected_dim = dims[0]
    mismatch_count = sum(1 for d in dims if d != expected_dim)
 
    zero_count = 0
    nan_count = 0
    for e in embeddings:
        if any(math.isnan(v) for v in e):
            nan_count += 1
        elif all(v == 0 for v in e):
            zero_count += 1
 
    return {
        "embedding_dimension": expected_dim,
        "zero_vector_count": zero_count,
        "nan_count": nan_count,
        "dimension_mismatch_count": mismatch_count,
    }
 
