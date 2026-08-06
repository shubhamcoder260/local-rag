import ollama

EMBEDDING_MODEL = "nomic-embed-text"

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
