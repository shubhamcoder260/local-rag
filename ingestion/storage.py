import chromadb
import re


VERSION = "chromadb-v2"

DB_PATH = "./chroma_storage"
client = chromadb.PersistentClient(path=DB_PATH)


def sanitize_collection_name(name: str) -> str:
    """Converts any filename/string into a safe chromaDB collection name."""

    name=name.lower()
    name=re.sub(r"[^a-z0-9_-]","_",name)
    name= name.strip("_-")
    if len(name) < 3:
        name=name + "_doc"
    return name[:63]

def get_or_create_collection(source_name:str):
    collection_name = sanitize_collection_name(source_name)
    return client.get_or_create_collection(name=collection_name)

def get_collection(collection_name: str):

    return client.get_collection(name=collection_name)


def _chunk_index(chunk_id: str, collection_name: str) -> int:
    """Extracts the trailing integer from a '{collection_name}_chunk_{i}'
    id. Returns -1 for anything that doesn't match the expected pattern,
    so malformed/foreign ids are never mistaken for stale chunks."""
    prefix = f"{collection_name}_chunk_"
    if not chunk_id.startswith(prefix):
        return -1
    try:
        return int(chunk_id[len(prefix):])
    except ValueError:
        return -1


def index_document(source_name: str, chunks: list, embeddings: list):
    """
    Stores an embedded document into ChromaDB.

    Before writing, removes any chunk ids left over from a PREVIOUS,
    LARGER version of this same document. upsert() only overwrites ids in
    the new range (0..len(chunks)-1) - if a document shrinks between
    re-ingestions (e.g. 20 chunks -> 12 chunks), ids 12-19 from the old
    version would otherwise remain in the collection forever, orphaned and
    silently returnable in search results. This was proven with real data
    during Phase 3 benchmarking (count_match=False on a shrunk
    re-ingestion) before being fixed here.
    """

    collection_name = sanitize_collection_name(source_name)
    collection = get_or_create_collection(source_name)

    new_count = len(chunks)
    all_ids = [
        f"{collection_name}_chunk_{i}"
        for i in range(new_count)
    ]

    all_metadatas = [
        {"source": source_name}
        for _ in chunks
    ]

    existing = collection.get()
    stale_ids = [
        eid for eid in existing.get("ids", [])
        if _chunk_index(eid, collection_name) >= new_count
    ]
    if stale_ids:
        collection.delete(ids=stale_ids)

    collection.upsert(
        ids=all_ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=all_metadatas,
    )

    return collection_name


def verify_storage(collection_name: str, chunks: list, embeddings: list) -> dict:
    """
    Verifies a write actually landed correctly and is retrievable - not
    just that upsert() didn't raise an exception. Two checks:

    1. Count check: does the collection's total stored count match what we
       just tried to store? Now that index_document() cleans up stale ids
       before writing, this should read True even after a document shrinks
       between re-ingestions - this check remains in place as an ongoing
       regression guard, not just a one-time diagnostic.

    2. Round-trip check: query the collection using one of the embeddings
       we just stored, and confirm the SAME chunk text comes back as the
       top match - the strongest evidence storage is genuinely retrievable,
       not just written.
    """
    collection = get_collection(collection_name)
    stored_count = collection.count()
    expected_count = len(chunks)
    count_match = stored_count == expected_count

    roundtrip_match = False
    if chunks and embeddings:
        probe_embedding = embeddings[0]
        probe_chunk = chunks[0]
        result = collection.query(query_embeddings=[probe_embedding], n_results=1)
        documents = result.get("documents") if result else None
        if documents and documents[0]:
            roundtrip_match = documents[0][0] == probe_chunk

    return {
        "stored_count": stored_count,
        "expected_count": expected_count,
        "count_match": count_match,
        "roundtrip_match": roundtrip_match,
    }


def list_documents():
    """
    Returns the names of all indexed document collections.
    """
    collections=client.list_collections()
    return [collection.name for collection in collections]

def clear_database():
    """
    Deletes every collection from the Chroma database.

    Returns:
        int: Number of deleted collections.
    """
    collections= client.list_collections()
    deleted=0

    for collection in collections:
        client.delete_collection(collection.name)
        deleted += 1

    return deleted