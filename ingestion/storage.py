import chromadb
import re

VERSION = "chromadb-v1"
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

def index_document(source_name: str, chunks: list, embeddings: list):
    """
    Stores an embedded document into ChromaDB.
    """

    collection_name = sanitize_collection_name(source_name)
    collection = get_or_create_collection(source_name)

    all_ids = [
        f"{collection_name}_chunk_{i}"
        for i in range(len(chunks))
    ]

    all_metadatas = [
        {"source": source_name}
        for _ in chunks
    ]

    collection.upsert(
        ids=all_ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=all_metadatas,
    )

    return collection_name

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