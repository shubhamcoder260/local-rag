from ingestion import chunker
from ingestion import embedder
from ingestion import extractor
from ingestion import storage


def ingest_document(text: str,source_name: str):        #text
    """
    Core ingestion pipeline.

    Takes plain text and indexes it into the knowledge base.
    """

    chunks = chunker.chunk_text(text)               #chunk

    embeddings = embedder.embed_in_batches(chunks)  #embed

    collection_name = storage.index_document(       #store
        source_name,
        chunks,
        embeddings,

    )

    return {
        "collection_name" : collection_name,
        "chunks": len(chunks),
    }


def ingest_pdf(file_bytes:bytes, source_name: str):
    """
    PDF ingestion pipeline.

    Extracts text from a PDF and passes it to the generic
    document ingestion pipeline.
    """

    text = extractor.extract_text_from_pdf(file_bytes)

    return ingest_document(
        text=text,
        source_name=source_name,
    )