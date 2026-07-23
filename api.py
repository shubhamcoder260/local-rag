import chromadb
import ollama
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import fitz
import re
from fastapi.responses import StreamingResponse
import json


app = FastAPI(title="Local RAG Application API")

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5-coder:7b"
DB_PATH = "./chroma_storage"
MAX_TEXT_LENGTH=500_000

client = chromadb.PersistentClient(path=DB_PATH)


class QueryRequest(BaseModel):
    question: str
    source_collection: str
    llm_model: str


class IngestRequest(BaseModel):
    text_content: str
    source_name: str

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

def chunk_text(text, max_chars=1500, overlap_chars=150):
    """
    Splits text into chunks bounded by character length, not word count.
    Falls back to hard character-slicing for any single 'word' that alone
    exceeds max_chars (e.g. CSV rows, DNA sequences, unbroken log lines).
    """

    words = text.split()
    chunks=[]
    current_chunk=[]
    current_length=0

    for word in words:
        if len(word) > max_chars:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk=[]
                current_length=0
            for i in range(0,len(word),max_chars - overlap_chars):
                chunks.append(word[i:i + max_chars])
            continue

        if current_length +len(word) +1>max_chars:
            chunks.append(" ".join(current_chunk))
            overlap_words=[]
            overlap_len=0
            for w in reversed(current_chunk):
                if overlap_len - len(w) > overlap_chars:
                    break
                overlap_words.insert(0,w)
                overlap_len += len(w)+1
            current_chunk=overlap_words
            current_length = overlap_len

        current_chunk.append(word)
        current_length += len(word) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks






@app.post("/ingest")
async def ingest_raw_text(payload: IngestRequest):
    """API endpoint to chunk, embed, and store raw text data into ChromaDB."""
    try:
        
        if len(payload.text_content) > MAX_TEXT_LENGTH: 
            raise HTTPException(status_code=413, detail=f"Text too large ({len(payload.text_content)} chars). Max supported is {MAX_TEXT_LENGTH} chars for now — try a smaller document.")


        chunks = chunk_text(payload.text_content)
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from the document."
            )
        collection_name=sanitize_collection_name(payload.source_name)
        doc_collection= client.get_or_create_collection(name=collection_name)

        embeddings=embed_in_batches(chunks)

        all_ids = [f"{payload.source_name}_chunks_{i}" for i in range(len(chunks))]
        all_metadatas = [{"source":payload.source_name} for _ in chunks]

        doc_collection.upsert(
            ids=all_ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=all_metadatas
        )
        return {"status": "success", "message": f"Ingested {len(chunks)} chunks into collection '{collection_name}' successfully."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/ask")
async def ask_rag(payload: QueryRequest):
    """API endpoint to query a specific document's collection and stream the answer."""
    try:
        query_response = ollama.embed(
            model=EMBEDDING_MODEL, 
            input=payload.question)
        query_embedding = query_response["embeddings"][0]

        try:
            doc_collection = client.get_collection(name=payload.source_collection)
        except Exception:
            raise HTTPException(status_code=404, detail=f"No document found with collection name '{payload.source_collection}'. Use GET /documents to see available options.")

        results = doc_collection.query(query_embeddings=[query_embedding], n_results=5)

        if not results or not results["documents"] or not results["documents"][0]:
            raise HTTPException(status_code=404, detail="No relevant context found in database.")

        retrieved_context = "\n\n".join(results["documents"][0])

        system_prompt = f"""You are a highly secure, private AI assistant.
Answer using ONLY the exact facts present in the context below. Do not add details, descriptions, dates, or explanations that are not explicitly written in the context, even if you know them from elsewhere. If the context only lists an item without details, state only that it is listed, without elaborating.

CONTEXT:
{retrieved_context}"""

        def stream_generator():
            metadata = {
                "question": payload.question,
                "source_collection": payload.source_collection,
                "context_used": retrieved_context
            }
            yield json.dumps(metadata) + "\n"

            response_stream = ollama.generate(
                model=payload.llm_model,
                prompt=payload.question,
                system=system_prompt,
                stream=True
            )
            for chunk in response_stream:
                yield chunk["response"]

        return StreamingResponse(stream_generator(), media_type="text/plain")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def extract_text_from_pdf(file_bytes):
    """Opens a PDF from raw bytes and extracts all text, page by page."""
    with fitz.open(stream=file_bytes,filetype="pdf") as doc:
        full_text=""
        for page in doc:
            full_text+= page.get_text()
    return full_text


@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...), source_name: str = None):
    """API endpoint to extract text from an uploaded PDF, batch-embed, and bulk-store it."""
    try:
        if source_name is None:
            source_name = file.filename

        file_bytes = await file.read()
        extracted_text = extract_text_from_pdf(file_bytes)


        if len(extracted_text) > MAX_TEXT_LENGTH:
            raise HTTPException(
                status_code=413,
                 detail=(
                    f"PDF too large ({len(extracted_text)} chars). "
                    f"Maximum supported is {MAX_TEXT_LENGTH} characters."
        )
    )

        chunks = chunk_text(extracted_text)
        if not chunks:
            raise HTTPException(
                            status_code=400,
                            detail="No text could be extracted from the document."
                        )
        collection_name = sanitize_collection_name(source_name)
        doc_collection = client.get_or_create_collection(name=collection_name)

        embeddings = embed_in_batches(chunks)

        all_ids = [f"{source_name}_chunk_{i}" for i in range(len(chunks))]
        all_metadatas = [{"source": source_name} for _ in chunks]

        doc_collection.upsert(
            ids=all_ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=all_metadatas
        )
        
        

        return {"status": "success", "message": f"Ingested {len(chunks)} chunks into collection '{collection_name}' successfully."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    



def sanitize_collection_name(name):
    """Converts any filename/string into a safe chromaDB collection name."""

    name=name.lower()
    name=re.sub(r"[^a-z0-9_-]","_",name)
    name= name.strip("_-")
    if len(name) < 3:
        name=name + "_doc"
    return name[:63]

@app.get("/documents")
async def list_documents():
    """List all currently ingested document collection."""
    try:
        collections=client.list_collections()
        names = [c.name for c in collections]

        return {"documents": names}
    except Exception as e:
        raise HTTPException(status_code=500, detail = str(e))

@app.delete("/knowledgebase")
async def clear_knowleedgebase():
    """Deletes every document collection and recreates an empty knowledge base."""

    try:
        collections = client.list_collections()

        deleted=0

        for collection in collections:
            client.delete_collection(collection.name)
            deleted+=1

        return {
            "status": "success",
            "message": f"Deleted{deleted} collection(s). Knowledge base is now empty."
        }
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))

@app.get("/models")
async def list_models():
    """Lists all ollama models currently available on this machine."""
    try:
        response=ollama.list()
        model_names = [model.model for model in response.models]
        return {"models":model_names}
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))