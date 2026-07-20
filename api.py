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

client = chromadb.PersistentClient(path=DB_PATH)


class QueryRequest(BaseModel):
    question: str
    source_collection: str
    llm_model: str


class IngestRequest(BaseModel):
    text_content: str
    source_name: str


def chunk_text(text, chunk_size=500, overlap=50):
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


@app.post("/ingest")
async def ingest_raw_text(payload: IngestRequest):
    """API endpoint to chunk, embed, and store raw text data into ChromaDB."""
    try:
        chunks = chunk_text(payload.text_content)
        collection_name=sanitize_collection_name(payload.source_name)
        doc_collection= client.get_or_create_collection(name=collection_name)
        for i, current_chunk in enumerate(chunks):
            response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=current_chunk)
            embedding = response["embedding"]
            chunk_id = f"{payload.source_name}_chunk_{i}"
            doc_collection.upsert(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[current_chunk],
                metadatas=[{"source": payload.source_name}]
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
        query_response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=payload.question)
        query_embedding = query_response["embedding"]

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
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    full_text=""
    for page in doc:
        full_text+= page.get_text()
    doc.close()
    return full_text

@app.post("/ingest/pdf") 
#whenever a client sends a POST request to /ingest/pdf , execute the function below#

async def ingest_pdf(file: UploadFile = File(...), source_name:str=None):
    """API endpoint to extract text from an uploaded PDF,vhunk it , embed it, and store it."""
    try:
        if source_name is None:
            source_name = file.filename
        
        file_bytes=await file.read()
        extracted_text = extract_text_from_pdf(file_bytes)

        chunks= chunk_text(extracted_text)
        collection_name= sanitize_collection_name(source_name)
        doc_collection= client.get_or_create_collection(name=collection_name)

        for i, current_chunk in enumerate(chunks):
            response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=current_chunk)
            embedding=response["embedding"]
            chunk_id = f"{source_name}_chunk_{i}"
            doc_collection.upsert(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[current_chunk],
                metadatas=[{"source": source_name}]
            )
        return {"status":"sucecss","message":f"Ingested {len(chunks)} chunks into collection '{collection_name}' successfully."}
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


@app.get("/models")
async def list_models():
    """Lists all ollama models currently available on this machine."""
    try:
        response=ollama.list()
        model_names = [model["model"] for model in response["models"]]
        return {"models":model_names}
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))