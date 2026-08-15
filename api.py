import ollama
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import json

from ingestion import chunker
from ingestion import embedder
from ingestion import extractor
from ingestion import storage
from ingestion import pipeline

from fastapi.responses import StreamingResponse, FileResponse

app = FastAPI(title="Local RAG Application API")




class QueryRequest(BaseModel):
    question: str
    source_collection: str
    llm_model: str


class IngestRequest(BaseModel):
    text_content: str
    source_name: str

class BenchmarkConfig(BaseModel):
    ingestion_enabled: bool


@app.post("/ingest")
async def ingest_raw_text(payload: IngestRequest):
    """API endpoint to chunk, embed, and store raw text data into ChromaDB."""
    try:

        result = pipeline.ingest_document(
            text=payload.text_content,
            source_name=payload.source_name,

            )
        return {"status": "success", 
                "message": (
                    f"Ingested {result['chunks']} chunks "
                    f"into collection '{result['collection_name']} 'successfully."
                )
                }
    except HTTPException:
        
        raise
    except pipeline.DocumentTooLargeError as e:
        raise HTTPException(status_code=413, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/ask")
async def ask_rag(payload: QueryRequest):
    """API endpoint to query a specific document's collection and stream the answer."""
    try:
        query_response = ollama.embed(
            model=embedder.EMBEDDING_MODEL, 
            input=payload.question)
        query_embedding = query_response["embeddings"][0]

        try:
            doc_collection = storage.get_collection(
                payload.source_collection)
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



@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...), source_name: str = None):
    """API endpoint to extract text from an uploaded PDF, batch-embed, and bulk-store it."""
    try:
        if source_name is None:
            source_name = file.filename

        file_bytes = await file.read()
        
        result = pipeline.ingest_pdf(
            file_bytes= file_bytes,
            source_name = source_name,
)
        
        return {"status": "success", 
                "message": (
                    f"Ingested {result['chunks']} chunks "
                    f"into collection '{result['collection_name']}' successfully."
                    )
                    }
    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/benchmark/config")
async def set_benchmark_config(payload: BenchmarkConfig):
    """Toggles ingestion benchmark logging on/off globally. Every ingestion
    call (UI, curl, bulk_ingest.py) respects this immediately - no need to
    change any caller."""
    pipeline.set_benchmark_enabled(payload.ingestion_enabled)
    return {"status": "success", "ingestion_enabled": payload.ingestion_enabled}
 
 
@app.get("/benchmark/config")
async def get_benchmark_config():
    """Returns the current ingestion benchmark on/off state."""
    return {"ingestion_enabled": pipeline.is_benchmark_enabled()}
 
 
@app.get("/benchmark/ingestion")
async def download_ingestion_benchmark():
    """Serves the current ingestion benchmark CSV, read fresh from disk on
    every request - never a cached/stale copy."""
    if not pipeline.LOG_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No ingestion benchmark data recorded yet."
        )
    return FileResponse(
        path=pipeline.LOG_PATH,
        media_type="text/csv",
        filename="ingestion_benchmark.csv",
    )
 


    
@app.get("/documents")
async def get_documents():
    """List all currently ingested document collection."""
    try:
        return {
            "documents": storage.list_documents()
    }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
    )
@app.delete("/knowledgebase")
async def clear_knowledgebase():
    """Deletes every document collection and recreates an empty knowledge base."""
    try:

        deleted = storage.clear_database()

        return {
            "status": "success",
            "message": f"Deleted {deleted} collection(s). Knowledge base is now empty."
    }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
    )


 
@app.get("/models")
async def list_models():
    """Lists all ollama models currently available on this machine."""
    try:
        response=ollama.list()
        model_names = [model.model for model in response.models]
        return {"models":model_names}
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))