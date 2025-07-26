from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import asyncio
import logging

from rag_main import BanglaRAGSystem

app = FastAPI()
logger = logging.getLogger("ragagent")

# Path to knowledge base (PDF or text file)
PDF_PATH = "HSC26-Bangla1st-Paper.pdf"
EMBEDDING_TYPE = "bangla_sbert"

# Instantiate the RAG system (not initialized yet)
rag = BanglaRAGSystem(pdf_path=PDF_PATH, embedding_type=EMBEDDING_TYPE)

class QueryRequest(BaseModel):
    session_id: str = None
    query: str

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing RAGagent knowledge base...")
    try:
        await rag.initialize()
        logger.info("RAGagent is ready.")
    except Exception as e:
        logger.error(f"Failed to initialize RAGagent: {e}")

@app.get("/status")
async def status():
    return {"status": "ready" if rag.is_initialized else "not_ready"}

@app.post("/chat")
async def chat_endpoint(body: QueryRequest):
    if not rag.is_initialized:
        return JSONResponse({"status": "not_ready", "msg": "RAG system is still loading. Please retry."}, status_code=503)
    sid = body.session_id or f"session_{id(body)}"
    result = await rag.query(query=body.query, session_id=sid, use_memory=True, max_results=5)
    return result

@app.get("/sessions")
def get_sessions():
    # Returns all session IDs
    return {"sessions": list(rag.memory_manager.conversations.keys())}

@app.get("/session/{session_id}")
def get_session_history(session_id: str):
    if session_id not in rag.memory_manager.conversations:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"history": rag.get_session_history(session_id)}

@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    rag.clear_session(session_id)
    return {"msg": f"Session {session_id} deleted."}

@app.delete("/sessions")
def delete_all_sessions():
    rag.memory_manager.conversations.clear()
    return {"msg": "All sessions deleted."}

if __name__ == "__main__":
    uvicorn.run("test_redis:app", host="0.0.0.0", port=8000, reload=True)